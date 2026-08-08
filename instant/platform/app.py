"""Authenticated OpenAI gateway with durable, receipt-backed telemetry."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..protocol import receipts
from ..protocol.epistula import (
    EpistulaError,
    ReplayGuard,
    generate_headers,
    verify_headers,
)
from ..protocol.keys import Signer
from ..protocol.schemas import ChatCompletionRequest
from ..protocol.ss58 import is_valid
from .state import PlatformState

_FORWARDED_RESPONSE_HEADERS = {
    "x-instant-receipt",
    "x-instant-receipt-sig",
    "x-instant-request-id",
    "cache-control",
    "x-accel-buffering",
}


@dataclass(slots=True)
class PlatformContext:
    signer: Signer
    miner_url: str
    miner_ss58: str
    http: httpx.AsyncClient
    state: PlatformState
    api_key_sha256: str
    validator_hotkeys: frozenset[str]
    miner_uid: int = 0
    stats_window_s: int = 3600
    validator_replay: ReplayGuard = field(default_factory=ReplayGuard)

    def __post_init__(self) -> None:
        self.miner_url = self.miner_url.rstrip("/")
        self.api_key_sha256 = self.api_key_sha256.lower()
        if not is_valid(self.miner_ss58):
            raise ValueError("miner_ss58 must be a valid Bittensor SS58 address")
        if len(self.api_key_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.api_key_sha256
        ):
            raise ValueError("api_key_sha256 must be a SHA-256 hex digest")
        if self.miner_uid < 0:
            raise ValueError("miner_uid must be non-negative")
        if self.stats_window_s < 1:
            raise ValueError("stats_window_s must be positive")


def create_app(ctx: PlatformContext) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await ctx.http.aclose()
            ctx.state.close()

    app = FastAPI(
        title="Instant platform",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.ctx = ctx

    async def health_payload() -> dict:
        try:
            response = await ctx.http.get(f"{ctx.miner_url}/health")
            reachable = response.status_code == 200
            payload = response.json() if reachable else None
            miner = payload if isinstance(payload, dict) else None
            reachable = miner is not None
        except (httpx.HTTPError, ValueError):
            miner = None
            reachable = False
        ready = bool(miner and miner.get("ready"))
        return {
            "status": "ok" if reachable and ready else "degraded",
            "ready": ready,
            "miner_url": ctx.miner_url,
            "miner_ss58": ctx.miner_ss58,
            "miner": miner,
        }

    @app.get("/health")
    async def health() -> dict:
        return await health_payload()

    @app.get("/livez")
    async def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        payload = await health_payload()
        return JSONResponse(payload, status_code=200 if payload["ready"] else 503)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if not _bearer_ok(ctx, request):
            return _bearer_error()
        try:
            response = await ctx.http.get(f"{ctx.miner_url}/manifest")
            response.raise_for_status()
            manifest = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return _error(502, "miner_unavailable", str(exc))
        if not isinstance(manifest, dict):
            return _error(502, "invalid_miner_manifest", "expected a JSON object")
        if manifest.get("hotkey") != ctx.miner_ss58:
            return _error(
                502,
                "miner_identity_mismatch",
                f"expected {ctx.miner_ss58}, got {manifest.get('hotkey')}",
            )
        if not isinstance(manifest.get("model_id"), str):
            return _error(502, "invalid_miner_manifest", "model_id is missing")
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": manifest.get("model_id"),
                        "object": "model",
                        "owned_by": manifest.get("hotkey"),
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        # Authenticate before reading, validating, signing, or forwarding a
        # potentially large request body.
        if not _bearer_ok(ctx, request):
            return _bearer_error()

        raw = await request.body()
        try:
            parsed = ChatCompletionRequest(**json.loads(raw))
        except (ValueError, TypeError) as exc:
            return _error(400, "bad_request", str(exc))

        request_id = str(uuid.uuid4())
        started_ms = _now_ms()
        started_perf = time.perf_counter()
        # Every PlatformState call commits to SQLite and is therefore blocking.
        # Measured on the OCI platform host, one committed write costs 2.5ms with
        # the connection's default synchronous=FULL, against 0.08ms for a thread
        # hop -- so calling these inline stalls the loop relaying every other
        # in-flight stream. Offload each one.
        await run_in_threadpool(
            ctx.state.begin,
            request_id=request_id,
            miner_hotkey=ctx.miner_ss58,
            started_ms=started_ms,
            stream=parsed.stream,
        )
        headers = {
            "Content-Type": "application/json",
            **generate_headers(
                ctx.signer,
                raw,
                signed_for=ctx.miner_ss58,
                request_uuid=request_id,
            ),
        }
        url = f"{ctx.miner_url}/v1/chat/completions"
        if not parsed.stream:
            try:
                upstream = await ctx.http.post(url, content=raw, headers=headers)
            except httpx.HTTPError as exc:
                elapsed_ms = _elapsed_ms(started_perf)
                await run_in_threadpool(
                    ctx.state.finish,
                    request_id=request_id,
                    finished_ms=_now_ms(),
                    status_code=None,
                    success=False,
                    total_ms=elapsed_ms,
                    error=f"miner unavailable: {exc}",
                )
                return _error(502, "miner_unavailable", str(exc))

            elapsed_ms = _elapsed_ms(started_perf)
            evidence = _verify_header_receipt(
                ctx, upstream, request_id=request_id, request_body=raw
            )
            success = 200 <= upstream.status_code < 300 and evidence.verified
            receipt = evidence.signed_receipt if evidence.verified else None
            await run_in_threadpool(
                ctx.state.finish,
                request_id=request_id,
                finished_ms=_now_ms(),
                status_code=upstream.status_code,
                success=success,
                clean_reject=upstream.status_code in {429, 503},
                ttft_ms=elapsed_ms if success else None,
                total_ms=elapsed_ms,
                prompt_tokens=receipt.receipt.prompt_tokens if receipt else 0,
                completion_tokens=receipt.receipt.completion_tokens if receipt else 0,
                signed_receipt=evidence.signed_receipt,
                receipt_seen=evidence.seen,
                receipt_verified=evidence.verified,
                error=None if success else evidence.error or f"miner HTTP {upstream.status_code}",
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_response_headers(upstream),
                media_type=_media_type(upstream, "application/json"),
            )

        try:
            outgoing = ctx.http.build_request("POST", url, content=raw, headers=headers)
            upstream = await ctx.http.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            elapsed_ms = _elapsed_ms(started_perf)
            await run_in_threadpool(
                ctx.state.finish,
                request_id=request_id,
                finished_ms=_now_ms(),
                status_code=None,
                success=False,
                total_ms=elapsed_ms,
                error=f"miner unavailable: {exc}",
            )
            return _error(502, "miner_unavailable", str(exc))
        if upstream.status_code >= 400:
            body = await upstream.aread()
            response_headers = _response_headers(upstream)
            media_type = _media_type(upstream, "application/json")
            await upstream.aclose()
            elapsed_ms = _elapsed_ms(started_perf)
            await run_in_threadpool(
                ctx.state.finish,
                request_id=request_id,
                finished_ms=_now_ms(),
                status_code=upstream.status_code,
                success=False,
                clean_reject=upstream.status_code in {429, 503},
                total_ms=elapsed_ms,
                error=f"miner HTTP {upstream.status_code}",
            )
            return Response(
                content=body,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=media_type,
            )

        observer = _StreamObserver(started_perf=started_perf)

        async def relay() -> AsyncIterator[bytes]:
            relay_error: str | None = None
            try:
                async for chunk in upstream.aiter_raw():
                    observer.feed(chunk)
                    yield chunk
            except httpx.HTTPError as exc:
                relay_error = f"miner stream failed: {exc}"
                raise
            finally:
                try:
                    await upstream.aclose()
                finally:
                    finished_ms = _now_ms()
                    total_ms = _elapsed_ms(started_perf)
                    evidence = observer.verify(
                        ctx,
                        request_id=request_id,
                        request_body=raw,
                    )
                    success = (
                        relay_error is None
                        and not observer.error_event
                        and evidence.verified
                    )
                    receipt = evidence.signed_receipt if evidence.verified else None
                    ttft_ms = observer.ttft_ms(total_ms) if success else None
                    tps_milli = (
                        _observed_tps_milli(
                            receipt.receipt.completion_tokens,
                            total_ms=total_ms,
                            ttft_ms=ttft_ms or 0,
                        )
                        if receipt and ttft_ms is not None
                        else None
                    )
                    await run_in_threadpool(
                        ctx.state.finish,
                        request_id=request_id,
                        finished_ms=finished_ms,
                        status_code=upstream.status_code,
                        success=success,
                        ttft_ms=ttft_ms,
                        total_ms=total_ms,
                        tps_milli=tps_milli,
                        prompt_tokens=receipt.receipt.prompt_tokens if receipt else 0,
                        completion_tokens=(
                            receipt.receipt.completion_tokens if receipt else 0
                        ),
                        signed_receipt=evidence.signed_receipt,
                        receipt_seen=evidence.seen,
                        receipt_verified=evidence.verified,
                        error=(
                            None
                            if success
                            else relay_error
                            or ("miner emitted an error event" if observer.error_event else None)
                            or evidence.error
                        ),
                    )

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=_media_type(upstream, "text/event-stream"),
        )

    @app.get("/validator/v1/stats")
    async def validator_stats(request: Request) -> Response:
        raw = await request.body()
        try:
            verified = verify_headers(
                dict(request.headers),
                raw,
                allowed_signers=ctx.validator_hotkeys,
                expected_signed_for=ctx.signer.ss58_address,
                replay_guard=ctx.validator_replay,
            )
            # verify_headers permits an absent Signed-For for generic Epistula
            # compatibility. This privileged endpoint requires an explicit
            # recipient so a valid request cannot be replayed at another host.
            if verified.signed_for != ctx.signer.ss58_address:
                raise EpistulaError("request must be signed for this platform")
        except EpistulaError as exc:
            return _error(401, "unauthorized", exc.reason)

        generated_at_ms = _now_ms()
        # The costliest state call by far: it holds the lock across every row in
        # the window, parses each stored receipt, and verifies signatures to build
        # a merkle root. Unlike begin/finish this is O(window), so inline it would
        # stall streaming relays for as long as the window is large.
        stats = await run_in_threadpool(
            ctx.state.stats,
            miner_hotkey=ctx.miner_ss58,
            miner_uid=ctx.miner_uid,
            window_start_ms=generated_at_ms - ctx.stats_window_s * 1000,
            window_end_ms=generated_at_ms,
            generated_at_ms=generated_at_ms,
        )
        body = json.dumps(
            stats.model_dump(), separators=(",", ":"), sort_keys=True
        ).encode()
        response_headers = generate_headers(
            ctx.signer, body, signed_for=verified.signed_by
        )
        return Response(
            content=body, media_type="application/json", headers=response_headers
        )

    return app


@dataclass(slots=True)
class _ReceiptEvidence:
    signed_receipt: receipts.SignedReceipt | None = None
    seen: bool = False
    verified: bool = False
    error: str | None = None


def _verify_header_receipt(
    ctx: PlatformContext,
    response: httpx.Response,
    *,
    request_id: str,
    request_body: bytes,
) -> _ReceiptEvidence:
    raw_receipt = response.headers.get(receipts.H_RECEIPT)
    signature = response.headers.get(receipts.H_RECEIPT_SIG)
    seen = raw_receipt is not None or signature is not None
    if raw_receipt is None or signature is None:
        return _ReceiptEvidence(seen=seen, error="miner receipt is missing")
    try:
        signed = receipts.SignedReceipt(
            receipt=receipts.Receipt.from_payload(json.loads(raw_receipt)),
            signature=signature,
        )
        _verify_receipt(
            ctx,
            signed,
            request_id=request_id,
            request_body=request_body,
            response_body=response.content,
        )
        return _ReceiptEvidence(signed_receipt=signed, seen=True, verified=True)
    except (KeyError, TypeError, ValueError, receipts.ReceiptError) as exc:
        return _ReceiptEvidence(seen=True, error=f"invalid miner receipt: {exc}")


class _StreamObserver:
    def __init__(self, *, started_perf: float) -> None:
        self.started_perf = started_perf
        self.first_content_perf: float | None = None
        self.assembled = bytearray()
        self.buffer = bytearray()
        self.receipt_payload: dict[str, Any] | None = None
        self.receipt_seen = False
        self.error_event = False

    def feed(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        while True:
            boundary = _frame_boundary(self.buffer)
            if boundary is None:
                return
            index, width = boundary
            frame = bytes(self.buffer[:index])
            del self.buffer[: index + width]
            self._frame(frame)

    def _frame(self, raw: bytes) -> None:
        lines = raw.replace(b"\r\n", b"\n").split(b"\n")
        event = next(
            (line[6:].strip().decode() for line in lines if line.startswith(b"event:")),
            None,
        )
        data_lines = [line[5:].strip() for line in lines if line.startswith(b"data:")]
        if not data_lines:
            return
        data = b"\n".join(data_lines)
        if event is not None:
            if event == receipts.SSE_RECEIPT_EVENT:
                self.receipt_seen = True
                try:
                    payload = json.loads(data)
                    self.receipt_payload = payload if isinstance(payload, dict) else None
                except ValueError:
                    self.receipt_payload = None
            elif event == "error":
                self.error_event = True
            return
        if data == b"[DONE]":
            return
        self.assembled.extend(data)
        if self.first_content_perf is None and _has_content(data):
            self.first_content_perf = time.perf_counter()

    def ttft_ms(self, total_ms: int) -> int:
        if self.first_content_perf is None:
            return total_ms
        return max(0, int((self.first_content_perf - self.started_perf) * 1000))

    def verify(
        self,
        ctx: PlatformContext,
        *,
        request_id: str,
        request_body: bytes,
    ) -> _ReceiptEvidence:
        if not self.receipt_seen or self.receipt_payload is None:
            detail = "miner emitted an error event" if self.error_event else "receipt missing"
            return _ReceiptEvidence(seen=self.receipt_seen, error=detail)
        try:
            signed = receipts.SignedReceipt.from_payload(self.receipt_payload)
            _verify_receipt(
                ctx,
                signed,
                request_id=request_id,
                request_body=request_body,
                response_body=bytes(self.assembled),
            )
            return _ReceiptEvidence(signed_receipt=signed, seen=True, verified=True)
        except (KeyError, TypeError, ValueError, receipts.ReceiptError) as exc:
            return _ReceiptEvidence(seen=True, error=f"invalid miner receipt: {exc}")


def _verify_receipt(
    ctx: PlatformContext,
    signed: receipts.SignedReceipt,
    *,
    request_id: str,
    request_body: bytes,
    response_body: bytes,
) -> None:
    receipts.verify(
        signed,
        expected_miner_hotkey=ctx.miner_ss58,
        request_body=request_body,
        response_body=response_body,
    )
    receipt = signed.receipt
    if receipt.request_id != request_id:
        raise receipts.ReceiptError("receipt request id does not match")
    if receipt.signer_of_request != ctx.signer.ss58_address:
        raise receipts.ReceiptError("receipt names a different request signer")


def _bearer_ok(ctx: PlatformContext, request: Request) -> bool:
    header = request.headers.get("authorization", "")
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return False
    digest = hashlib.sha256(token.encode()).hexdigest()
    return hmac.compare_digest(digest, ctx.api_key_sha256)


def _bearer_error() -> JSONResponse:
    return JSONResponse(
        {"error": "unauthorized", "detail": "a valid Bearer API key is required"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _frame_boundary(buffer: bytearray) -> tuple[int, int] | None:
    lf = buffer.find(b"\n\n")
    crlf = buffer.find(b"\r\n\r\n")
    choices = [(lf, 2), (crlf, 4)]
    valid = [choice for choice in choices if choice[0] >= 0]
    return min(valid, default=None, key=lambda choice: choice[0])


def _has_content(raw: bytes) -> bool:
    try:
        payload = json.loads(raw)
    except ValueError:
        return False
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
    return False


def _observed_tps_milli(tokens: int, *, total_ms: int, ttft_ms: int) -> int:
    generation_ms = max(1, total_ms - ttft_ms)
    return max(0, int(tokens)) * 1_000_000 // generation_ms


def _now_ms() -> int:
    return int(time.time() * 1000)


def _elapsed_ms(started_perf: float) -> int:
    return max(0, int((time.perf_counter() - started_perf) * 1000))


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() in _FORWARDED_RESPONSE_HEADERS
    }


def _media_type(response: httpx.Response, default: str) -> str:
    return response.headers.get("content-type", default).split(";", 1)[0]


def _error(status: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)
