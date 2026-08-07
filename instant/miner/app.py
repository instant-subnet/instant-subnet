"""The miner's HTTP surface (DESIGN.md §4).

``POST /v1/chat/completions``  Epistula-gated. Streams or completes, and
                              signs a receipt over what it served.
``GET  /health``              Unauthenticated. Up, ready, which model.
``GET  /capacity``            Unauthenticated. What it will accept now.
``GET  /manifest``            Unauthenticated. Model, weights, image, last
                              attestation.
``POST /attest``              Epistula-gated. Answers a fresh challenge.

The app is built by :func:`create_app` from an explicit
:class:`MinerContext`, with no globals and no import-time side effects.
That is what lets the test suite run the whole surface against a fake
upstream and a stub attestor, in-process, in milliseconds — and it is why
the endpoint tests below the line in ``tests/test_miner_app.py`` exercise
real HTTP rather than calling handler functions directly.

Three endpoints are deliberately unauthenticated. A validator needs to tell
"this miner is down" from "this miner rejected me", and if learning that
requires a valid signature then during the incident where the difference
matters, the two are indistinguishable. None of them reveal anything a
metagraph scrape does not already.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid as uuidlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..protocol import receipts
from ..protocol.attestation import NONCE_BYTES
from ..protocol.epistula import EpistulaError
from ..protocol.keys import Signer
from ..protocol.schemas import (
    AttestRequest,
    CapacityResponse,
    ChatCompletionRequest,
    HealthResponse,
    ManifestResponse,
)
from .attest import AttestationCache, AttestationUnavailable
from .auth import Authenticator
from .upstream import StreamOutcome, UpstreamError, VllmClient

log = logging.getLogger("instant.miner")

__version__ = "0.1.0"


@dataclass
class MinerContext:
    """Everything the handlers need, injected rather than imported."""

    signer: Signer
    auth: Authenticator
    vllm: VllmClient
    attestation: AttestationCache
    model_id: str
    max_model_len: int
    max_concurrent: int
    attestation_mode: str
    weights_digest: str = ""
    image_digest: str = ""
    started_at: float = field(default_factory=time.monotonic)
    in_flight: int = 0

    @property
    def uptime_s(self) -> int:
        return int(time.monotonic() - self.started_at)


def create_app(
    ctx: MinerContext,
    *,
    background: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Build the miner app.

    ``background`` is an optional coroutine factory started with the server
    and cancelled with it — in production that is the metagraph refresh loop.
    It is a parameter rather than an import so that the tests get an app with
    no background work in it and therefore no reason to be flaky.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task: asyncio.Task | None = None
        if background is not None:
            task = asyncio.create_task(background())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await ctx.vllm.aclose()

    app = FastAPI(
        title="Instant miner",
        version=__version__,
        docs_url=None,       # nothing to document for an unauthenticated visitor
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    app.include_router(_router(ctx))
    return app


def _router(ctx: MinerContext) -> APIRouter:
    router = APIRouter()

    # --- unauthenticated -------------------------------------------------

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        ready = await ctx.vllm.health()
        if not ready:
            status = "loading" if ctx.uptime_s < 600 else "degraded"
        elif ctx.auth.accept.is_stale:
            # Serving fine, but the accept-list has not refreshed. Saying
            # "ok" here would hide the failure that is about to cause
            # rejections.
            status = "degraded"
        else:
            status = "ok"
        return HealthResponse(
            status=status,  # type: ignore[arg-type]
            ready=ready,
            version=__version__,
            model_id=ctx.model_id,
            uptime_s=ctx.uptime_s,
            attestation_mode=ctx.attestation_mode,  # type: ignore[arg-type]
        )

    @router.get("/capacity", response_model=CapacityResponse)
    async def capacity() -> CapacityResponse:
        return CapacityResponse(
            max_concurrent=ctx.max_concurrent,
            in_flight=ctx.in_flight,
            queue_depth=max(0, ctx.in_flight - ctx.max_concurrent),
            max_model_len=ctx.max_model_len,
            accepting=ctx.in_flight < ctx.max_concurrent,
        )

    @router.get("/manifest", response_model=ManifestResponse)
    async def manifest() -> ManifestResponse:
        last = ctx.attestation.last
        return ManifestResponse(
            hotkey=ctx.signer.ss58_address,
            model_id=ctx.model_id,
            weights_digest=ctx.weights_digest or (last.weights_digest if last else ""),
            image_digest=ctx.image_digest or (last.image_digest if last else ""),
            max_model_len=ctx.max_model_len,
            attestation_id=last.digest() if last else None,
            attested_at_ms=ctx.attestation.last_generated_ms or None,
        )

    # --- authenticated ---------------------------------------------------

    @router.post("/attest")
    async def attest(request: Request) -> Response:
        raw = await request.body()
        try:
            verified = ctx.auth.verify(dict(request.headers), raw)
        except EpistulaError as exc:
            return _error(401, "unauthorized", exc.reason)

        try:
            parsed = AttestRequest(**json.loads(raw or b"{}"))
        except (ValueError, TypeError) as exc:
            return _error(400, "bad_request", str(exc))

        # The nonce must be the verifier's, and it must be full length. A
        # short nonce is a nonce someone could have precomputed against.
        try:
            nonce = bytes.fromhex(parsed.nonce.removeprefix("0x"))
        except ValueError:
            return _error(400, "bad_request", "nonce is not hex")
        if len(nonce) != NONCE_BYTES:
            return _error(
                400, "bad_request", f"nonce must be {NONCE_BYTES} bytes"
            )

        try:
            bundle = await ctx.attestation.get(
                nonce.hex(), force=parsed.force_refresh
            )
        except AttestationUnavailable as exc:
            # 503, not 500: the miner is working, the attestation path is
            # not, and a verifier should treat that as "try again" rather
            # than "this miner is broken".
            log.warning("attestation unavailable for %s: %s", verified.signed_by, exc)
            return _error(503, "attestation_unavailable", str(exc))

        return JSONResponse(
            {"bundle": bundle.to_payload(), "generated_ms": bundle.generated_at_ms}
        )

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        raw = await request.body()

        try:
            verified = ctx.auth.verify(dict(request.headers), raw)
        except EpistulaError as exc:
            return _error(401, "unauthorized", exc.reason)

        try:
            parsed = ChatCompletionRequest(**json.loads(raw))
        except ValueError as exc:
            return _error(400, "bad_request", _short(str(exc)))

        if parsed.model != ctx.model_id:
            # Answering for a model we are not running would make the
            # subnet's central claim -- you got the model you asked for --
            # false, and no amount of attestation downstream repairs that.
            return _error(
                400,
                "model_mismatch",
                f"this miner serves {ctx.model_id}, not {parsed.model}",
            )

        if ctx.in_flight >= ctx.max_concurrent:
            return _error(
                429,
                "at_capacity",
                f"{ctx.in_flight} in flight, limit {ctx.max_concurrent}",
            )

        request_id = verified.uuid or str(uuidlib.uuid4())
        payload = parsed.model_dump(exclude_none=True)
        started_ms = int(time.time() * 1000)

        if parsed.stream:
            return _stream_response(ctx, payload, raw, request_id, verified.signed_by,
                                    started_ms)

        ctx.in_flight += 1
        try:
            data, outcome = await ctx.vllm.complete(payload)
        except UpstreamError as exc:
            log.warning("upstream failed for %s: %s", request_id, exc)
            return _error(exc.status, "upstream_error", str(exc))
        finally:
            ctx.in_flight -= 1

        body = json.dumps(data, separators=(",", ":")).encode()
        signed = _sign_receipt(ctx, request_id, verified.signed_by, raw, body,
                               outcome, started_ms)
        return Response(
            content=body,
            media_type="application/json",
            headers={
                receipts.H_RECEIPT: json.dumps(
                    signed.receipt.to_payload(), separators=(",", ":")
                ),
                receipts.H_RECEIPT_SIG: signed.signature,
                "X-Instant-Request-Id": request_id,
            },
        )

    return router


def _stream_response(
    ctx: MinerContext,
    payload: dict,
    raw_request: bytes,
    request_id: str,
    signed_by: str,
    started_ms: int,
) -> StreamingResponse:
    """Stream tokens, then the receipt as the final SSE event.

    The receipt goes last because it commits to the response hash, which is
    not known until the last token is out. A client that only wants tokens
    ignores the ``receipt`` event; a validator reads it. Trailers would be
    the tidier mechanism and are not reliably delivered through the
    intermediaries that sit between us and a browser, so: an SSE event.

    If the upstream fails mid-stream there is no receipt — the miner did not
    complete the work and must not sign as though it had. The client gets an
    ``error`` event and the observer records a failure, which is the honest
    outcome.
    """
    outcome = StreamOutcome()

    async def body_iter():
        ctx.in_flight += 1
        try:
            async for chunk in ctx.vllm.stream(payload, outcome):
                yield chunk
        except UpstreamError as exc:
            log.warning("stream failed for %s: %s", request_id, exc)
            yield _sse("error", {"error": "upstream_error", "detail": str(exc)})
            return
        finally:
            ctx.in_flight -= 1

        yield b"data: [DONE]\n\n"

        signed = _sign_receipt(
            ctx, request_id, signed_by, raw_request, bytes(outcome.assembled),
            outcome, started_ms,
        )
        yield _sse(receipts.SSE_RECEIPT_EVENT, signed.to_payload())

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers the whole stream and every miner
            # behind a default proxy config reports a TTFT equal to its
            # total time. It has happened to every subnet that streams.
            "X-Accel-Buffering": "no",
            "X-Instant-Request-Id": request_id,
        },
    )


def _sign_receipt(
    ctx: MinerContext,
    request_id: str,
    signed_by: str,
    request_body: bytes,
    response_body: bytes,
    outcome: StreamOutcome,
    started_ms: int,
) -> receipts.SignedReceipt:
    last = ctx.attestation.last
    return receipts.build(
        ctx.signer,
        request_id=request_id,
        signer_of_request=signed_by,
        request_body=request_body,
        response_body=response_body,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        ttft_ms_self=outcome.ttft_ms,
        total_ms_self=outcome.total_ms,
        started_at_ms=started_ms,
        finished_at_ms=int(time.time() * 1000),
        attestation_id=last.digest() if last else None,
    )


def _sse(event: str, data: dict) -> bytes:
    return (
        f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
    ).encode()


def _error(status: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def _short(message: str, limit: int = 400) -> str:
    """Validation errors from pydantic are long. Callers get the useful part."""
    flat = " ".join(message.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
