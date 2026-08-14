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

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..protocol import receipts, sse
from ..protocol.epistula import (
    EpistulaError,
    ReplayGuard,
    generate_headers,
    verify_headers,
)
from ..protocol.keys import Signer
from ..protocol.schemas import ChatCompletionRequest
from ..protocol.ss58 import is_valid
from . import public
from .state import KeyConflictError, KeyRevokedError, PlatformState

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
    #: Hard ceiling used to reject impossible miner-reported token counts.
    #: The signed receipt binds a count to the response; it does not make an
    #: out-of-range value true.
    model_max_len: int = 131_072
    miner_uid: int = 0
    stats_window_s: int = 3600
    # How long one computed public stats body is reused. Injectable so tests
    # can pin it rather than racing a ten-second wall clock.
    public_stats_ttl_s: float = public.DEFAULT_TTL_S
    #: Pepper for customer key digests. Empty disables DB-backed keys entirely,
    #: leaving only the env service credential.
    api_key_pepper: str = ""
    #: Shared secret the control plane presents to the key-management routes.
    #: Empty refuses every admin call, so a blank env var cannot open issuance.
    admin_token: str = ""
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
        if self.model_max_len < 1:
            raise ValueError("model_max_len must be positive")
        if self.stats_window_s < 1:
            raise ValueError("stats_window_s must be positive")
        if self.public_stats_ttl_s < 0:
            raise ValueError("public_stats_ttl_s must not be negative")


@dataclass(frozen=True, slots=True)
class _ApiPrincipal:
    """Authenticated caller identity safe to persist with telemetry.

    The legacy environment credential intentionally has no customer key id.
    A dashboard-issued key carries only its opaque id; the raw credential and
    digest never enter a request row.
    """

    api_key_id: str | None


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
        if await _authenticate_bearer(ctx, request) is None:
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
        principal = await _authenticate_bearer(ctx, request)
        if principal is None:
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
            api_key_id=principal.api_key_id,
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
                return _error(
                    502, "miner_unavailable", str(exc), _provenance_headers(ctx)
                )

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
            if 200 <= upstream.status_code < 300 and not evidence.verified:
                return _error(
                    502,
                    "invalid_miner_receipt",
                    evidence.error or "miner response could not be verified",
                    _provenance_headers(ctx),
                )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=(
                    _response_headers(upstream, ctx, request_id=request_id)
                    if evidence.verified
                    else _unverified_response_headers(ctx, request_id=request_id)
                ),
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
            return _error(
                502, "miner_unavailable", str(exc), _provenance_headers(ctx)
            )
        if upstream.status_code >= 400:
            body = await upstream.aread()
            response_headers = _stream_response_headers(
                upstream, ctx, request_id=request_id
            )
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
            completed = False
            try:
                async for chunk in upstream.aiter_raw():
                    for customer_frame in observer.feed(chunk):
                        yield customer_frame
                completed = True
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
                        and 200 <= upstream.status_code < 300
                        and not observer.error_event
                        and observer.protocol_error is None
                        and evidence.verified
                    )
                    receipt = evidence.signed_receipt if evidence.verified else None
                    ttft_ms = observer.ttft_ms(total_ms) if success else None
                    tps_milli = (
                        sse.observed_tps_milli(
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
                            or observer.protocol_error
                            or evidence.error
                        ),
                    )

            # Commit the verified result before publishing the terminal marker.
            # Standard OpenAI clients stop reading as soon as they see [DONE],
            # so work placed after this yield is not guaranteed to run.
            if completed and success:
                yield b"data: [DONE]\n\n"
            elif completed:
                detail = evidence.error or "miner stream could not be verified"
                payload = json.dumps(
                    {
                        "error": {
                            "type": "invalid_miner_receipt",
                            "message": detail,
                        }
                    },
                    separators=(",", ":"),
                ).encode()
                yield b"data: " + payload + b"\n\n"

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            headers=_stream_response_headers(upstream, ctx, request_id=request_id),
            media_type=_media_type(upstream, "text/event-stream"),
        )

    @app.post("/admin/v1/keys")
    async def admin_register_key(request: Request) -> Response:
        """Register a key the control plane has just issued.

        The raw key arrives once, is digested here, and is never stored or
        logged. The gateway keeps its own digest so the two services share no
        secret: the control plane cannot compute what we hold, and we cannot
        reproduce what it shows the operator.

        nginx has no location for this path and ends in ``return 404``, so it is
        unreachable from the internet; the admin token guards it against other
        processes on this host.
        """
        if not _admin_ok(ctx, request):
            return _error(401, "unauthorized", "admin token required")
        if not ctx.api_key_pepper:
            return _error(
                503,
                "keys_unavailable",
                "INSTANT_PLATFORM_API_KEY_PEPPER is not configured",
            )
        try:
            body = json.loads(await request.body())
        except ValueError as exc:
            return _error(400, "bad_request", str(exc))
        if not isinstance(body, dict):
            return _error(400, "bad_request", "expected a JSON object")
        token = body.get("key")
        key_id = body.get("key_id")
        if not isinstance(token, str) or len(token) < 16:
            return _error(400, "bad_request", "key must be at least 16 characters")
        if not isinstance(key_id, str) or not key_id:
            return _error(400, "bad_request", "key_id is required")
        label = body.get("label") or ""
        if not isinstance(label, str) or len(label) > 200:
            return _error(400, "bad_request", "label must be a string under 200 chars")
        try:
            status = await run_in_threadpool(
                ctx.state.register_key,
                key_id=key_id,
                prefix=token[:8],
                last4=token[-4:],
                digest=key_digest(token, ctx.api_key_pepper),
                label=label,
                created_ms=_now_ms(),
            )
        except KeyRevokedError as exc:
            return _error(409, "key_revoked", str(exc))
        except KeyConflictError as exc:
            return _error(409, "key_conflict", str(exc))
        return JSONResponse(
            {"key_id": key_id, "status": status},
            status_code=201 if status == "registered" else 200,
        )

    @app.delete("/admin/v1/keys/{key_id}")
    async def admin_revoke_key(key_id: str, request: Request) -> Response:
        """Revoke a key. Must be called whenever the UI revokes one, or the
        dashboard would report a key as dead while the API still serves it."""
        if not _admin_ok(ctx, request):
            return _error(401, "unauthorized", "admin token required")
        revoked = await run_in_threadpool(
            ctx.state.revoke_key, key_id=key_id, revoked_ms=_now_ms()
        )
        if not revoked:
            return _error(404, "not_found", "no active key with that id")
        return JSONResponse({"key_id": key_id, "status": "revoked"})

    @app.get("/admin/v1/keys")
    async def admin_list_keys(request: Request) -> Response:
        """Safe key metadata and receipt-backed lifetime usage; never digests."""
        if not _admin_ok(ctx, request):
            return _error(401, "unauthorized", "admin token required")
        keys = await run_in_threadpool(ctx.state.list_keys)
        return JSONResponse({"keys": keys})

    @app.get("/admin/v1/keys/{key_id}/usage")
    async def admin_key_usage(key_id: str, request: Request) -> Response:
        """Return one key's lifetime usage, including after it is revoked.

        The control plane asks only for ids already owned by its signed-in
        user. Echoing the stored prefix and last four characters lets it reject
        an accidental id mismatch without exposing a credential or digest.
        """
        if not _admin_ok(ctx, request):
            return _error(401, "unauthorized", "admin token required")
        usage = await run_in_threadpool(ctx.state.key_usage, key_id)
        if usage is None:
            return _error(404, "not_found", "no key with that id")
        return JSONResponse(usage)

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

    public_stats_cache = public.PublicStatsCache(ctx.public_stats_ttl_s)

    # Loopback only. nginx never proxies this route: the control plane calls
    # it and owns the browser-facing contract, so the inference data plane
    # gains no public surface. See instant/platform/public.py.
    #
    # Deliberately `def`, not `async def`. ctx.state.stats() is a synchronous
    # SQLite read that holds the state lock and parses every stored receipt in
    # the window; awaiting it on the event loop would stall in-flight streaming
    # completions for the duration. FastAPI runs a sync route in a threadpool,
    # which keeps the inference data plane responsive while this one works.
    @app.get("/public/v1/stats")
    def public_stats() -> Response:
        def compute() -> bytes:
            generated_at_ms = _now_ms()
            return public.encode(
                public.project(
                    ctx.state.stats(
                        miner_hotkey=ctx.miner_ss58,
                        miner_uid=ctx.miner_uid,
                        window_start_ms=generated_at_ms - ctx.stats_window_s * 1000,
                        window_end_ms=generated_at_ms,
                        generated_at_ms=generated_at_ms,
                    )
                )
            )

        return Response(
            content=public_stats_cache.get(compute),
            media_type="application/json",
            # The cache above already collapses concurrent readers, so a
            # browser cache on top of it would add staleness without saving
            # this process any work. The page measures age from its own fetch
            # time (a client clock cannot be trusted), so a cached response is
            # already up to one TTL older than the page will claim; layering a
            # second cache would widen that gap further.
            headers={"cache-control": "no-store"},
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


class _StreamObserver(sse.StreamCommitment):
    """The platform's view of a streamed miner response.

    Frame parsing and the assembled-bytes commitment live in
    :mod:`instant.protocol.sse` so that this observer and the validator's direct
    prober cannot drift on which bytes a receipt covers. Only the platform's
    verification policy is here.
    """

    def verify(
        self,
        ctx: PlatformContext,
        *,
        request_id: str,
        request_body: bytes,
    ) -> _ReceiptEvidence:
        if self.protocol_error is not None:
            return _ReceiptEvidence(seen=self.receipt_seen, error=self.protocol_error)
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
    _validate_receipt_token_counts(ctx, receipt, request_body=request_body)
    if receipt.request_id != request_id:
        raise receipts.ReceiptError("receipt request id does not match")
    if receipt.signer_of_request != ctx.signer.ss58_address:
        raise receipts.ReceiptError("receipt names a different request signer")


def _validate_receipt_token_counts(
    ctx: PlatformContext,
    receipt: receipts.Receipt,
    *,
    request_body: bytes,
) -> None:
    """Reject impossible receipt counters before they reach SQLite or TPS.

    Counts remain miner-reported until independent tokenization or an attested
    inference worker is in place. These bounds are still load-bearing: a
    signed negative or arbitrarily large integer must not become plausible UI
    telemetry, overflow SQLite, or distort observed-throughput calculations.
    """
    counts = (receipt.prompt_tokens, receipt.completion_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise receipts.ReceiptError("receipt token counts must be integers")
    if any(value < 0 for value in counts):
        raise receipts.ReceiptError("receipt token counts must be non-negative")
    if any(value > ctx.model_max_len for value in counts):
        raise receipts.ReceiptError("receipt token count exceeds model context")
    if sum(counts) > ctx.model_max_len:
        raise receipts.ReceiptError("receipt total tokens exceed model context")

    try:
        parsed = ChatCompletionRequest.model_validate_json(request_body)
    except ValueError as exc:
        # The route has already validated this body. Failing closed here keeps
        # the evidence check correct if a future caller bypasses that route.
        raise receipts.ReceiptError("receipt request body is not a valid request") from exc
    requested_limits = [
        value
        for value in (parsed.max_tokens, parsed.max_completion_tokens)
        if value is not None
    ]
    if requested_limits and receipt.completion_tokens > min(requested_limits):
        raise receipts.ReceiptError("receipt completion tokens exceed request limit")


def key_digest(token: str, pepper: str) -> str:
    """Digest a customer key for storage and lookup.

    HMAC under a server-held pepper rather than a bare hash: the key is 32
    random bytes, so brute force is not the threat -- the pepper is what stops
    a stolen database from being replayed against the running gateway.
    """
    return hmac.new(pepper.encode(), token.encode(), hashlib.sha256).hexdigest()


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return None
    return token


async def _authenticate_bearer(
    ctx: PlatformContext, request: Request
) -> _ApiPrincipal | None:
    token = _bearer_token(request)
    if token is None:
        return None
    # The env key is the control plane's service credential, checked first and
    # in constant time so an empty key table never locks the gateway out.
    if hmac.compare_digest(
        hashlib.sha256(token.encode()).hexdigest(), ctx.api_key_sha256
    ):
        return _ApiPrincipal(api_key_id=None)
    if not ctx.api_key_pepper:
        return None
    key_id = await run_in_threadpool(
        ctx.state.active_key_id, key_digest(token, ctx.api_key_pepper)
    )
    return _ApiPrincipal(api_key_id=key_id) if key_id is not None else None


def _admin_ok(ctx: PlatformContext, request: Request) -> bool:
    """Authorise a control-plane call to the key-management routes.

    Constant-time, and refuses outright when no token is configured so a blank
    environment variable cannot silently open key issuance to any local caller.
    """
    if not ctx.admin_token:
        return False
    presented = request.headers.get("x-instant-admin-token", "")
    if not presented:
        return False
    return hmac.compare_digest(presented, ctx.admin_token)


def _bearer_error() -> JSONResponse:
    return JSONResponse(
        {"error": "unauthorized", "detail": "a valid Bearer API key is required"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _elapsed_ms(started_perf: float) -> int:
    return max(0, int((time.perf_counter() - started_perf) * 1000))


def _provenance_headers(ctx: PlatformContext) -> dict[str, str]:
    """Name the miner that handled a request.

    Written from ``ctx`` and never forwarded from upstream. A miner is an
    economically motivated adversary; letting it name the hotkey that served a
    request would let it credit another operator, or discredit one. Neither
    header is in ``_FORWARDED_RESPONSE_HEADERS``, so an upstream copy is dropped
    before this is applied.

    Deliberately absent: ``X-Instant-Ttft-Ms``. On the streaming path headers
    flush before the first token, so the value does not exist yet; on the
    non-streaming path there is no observable first-token event and the recorded
    figure is total latency under another name. Measured TTFT reaches consumers
    through the receipt and the validator stats window instead.
    """
    return {
        "X-Instant-Miner-Uid": str(ctx.miner_uid),
        "X-Instant-Miner-Hotkey": ctx.miner_ss58,
    }


def _response_headers(
    response: httpx.Response,
    ctx: PlatformContext | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, str]:
    """Forward the miner's allow-listed headers, then stamp our own provenance."""
    headers = {
        name: value
        for name, value in response.headers.items()
        if name.lower() in _FORWARDED_RESPONSE_HEADERS
    }
    if ctx is not None:
        headers.update(_provenance_headers(ctx))
    if request_id is not None:
        headers["X-Instant-Request-Id"] = request_id
    return headers


def _unverified_response_headers(
    ctx: PlatformContext, *, request_id: str
) -> dict[str, str]:
    headers = _provenance_headers(ctx)
    headers["X-Instant-Request-Id"] = request_id
    return headers


def _stream_response_headers(
    response: httpx.Response, ctx: PlatformContext, *, request_id: str
) -> dict[str, str]:
    """Headers safe to expose before stream receipt verification completes."""
    headers = {
        name: value
        for name, value in response.headers.items()
        if name.lower() in {"cache-control", "x-accel-buffering"}
    }
    headers.update(_provenance_headers(ctx))
    # The platform generated this id. Never let an upstream miner stamp a
    # conflicting value into a response whose receipt has not yet verified.
    headers["X-Instant-Request-Id"] = request_id
    return headers


def _media_type(response: httpx.Response, default: str) -> str:
    return response.headers.get("content-type", default).split(";", 1)[0]


def _error(
    status: int,
    error: str,
    detail: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"error": error, "detail": detail}, status_code=status, headers=headers
    )
