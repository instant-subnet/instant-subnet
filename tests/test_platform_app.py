"""Authenticated platform -> miner relay and receipt-backed telemetry."""

from __future__ import annotations

import hashlib
import json
import time

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from instant.platform.app import PlatformContext, create_app
from instant.platform.state import open_state
from instant.protocol import receipts
from instant.protocol.epistula import generate_headers, verify_headers

MODEL = "instant/mock-echo"
API_KEY = "isk_test_platform_key"
API_KEY_HASH = hashlib.sha256(API_KEY.encode()).hexdigest()
PEPPER = "test-pepper-not-a-real-one"
ADMIN_TOKEN = "test-admin-token"
ADMIN = {"x-instant-admin-token": ADMIN_TOKEN}
AUTH = {"authorization": f"Bearer {API_KEY}"}


def body_for(*, stream: bool = False, max_tokens: int | None = None) -> bytes:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return json.dumps(
        payload,
        separators=(",", ":"),
    ).encode()


SPOOFED_UID = "999"
SPOOFED_HOTKEY = "5SpoofedHotkeyThatMustNeverReachAClient"


def _spoofed(app: FastAPI) -> dict[str, str]:
    """A hostile miner claiming to be someone else."""
    if not app.state.spoof_provenance:
        return {}
    return {
        "X-Instant-Miner-Uid": SPOOFED_UID,
        "X-Instant-Miner-Hotkey": SPOOFED_HOTKEY,
    }


@pytest.fixture
def miner_app(miner_key, platform_key):
    app = FastAPI()
    app.state.requests = []
    app.state.corrupt_receipt = False
    app.state.spoof_provenance = False
    app.state.empty_receipt_event = False
    app.state.unknown_stream_event = False
    app.state.data_error = False
    app.state.prompt_tokens = None
    app.state.completion_tokens = None

    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": True, "model_id": MODEL}

    @app.get("/manifest")
    async def manifest():
        return {"hotkey": miner_key.ss58_address, "model_id": MODEL}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        raw = await request.body()
        verified = verify_headers(
            dict(request.headers),
            raw,
            allowed_signers={platform_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        app.state.requests.append((raw, verified))
        started = int(time.time() * 1000)
        if json.loads(raw).get("stream"):
            data = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": {"content": "hi"}}],
                },
                separators=(",", ":"),
            ).encode()
            signed = receipts.build(
                miner_key,
                request_id=verified.uuid,
                signer_of_request=platform_key.ss58_address,
                request_body=raw,
                response_body=data,
                prompt_tokens=(
                    1
                    if app.state.prompt_tokens is None
                    else app.state.prompt_tokens
                ),
                completion_tokens=(
                    1
                    if app.state.completion_tokens is None
                    else app.state.completion_tokens
                ),
                ttft_ms_self=2,
                total_ms_self=3,
                started_at_ms=started,
                finished_at_ms=started + 3,
                attestation_id="attest-1",
            )
            if app.state.corrupt_receipt:
                signed.signature = "00" * 64

            async def chunks():
                yield b": keep-alive\n\n"
                yield b"data: " + data + b"\n\n"
                if app.state.unknown_stream_event:
                    yield b"event: bonus\ndata: uncommitted customer bytes\n\n"
                if app.state.empty_receipt_event:
                    yield b"event: receipt\n\n"
                if app.state.data_error:
                    yield b'data: {"error":{"message":"upstream failed"}}\n\n'
                yield (
                    b"event: receipt\ndata: "
                    + json.dumps(signed.to_payload(), separators=(",", ":")).encode()
                    + b"\n\n"
                )
                yield b"data: [DONE]\n\n"

            stream_headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                receipts.H_RECEIPT: "unverified-header-receipt",
                receipts.H_RECEIPT_SIG: "unverified-header-signature",
            }
            stream_headers.update(_spoofed(app))
            return StreamingResponse(
                chunks(),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        response_body = b'{"id":"answer"}'
        signed = receipts.build(
            miner_key,
            request_id=verified.uuid,
            signer_of_request=platform_key.ss58_address,
            request_body=raw,
            response_body=response_body,
            prompt_tokens=(
                2 if app.state.prompt_tokens is None else app.state.prompt_tokens
            ),
            completion_tokens=(
                3
                if app.state.completion_tokens is None
                else app.state.completion_tokens
            ),
            ttft_ms_self=2,
            total_ms_self=3,
            started_at_ms=started,
            finished_at_ms=started + 3,
            attestation_id="attest-1",
        )
        signature = "00" * 64 if app.state.corrupt_receipt else signed.signature
        plain_headers = {
            receipts.H_RECEIPT: json.dumps(
                signed.receipt.to_payload(), separators=(",", ":")
            ),
            receipts.H_RECEIPT_SIG: signature,
            "X-Instant-Request-Id": verified.uuid,
        }
        plain_headers.update(_spoofed(app))
        return Response(
            content=response_body,
            media_type="application/json",
            headers=plain_headers,
        )

    return app


def platform_context(miner_app, miner_key, platform_key, validator_key, state_path):
    return PlatformContext(
        signer=platform_key,
        miner_url="http://miner",
        miner_ss58=miner_key.ss58_address,
        http=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=miner_app), base_url="http://miner"
        ),
        state=open_state(state_path),
        api_key_sha256=API_KEY_HASH,
        validator_hotkeys=frozenset({validator_key.ss58_address}),
        miner_uid=7,
        stats_window_s=3600,
        api_key_pepper=PEPPER,
        admin_token=ADMIN_TOKEN,
    )


@pytest.fixture
def client(miner_app, miner_key, platform_key, validator_key, tmp_path):
    ctx = platform_context(
        miner_app, miner_key, platform_key, validator_key, tmp_path / "platform.sqlite3"
    )
    with TestClient(create_app(ctx)) as test_client:
        test_client.headers.update(AUTH)
        yield test_client


def test_health_is_public_but_openai_routes_require_bearer(client, miner_app):
    assert client.get("/health", headers={"authorization": ""}).status_code == 200
    models = client.get("/v1/models", headers={"authorization": ""})
    completion = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"authorization": "", "content-type": "application/json"},
    )
    assert models.status_code == completion.status_code == 401
    assert models.headers["www-authenticate"] == "Bearer"
    assert miner_app.state.requests == []


def test_health_and_models_reach_the_configured_miner(client, miner_key):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["ready"] is True
    assert client.get("/livez").status_code == 200
    assert client.get("/readyz").status_code == 200

    models = client.get("/v1/models").json()
    assert models["data"] == [
        {"id": MODEL, "object": "model", "owned_by": miner_key.ss58_address}
    ]


def test_models_rejects_a_different_miner_identity(
    miner_app, miner_key, platform_key, validator_key, tmp_path
):
    wrong_miner = type(miner_key).generate()
    ctx = platform_context(
        miner_app, wrong_miner, platform_key, validator_key, tmp_path / "wrong.sqlite3"
    )
    with TestClient(create_app(ctx)) as mismatch_client:
        response = mismatch_client.get("/v1/models", headers=AUTH)
    assert response.status_code == 502
    assert response.json()["error"] == "miner_identity_mismatch"


def test_nonstream_request_is_signed_and_receipt_verified(
    client, miner_app, platform_key, validator_key
):
    raw = body_for()
    response = client.post(
        "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 200
    assert response.content == b'{"id":"answer"}'
    forwarded, verified = miner_app.state.requests[-1]
    assert forwarded == raw
    assert verified.signed_by == platform_key.ss58_address

    stats = _stats(client, validator_key, platform_key).json()["miners"][0]
    assert (stats["requests"], stats["successes"], stats["failures"]) == (1, 1, 0)
    assert (stats["receipts_seen"], stats["receipts_verified"]) == (1, 1)
    assert (stats["prompt_tokens"], stats["completion_tokens"]) == (2, 3)


def test_invalid_receipt_fails_closed_and_counts_as_failure(
    client, miner_app, validator_key, platform_key
):
    miner_app.state.corrupt_receipt = True
    response = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 502
    assert response.json()["error"] == "invalid_miner_receipt"
    assert receipts.H_RECEIPT not in response.headers
    assert receipts.H_RECEIPT_SIG not in response.headers
    stats = _stats(client, validator_key, platform_key).json()["miners"][0]
    assert (stats["successes"], stats["failures"]) == (0, 1)
    assert (stats["receipts_seen"], stats["receipts_verified"]) == (1, 0)


def test_stream_relay_hides_internal_receipt_and_records_before_done(
    client, validator_key, platform_key
):
    response = client.post(
        "/v1/chat/completions",
        content=body_for(stream=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    assert response.text.count("data: [DONE]") == 1
    assert "event: receipt" not in response.text
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert receipts.H_RECEIPT not in response.headers
    assert receipts.H_RECEIPT_SIG not in response.headers
    assert response.headers["x-instant-request-id"]

    stats = _stats(client, validator_key, platform_key).json()["miners"][0]
    assert stats["successes"] == 1
    assert stats["ttft_p95_ms"] >= 0
    assert stats["tokens_per_s_p50"] >= 0


def test_corrupt_stream_receipt_never_publishes_done(client, miner_app):
    miner_app.state.corrupt_receipt = True
    response = client.post(
        "/v1/chat/completions",
        content=body_for(stream=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200  # headers were already streamed
    assert "data: [DONE]" not in response.text
    assert "invalid_miner_receipt" in response.text
    assert "event: receipt" not in response.text


@pytest.mark.parametrize(
    "fault", ["unknown_stream_event", "empty_receipt_event", "data_error"]
)
def test_uncommitted_or_malformed_stream_events_fail_closed(client, miner_app, fault):
    setattr(miner_app.state, fault, True)
    response = client.post(
        "/v1/chat/completions",
        content=body_for(stream=True),
        headers={"content-type": "application/json"},
    )
    assert "data: [DONE]" not in response.text
    assert "invalid_miner_receipt" in response.text
    assert "uncommitted customer bytes" not in response.text
    assert "event: receipt" not in response.text


def test_stream_heartbeats_are_preserved(client):
    response = client.post(
        "/v1/chat/completions",
        content=body_for(stream=True),
        headers={"content-type": "application/json"},
    )
    assert ": keep-alive\n\n" in response.text
    assert "data: [DONE]" in response.text


async def test_openai_sdk_stops_at_done_after_telemetry_is_committed(
    miner_app, miner_key, platform_key, validator_key, tmp_path
):
    AsyncOpenAI = pytest.importorskip("openai").AsyncOpenAI
    ctx = platform_context(
        miner_app,
        miner_key,
        platform_key,
        validator_key,
        tmp_path / "sdk.sqlite3",
    )
    app = create_app(ctx)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://platform",
    )
    sdk = AsyncOpenAI(
        base_url="http://platform/v1",
        api_key=API_KEY,
        http_client=http,
    )
    try:
        stream = await sdk.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        content = ""
        async for chunk in stream:
            content += chunk.choices[0].delta.content or ""
        assert content == "hi"

        generated_at = int(time.time() * 1000)
        stats = ctx.state.stats(
            miner_hotkey=miner_key.ss58_address,
            miner_uid=7,
            window_start_ms=0,
            window_end_ms=generated_at,
            generated_at_ms=generated_at,
        ).miners[0]
        assert (stats.requests, stats.successes, stats.receipts_verified) == (1, 1, 1)
    finally:
        await sdk.close()
        await ctx.http.aclose()
        ctx.state.close()


def test_stats_requires_epistula_and_signs_exact_response_bytes(
    client, validator_key, platform_key
):
    assert client.get("/validator/v1/stats").status_code == 401
    response = _stats(client, validator_key, platform_key)
    assert response.status_code == 200
    verified = verify_headers(
        response.headers,
        response.content,
        allowed_signers={platform_key.ss58_address},
        expected_signed_for=validator_key.ss58_address,
    )
    assert verified.signed_by == platform_key.ss58_address
    payload = response.json()
    assert payload["miners"][0]["uid"] == 7
    assert payload["block_start"] == payload["block_end"] == 0


def test_stats_rejects_replay(client, validator_key, platform_key):
    headers = generate_headers(
        validator_key, b"", signed_for=platform_key.ss58_address
    )
    assert client.get("/validator/v1/stats", headers=headers).status_code == 200
    replay = client.get("/validator/v1/stats", headers=headers)
    assert replay.status_code == 401
    assert "replayed" in replay.json()["detail"]


def test_invalid_request_is_rejected_before_it_reaches_the_miner(client, miner_app):
    response = client.post(
        "/v1/chat/completions",
        content=b'{"model":"missing-messages"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert miner_app.state.requests == []


def test_public_stats_is_readable_without_a_signature(client):
    # The metrics page is a browser with no key and no wallet. If this needs
    # credentials, the page has no data source at all.
    response = client.get("/public/v1/stats", headers={"authorization": ""})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"

    payload = response.json()
    assert payload["miners"][0]["uid"] == 7
    assert "receipt_merkle_root" not in payload
    assert "block_start" not in payload
    assert "hotkey" not in payload["miners"][0]


def test_public_stats_agrees_with_the_signed_validator_response(
    client, validator_key, platform_key
):
    client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"content-type": "application/json"},
    )
    payload = client.get("/public/v1/stats", headers={"authorization": ""}).json()
    assert payload["miners"][0]["successes"] == 1
    assert payload["total_requests"] == 1

    # Same window, same aggregation, so the page can never quietly disagree
    # with what the validator is scoring on.
    signed = _stats(client, validator_key, platform_key).json()["miners"][0]
    assert payload["miners"][0]["requests"] == signed["requests"]
    assert payload["miners"][0]["tokens_per_s_p50"] == signed["tokens_per_s_p50"]


def test_public_stats_is_cached_so_readers_cannot_amplify_the_query(
    miner_app, miner_key, platform_key, validator_key, tmp_path
):
    # A long TTL rather than the default, so the assertion is about caching
    # and not about whether this test ran inside ten seconds.
    ctx = platform_context(
        miner_app, miner_key, platform_key, validator_key, tmp_path / "cached.sqlite3"
    )
    ctx.public_stats_ttl_s = 3600.0
    with TestClient(create_app(ctx)) as cached_client:
        cached_client.headers.update(AUTH)
        first = cached_client.get("/public/v1/stats", headers={"authorization": ""})
        cached_client.post(
            "/v1/chat/completions",
            content=body_for(),
            headers={"content-type": "application/json"},
        )
        second = cached_client.get("/public/v1/stats", headers={"authorization": ""})

    # Within the TTL the second reader gets the first reader's bytes. That is
    # the trade being made: a page polling every 10s costs one stats() call
    # per 10s no matter how many tabs are open, at the price of showing a
    # request up to a TTL late.
    assert first.content == second.content


def test_public_stats_recomputes_once_the_ttl_lapses(
    miner_app, miner_key, platform_key, validator_key, tmp_path
):
    ctx = platform_context(
        miner_app, miner_key, platform_key, validator_key, tmp_path / "uncached.sqlite3"
    )
    ctx.public_stats_ttl_s = 0.0
    with TestClient(create_app(ctx)) as fresh_client:
        fresh_client.headers.update(AUTH)
        before = fresh_client.get("/public/v1/stats", headers={"authorization": ""})
        fresh_client.post(
            "/v1/chat/completions",
            content=body_for(),
            headers={"content-type": "application/json"},
        )
        after = fresh_client.get("/public/v1/stats", headers={"authorization": ""})

    assert before.json()["total_requests"] == 0
    assert after.json()["total_requests"] == 1


def test_public_stats_does_not_weaken_the_validator_route(client, validator_key):
    assert client.get("/validator/v1/stats").status_code == 401
    signed = _stats(client, validator_key, validator_key)
    assert signed.status_code == 401


def _stats(client: TestClient, validator_key, platform_key) -> httpx.Response:
    return client.get(
        "/validator/v1/stats",
        headers=generate_headers(
            validator_key, b"", signed_for=platform_key.ss58_address
        ),
    )


def _thread_probe(monkeypatch, client: TestClient) -> tuple[set[int], set[int]]:
    """Record which threads run loop code versus blocking state code."""
    import threading

    from instant.platform import app as app_module

    loop_threads: set[int] = set()
    state_threads: set[int] = set()

    real_now = app_module._now_ms

    def spy_now() -> int:
        # Called inline from the async handlers, so it names the loop thread.
        loop_threads.add(threading.get_ident())
        return real_now()

    monkeypatch.setattr(app_module, "_now_ms", spy_now)

    state = client.app.state.ctx.state
    for name in ("begin", "finish", "stats"):
        real = getattr(state, name)

        def spy(*args, _real=real, **kwargs):
            state_threads.add(threading.get_ident())
            return _real(*args, **kwargs)

        monkeypatch.setattr(state, name, spy)

    return loop_threads, state_threads


def test_state_writes_do_not_run_on_the_event_loop(client, monkeypatch):
    """Every PlatformState call commits to SQLite while holding its lock.

    On the deployed host one committed write costs ~2.5ms, so running these on
    the event loop stalls the relay of every concurrent stream for that long.
    They must be offloaded to a worker thread.
    """
    loop_threads, state_threads = _thread_probe(monkeypatch, client)

    response = client.post("/v1/chat/completions", content=body_for())

    assert response.status_code == 200
    assert loop_threads, "probe never observed the event loop thread"
    assert state_threads, "probe never observed a state call"
    assert state_threads.isdisjoint(loop_threads)


def test_stats_aggregation_does_not_run_on_the_event_loop(
    client, validator_key, platform_key, monkeypatch
):
    """stats() is O(window) and verifies every stored receipt -- the costliest
    of the three, and the one most likely to stall an in-flight stream."""
    loop_threads, state_threads = _thread_probe(monkeypatch, client)

    response = _stats(client, validator_key, platform_key)

    assert response.status_code == 200
    assert loop_threads and state_threads
    assert state_threads.isdisjoint(loop_threads)
def _provenance(response: httpx.Response) -> tuple[str | None, str | None]:
    return (
        response.headers.get("x-instant-miner-uid"),
        response.headers.get("x-instant-miner-hotkey"),
    )


def test_successful_responses_name_the_miner_that_served_them(client, miner_key):
    uid = str(client.app.state.ctx.miner_uid)
    plain = client.post("/v1/chat/completions", content=body_for())
    assert plain.status_code == 200
    assert _provenance(plain) == (uid, miner_key.ss58_address)

    with client.stream(
        "POST", "/v1/chat/completions", content=body_for(stream=True)
    ) as streamed:
        assert streamed.status_code == 200
        assert _provenance(streamed) == (uid, miner_key.ss58_address)


def test_a_miner_cannot_claim_another_operators_identity(client, miner_app, miner_key):
    uid = str(client.app.state.ctx.miner_uid)
    """The header is written from our config, never forwarded from upstream."""
    miner_app.state.spoof_provenance = True

    plain = client.post("/v1/chat/completions", content=body_for())
    assert _provenance(plain) == (uid, miner_key.ss58_address)
    assert SPOOFED_HOTKEY not in plain.headers.values()

    with client.stream(
        "POST", "/v1/chat/completions", content=body_for(stream=True)
    ) as streamed:
        assert _provenance(streamed) == (uid, miner_key.ss58_address)
        assert SPOOFED_HOTKEY not in streamed.headers.values()


def test_an_unreachable_miner_is_still_named(client, miner_key, monkeypatch):
    """502s are exactly when a caller needs to know which miner failed them."""
    ctx = client.app.state.ctx
    uid = str(ctx.miner_uid)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(
        ctx,
        "http",
        httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://miner"),
    )

    plain = client.post("/v1/chat/completions", content=body_for())
    assert plain.status_code == 502
    assert _provenance(plain) == (uid, miner_key.ss58_address)

    streamed = client.post("/v1/chat/completions", content=body_for(stream=True))
    assert streamed.status_code == 502
    assert _provenance(streamed) == (uid, miner_key.ss58_address)


def test_requests_that_never_reached_a_miner_are_not_attributed_to_one(client):
    """A rejected request was served by nobody; saying otherwise would be false,
    and on 401 it would hand miner identity to an unauthenticated caller."""
    malformed = client.post(
        "/v1/chat/completions",
        content=b'{"model":"missing-messages"}',
        headers={"content-type": "application/json"},
    )
    assert malformed.status_code == 400
    assert _provenance(malformed) == (None, None)

    unauthorized = client.post(
        "/v1/chat/completions", content=body_for(), headers={"authorization": "Bearer x"}
    )
    assert unauthorized.status_code == 401
    assert _provenance(unauthorized) == (None, None)


# --- customer API keys -------------------------------------------------------


CUSTOMER_KEY = "isk_customer_key_that_is_long_enough"


def _register(client: TestClient, key: str, key_id: str = "k_1", **extra):
    payload = {"key": key, "key_id": key_id, "label": "ditto"}
    payload.update(extra)
    return client.post("/admin/v1/keys", json=payload, headers=ADMIN)


def test_a_registered_key_is_accepted_for_inference(client):
    """The point of the whole exercise: a key minted in the dashboard works."""
    before = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"authorization": f"Bearer {CUSTOMER_KEY}"},
    )
    assert before.status_code == 401

    assert _register(client, CUSTOMER_KEY).status_code == 201

    after = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"authorization": f"Bearer {CUSTOMER_KEY}"},
    )
    assert after.status_code == 200

    usage = client.get("/admin/v1/keys/k_1/usage", headers=ADMIN).json()
    assert usage["key_id"] == "k_1"
    assert usage["prefix"] == CUSTOMER_KEY[:8]
    assert usage["last4"] == CUSTOMER_KEY[-4:]
    assert isinstance(usage["last_used_ms"], int)
    assert usage["requests"] == 1
    assert usage["successful_requests"] == usage["verified_requests"] == 1
    assert usage["prompt_tokens"] == 2
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == 5
    assert "digest" not in usage
    listed = client.get("/admin/v1/keys", headers=ADMIN).json()["keys"]
    assert listed == [usage]


def test_streaming_usage_is_attributed_to_the_authenticated_key(client):
    assert _register(client, CUSTOMER_KEY).status_code == 201
    auth = {
        "authorization": f"Bearer {CUSTOMER_KEY}",
        # A caller-provided identity hint must never override authentication.
        "x-instant-api-key-id": "somebody-elses-key",
    }
    with client.stream(
        "POST", "/v1/chat/completions", content=body_for(stream=True), headers=auth
    ) as response:
        assert response.status_code == 200
        assert b"data: [DONE]" in b"".join(response.iter_bytes())

    usage = client.get("/admin/v1/keys/k_1/usage", headers=ADMIN).json()
    assert usage["requests"] == usage["verified_requests"] == 1
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == 2


def test_unverified_failures_update_last_use_but_cannot_inflate_token_usage(
    client, miner_app
):
    assert _register(client, CUSTOMER_KEY).status_code == 201
    miner_app.state.corrupt_receipt = True
    response = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"authorization": f"Bearer {CUSTOMER_KEY}"},
    )
    assert response.status_code == 502

    usage = client.get("/admin/v1/keys/k_1/usage", headers=ADMIN).json()
    assert isinstance(usage["last_used_ms"], int)
    assert usage["requests"] == 1
    assert usage["successful_requests"] == usage["verified_requests"] == 0
    assert usage["prompt_tokens"] == usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens", "body"),
    [
        (-1, 1, body_for()),
        (131_072, 1, body_for()),
        (1, 3, body_for(max_tokens=2)),
    ],
)
def test_impossible_miner_token_counts_fail_closed(
    client, miner_app, prompt_tokens, completion_tokens, body
):
    assert _register(client, CUSTOMER_KEY).status_code == 201
    miner_app.state.prompt_tokens = prompt_tokens
    miner_app.state.completion_tokens = completion_tokens

    response = client.post(
        "/v1/chat/completions",
        content=body,
        headers={"authorization": f"Bearer {CUSTOMER_KEY}"},
    )

    assert response.status_code == 502
    usage = client.get("/admin/v1/keys/k_1/usage", headers=ADMIN).json()
    assert usage["requests"] == 1
    assert usage["verified_requests"] == 0
    assert usage["total_tokens"] == 0


def test_service_credential_requests_are_not_misattributed_to_customer_keys(client):
    assert _register(client, CUSTOMER_KEY).status_code == 201
    assert client.post("/v1/chat/completions", content=body_for()).status_code == 200

    usage = client.get("/admin/v1/keys/k_1/usage", headers=ADMIN).json()
    assert usage["last_used_ms"] is None
    assert usage["requests"] == usage["total_tokens"] == 0


def test_revoking_a_key_stops_it_serving_traffic(client):
    """A key the dashboard calls revoked must stop working, or the UI is lying."""
    _register(client, CUSTOMER_KEY)
    auth = {"authorization": f"Bearer {CUSTOMER_KEY}"}
    assert client.post("/v1/chat/completions", content=body_for(), headers=auth).status_code == 200

    revoked = client.request("DELETE", "/admin/v1/keys/k_1", headers=ADMIN)
    assert revoked.status_code == 200

    assert client.post("/v1/chat/completions", content=body_for(), headers=auth).status_code == 401
    # Revoking twice is not success -- it would hide a bug in the caller.
    assert client.request("DELETE", "/admin/v1/keys/k_1", headers=ADMIN).status_code == 404
    historical = client.get("/admin/v1/keys/k_1/usage", headers=ADMIN)
    assert historical.status_code == 200
    assert historical.json()["revoked_ms"] is not None
    assert historical.json()["requests"] == 1


def test_the_raw_key_is_never_stored(client, tmp_path):
    """A database that can reproduce a customer's key is a breach waiting to happen."""
    _register(client, CUSTOMER_KEY)
    keys = client.get("/admin/v1/keys", headers=ADMIN).json()["keys"]
    assert keys[0]["prefix"] == CUSTOMER_KEY[:8]
    assert keys[0]["last4"] == CUSTOMER_KEY[-4:]
    assert "digest" not in keys[0]

    blob = (tmp_path / "platform.sqlite3").read_bytes()
    assert CUSTOMER_KEY.encode() not in blob


def test_admin_routes_refuse_without_the_token(client):
    assert client.post("/admin/v1/keys", json={"key": CUSTOMER_KEY, "key_id": "x"}).status_code == 401
    assert client.post(
        "/admin/v1/keys",
        json={"key": CUSTOMER_KEY, "key_id": "x"},
        headers={"x-instant-admin-token": "wrong"},
    ).status_code == 401
    assert client.get("/admin/v1/keys", headers={"x-instant-admin-token": "wrong"}).status_code == 401
    assert client.get("/admin/v1/keys/x/usage").status_code == 401
    assert (
        client.get(
            "/admin/v1/keys/x/usage",
            headers={"x-instant-admin-token": "wrong"},
        ).status_code
        == 401
    )
    assert client.get("/admin/v1/keys/x/usage", headers=ADMIN).status_code == 404


def test_the_service_credential_still_works_alongside_minted_keys(client):
    """The control plane's relay must not break when customer keys exist."""
    _register(client, CUSTOMER_KEY)
    assert client.post("/v1/chat/completions", content=body_for()).status_code == 200


def test_exact_key_registration_is_retry_safe_but_collisions_are_conflicts(client):
    created = _register(client, CUSTOMER_KEY, key_id="stable")
    assert created.status_code == 201
    assert created.json()["status"] == "registered"

    retry = _register(client, CUSTOMER_KEY, key_id="stable")
    assert retry.status_code == 200
    assert retry.json() == {"key_id": "stable", "status": "already_registered"}

    digest_collision = _register(client, CUSTOMER_KEY, key_id="different-id")
    id_collision = _register(
        client,
        "isk_a_completely_different_customer_key",
        key_id="stable",
    )
    assert digest_collision.status_code == id_collision.status_code == 409
    assert digest_collision.json()["error"] == "key_conflict"
    assert id_collision.json()["error"] == "key_conflict"


def test_revoked_key_registration_retry_is_refused(client):
    assert _register(client, CUSTOMER_KEY, key_id="dead").status_code == 201
    assert client.delete("/admin/v1/keys/dead", headers=ADMIN).status_code == 200
    retry = _register(client, CUSTOMER_KEY, key_id="dead")
    assert retry.status_code == 409
    assert retry.json()["error"] == "key_revoked"


def test_dynamic_key_lookup_does_not_run_on_the_event_loop(
    client, monkeypatch
):
    import threading

    from instant.platform import app as app_module

    assert _register(client, CUSTOMER_KEY).status_code == 201
    loop_threads: set[int] = set()
    lookup_threads: set[int] = set()
    real_now = app_module._now_ms
    real_lookup = client.app.state.ctx.state.active_key_id

    def spy_now() -> int:
        loop_threads.add(threading.get_ident())
        return real_now()

    def spy_lookup(digest: str) -> str | None:
        lookup_threads.add(threading.get_ident())
        return real_lookup(digest)

    monkeypatch.setattr(app_module, "_now_ms", spy_now)
    monkeypatch.setattr(client.app.state.ctx.state, "active_key_id", spy_lookup)
    response = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"authorization": f"Bearer {CUSTOMER_KEY}"},
    )
    assert response.status_code == 200
    assert loop_threads and lookup_threads
    assert loop_threads.isdisjoint(lookup_threads)


def test_a_key_from_another_pepper_is_rejected(client):
    """The pepper is what stops a stolen digest being replayed elsewhere."""
    assert _register(client, CUSTOMER_KEY).status_code == 201
    client.app.state.ctx.api_key_pepper = "a-different-pepper"
    response = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"authorization": f"Bearer {CUSTOMER_KEY}"},
    )
    assert response.status_code == 401


def test_runtime_key_secrets_are_required_even_for_preflight(monkeypatch):
    from instant.platform.main import _required_secret

    monkeypatch.delenv("INSTANT_TEST_SECRET", raising=False)
    with pytest.raises(ValueError, match="INSTANT_TEST_SECRET"):
        _required_secret("INSTANT_TEST_SECRET")
    monkeypatch.setenv("INSTANT_TEST_SECRET", "short")
    with pytest.raises(ValueError, match="at least 32"):
        _required_secret("INSTANT_TEST_SECRET")
    monkeypatch.setenv("INSTANT_TEST_SECRET", "x" * 32)
    assert _required_secret("INSTANT_TEST_SECRET") == "x" * 32
