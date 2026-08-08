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
AUTH = {"authorization": f"Bearer {API_KEY}"}


def body_for(*, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        },
        separators=(",", ":"),
    ).encode()


@pytest.fixture
def miner_app(miner_key, platform_key):
    app = FastAPI()
    app.state.requests = []
    app.state.corrupt_receipt = False

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
                prompt_tokens=1,
                completion_tokens=1,
                ttft_ms_self=2,
                total_ms_self=3,
                started_at_ms=started,
                finished_at_ms=started + 3,
                attestation_id="attest-1",
            )
            if app.state.corrupt_receipt:
                signed.signature = "00" * 64

            async def chunks():
                yield b"data: " + data + b"\n\n"
                yield b"data: [DONE]\n\n"
                yield (
                    b"event: receipt\ndata: "
                    + json.dumps(signed.to_payload(), separators=(",", ":")).encode()
                    + b"\n\n"
                )

            return StreamingResponse(
                chunks(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response_body = b'{"id":"answer"}'
        signed = receipts.build(
            miner_key,
            request_id=verified.uuid,
            signer_of_request=platform_key.ss58_address,
            request_body=raw,
            response_body=response_body,
            prompt_tokens=2,
            completion_tokens=3,
            ttft_ms_self=2,
            total_ms_self=3,
            started_at_ms=started,
            finished_at_ms=started + 3,
            attestation_id="attest-1",
        )
        signature = "00" * 64 if app.state.corrupt_receipt else signed.signature
        return Response(
            content=response_body,
            media_type="application/json",
            headers={
                receipts.H_RECEIPT: json.dumps(
                    signed.receipt.to_payload(), separators=(",", ":")
                ),
                receipts.H_RECEIPT_SIG: signature,
                "X-Instant-Request-Id": verified.uuid,
            },
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


def test_invalid_receipt_is_forwarded_but_counts_as_failure(
    client, miner_app, validator_key, platform_key
):
    miner_app.state.corrupt_receipt = True
    response = client.post(
        "/v1/chat/completions",
        content=body_for(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    stats = _stats(client, validator_key, platform_key).json()["miners"][0]
    assert (stats["successes"], stats["failures"]) == (0, 1)
    assert (stats["receipts_seen"], stats["receipts_verified"]) == (1, 0)


def test_stream_relay_keeps_receipt_and_records_external_metrics(
    client, validator_key, platform_key
):
    response = client.post(
        "/v1/chat/completions",
        content=body_for(stream=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "event: receipt" in response.text
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    stats = _stats(client, validator_key, platform_key).json()["miners"][0]
    assert stats["successes"] == 1
    assert stats["ttft_p95_ms"] >= 0
    assert stats["tokens_per_s_p50"] >= 0


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


def _stats(client: TestClient, validator_key, platform_key) -> httpx.Response:
    return client.get(
        "/validator/v1/stats",
        headers=generate_headers(
            validator_key, b"", signed_for=platform_key.ss58_address
        ),
    )
