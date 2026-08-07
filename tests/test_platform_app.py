"""The first real platform -> miner hop."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from instant.platform.app import PlatformContext, create_app
from instant.protocol.epistula import verify_headers

MODEL = "openai/gpt-oss-20b"


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
        if json.loads(raw).get("stream"):
            async def chunks():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b"data: [DONE]\n\n"
                yield b'event: receipt\ndata: {"receipt":"signed"}\n\n'

            return StreamingResponse(
                chunks(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return Response(
            content=b'{"id":"answer"}',
            media_type="application/json",
            headers={
                "X-Instant-Receipt": "receipt-json",
                "X-Instant-Receipt-Sig": "receipt-signature",
                "X-Instant-Request-Id": "request-1",
            },
        )

    return app


@pytest.fixture
def client(miner_app, miner_key, platform_key):
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=miner_app),
        base_url="http://miner",
    )
    ctx = PlatformContext(
        signer=platform_key,
        miner_url="http://miner",
        miner_ss58=miner_key.ss58_address,
        http=upstream,
    )
    with TestClient(create_app(ctx)) as test_client:
        yield test_client


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


def test_models_rejects_a_different_miner_identity(miner_app, miner_key, platform_key):
    wrong_miner = type(miner_key).generate()
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=miner_app), base_url="http://miner"
    )
    ctx = PlatformContext(
        signer=platform_key,
        miner_url="http://miner",
        miner_ss58=wrong_miner.ss58_address,
        http=upstream,
    )
    with TestClient(create_app(ctx)) as mismatch_client:
        response = mismatch_client.get("/v1/models")
    assert response.status_code == 502
    assert response.json()["error"] == "miner_identity_mismatch"


def test_nonstream_request_is_signed_over_the_exact_bytes(
    client, miner_app, platform_key
):
    raw = body_for()
    response = client.post(
        "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 200
    assert response.content == b'{"id":"answer"}'
    assert response.headers["x-instant-receipt"] == "receipt-json"
    forwarded, verified = miner_app.state.requests[-1]
    assert forwarded == raw
    assert verified.signed_by == platform_key.ss58_address


def test_stream_relay_keeps_the_final_receipt_event(client):
    response = client.post(
        "/v1/chat/completions",
        content=body_for(stream=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "event: receipt" in response.text
    assert '"receipt":"signed"' in response.text
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_invalid_request_is_rejected_before_it_reaches_the_miner(client, miner_app):
    response = client.post(
        "/v1/chat/completions",
        content=b'{"model":"missing-messages"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert miner_app.state.requests == []
