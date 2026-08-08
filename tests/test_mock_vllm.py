from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from instant.mock_vllm.app import create_app

MODEL = "instant/mock-echo"


def request_body(*, model: str = MODEL, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello localnet"}],
        "stream": stream,
    }


def test_mock_exposes_vllm_health_and_model_surfaces():
    client = TestClient(create_app(model_id=MODEL, first_token_delay_ms=0))
    assert client.get("/health").json() == {
        "status": "ok",
        "ready": True,
        "kind": "mock",
    }
    assert client.get("/v1/models").json()["data"][0]["id"] == MODEL


def test_nonstream_response_is_openai_shaped_and_obviously_mocked():
    client = TestClient(create_app(model_id=MODEL, first_token_delay_ms=0))
    response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    assert body["choices"][0]["message"]["content"].startswith("[mock]")
    assert body["usage"]["completion_tokens"] > 0


def test_mock_rejects_a_real_model_identity():
    client = TestClient(create_app(model_id=MODEL, first_token_delay_ms=0))
    response = client.post(
        "/v1/chat/completions",
        json=request_body(model="openai/gpt-oss-20b"),
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_stream_has_role_content_finish_usage_and_done_frames():
    client = TestClient(
        create_app(model_id=MODEL, first_token_delay_ms=0, token_delay_ms=0)
    )
    text = client.post("/v1/chat/completions", json=request_body(stream=True)).text
    payloads = [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: {")
    ]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert any(item.get("usage") for item in payloads)
    assert any(
        choice.get("finish_reason") == "stop"
        for item in payloads
        for choice in item.get("choices", [])
    )
    assert text.rstrip().endswith("data: [DONE]")


def test_configured_first_token_delay_is_real_not_metadata():
    client = TestClient(
        create_app(model_id=MODEL, first_token_delay_ms=40, token_delay_ms=0)
    )
    started = time.perf_counter()
    response = client.post("/v1/chat/completions", json=request_body(stream=True))
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed >= 0.035
