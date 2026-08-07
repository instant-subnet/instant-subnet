"""Boundary validation. These models are the only thing between a hostile
payload and vLLM, so the tests are mostly about what they refuse."""

import pytest
from pydantic import ValidationError

from instant.protocol.schemas import (
    AttestRequest,
    CapacityResponse,
    ChatCompletionRequest,
    HealthResponse,
    ManifestResponse,
    MinerStatsWindow,
    StatsResponse,
)

MINIMAL = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "hi"}]}


def test_minimal_request_parses():
    r = ChatCompletionRequest(**MINIMAL)
    assert r.stream is False
    assert r.n == 1


def test_unknown_field_is_rejected():
    # A silently-ignored 'temperture' typo means two miners sample
    # differently for the same request and the scores stop meaning anything.
    with pytest.raises(ValidationError, match="temperture"):
        ChatCompletionRequest(**MINIMAL, temperture=0.7)


def test_empty_messages_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])


def test_bad_role_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "wizard", "content": "hi"}])


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("max_tokens", 0),
        ("max_tokens", 200_000),
        ("presence_penalty", 3),
        ("frequency_penalty", -3),
    ],
)
def test_out_of_range_sampling_params(field, value):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**MINIMAL, **{field: value})


def test_n_greater_than_one_is_rejected():
    # One completion per request keeps the receipt one-to-one with the work.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**MINIMAL, n=2)


def test_stop_sequences_are_bounded():
    ChatCompletionRequest(**MINIMAL, stop=["a", "b", "c", "d"])
    with pytest.raises(ValidationError, match="4 stop"):
        ChatCompletionRequest(**MINIMAL, stop=["a", "b", "c", "d", "e"])


def test_reasoning_effort_passthrough():
    assert ChatCompletionRequest(**MINIMAL, reasoning_effort="high").reasoning_effort == "high"
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**MINIMAL, reasoning_effort="maximum")


def test_multimodal_content_shape_is_allowed():
    ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )


def test_health_response():
    h = HealthResponse(
        status="loading", ready=False, version="0.1.0",
        model_id="openai/gpt-oss-120b", uptime_s=3, attestation_mode="hard",
    )
    assert h.ready is False
    with pytest.raises(ValidationError):
        HealthResponse(
            status="fine", ready=True, version="0.1.0", model_id="m",
            uptime_s=1, attestation_mode="hard",
        )


def test_attestation_mode_is_constrained():
    with pytest.raises(ValidationError):
        HealthResponse(
            status="ok", ready=True, version="0.1.0", model_id="m",
            uptime_s=1, attestation_mode="maybe",
        )


def test_capacity_rejects_negative_counts():
    CapacityResponse(max_concurrent=8, in_flight=0, queue_depth=0,
                     max_model_len=131072, accepting=True)
    with pytest.raises(ValidationError):
        CapacityResponse(max_concurrent=8, in_flight=-1, queue_depth=0,
                         max_model_len=131072, accepting=True)


def test_manifest_optional_attestation():
    m = ManifestResponse(
        hotkey="5Grw", model_id="openai/gpt-oss-120b",
        weights_digest="sha256:" + "11" * 32,
        image_digest="sha256:" + "22" * 32, max_model_len=131072,
    )
    assert m.attestation_id is None


def test_attest_request_nonce_length():
    AttestRequest(nonce="ab" * 32)
    with pytest.raises(ValidationError):
        AttestRequest(nonce="ab")


def test_success_rate_is_basis_points():
    kw = dict(
        hotkey="5Grw", uid=1, requests=100, successes=99, failures=1,
        ttft_p50_ms=40, ttft_p95_ms=90, tokens_per_s_p50=120,
        tokens_per_s_p95=180, prompt_tokens=1, completion_tokens=1,
        receipts_seen=99, receipts_verified=99, attestation_ok=True,
    )
    MinerStatsWindow(success_rate_bps=9_900, **kw)
    with pytest.raises(ValidationError):
        MinerStatsWindow(success_rate_bps=10_001, **kw)


def test_stats_response_shape():
    s = StatsResponse(
        window_start_ms=1, window_end_ms=2, block_start=10, block_end=20,
        miners=[], receipt_merkle_root="sha256:" + "00" * 32,
        total_requests=0, generated_at_ms=3,
    )
    assert s.miners == []
