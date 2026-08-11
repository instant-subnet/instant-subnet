"""Authenticated platform telemetry and one-shot scoring orchestration."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from instant.protocol import receipts
from instant.protocol.canonical import canonical_json
from instant.protocol.epistula import generate_headers, verify_headers
from instant.validator.coordinator import ScoringCoordinator
from instant.validator.score import load_scoring_config
from instant.validator.state import ProbeResult, open_state
from instant.validator.telemetry import PlatformTelemetryClient, TelemetryError


def stats_payload(now_ms, miner_key, **miner_overrides):
    miner = {
        "hotkey": miner_key.ss58_address,
        "uid": 7,
        "requests": 100,
        "successes": 99,
        "failures": 1,
        "clean_rejects": 1,
        "served": 99,
        "ttft_p50_ms": 100,
        "ttft_p95_ms": 200,
        "tokens_per_s_p50": 120,
        "tokens_per_s_p95": 140,
        "success_rate_bps": 9_900,
        "prompt_tokens": 1_000,
        "completion_tokens": 2_000,
        "receipts_seen": 99,
        "receipts_verified": 99,
        "attestation_ok": False,
        "attestation_id": None,
    }
    miner.update(miner_overrides)
    return {
        "window_start_ms": now_ms - 60_000,
        "window_end_ms": now_ms,
        "block_start": 0,
        "block_end": 0,
        "miners": [miner],
        "receipt_merkle_root": "sha256:" + "00" * 32,
        "total_requests": miner["requests"],
        "generated_at_ms": now_ms,
    }


def signed_stats_app(payload, platform_key, validator_key, *, response_signer=None):
    app = FastAPI()
    seen = {}

    @app.get("/validator/v1/stats")
    async def stats(request: Request):
        seen["request"] = verify_headers(
            request.headers,
            b"",
            expected_signed_for=platform_key.ss58_address,
        )
        raw = canonical_json(payload)
        headers = generate_headers(
            response_signer or platform_key,
            raw,
            signed_for=validator_key.ss58_address,
            timestamp_ms=payload["generated_at_ms"],
        )
        return Response(content=raw, media_type="application/json", headers=headers)

    return app, seen


@pytest.mark.asyncio
async def test_signed_stats_are_roster_bound_and_mapped(
    platform_key, validator_key, miner_key, now_ms
):
    app, seen = signed_stats_app(
        stats_payload(now_ms, miner_key), platform_key, validator_key
    )
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    )
    client = PlatformTelemetryClient(
        base_url="http://platform",
        platform_hotkey=platform_key.ss58_address,
        validator_signer=validator_key,
        http=http,
    )

    batch = await client.fetch([(7, miner_key.ss58_address)], now_ms=now_ms)

    assert seen["request"].signed_by == validator_key.ss58_address
    assert batch.epoch == now_ms // 60_000
    row = batch.rows[0]
    assert (row.clean_rejects, row.served) == (1, 99)
    assert (row.ttft_p95_ms, row.tokens_per_s_p50) == (200, 120)
    await http.aclose()


@pytest.mark.asyncio
async def test_response_must_be_signed_by_configured_platform(
    platform_key, stranger_key, validator_key, miner_key, now_ms
):
    app, _ = signed_stats_app(
        stats_payload(now_ms, miner_key),
        platform_key,
        validator_key,
        response_signer=stranger_key,
    )
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    )
    client = PlatformTelemetryClient(
        base_url="http://platform",
        platform_hotkey=platform_key.ss58_address,
        validator_signer=validator_key,
        http=http,
    )
    with pytest.raises(TelemetryError, match="signature rejected"):
        await client.fetch([(7, miner_key.ss58_address)], now_ms=now_ms)
    await http.aclose()


@pytest.mark.asyncio
async def test_stale_uid_mapping_is_rejected(
    platform_key, validator_key, miner_key, now_ms
):
    app, _ = signed_stats_app(
        stats_payload(now_ms, miner_key), platform_key, validator_key
    )
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    )
    client = PlatformTelemetryClient(
        base_url="http://platform",
        platform_hotkey=platform_key.ss58_address,
        validator_signer=validator_key,
        http=http,
    )
    with pytest.raises(TelemetryError, match="now uid 8"):
        await client.fetch([(8, miner_key.ss58_address)], now_ms=now_ms)
    await http.aclose()


@pytest.mark.asyncio
async def test_inconsistent_signed_counts_are_still_rejected(
    platform_key, validator_key, miner_key, now_ms
):
    payload = stats_payload(now_ms, miner_key, failures=2)
    app, _ = signed_stats_app(payload, platform_key, validator_key)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    )
    client = PlatformTelemetryClient(
        base_url="http://platform",
        platform_hotkey=platform_key.ss58_address,
        validator_signer=validator_key,
        http=http,
    )
    with pytest.raises(TelemetryError, match="counts do not balance"):
        await client.fetch([(7, miner_key.ss58_address)], now_ms=now_ms)
    await http.aclose()


@pytest.mark.asyncio
async def test_coordinator_persists_and_scores_once_idempotently(
    platform_key, validator_key, miner_key, now_ms, monkeypatch
):
    payload = stats_payload(now_ms, miner_key)
    app, _ = signed_stats_app(payload, platform_key, validator_key)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    )
    client = PlatformTelemetryClient(
        base_url="http://platform",
        platform_hotkey=platform_key.ss58_address,
        validator_signer=validator_key,
        http=http,
    )
    state = open_state(":memory:")
    epoch = now_ms // 60_000
    state.record_probes(
        [
            ProbeResult(
                epoch=epoch,
                uid=7,
                hotkey=miner_key.ss58_address,
                source="direct",
                outcome="success",
                observed_ms=300,
                ttft_ms=100,
                tps_milli=120_000,
            )
            for _ in range(20)
        ]
    )
    config = load_scoring_config()
    coordinator = ScoringCoordinator(
        state=state,
        telemetry=client,
        config=config,
        attestation_mode="off",
    )
    snapshot = SimpleNamespace(
        chain_connected=True,
        own_hotkey=validator_key.ss58_address,
        miners=(SimpleNamespace(uid=7, hotkey=miner_key.ss58_address),),
    )
    # The coordinator does not accept a clock injection; pin the fetcher's
    # validation boundary while retaining fresh, unique response signatures.
    import instant.validator.telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "_now_ms", lambda: now_ms)

    first, _ = await coordinator.run_once(snapshot)
    carried = state.carry_forward()
    second, _ = await coordinator.run_once(snapshot)

    assert first.empty is False
    assert first.total_weight == 65_535
    assert second.reused is True
    assert state.carry_forward() == carried
    assert state.telemetry_for_epoch(epoch)[0]["requests"] == 100
    await coordinator.aclose()
    state.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_score_once_runs_twenty_real_receipt_verified_direct_probes(
    platform_key, validator_key, miner_key, now_ms, monkeypatch
):
    platform_app, _ = signed_stats_app(
        stats_payload(now_ms, miner_key), platform_key, validator_key
    )
    platform_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=platform_app),
        base_url="http://platform",
    )
    telemetry = PlatformTelemetryClient(
        base_url="http://platform",
        platform_hotkey=platform_key.ss58_address,
        validator_signer=validator_key,
        http=platform_http,
    )
    async def miner_handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers=[validator_key.ss58_address],
            expected_signed_for=miner_key.ss58_address,
        )
        # Each probe carries its own nonce-prefixed challenge and the probe
        # streams, so a working miner has to answer over SSE and sign the
        # concatenated data payloads it actually sent.
        prompt = json.loads(request_body)["messages"][0]["content"]
        first, last = re.search(r"from (\d+) to (\d+)", prompt).groups()
        answer = " ".join(str(v) for v in range(int(first), int(last) + 1))
        frames = [
            json.dumps(
                {"choices": [{"delta": {"content": piece}}]}, separators=(",", ":")
            ).encode()
            for piece in re.findall(r"\s*\S+", answer)
        ]
        signed = receipts.build(
            miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=b"".join(frames),
            prompt_tokens=5,
            completion_tokens=4,
            ttft_ms_self=1,
            total_ms_self=2,
            started_at_ms=now_ms,
            finished_at_ms=now_ms + 2,
        )
        wire = b"".join(b"data: " + frame + b"\n\n" for frame in frames)
        wire += b"data: [DONE]\n\n"
        wire += (
            b"event: "
            + receipts.SSE_RECEIPT_EVENT.encode()
            + b"\ndata: "
            + json.dumps(signed.to_payload(), separators=(",", ":")).encode()
            + b"\n\n"
        )
        return httpx.Response(
            200, content=wire, headers={"Content-Type": "text/event-stream"}
        )

    probe_http = httpx.AsyncClient(transport=httpx.MockTransport(miner_handler))
    state = open_state(":memory:")
    config = load_scoring_config()
    coordinator = ScoringCoordinator(
        state=state,
        telemetry=telemetry,
        config=config,
        attestation_mode="off",
        probe_http=probe_http,
        probe_signer=validator_key,
        probe_model="instant/mock-echo",
    )
    snapshot = SimpleNamespace(
        chain_connected=True,
        own_hotkey=validator_key.ss58_address,
        miners=(
            SimpleNamespace(
                uid=7,
                hotkey=miner_key.ss58_address,
                url="http://miner.test:8091",
            ),
        ),
    )
    import instant.validator.telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "_now_ms", lambda: now_ms)

    run, _ = await coordinator.run_once(snapshot)

    assert run.direct_probe_attempts == config.probe.direct_count == 20
    assert run.direct_probe_successes == 20
    assert run.empty is False
    assert run.total_weight == 65_535
    assert state.probe_summary(run.epoch, source="direct") == {
        "attempts": 20,
        "successes": 20,
    }
    await coordinator.aclose()
    state.close()
    await platform_http.aclose()
