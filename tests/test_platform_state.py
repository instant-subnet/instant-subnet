from __future__ import annotations

from instant.platform.state import open_state
from instant.protocol import receipts


def test_platform_telemetry_survives_reopen(tmp_path, miner_key, platform_key):
    path = tmp_path / "nested" / "platform.sqlite3"
    request_id = "request-1"
    request_body = b"request"
    response_body = b"response"
    signed = receipts.build(
        miner_key,
        request_id=request_id,
        signer_of_request=platform_key.ss58_address,
        request_body=request_body,
        response_body=response_body,
        prompt_tokens=7,
        completion_tokens=11,
        ttft_ms_self=1,
        total_ms_self=2,
        started_at_ms=1_000,
        finished_at_ms=1_010,
        attestation_id="attest-1",
    )
    state = open_state(path)
    state.begin(
        request_id=request_id,
        miner_hotkey=miner_key.ss58_address,
        started_ms=1_000,
        stream=True,
    )
    state.finish(
        request_id=request_id,
        finished_ms=1_100,
        status_code=200,
        success=True,
        ttft_ms=25,
        total_ms=100,
        tps_milli=110_000,
        prompt_tokens=7,
        completion_tokens=11,
        signed_receipt=signed,
        receipt_seen=True,
        receipt_verified=True,
    )
    state.close()

    reopened = open_state(path)
    stats = reopened.stats(
        miner_hotkey=miner_key.ss58_address,
        miner_uid=3,
        window_start_ms=0,
        window_end_ms=2_000,
        generated_at_ms=2_000,
    )
    reopened.close()
    miner = stats.miners[0]
    assert (miner.requests, miner.successes, miner.failures) == (1, 1, 0)
    assert (miner.ttft_p50_ms, miner.ttft_p95_ms) == (25, 25)
    assert (miner.tokens_per_s_p50, miner.tokens_per_s_p95) == (110, 110)
    assert (miner.receipts_seen, miner.receipts_verified) == (1, 1)
    assert miner.attestation_ok is True
    assert stats.receipt_merkle_root == receipts.merkle_root([signed])


def test_stats_invariants_and_nearest_rank_percentiles(
    tmp_path, miner_key, platform_key
):
    state = open_state(tmp_path / "stats.sqlite3")
    for index, (success, clean, ttft, tps) in enumerate(
        [
            (True, False, 10, 10_000),
            (True, False, 20, 20_000),
            (False, True, None, None),
        ]
    ):
        request_id = f"request-{index}"
        signed = (
            receipts.build(
                miner_key,
                request_id=request_id,
                signer_of_request=platform_key.ss58_address,
                request_body=b"request",
                response_body=b"response",
                prompt_tokens=1,
                completion_tokens=1,
                ttft_ms_self=1,
                total_ms_self=2,
                started_at_ms=1_000,
                finished_at_ms=1_002,
            )
            if success
            else None
        )
        state.begin(
            request_id=request_id,
            miner_hotkey=miner_key.ss58_address,
            started_ms=1_000 + index,
            stream=True,
        )
        state.finish(
            request_id=request_id,
            finished_ms=1_100 + index,
            status_code=200 if success else 429,
            success=success,
            clean_reject=clean,
            ttft_ms=ttft,
            tps_milli=tps,
            signed_receipt=signed,
            receipt_seen=success,
            receipt_verified=success,
        )
    stats = state.stats(
        miner_hotkey=miner_key.ss58_address,
        miner_uid=9,
        window_start_ms=0,
        window_end_ms=2_000,
        generated_at_ms=2_000,
    ).miners[0]
    state.close()
    assert stats.successes + stats.failures == stats.requests == 3
    assert stats.clean_rejects == 1
    assert stats.served == 2
    assert stats.success_rate_bps == 6_666
    assert stats.ttft_p50_ms == 10
    assert stats.ttft_p95_ms == 20


def test_an_unfinished_attempt_is_a_failure_not_forgotten(tmp_path, miner_key):
    state = open_state(tmp_path / "pending.sqlite3")
    state.begin(
        request_id="interrupted",
        miner_hotkey=miner_key.ss58_address,
        started_ms=1_000,
        stream=True,
    )
    stats = state.stats(
        miner_hotkey=miner_key.ss58_address,
        miner_uid=0,
        window_start_ms=0,
        window_end_ms=2_000,
        generated_at_ms=2_000,
    ).miners[0]
    state.close()
    assert (stats.requests, stats.successes, stats.failures) == (1, 0, 1)
