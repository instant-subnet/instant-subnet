from __future__ import annotations

import pytest

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


def test_a_v1_database_migrates_without_losing_telemetry(tmp_path):
    """The deployed database holds request and receipt history that scoring
    reads. A migration that drops it would silently erase evidence a miner has
    already been paid for, so prove the rows survive."""
    import sqlite3

    from instant.platform import state as state_mod

    path = tmp_path / "v1.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(state_mod._SCHEMA.replace(state_mod._SCHEMA_KEYS, ""))
    db.execute("PRAGMA user_version=1")
    db.execute(
        "INSERT INTO requests (request_id, miner_hotkey, started_ms, stream, success) "
        "VALUES ('req-1','5Miner',1000,0,1)"
    )
    db.commit()
    db.close()

    migrated = open_state(path)
    try:
        rows = migrated._db.execute("SELECT request_id FROM requests").fetchall()
        assert [r[0] for r in rows] == ["req-1"], "telemetry lost during migration"
        version = migrated._db.execute("PRAGMA user_version").fetchone()[0]
        assert version == state_mod.SCHEMA_VERSION
        # And the new table is usable.
        migrated.register_key(
            key_id="k", prefix="isk_abcd", last4="wxyz",
            digest="d" * 64, label="", created_ms=1,
        )
        assert migrated.key_is_active("d" * 64)
    finally:
        migrated.close()


def test_a_future_schema_is_refused_rather_than_guessed(tmp_path):
    import sqlite3

    path = tmp_path / "future.sqlite3"
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version=99")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="newer than this build"):
        open_state(path)


def test_registering_a_revoked_digest_does_not_resurrect_it(tmp_path):
    """Registration must never undo a revocation, or revoke is bypassable."""
    state = open_state(tmp_path / "keys.sqlite3")
    try:
        digest = "a" * 64
        state.register_key(key_id="k1", prefix="isk_aaaa", last4="zzzz",
                           digest=digest, label="first", created_ms=1)
        assert state.key_is_active(digest)
        assert state.revoke_key(key_id="k1", revoked_ms=2)
        assert not state.key_is_active(digest)

        state.register_key(key_id="k1", prefix="isk_aaaa", last4="zzzz",
                           digest=digest, label="second", created_ms=3)
        assert not state.key_is_active(digest), "a revoked key came back to life"
    finally:
        state.close()
