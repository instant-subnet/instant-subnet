from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from instant.platform.state import UNFINISHED_GRACE_MS, KeyRevokedError, open_state
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
    # A miner-supplied attestation id is not verified hardware evidence.
    assert miner.attestation_ok is False
    assert miner.attestation_id is None
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


def test_an_unfinished_attempt_is_excluded_until_it_closes(tmp_path, miner_key):
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
    assert (stats.requests, stats.successes, stats.failures) == (0, 0, 0)


def test_a_stale_unfinished_attempt_eventually_counts_as_failure(tmp_path, miner_key):
    state = open_state(tmp_path / "stale-pending.sqlite3")
    state.begin(
        request_id="crash-orphan",
        miner_hotkey=miner_key.ss58_address,
        started_ms=1_000,
        stream=True,
    )
    generated_at = 1_000 + UNFINISHED_GRACE_MS + 1
    stats = state.stats(
        miner_hotkey=miner_key.ss58_address,
        miner_uid=0,
        window_start_ms=0,
        window_end_ms=generated_at,
        generated_at_ms=generated_at,
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
    db.executescript(state_mod._SCHEMA_REQUESTS_V1)
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


def test_a_failed_migration_rolls_back_ddl_and_version(tmp_path, monkeypatch):
    import sqlite3

    from instant.platform import state as state_mod

    path = tmp_path / "broken-v1.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(state_mod._SCHEMA_REQUESTS_V1)
    db.execute("PRAGMA user_version=1")
    db.commit()
    db.close()

    monkeypatch.setitem(
        state_mod._MIGRATIONS,
        1,
        "CREATE TABLE must_rollback (value TEXT); THIS IS NOT SQL;",
    )
    with pytest.raises(sqlite3.DatabaseError):
        open_state(path)

    check = sqlite3.connect(path)
    try:
        version = check.execute("PRAGMA user_version").fetchone()[0]
        table = check.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='must_rollback'"
        ).fetchone()
    finally:
        check.close()
    assert version == 1
    assert table is None


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

        with pytest.raises(KeyRevokedError):
            state.register_key(
                key_id="k1",
                prefix="isk_aaaa",
                last4="zzzz",
                digest=digest,
                label="second",
                created_ms=3,
            )
        assert not state.key_is_active(digest), "a revoked key came back to life"
    finally:
        state.close()


def test_a_v2_database_adds_nullable_key_attribution_without_backfill(tmp_path):
    """Old telemetry cannot be safely guessed onto a newly registered key."""
    import sqlite3

    from instant.platform import state as state_mod

    path = tmp_path / "v2.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(state_mod._SCHEMA_REQUESTS_V1 + state_mod._SCHEMA_KEYS)
    db.execute("PRAGMA user_version=2")
    db.execute(
        "INSERT INTO api_keys "
        "(key_id, prefix, last4, digest, label, created_ms) "
        "VALUES ('customer','isk_test','last','digest','ditto',1)"
    )
    db.execute(
        "INSERT INTO requests (request_id, miner_hotkey, started_ms, stream, success) "
        "VALUES ('historical','5Miner',1000,0,1)"
    )
    db.commit()
    db.close()

    migrated = open_state(path)
    try:
        row = migrated._db.execute(
            "SELECT api_key_id FROM requests WHERE request_id='historical'"
        ).fetchone()
        assert row[0] is None
        assert migrated.key_usage("customer")["requests"] == 0
        assert (
            migrated._db.execute("PRAGMA user_version").fetchone()[0]
            == state_mod.SCHEMA_VERSION
        )
    finally:
        migrated.close()


def test_a_failed_v2_migration_rolls_back_column_index_and_version(
    tmp_path, monkeypatch
):
    import sqlite3

    from instant.platform import state as state_mod

    path = tmp_path / "broken-v2.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(state_mod._SCHEMA_REQUESTS_V1 + state_mod._SCHEMA_KEYS)
    db.execute("PRAGMA user_version=2")
    db.commit()
    db.close()

    monkeypatch.setitem(
        state_mod._MIGRATIONS,
        2,
        "ALTER TABLE requests ADD COLUMN api_key_id TEXT; THIS IS NOT SQL;",
    )
    with pytest.raises(sqlite3.DatabaseError):
        open_state(path)

    check = sqlite3.connect(path)
    try:
        version = check.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1] for row in check.execute("PRAGMA table_info(requests)").fetchall()
        }
        index = check.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='index' AND name='requests_by_api_key'"
        ).fetchone()
    finally:
        check.close()
    assert version == 2
    assert "api_key_id" not in columns
    assert index is None


def test_per_key_usage_is_exact_under_concurrency_and_ignores_unverified_tokens(
    tmp_path, miner_key, platform_key
):
    state = open_state(tmp_path / "concurrent-key-usage.sqlite3")
    try:
        state.register_key(
            key_id="key-a",
            prefix="isk_alpha",
            last4="aaaa",
            digest="a" * 64,
            label="alpha",
            created_ms=1,
        )
        state.register_key(
            key_id="key-b",
            prefix="isk_bravo",
            last4="bbbb",
            digest="b" * 64,
            label="bravo",
            created_ms=2,
        )

        work = []
        for index in range(40):
            request_id = f"a-{index}"
            success = index % 5 != 0
            signed = (
                receipts.build(
                    miner_key,
                    request_id=request_id,
                    signer_of_request=platform_key.ss58_address,
                    request_body=b"request",
                    response_body=b"response",
                    prompt_tokens=2,
                    completion_tokens=3,
                    ttft_ms_self=1,
                    total_ms_self=2,
                    started_at_ms=1_000 + index,
                    finished_at_ms=1_010 + index,
                )
                if success
                else None
            )
            work.append(
                (request_id, "key-a", 1_000 + index, success, signed, 2, 3)
            )
        for index in range(30):
            request_id = f"b-{index}"
            signed = receipts.build(
                miner_key,
                request_id=request_id,
                signer_of_request=platform_key.ss58_address,
                request_body=b"request",
                response_body=b"response",
                prompt_tokens=1,
                completion_tokens=1,
                ttft_ms_self=1,
                total_ms_self=2,
                started_at_ms=2_000 + index,
                finished_at_ms=2_010 + index,
            )
            work.append((request_id, "key-b", 2_000 + index, True, signed, 1, 1))

        def write(record):
            request_id, key_id, started_ms, success, signed, prompt, completion = record
            state.begin(
                request_id=request_id,
                miner_hotkey=miner_key.ss58_address,
                started_ms=started_ms,
                stream=True,
                api_key_id=key_id,
            )
            state.finish(
                request_id=request_id,
                finished_ms=started_ms + 10,
                status_code=200 if success else 502,
                success=success,
                prompt_tokens=prompt if success else 100_000,
                completion_tokens=completion if success else 100_000,
                signed_receipt=signed,
                receipt_seen=success,
                receipt_verified=success,
                error=None if success else "invalid receipt",
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(write, work))

        # A service-credential request remains deliberately unattributed.
        state.begin(
            request_id="service",
            miner_hotkey=miner_key.ss58_address,
            started_ms=9_999,
            stream=False,
            api_key_id=None,
        )
        state.finish(
            request_id="service",
            finished_ms=10_000,
            status_code=500,
            success=False,
            prompt_tokens=100_000,
            completion_tokens=100_000,
        )

        listed = {row["key_id"]: row for row in state.list_keys()}
        assert listed["key-a"] == {
            "key_id": "key-a",
            "prefix": "isk_alpha",
            "last4": "aaaa",
            "label": "alpha",
            "created_ms": 1,
            "revoked_ms": None,
            "last_used_ms": 1_039,
            "requests": 40,
            "successful_requests": 32,
            "verified_requests": 32,
            "prompt_tokens": 64,
            "completion_tokens": 96,
            "total_tokens": 160,
        }
        assert listed["key-b"]["last_used_ms"] == 2_029
        assert listed["key-b"]["requests"] == 30
        assert listed["key-b"]["prompt_tokens"] == 30
        assert listed["key-b"]["completion_tokens"] == 30
        assert listed["key-b"]["total_tokens"] == 60

        assert state.revoke_key(key_id="key-a", revoked_ms=3_000)
        retained = state.key_usage("key-a")
        assert retained is not None
        assert retained["revoked_ms"] == 3_000
        assert retained["last_used_ms"] == 1_039
        assert retained["total_tokens"] == 160
        assert state.key_usage("missing") is None
    finally:
        state.close()
