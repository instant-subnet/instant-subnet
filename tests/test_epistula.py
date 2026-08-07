"""Epistula v2 round-trips, and every rejection path.

The interesting tests here are the negative ones. A signing layer that
accepts valid requests is easy; the value is in what it refuses.
"""

import pytest

from instant.protocol import epistula
from instant.protocol.epistula import (
    ALLOWED_DELTA_MS,
    ALLOWED_FUTURE_MS,
    EpistulaError,
    ReplayGuard,
    generate_headers,
    message_for,
    verify_headers,
)

BODY = b'{"model":"openai/gpt-oss-120b","messages":[]}'


def test_round_trip(miner_key, platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    v = verify_headers(h, BODY, now_ms=now_ms)
    assert v.signed_by == platform_key.ss58_address
    assert v.signed_for is None
    assert v.timestamp_ms == now_ms


def test_signed_for_round_trip(miner_key, platform_key, now_ms):
    h = generate_headers(
        platform_key, BODY, miner_key.ss58_address, timestamp_ms=now_ms
    )
    v = verify_headers(
        h, BODY, expected_signed_for=miner_key.ss58_address, now_ms=now_ms
    )
    assert v.signed_for == miner_key.ss58_address


def test_signed_for_emits_three_secret_signatures(platform_key, miner_key, now_ms):
    h = generate_headers(
        platform_key, BODY, miner_key.ss58_address, timestamp_ms=now_ms
    )
    for i in range(3):
        assert epistula.H_SECRET.format(i) in h


def test_no_signed_for_means_no_secret_signatures(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    assert epistula.H_SECRET.format(0) not in h
    assert epistula.H_SIGNED_FOR not in h


def test_message_uses_empty_string_for_absent_signed_for():
    # This is the bug that verifies locally and fails everywhere else.
    m = message_for(b"x", "uuid-1", 123, None)
    assert m.endswith(".123.")
    assert "None" not in m


def test_body_tampering_is_caught(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    with pytest.raises(EpistulaError, match="signature mismatch"):
        verify_headers(h, BODY + b" ", now_ms=now_ms)


def test_stale_request_is_rejected(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    with pytest.raises(EpistulaError, match="stale"):
        verify_headers(h, BODY, now_ms=now_ms + ALLOWED_DELTA_MS + 1)


def test_request_at_the_edge_of_the_window_is_accepted(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    verify_headers(h, BODY, now_ms=now_ms + ALLOWED_DELTA_MS)


def test_future_request_is_rejected(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms + 60_000)
    with pytest.raises(EpistulaError, match="future"):
        verify_headers(h, BODY, now_ms=now_ms)


def test_small_forward_skew_is_tolerated(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms + ALLOWED_FUTURE_MS)
    verify_headers(h, BODY, now_ms=now_ms)


@pytest.mark.parametrize(
    "drop,expected",
    [
        (epistula.H_SIGNED_BY, "Signed-By"),
        (epistula.H_SIGNATURE, "Request-Signature"),
        (epistula.H_UUID, "Uuid"),
        (epistula.H_TIMESTAMP, "Timestamp"),
    ],
)
def test_missing_headers_are_named(platform_key, now_ms, drop, expected):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    del h[drop]
    with pytest.raises(EpistulaError, match=expected):
        verify_headers(h, BODY, now_ms=now_ms)


def test_non_integer_timestamp(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    h[epistula.H_TIMESTAMP] = "soon"
    with pytest.raises(EpistulaError, match="not an integer"):
        verify_headers(h, BODY, now_ms=now_ms)


def test_wrong_version_is_rejected(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    h[epistula.H_VERSION] = "1"
    with pytest.raises(EpistulaError, match="version"):
        verify_headers(h, BODY, now_ms=now_ms)


def test_header_lookup_is_case_insensitive(platform_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    lowered = {k.lower(): v for k, v in h.items()}
    assert verify_headers(lowered, BODY, now_ms=now_ms).signed_by


def test_accept_list_blocks_unknown_signers(platform_key, stranger_key, now_ms):
    h = generate_headers(stranger_key, BODY, timestamp_ms=now_ms)
    with pytest.raises(EpistulaError, match="accept-list"):
        verify_headers(
            h, BODY, allowed_signers=[platform_key.ss58_address], now_ms=now_ms
        )


def test_accept_list_admits_known_signers(platform_key, stranger_key, now_ms):
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    verify_headers(
        h,
        BODY,
        allowed_signers=[stranger_key.ss58_address, platform_key.ss58_address],
        now_ms=now_ms,
    )


def test_request_for_another_recipient_is_rejected(
    platform_key, miner_key, stranger_key, now_ms
):
    # A perfectly valid signature, addressed to someone else. This is the
    # laundering attack Signed-For exists to stop.
    h = generate_headers(
        platform_key, BODY, stranger_key.ss58_address, timestamp_ms=now_ms
    )
    with pytest.raises(EpistulaError, match="different recipient"):
        verify_headers(
            h, BODY, expected_signed_for=miner_key.ss58_address, now_ms=now_ms
        )


def test_replay_is_rejected(platform_key, now_ms):
    guard = ReplayGuard()
    h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
    verify_headers(h, BODY, replay_guard=guard, now_ms=now_ms)
    with pytest.raises(EpistulaError, match="replayed"):
        verify_headers(h, BODY, replay_guard=guard, now_ms=now_ms)


def test_replay_guard_sweeps_old_entries(platform_key, now_ms):
    guard = ReplayGuard()
    for i in range(100):
        guard.check_and_record(f"uuid-{i}", now_ms, now_ms)
    assert len(guard) == 100
    # Far enough ahead that nothing recorded could still be accepted.
    guard.check_and_record("fresh", now_ms + 60_000, now_ms + 60_000)
    assert len(guard) == 1


def test_replay_guard_does_not_grow_without_bound(platform_key, now_ms):
    guard = ReplayGuard()
    for i in range(5_000):
        t = now_ms + i * 10
        guard.check_and_record(f"uuid-{i}", t, t)
    # 5000 requests over 50s, with a 10s window: memory tracks rate, not total.
    assert len(guard) < 2_000


def test_distinct_uuids_are_not_replays(platform_key, now_ms):
    guard = ReplayGuard()
    for _ in range(5):
        h = generate_headers(platform_key, BODY, timestamp_ms=now_ms)
        verify_headers(h, BODY, replay_guard=guard, now_ms=now_ms)
