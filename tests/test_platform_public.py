"""The public stats projection: what it drops, and what it costs to serve."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from instant.platform.public import (
    PublicStats,
    PublicStatsCache,
    encode,
    project,
)
from instant.protocol.schemas import MinerStatsWindow, StatsResponse

GOLDEN = Path(__file__).parent / "data" / "public_stats_example.json"


def sample_stats(now_ms: int) -> StatsResponse:
    """A window with enough shape that dropped fields would be noticed."""
    return StatsResponse(
        window_start_ms=now_ms - 3_600_000,
        window_end_ms=now_ms,
        block_start=0,
        block_end=0,
        generated_at_ms=now_ms,
        total_requests=1204,
        receipt_merkle_root="ab" * 32,
        miners=[
            MinerStatsWindow(
                hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                uid=1,
                requests=1204,
                successes=1197,
                failures=7,
                clean_rejects=3,
                served=1197,
                ttft_p50_ms=184,
                ttft_p95_ms=412,
                tokens_per_s_p50=42,
                tokens_per_s_p95=51,
                success_rate_bps=9941,
                prompt_tokens=21044,
                completion_tokens=88301,
                receipts_seen=1204,
                receipts_verified=1197,
                attestation_ok=False,
                attestation_id=None,
            )
        ],
    )


def test_projection_drops_the_validator_only_fields(now_ms):
    payload = json.loads(encode(project(sample_stats(now_ms))))

    # The merkle root is the validator's audit anchor over the receipt set. It
    # is not a public commitment and must not leak into a browser contract.
    assert "receipt_merkle_root" not in payload
    # Never populated by the platform, so publishing them would be noise that
    # someone eventually mistakes for a real block range.
    assert "block_start" not in payload
    assert "block_end" not in payload

    miner = payload["miners"][0]
    assert "hotkey" not in miner
    assert "served" not in miner


def test_projection_preserves_every_number_unchanged(now_ms):
    source = sample_stats(now_ms)
    projected = project(source)
    miner, public_miner = source.miners[0], projected.miners[0]

    # Units are the aggregation's, untouched: whole tokens/s, milliseconds,
    # and basis points. A conversion here would silently disagree with what
    # the validator is scoring on.
    assert public_miner.tokens_per_s_p50 == miner.tokens_per_s_p50 == 42
    assert public_miner.ttft_p50_ms == miner.ttft_p50_ms == 184
    assert public_miner.success_rate_bps == miner.success_rate_bps == 9941
    assert projected.total_requests == source.total_requests
    assert projected.generated_at_ms == source.generated_at_ms


def test_attestation_fields_are_carried_even_though_they_are_empty(now_ms):
    # Attestation is off, so these are false/null today. They are in the
    # contract so that surfacing attestation later is a frontend-only change.
    projected = project(sample_stats(now_ms))
    assert projected.miners[0].attestation_ok is False
    assert projected.miners[0].attestation_id is None


def test_public_model_rejects_unknown_fields():
    with pytest.raises(ValueError):
        PublicStats(
            generated_at_ms=0,
            window_start_ms=0,
            window_end_ms=0,
            total_requests=0,
            miners=[],
            receipt_merkle_root="ab" * 32,
        )


def test_golden_file_matches_the_projection(now_ms):
    """The golden file is the contract, readable in the diff.

    It is also what the frontend develops against, so a change here that
    nobody noticed is a change that breaks the page.
    """
    assert GOLDEN.read_bytes() == encode(project(sample_stats(now_ms))) + b"\n"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_serves_one_computation_per_ttl():
    clock = _Clock()
    cache = PublicStatsCache(ttl_s=10.0, clock=clock)
    calls = []

    def compute() -> bytes:
        calls.append(clock.now)
        return f"body-{len(calls)}".encode()

    assert cache.get(compute) == b"body-1"
    clock.now = 9.9
    assert cache.get(compute) == b"body-1"
    assert len(calls) == 1

    clock.now = 10.0
    assert cache.get(compute) == b"body-2"
    assert len(calls) == 2


def test_cache_with_zero_ttl_always_recomputes():
    clock = _Clock()
    cache = PublicStatsCache(ttl_s=0.0, clock=clock)
    calls = []

    def compute() -> bytes:
        calls.append(1)
        return b"body"

    cache.get(compute)
    cache.get(compute)
    assert len(calls) == 2


def test_cache_rejects_a_negative_ttl():
    with pytest.raises(ValueError):
        PublicStatsCache(ttl_s=-1.0)
