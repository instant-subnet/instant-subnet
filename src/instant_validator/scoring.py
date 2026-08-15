"""One deterministic MVP scoring function."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .report import MinerMetrics

BPS = 10_000
MAX_WEIGHT_U16 = 65_535
TTFT_P95_TARGET_MS = 750
TOKENS_PER_SECOND_TARGET = 50
SCORING_VERSION = 1


def score_miner(miner: MinerMetrics) -> int:
    """Score a miner from 0 to 10,000 using integer arithmetic only.

    Success rate and verified TOPLOC coverage each contribute 40%. TTFT and
    throughput each contribute 10%. A miner with no successful, verified work
    receives zero.
    """

    if miner.successes == 0 or miner.toploc_verified == 0:
        return 0
    success = miner.successes * BPS // miner.requests
    proof = miner.toploc_verified * BPS // miner.requests
    latency = min(BPS, TTFT_P95_TARGET_MS * BPS // miner.ttft_p95_ms)
    throughput = min(
        BPS,
        miner.tokens_per_second_p50 * BPS // TOKENS_PER_SECOND_TARGET,
    )
    return (success * 40 + proof * 40 + latency * 10 + throughput * 10) // 100


def score_miners(miners: Iterable[MinerMetrics]) -> dict[int, int]:
    """Return the score for each unique UID."""

    return {miner.uid: score_miner(miner) for miner in miners}


def normalize_weights(scores: Mapping[int, int]) -> dict[int, int]:
    """Scale positive scores so the highest emitted u16 weight is 65,535."""

    positive = {int(uid): int(score) for uid, score in scores.items() if score > 0}
    if not positive:
        return {}
    highest = max(positive.values())
    return {
        uid: (positive[uid] * MAX_WEIGHT_U16 + highest // 2) // highest
        for uid in sorted(positive)
    }
