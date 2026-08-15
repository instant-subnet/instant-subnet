from dataclasses import replace

from instant_validator.scoring import (
    MAX_WEIGHT_U16,
    normalize_weights,
    score_miner,
    score_miners,
)


def test_golden_metrics_produce_exact_scores_and_weights(parsed_report):
    scores = score_miners(parsed_report.miners)
    weights = normalize_weights(scores)

    assert scores == {12: 9840, 37: 8753}
    assert max(weights.values()) == MAX_WEIGHT_U16
    assert weights == {12: 65_535, 37: 58_296}


def test_better_metrics_never_reduce_score(parsed_report):
    baseline = parsed_report.miners[1]
    better = replace(
        baseline,
        successes=48,
        failures=2,
        ttft_p95_ms=500,
        tokens_per_second_p50=70,
        toploc_verified=48,
        toploc_failed=1,
        toploc_timed_out=1,
    )
    assert score_miner(better) > score_miner(baseline)


def test_no_success_or_no_verified_toploc_gets_zero(parsed_report):
    miner = parsed_report.miners[0]
    assert (
        score_miner(
            replace(
                miner,
                successes=0,
                failures=miner.requests,
                ttft_p50_ms=0,
                ttft_p95_ms=0,
                tokens_per_second_p50=0,
            )
        )
        == 0
    )
    assert (
        score_miner(
            replace(
                miner,
                toploc_verified=0,
                toploc_failed=miner.requests,
                toploc_timed_out=0,
            )
        )
        == 0
    )


def test_normalization_is_deterministic_and_drops_zero_scores():
    assert normalize_weights({9: 1, 3: 1, 7: 0}) == {3: 65_535, 9: 65_535}
    assert normalize_weights({1: 0}) == {}
