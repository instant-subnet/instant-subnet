"""Scoring. DESIGN.md §8.

Two kinds of test live here and they are testing different fears.

The first kind is arithmetic: does ``latency_bps`` implement the formula in
§8.1. Those are quick and boring and they are not the interesting ones.

The second kind is determinism. This module decides emissions, and two
validators that compute *nearly* the same weight vector are worse off than
two that compute wildly different ones, because the near-miss looks like
agreement right up until it costs someone their vtrust. So there are tests
here that assert byte-identical output under reordering, tests that assert
the apportioned weights sum to exactly 65535, and a test that the whole
pipeline never touches a float — asserted by walking the results rather than
by trusting review.
"""

from __future__ import annotations

import dataclasses
import math
import random
from copy import deepcopy

import pytest

from instant.common.config import ConfigError
from instant.validator.score import (
    BPS_ONE,
    EMPTY,
    LOG2_FRAC_BITS,
    MAX_WEIGHT_U16,
    Components,
    MinerObservations,
    ScoringConfig,
    SourceMix,
    SourceSample,
    blend,
    capacity_bps,
    credited_served,
    cube_normalise,
    ema,
    gate_bps,
    latency_bps,
    load_scoring_config,
    log2_fp,
    penalty_bps,
    percentile,
    reliability_bps,
    score_epoch,
    throughput_bps,
)


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    """The *shipped* config, not a fixture-built one.

    Every test below therefore doubles as a check that config/slo.toml and
    config/scoring.toml are internally consistent and load at all.
    """
    return load_scoring_config()


def probes(
    n: int = 25, *, ttft_ms: int = 250, tps_milli: int = 100_000, served: int = 0
) -> SourceSample:
    """A source that made ``n`` attempts and succeeded at all of them."""
    return SourceSample(
        attempts=n,
        successes=n,
        ttft_ms=(ttft_ms,) * n,
        tps_milli=(tps_milli,) * n,
        served=served,
    )


def healthy(uid: int = 0, hotkey: str = "5Healthy", **overrides) -> MinerObservations:
    """A miner meeting every target exactly, which should score 10000."""
    base = {
        "shadow": probes(),
        "direct": probes(),
        "telemetry": probes(served=100),
        "attested": True,
    }
    base.update(overrides)
    return MinerObservations(uid=uid, hotkey=hotkey, **base)  # type: ignore[arg-type]


# --- percentiles ------------------------------------------------------------


def test_p95_is_nearest_rank_not_interpolated():
    values = list(range(1, 21))  # 1..20
    # ceil(0.95 * 20) = 19 -> the 19th smallest, which is 19.
    assert percentile(values, 95) == 19
    # An interpolating definition gives 19.05 here. We want an observation.
    assert isinstance(percentile(values, 95), int)


@pytest.mark.parametrize(
    ("n", "p", "expected_rank"),
    [
        (25, 95, 24),  # ceil(23.75); truncating would give 23
        (7, 95, 7),  # ceil(6.65);  truncating would give 6
        (13, 50, 7),  # ceil(6.5);   truncating would give 6
        (20, 95, 19),  # exact — the case that hides the bug
    ],
)
def test_the_rank_rounds_up_not_down(n, p, expected_rank):
    """``ceil``, specifically.

    Truncating instead agrees with ceiling whenever ``p*n`` divides by 100,
    which for p95 means every multiple of 20 probes — including the round
    numbers a test is most likely to use. So the sizes here are chosen to be
    exactly the ones where the two definitions disagree. Rounding down would
    report a p95 that is really a p92, i.e. would quietly forgive a miner's
    worst tail.
    """
    values = list(range(1, n + 1))  # value == rank, so the assert reads directly
    assert percentile(values, p) == expected_rank


def test_percentile_ignores_input_order():
    values = [90, 10, 50, 30, 70]
    assert percentile(values, 50) == percentile(sorted(values, reverse=True), 50)


def test_percentile_of_one_sample_is_that_sample():
    for p in (1, 50, 95, 100):
        assert percentile([417], p) == 417


def test_p100_is_the_maximum():
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_percentile_of_nothing_is_an_error():
    # Silently returning 0 would score a miner that answered no probes as
    # infinitely fast.
    with pytest.raises(ValueError, match="empty sample"):
        percentile([], 95)


@pytest.mark.parametrize("p", [0, -5, 101])
def test_percentile_rejects_impossible_p(p):
    with pytest.raises(ValueError):
        percentile([1, 2, 3], p)


# --- the fixed-point logarithm ----------------------------------------------


@pytest.mark.parametrize("exponent", range(0, 32))
def test_log2_of_a_power_of_two_is_exact(exponent):
    assert log2_fp(1 << exponent) == exponent << LOG2_FRAC_BITS


@pytest.mark.parametrize("x", [3, 5, 7, 100, 1001, 65_537, 10_000_000])
def test_log2_matches_the_real_thing(x):
    ours = log2_fp(x) / (1 << LOG2_FRAC_BITS)
    assert abs(ours - math.log2(x)) < 1e-6


def test_log2_is_monotonic():
    previous = -1
    for x in range(1, 5000):
        current = log2_fp(x)
        assert current > previous
        previous = current


def test_log2_below_one_is_an_error():
    # Callers pass 1 + count precisely so this cannot happen; if it does, the
    # caller forgot the +1 and we want to know rather than return 0.
    with pytest.raises(ValueError, match=r"1 \+ count"):
        log2_fp(0)


# --- latency and throughput -------------------------------------------------


def test_hitting_the_latency_target_exactly_earns_full_marks():
    assert latency_bps(250, 250) == BPS_ONE


def test_twice_the_target_earns_half():
    assert latency_bps(500, 250) == BPS_ONE // 2


def test_beating_the_target_earns_no_bonus():
    # Deliberate. A user cannot perceive 40ms vs 80ms TTFT, and paying for
    # it would push miners to optimise the probe rather than the product.
    assert latency_bps(50, 250) == latency_bps(250, 250) == BPS_ONE


def test_a_very_slow_miner_earns_almost_nothing_but_not_negative():
    assert 0 < latency_bps(30_000, 250) < 100


def test_throughput_scales_linearly_to_the_target():
    target = 100_000  # 100 tok/s in milli-tokens
    assert throughput_bps(100_000, target) == BPS_ONE
    assert throughput_bps(50_000, target) == BPS_ONE // 2
    assert throughput_bps(250_000, target) == BPS_ONE  # clamped


def test_zero_throughput_is_zero_not_an_error():
    assert throughput_bps(0, 100_000) == 0


# --- reliability ------------------------------------------------------------


def test_all_successes_is_full_reliability():
    assert reliability_bps(SourceSample(attempts=50, successes=50)) == BPS_ONE


def test_a_clean_reject_costs_half_a_failure():
    """The sign fix, pinned.

    §8.1 writes this as ``(successes - 0.5*clean_rejects)/attempts``, which
    with disjoint fields makes an honest 503 *worse* than a hung connection
    — the opposite of the intent stated in the same section. This test is
    what stops anyone restoring the literal formula from the doc.
    """
    hangs = SourceSample(attempts=100, successes=90)  # 10 timeouts
    sheds = SourceSample(attempts=100, successes=90, clean_rejects=10)
    assert reliability_bps(hangs) == 9000
    assert reliability_bps(sheds) == 9500
    assert reliability_bps(sheds) > reliability_bps(hangs)


def test_reliability_of_a_source_that_tried_nothing_is_unknown():
    # Not zero. Zero would blend in as "this miner failed", when the truth
    # is that this source never asked.
    assert reliability_bps(EMPTY) is None


def test_a_miner_that_is_up_but_full_earns_half_credit():
    # Not zero, because it answered every single request honestly and
    # promptly; not full marks, because it answered none of them with
    # tokens. Exactly between a working miner and a dead one.
    everything_shed = SourceSample(attempts=10, successes=0, clean_rejects=10)
    assert reliability_bps(everything_shed) == BPS_ONE // 2


def test_a_miner_that_answers_nothing_at_all_earns_nothing():
    assert reliability_bps(SourceSample(attempts=10)) == 0


def test_a_sample_cannot_claim_more_outcomes_than_attempts():
    with pytest.raises(ValueError, match="out of 5 attempts"):
        SourceSample(attempts=5, successes=4, clean_rejects=3)


# --- capacity ---------------------------------------------------------------


def test_the_largest_miner_defines_the_ceiling():
    assert capacity_bps(1000, 1000) == BPS_ONE


def test_serving_nothing_earns_no_capacity():
    assert capacity_bps(0, 1000) == 0
    assert capacity_bps(0, 0) == 0


def test_nobody_serving_anything_gives_nobody_capacity():
    # Day one, before the platform has users. Everyone gets 0, which is the
    # same for everyone and therefore does not distort the ranking.
    assert capacity_bps(0, 0) == capacity_bps(5, 0) == 0


def test_capacity_is_logarithmic_not_linear():
    # A miner serving a tenth of the leader's traffic earns far more than a
    # tenth of the capacity term. That is the whole point: it keeps small
    # honest operators viable while still rewarding scale.
    tenth = capacity_bps(100, 1000)
    assert tenth > 6000
    assert tenth < BPS_ONE


def test_capacity_matches_the_real_logarithm():
    for served, ceiling in [(1, 1000), (10, 1000), (500, 1000), (999, 1000)]:
        expected = math.log(1 + served) / math.log(1 + ceiling) * BPS_ONE
        assert abs(capacity_bps(served, ceiling) - expected) <= 1


def test_the_log_base_does_not_matter():
    # The ratio cancels it, which is why log2 is safe to use where the
    # design says log.
    for base in (2.0, math.e, 10.0):
        expected = (math.log(1 + 250, base) / math.log(1 + 4000, base)) * BPS_ONE
        assert abs(capacity_bps(250, 4000) - expected) <= 1


def test_capacity_credits_telemetry_only_under_the_shipped_config(config):
    obs = MinerObservations(
        uid=1,
        hotkey="5A",
        shadow=SourceSample(served=999_999),
        direct=SourceSample(served=999_999),
        telemetry=SourceSample(served=42),
    )
    # Probes are traffic we generate. Counting them as demonstrated capacity
    # would let a miner earn capacity for answering us.
    assert credited_served(obs, config.capacity_sources) == 42


# --- blending sources -------------------------------------------------------


def test_blend_is_the_weighted_mean(config):
    values = {"shadow": 10_000, "direct": 0, "telemetry": 0}
    # 6500/10000 of 10000
    assert blend(values, config.latency_sources) == 6500


def test_a_missing_source_renormalises_rather_than_scoring_zero(config):
    # The day-one case: a miner with no user traffic has no telemetry. If
    # absence counted as zero it would score 8000 while being perfect.
    values: dict[str, int | None] = {
        "shadow": 10_000,
        "direct": 10_000,
        "telemetry": None,
    }
    assert blend(values, config.latency_sources) == 10_000


def test_a_present_zero_is_not_the_same_as_a_missing_source(config):
    absent: dict[str, int | None] = {"shadow": 10_000, "direct": 10_000, "telemetry": None}
    present: dict[str, int | None] = {"shadow": 10_000, "direct": 10_000, "telemetry": 0}
    assert blend(absent, config.latency_sources) > blend(present, config.latency_sources)


def test_blend_with_nothing_observed_is_unknown(config):
    assert blend({"shadow": None, "direct": None, "telemetry": None}, config.latency_sources) is None


def test_a_zero_weighted_source_is_ignored_even_when_present():
    mix = SourceMix(shadow=0, direct=0, telemetry=BPS_ONE)
    values: dict[str, int | None] = {"shadow": 10_000, "direct": 10_000, "telemetry": 100}
    assert blend(values, mix) == 100


# --- smoothing --------------------------------------------------------------


def test_a_miner_we_have_never_seen_takes_its_observation_whole():
    # Starting newcomers at zero would make registration itself a penalty
    # paid over several epochs.
    assert ema(None, 8_000, 3_000) == 8_000


def test_the_ema_moves_alpha_of_the_way():
    assert ema(0, 10_000, 3_000) == 3_000
    assert ema(10_000, 0, 3_000) == 7_000


def test_one_bad_epoch_does_not_delete_a_miner():
    smoothed = 10_000
    smoothed = ema(smoothed, 0, 3_000)
    assert smoothed == 7_000, "a single outage should cost 30%, not everything"


def test_the_ema_converges():
    value = 0
    for _ in range(100):
        value = ema(value, 10_000, 3_000)
    # It stalls one unit short of the ceiling, because at 9999 the true next
    # value is 9999.3 and that rounds back to 9999. One basis point out of
    # ten thousand, and it is a stable fixed point rather than a drift.
    assert value == 9_999


def test_the_ema_does_not_drift_downward():
    """Rounding half up, not truncating.

    Truncation biases every step in the same direction. Over an epoch's
    worth of miners and a week's worth of epochs that is a systematic shave
    on everyone's score, amplified by the cube in §8.3. This asserts the
    rounding directly: a value that should tick up must tick up.
    """
    # 3000*7000 + 7000*6667 = 21_000_000 + 46_669_000 = 67_669_000 -> 6766.9
    assert ema(6_667, 7_000, 3_000) == 6_767  # truncation would give 6766


# --- normalisation ----------------------------------------------------------


def test_weights_sum_to_exactly_the_u16_total():
    # Floor division alone loses up to one unit per miner. Two validators
    # that lose a different number of units have different vectors.
    for count in (1, 2, 3, 7, 64, 256):
        scores = {uid: 1_000 + uid for uid in range(count)}
        assert sum(cube_normalise(scores).values()) == MAX_WEIGHT_U16


def test_a_single_miner_takes_everything():
    assert cube_normalise({4: 5_000}) == {4: MAX_WEIGHT_U16}


def test_all_zero_scores_produce_all_zero_weights():
    # Not an even split. Nobody earned anything, and saying so is the point.
    assert cube_normalise({1: 0, 2: 0}) == {1: 0, 2: 0}


def test_cubing_makes_being_fastest_worth_it():
    linear_share = 10_000 / (10_000 + 8_000)  # 0.556
    weights = cube_normalise({1: 10_000, 2: 8_000})
    cubed_share = weights[1] / MAX_WEIGHT_U16
    assert cubed_share > 0.65
    assert cubed_share > linear_share
    # ...but the slower miner still earns enough to stay online.
    assert weights[2] > MAX_WEIGHT_U16 // 10


def test_the_remainder_tie_break_is_total_and_reproducible():
    # Identical scores means identical remainders; without a tie-break the
    # leftover units would land wherever dict iteration put them.
    scores = {uid: 1_000 for uid in range(7)}
    first = cube_normalise(scores)
    shuffled = dict(sorted(scores.items(), key=lambda kv: -kv[0]))
    assert cube_normalise(shuffled) == first
    assert sum(first.values()) == MAX_WEIGHT_U16


def test_normalising_nothing_is_empty_not_an_error():
    assert cube_normalise({}) == {}


# --- the gate ---------------------------------------------------------------


def test_a_healthy_attested_miner_passes(config):
    gate, reasons = gate_bps(healthy(), config, attestation_mode="hard")
    assert gate == config.gate_pass_bps
    assert reasons == ()


def test_no_attestation_is_fatal_in_hard_mode(config):
    gate, reasons = gate_bps(healthy(attested=False), config, attestation_mode="hard")
    assert gate == 0
    assert "attestation" in reasons[0]


def test_no_attestation_is_survivable_in_warn_mode(config):
    gate, _ = gate_bps(healthy(attested=False), config, attestation_mode="warn")
    assert gate == config.gate_warn_bps
    assert 0 < gate < config.gate_pass_bps


def test_attestation_is_not_checked_at_all_in_off_mode(config):
    # guards.py refuses to let this mode run on finney; here it just has to
    # not block a laptop.
    gate, reasons = gate_bps(healthy(attested=False), config, attestation_mode="off")
    assert gate == config.gate_pass_bps
    assert "not checked" in reasons[0]


def test_too_few_probes_means_no_score(config):
    thin = healthy(shadow=probes(2), direct=probes(2))
    gate, reasons = gate_bps(thin, config, attestation_mode="hard")
    assert gate == 0
    assert "insufficient probes" in reasons[0]


def test_telemetry_does_not_count_towards_the_probe_minimum(config):
    # Telemetry is the miner talking about itself. A miner could otherwise
    # meet the minimum by reporting traffic nobody verified.
    self_reported = MinerObservations(
        uid=1, hotkey="5A", telemetry=probes(10_000), attested=True
    )
    gate, reasons = gate_bps(self_reported, config, attestation_mode="hard")
    assert gate == 0
    assert "insufficient probes" in reasons[0]


def test_cooldown_is_reported_before_probe_count(config):
    # A miner in cooldown is not being probed, so telling it that it failed
    # for want of probes would send it chasing a symptom.
    out = MinerObservations(uid=1, hotkey="5A", cooldown_epochs_left=1, attested=True)
    gate, reasons = gate_bps(out, config, attestation_mode="hard")
    assert gate == 0
    assert "cooldown" in reasons[0]


# --- penalties --------------------------------------------------------------


def test_no_penalty_is_a_multiplier_of_one(config):
    assert penalty_bps(healthy(), config) == (BPS_ONE, ())


def test_a_receipt_mismatch_zeroes_the_epoch(config):
    multiplier, reasons = penalty_bps(healthy(receipts_mismatched=True), config)
    assert multiplier == 0
    assert "receipt mismatch" in reasons[0]


def test_overclaiming_capacity_costs_twenty_percent(config):
    multiplier, reasons = penalty_bps(healthy(capacity_overclaimed=True), config)
    assert multiplier == 8_000
    assert "overclaim" in reasons[0]


def test_penalties_compose_multiplicatively_and_cannot_become_a_bonus(config):
    both = healthy(receipts_mismatched=True, capacity_overclaimed=True)
    multiplier, reasons = penalty_bps(both, config)
    assert multiplier == 0
    assert len(reasons) == 2


# --- the whole epoch --------------------------------------------------------


def test_a_miner_meeting_every_target_scores_full_marks(config):
    result = score_epoch([healthy()], config)
    (score,) = result.scores
    assert score.components.latency == BPS_ONE
    assert score.components.throughput == BPS_ONE
    assert score.components.reliability == BPS_ONE
    assert score.components.capacity == BPS_ONE
    assert score.components.quality == BPS_ONE
    assert score.score_bps == BPS_ONE
    assert score.weight_u16 == MAX_WEIGHT_U16


def test_the_component_weights_are_the_design_ones(config):
    # Perfect on everything except latency, which is floored. What remains
    # is exactly 1 - 0.45 of the score, which pins §8.1's 0.45 rather than
    # trusting the config to have been transcribed correctly.
    glacial = 10_000_000
    slow = healthy(
        shadow=probes(ttft_ms=glacial),
        direct=probes(ttft_ms=glacial),
        telemetry=probes(ttft_ms=glacial, served=100),
    )
    (score,) = score_epoch([slow], config).scores
    assert score.components.latency == 0
    assert score.components.quality == 5_500


def test_latency_is_blended_across_sources_not_taken_from_the_worst(config):
    # Only the probes are slow; the platform's telemetry says the miner is
    # fast for real users. Telemetry is 20% of the latency term, so the
    # miner keeps 20% of it rather than none.
    mixed = healthy(
        shadow=probes(ttft_ms=10_000_000),
        direct=probes(ttft_ms=10_000_000),
        telemetry=probes(ttft_ms=250, served=100),
    )
    (score,) = score_epoch([mixed], config).scores
    assert score.components.latency == 2_000


def test_the_faster_miner_earns_more(config):
    fast = healthy(uid=1, hotkey="5Fast")
    slow = healthy(
        uid=2,
        hotkey="5Slow",
        shadow=probes(ttft_ms=1_000),
        direct=probes(ttft_ms=1_000),
        telemetry=probes(served=100),
    )
    result = score_epoch([fast, slow], config)
    weights = result.weights
    assert weights[1] > weights[2]
    assert sum(weights.values()) == MAX_WEIGHT_U16


def test_a_gated_out_miner_earns_nothing_this_epoch_however_good_its_history(config):
    """The reason the EMA is applied before the gate and not after.

    Smoothing a *quality measurement* across epochs is right — one bad probe
    round should not delete a miner. Smoothing the *verdict* is not: it would
    pay a miner that just lost its attestation 70% of last week's emissions,
    which makes the gate a suggestion.
    """
    history = {"5A": 10_000}
    lost_attestation = healthy(uid=1, hotkey="5A", attested=False)
    result = score_epoch(
        [lost_attestation], config, attestation_mode="hard", carry_forward=history
    )
    (score,) = result.scores
    assert score.smoothed_quality_bps > 9_000, "quality is still being tracked"
    assert score.gate_bps == 0
    assert score.score_bps == 0
    assert score.weight_u16 == 0
    assert result.is_empty


def test_the_ema_carries_across_epochs(config):
    first = score_epoch([healthy(hotkey="5A")], config)
    assert first.carry_forward == {"5A": BPS_ONE}

    # A total outage: probes still ran, everything failed.
    outage = MinerObservations(
        uid=0,
        hotkey="5A",
        shadow=SourceSample(attempts=25, successes=0),
        direct=SourceSample(attempts=25, successes=0),
        attested=True,
    )
    second = score_epoch([outage], config, carry_forward=first.carry_forward)
    (score,) = second.scores
    assert score.components.quality == 0
    assert score.smoothed_quality_bps == 7_000  # 30% of the way down, not all of it


def test_a_recycled_uid_does_not_inherit_the_previous_miner_s_reputation(config):
    """Keyed by hotkey, not uid — and this is the test that says why.

    The metagraph reassigns uids on deregistration. If the EMA were keyed by
    uid, deregistering a well-scored neighbour and taking its slot would
    inherit its score. That is not merely unfair, it is a strategy.
    """
    history = {"5Old": 10_000}
    newcomer = MinerObservations(
        uid=0,  # same uid the old miner held
        hotkey="5New",
        shadow=probes(),
        direct=probes(),
        attested=True,
    )
    result = score_epoch([newcomer], config, carry_forward=history)
    (score,) = result.scores
    # Its own measurement, whole — neither inherited nor penalised.
    assert score.smoothed_quality_bps == score.components.quality
    assert "5Old" not in result.carry_forward


def test_two_validators_seeing_the_same_epoch_agree_exactly(config):
    """The consensus property, asserted rather than argued.

    Same observations, different order, different process: identical weight
    vector. Not close — identical.
    """
    miners = [
        healthy(
            uid=uid,
            hotkey=f"5Miner{uid}",
            shadow=probes(20 + uid, ttft_ms=200 + uid * 37),
            direct=probes(20 + uid, ttft_ms=210 + uid * 31),
            telemetry=probes(served=uid * 97 + 3),
        )
        for uid in range(1, 33)
    ]
    reference = score_epoch(miners, config)

    rng = random.Random(20260803)
    for _ in range(8):
        shuffled = miners[:]
        rng.shuffle(shuffled)
        assert score_epoch(shuffled, config).weights == reference.weights

    assert sum(reference.weights.values()) == MAX_WEIGHT_U16


def test_nothing_in_a_result_is_a_float(config):
    """Walk the output rather than trusting review.

    A float that survives to the weight vector is the exact bug this whole
    module's integer discipline exists to prevent, and it would not show up
    as a test failure anywhere else — it would show up as vtrust drift weeks
    later.
    """
    miners = [
        healthy(uid=uid, hotkey=f"5M{uid}", telemetry=probes(served=uid * 13))
        for uid in range(1, 6)
    ]
    result = score_epoch(miners, config)
    for score in result.scores:
        for key, value in score.as_dict().items():
            if isinstance(value, (str, list)):
                continue
            assert isinstance(value, int) and not isinstance(value, bool), key
    for value in result.weights.values():
        assert isinstance(value, int)
    for value in result.carry_forward.values():
        assert isinstance(value, int)


def test_scores_come_back_ordered_by_uid(config):
    miners = [healthy(uid=uid, hotkey=f"5M{uid}") for uid in (9, 2, 7, 1)]
    result = score_epoch(miners, config)
    assert [s.uid for s in result.scores] == [1, 2, 7, 9]


def test_an_empty_epoch_is_empty_not_a_crash(config):
    result = score_epoch([], config)
    assert result.scores == ()
    assert result.weights == {}
    assert result.is_empty


def test_the_result_carries_both_config_versions(config):
    # Surfaced on /scores so that "why do we disagree" starts with "do we
    # disagree about the constants" instead of ending there.
    result = score_epoch([healthy()], config)
    assert result.config_version == config.version
    assert result.slo_version == config.slo.version


def test_a_score_explains_itself(config):
    (score,) = score_epoch([healthy(attested=False)], config).scores
    assert score.reasons, "a miner earning zero must be told why"
    assert score.as_dict()["weight_u16"] == 0


# --- configuration ----------------------------------------------------------


def slo_table() -> dict:
    return deepcopy(
        {
            "version": {"id": 1},
            "latency": {"ttft_p95_target_ms": 250},
            "throughput": {"tokens_per_s_p50_target": 100},
            "availability": {"success_rate_min_bps": 9900},
            "attestation": {"max_age_ms": 3_600_000},
        }
    )


def scoring_table() -> dict:
    return deepcopy(
        {
            "version": {"id": 1},
            "weights": {
                "latency": 4500,
                "throughput": 2000,
                "reliability": 2500,
                "capacity": 1000,
            },
            "sources": {
                "latency_shadow": 6500,
                "latency_direct": 1500,
                "latency_telemetry": 2000,
                "throughput_shadow": 6500,
                "throughput_direct": 1500,
                "throughput_telemetry": 2000,
                "reliability_shadow": 4000,
                "reliability_direct": 4000,
                "reliability_telemetry": 2000,
                "capacity_shadow": 0,
                "capacity_direct": 0,
                "capacity_telemetry": 10000,
            },
            "gate": {
                "pass_bps": 10000,
                "warn_bps": 1500,
                "fail_bps": 0,
                "min_probe_successes": 12,
                "gate_out_after_misses": 2,
                "gate_out_cooldown_epochs": 1,
            },
            "normalisation": {"exponent": 3, "ema_alpha_bps": 3000},
            "probe": {
                "direct_interval_s": 60,
                "shadow_interval_s": 30,
                "timeout_s": 30,
                "max_concurrent_probes": 16,
                "max_tokens": 256,
                "prompt_nonce_bytes": 16,
                "direct_count": 20,
            },
            "penalty": {"receipt_mismatch": 0, "capacity_overclaim": 8000},
        }
    )


def test_the_shipped_config_matches_the_design(config):
    # The numbers in DESIGN.md §8.1, which is what miners will read.
    assert (config.w_latency, config.w_throughput) == (4500, 2000)
    assert (config.w_reliability, config.w_capacity) == (2500, 1000)
    assert config.slo.ttft_p95_target_ms == 250
    assert config.slo.tokens_per_s_p50_target == 100
    assert config.exponent == 3


def test_component_weights_that_do_not_sum_to_one_are_refused():
    table = scoring_table()
    table["weights"]["latency"] = 5000
    with pytest.raises(ConfigError, match="sums to 10500"):
        load_scoring_config(slo_table(), table)


def test_a_source_mix_that_does_not_sum_to_one_is_refused():
    table = scoring_table()
    table["sources"]["latency_telemetry"] = 3000
    with pytest.raises(ConfigError, match=r"latency_\* sums to 11000"):
        load_scoring_config(slo_table(), table)


def test_a_float_in_the_config_is_refused():
    # The entire determinism argument rests on there being no floats. A
    # config that quietly introduces one would defeat it from outside the
    # code.
    table = scoring_table()
    table["normalisation"]["ema_alpha_bps"] = 0.3
    with pytest.raises(ConfigError, match="must be an integer"):
        load_scoring_config(slo_table(), table)


def test_a_zero_ema_alpha_is_refused():
    table = scoring_table()
    table["normalisation"]["ema_alpha_bps"] = 0
    with pytest.raises(ConfigError, match="freeze every score"):
        load_scoring_config(slo_table(), table)


def test_an_exponent_below_one_is_refused():
    table = scoring_table()
    table["normalisation"]["exponent"] = 0
    with pytest.raises(ConfigError, match="at least 1"):
        load_scoring_config(slo_table(), table)


def test_a_zero_min_probe_successes_is_refused():
    table = scoring_table()
    table["gate"]["min_probe_successes"] = 0
    with pytest.raises(ConfigError, match="at least 1"):
        load_scoring_config(slo_table(), table)


def test_a_success_floor_equal_to_the_attempt_count_is_refused():
    # The bug this guard exists for: 20 attempts and a floor of 20 successes
    # means one transient miss zeroes a healthy miner for the epoch.
    table = scoring_table()
    table["gate"]["min_probe_successes"] = table["probe"]["direct_count"]
    with pytest.raises(ConfigError, match="must be strictly below"):
        load_scoring_config(slo_table(), table)


def test_a_success_floor_above_the_attempt_count_is_refused():
    table = scoring_table()
    table["gate"]["min_probe_successes"] = table["probe"]["direct_count"] + 1
    with pytest.raises(ConfigError, match="must be strictly below"):
        load_scoring_config(slo_table(), table)


def test_the_shipped_config_leaves_the_gate_real_headroom():
    # Guards the actual config file, not a fixture: a shipped config whose
    # floor crept back up to the attempt count would reintroduce the bug.
    config = load_scoring_config()
    assert config.min_probe_successes < config.probe.direct_count
    assert config.probe.direct_count - config.min_probe_successes >= 4


def test_a_zero_latency_target_is_refused():
    # It is a divisor in §8.1.
    table = slo_table()
    table["latency"]["ttft_p95_target_ms"] = 0
    with pytest.raises(ConfigError, match="must be positive"):
        load_scoring_config(table, scoring_table())


def test_a_missing_section_names_itself():
    table = scoring_table()
    del table["gate"]
    with pytest.raises(ConfigError, match=r"missing the \[gate\] section"):
        load_scoring_config(slo_table(), table)


def test_a_missing_key_names_itself():
    table = scoring_table()
    del table["probe"]["timeout_s"]
    with pytest.raises(ConfigError, match="'timeout_s'"):
        load_scoring_config(slo_table(), table)


def test_config_objects_are_frozen(config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.w_latency = 9999  # type: ignore[misc]


def test_components_serialise_flat_for_the_scores_endpoint():
    keys = Components(1, 2, 3, 4, 5).as_dict().keys()
    assert all(k.endswith("_bps") for k in keys)


def test_a_probe_budget_too_small_for_an_answer_is_refused():
    # An undersized budget fails silently in the worst way: the answer is
    # truncated, every probe fails its content check, and a healthy miner is
    # gated out with nothing but "content mismatch" to show for it.
    table = scoring_table()
    table["probe"]["max_tokens"] = 8
    with pytest.raises(ConfigError, match="below 160"):
        load_scoring_config(slo_table(), table)


def test_the_shipped_probe_budget_can_hold_an_answer():
    config = load_scoring_config()
    assert config.probe.max_tokens >= 160


def test_scoring_does_not_apply_the_burn(config):
    # The burn belongs to weights.py. If it ever leaks into scoring it would
    # enter carry_forward and compound through the EMA, ratcheting every
    # restart further toward a full burn -- the failure resi guards against by
    # skipping the burn uid during consensus bootstrap.
    observation = MinerObservations(
        uid=1,
        hotkey="5MinerHotkey",
        direct=SourceSample(
            attempts=20, successes=20, ttft_ms=(100,) * 20, tps_milli=(120_000,) * 20
        ),
        attested=False,
    )
    result = score_epoch([observation], config, attestation_mode="off")
    assert sum(result.weights.values()) == MAX_WEIGHT_U16
    assert result.weights == {1: MAX_WEIGHT_U16}
