"""Scoring: observations in, weight vector out. DESIGN.md §8.

Every number in this module is an integer. Not "mostly integers" — there is
no float anywhere in the path from a probe timing to a u16 weight, including
the logarithm in the capacity term. The reason is consensus, not taste: two
validators that observe the same miners must emit *identical* weight
vectors, and floating point does not guarantee that across CPUs, libms, or
Python builds. A vector that differs in the last place is a vtrust penalty
that nobody can reproduce or explain.

The conventions that follow from that:

* Ratios are basis points. ``BPS_ONE`` (10000) is 1.0.
* Durations are integer milliseconds.
* Throughput is *milli-tokens per second* (``tps_milli``) — tokens/sec times
  1000 — so that a stream producing 37 tokens in 412 ms is 89805, not
  89.805825242718.
* Percentiles are nearest-rank with no interpolation, so a percentile is
  always an observed value and never an average of two.
* The logarithm is a fixed-point ``log2`` computed by repeated squaring
  (:func:`log2_fp`). The base cancels in the capacity ratio, so log2 is as
  good as ln and is exactly representable in binary fixed point.

Nothing here does I/O or reads a clock. Feed it observations, get scores.
That is what makes the whole of §8 testable in microseconds without a chain,
a GPU, or a miner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from instant.common.config import ConfigError, load_toml

#: One, in basis points.
BPS_ONE = 10_000

#: Bittensor weights are u16. This is the total the vector must apportion.
MAX_WEIGHT_U16 = 65_535

#: Fractional bits in the fixed-point logarithm. 24 puts the error at 2^-24,
#: roughly six decimal digits below the basis point we round to — the
#: capacity term is exact for every input we will ever see.
LOG2_FRAC_BITS = 24

#: The three ways we learn anything about a miner (§8.2), in a fixed order
#: so that iteration is deterministic even where the arithmetic is not
#: order-sensitive.
SOURCES: tuple[str, ...] = ("shadow", "direct", "telemetry")

AttestationMode = Literal["hard", "warn", "off"]

#: Smallest ``[probe].max_tokens`` that can hold a probe answer.
#:
#: Duplicated as a number rather than imported from ``validator.probe`` because
#: ``probe`` imports ``state`` which imports this module; the constant is small
#: and the cycle is not worth it.
#:
#: Measured against the launch H200: the largest challenge this prober emits (40
#: integers) costs 97 completion tokens including its reasoning preamble, at
#: either two or three digits. 160 is that with comfortable headroom. The floor
#: matters because an undersized budget does not degrade gracefully — the answer
#: is truncated, every probe fails its content check, and a healthy miner is
#: gated out with nothing in the logs but "content mismatch".
MIN_PROBE_MAX_TOKENS = 160


# --- configuration ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SloTargets:
    """``config/slo.toml`` — the §2.4 targets scoring is calibrated against."""

    version: int
    ttft_p95_target_ms: int
    tokens_per_s_p50_target: int
    success_rate_min_bps: int
    attestation_max_age_ms: int

    @property
    def tps_p50_target_milli(self) -> int:
        """The throughput target in the unit throughput is measured in."""
        return self.tokens_per_s_p50_target * 1000


@dataclass(frozen=True, slots=True)
class SourceMix:
    """How much of one component comes from each source, in basis points."""

    shadow: int
    direct: int
    telemetry: int

    def weight_of(self, source: str) -> int:
        if source == "shadow":
            return self.shadow
        if source == "direct":
            return self.direct
        if source == "telemetry":
            return self.telemetry
        raise KeyError(source)

    @property
    def total(self) -> int:
        return self.shadow + self.direct + self.telemetry


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """``[probe]`` — read by the prober, carried here so there is one loader."""

    direct_interval_s: int
    shadow_interval_s: int
    timeout_s: int
    max_concurrent_probes: int
    max_tokens: int
    prompt_nonce_bytes: int
    #: Direct probe *attempts* per miner per epoch. Distinct from
    #: ``gate.min_probe_successes``, which is how many of them must succeed.
    #: These were once the same number, which meant a single transient miss —
    #: one timeout, one honest 429 while the miner was full — zeroed a healthy
    #: miner for the epoch, and two in a row tripped it into a cooldown.
    direct_count: int


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Everything §8 needs, validated at load.

    ``version`` is the consensus-visible one. If two validators disagree on
    weights, the first question is whether they disagree on this number, so
    it is logged at boot and surfaced on ``/scores``.
    """

    version: int
    slo: SloTargets

    w_latency: int
    w_throughput: int
    w_reliability: int
    w_capacity: int

    latency_sources: SourceMix
    throughput_sources: SourceMix
    reliability_sources: SourceMix
    capacity_sources: SourceMix

    gate_pass_bps: int
    gate_warn_bps: int
    gate_fail_bps: int
    min_probe_successes: int
    gate_out_after_misses: int
    gate_out_cooldown_epochs: int

    exponent: int
    ema_alpha_bps: int

    probe: ProbeConfig

    penalty_receipt_mismatch: int
    penalty_capacity_overclaim: int


def _require(table: Mapping[str, Any], section: str, key: str, filename: str) -> Any:
    if section not in table:
        raise ConfigError(f"{filename} is missing the [{section}] section")
    if key not in table[section]:
        raise ConfigError(f"{filename} [{section}] is missing {key!r}")
    value = table[section][key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(
            f"{filename} [{section}].{key} = {value!r} must be an integer. "
            "Scoring does no floating-point arithmetic; see this module's "
            "docstring for why."
        )
    return value


def _mix(table: Mapping[str, Any], component: str, filename: str) -> SourceMix:
    mix = SourceMix(
        shadow=_require(table, "sources", f"{component}_shadow", filename),
        direct=_require(table, "sources", f"{component}_direct", filename),
        telemetry=_require(table, "sources", f"{component}_telemetry", filename),
    )
    if mix.total != BPS_ONE:
        raise ConfigError(
            f"{filename} [sources] {component}_* sums to {mix.total}, not "
            f"{BPS_ONE}. Each component's sources are a mix, not a set of "
            "independent multipliers."
        )
    return mix


def load_scoring_config(
    slo_table: Mapping[str, Any] | None = None,
    scoring_table: Mapping[str, Any] | None = None,
) -> ScoringConfig:
    """Load and validate ``config/slo.toml`` and ``config/scoring.toml``.

    The tables can be passed in for tests. In production both are read from
    disk, once, at boot — a validator that would compute the wrong weights
    should fail to start rather than start and be wrong.
    """
    slo_raw = load_toml("slo.toml") if slo_table is None else slo_table
    raw = load_toml("scoring.toml") if scoring_table is None else scoring_table

    slo = SloTargets(
        version=_require(slo_raw, "version", "id", "slo.toml"),
        ttft_p95_target_ms=_require(slo_raw, "latency", "ttft_p95_target_ms", "slo.toml"),
        tokens_per_s_p50_target=_require(
            slo_raw, "throughput", "tokens_per_s_p50_target", "slo.toml"
        ),
        success_rate_min_bps=_require(
            slo_raw, "availability", "success_rate_min_bps", "slo.toml"
        ),
        attestation_max_age_ms=_require(slo_raw, "attestation", "max_age_ms", "slo.toml"),
    )
    if slo.ttft_p95_target_ms <= 0 or slo.tokens_per_s_p50_target <= 0:
        raise ConfigError(
            "slo.toml targets must be positive; they are divisors in §8.1"
        )

    config = ScoringConfig(
        version=_require(raw, "version", "id", "scoring.toml"),
        slo=slo,
        w_latency=_require(raw, "weights", "latency", "scoring.toml"),
        w_throughput=_require(raw, "weights", "throughput", "scoring.toml"),
        w_reliability=_require(raw, "weights", "reliability", "scoring.toml"),
        w_capacity=_require(raw, "weights", "capacity", "scoring.toml"),
        latency_sources=_mix(raw, "latency", "scoring.toml"),
        throughput_sources=_mix(raw, "throughput", "scoring.toml"),
        reliability_sources=_mix(raw, "reliability", "scoring.toml"),
        capacity_sources=_mix(raw, "capacity", "scoring.toml"),
        gate_pass_bps=_require(raw, "gate", "pass_bps", "scoring.toml"),
        gate_warn_bps=_require(raw, "gate", "warn_bps", "scoring.toml"),
        gate_fail_bps=_require(raw, "gate", "fail_bps", "scoring.toml"),
        min_probe_successes=_require(
            raw, "gate", "min_probe_successes", "scoring.toml"
        ),
        gate_out_after_misses=_require(raw, "gate", "gate_out_after_misses", "scoring.toml"),
        gate_out_cooldown_epochs=_require(
            raw, "gate", "gate_out_cooldown_epochs", "scoring.toml"
        ),
        exponent=_require(raw, "normalisation", "exponent", "scoring.toml"),
        ema_alpha_bps=_require(raw, "normalisation", "ema_alpha_bps", "scoring.toml"),
        probe=ProbeConfig(
            direct_interval_s=_require(raw, "probe", "direct_interval_s", "scoring.toml"),
            shadow_interval_s=_require(raw, "probe", "shadow_interval_s", "scoring.toml"),
            timeout_s=_require(raw, "probe", "timeout_s", "scoring.toml"),
            max_concurrent_probes=_require(
                raw, "probe", "max_concurrent_probes", "scoring.toml"
            ),
            max_tokens=_require(raw, "probe", "max_tokens", "scoring.toml"),
            prompt_nonce_bytes=_require(raw, "probe", "prompt_nonce_bytes", "scoring.toml"),
            direct_count=_require(raw, "probe", "direct_count", "scoring.toml"),
        ),
        penalty_receipt_mismatch=_require(
            raw, "penalty", "receipt_mismatch", "scoring.toml"
        ),
        penalty_capacity_overclaim=_require(
            raw, "penalty", "capacity_overclaim", "scoring.toml"
        ),
    )

    component_total = (
        config.w_latency + config.w_throughput + config.w_reliability + config.w_capacity
    )
    if component_total != BPS_ONE:
        raise ConfigError(
            f"scoring.toml [weights] sums to {component_total}, not {BPS_ONE}. "
            "The four components are a partition of the score, so a sum that "
            "is not 10000 silently rescales every miner."
        )
    if not 0 < config.ema_alpha_bps <= BPS_ONE:
        raise ConfigError(
            f"scoring.toml [normalisation].ema_alpha_bps = {config.ema_alpha_bps} "
            f"must be in (0, {BPS_ONE}]. Zero would freeze every score at its "
            "initial value forever."
        )
    if config.exponent < 1:
        raise ConfigError(
            f"scoring.toml [normalisation].exponent = {config.exponent} must be "
            "at least 1; below that the fastest miner earns least."
        )
    if config.min_probe_successes < 1:
        raise ConfigError(
            "scoring.toml [gate].min_probe_successes must be at least 1, or a "
            "miner that answered nothing is scored on nothing."
        )
    if config.probe.direct_count < 1:
        raise ConfigError(
            "scoring.toml [probe].direct_count must be at least 1; it is how "
            "many probes the prober actually sends."
        )
    if config.probe.max_tokens < MIN_PROBE_MAX_TOKENS:
        raise ConfigError(
            f"scoring.toml [probe].max_tokens = {config.probe.max_tokens} is "
            f"below {MIN_PROBE_MAX_TOKENS}. A probe answer costs about 97 "
            "completion tokens including gpt-oss's reasoning preamble, and a "
            "budget too small to hold it does not fail loudly: the answer is "
            "truncated, every probe fails its content check, and a healthy "
            "miner is gated out for what looks like serving wrong output."
        )
    if config.min_probe_successes >= config.probe.direct_count:
        raise ConfigError(
            f"scoring.toml [gate].min_probe_successes "
            f"({config.min_probe_successes}) must be strictly below "
            f"[probe].direct_count ({config.probe.direct_count}). Equal values "
            "leave a miner no headroom at all: one timeout, or one honest 429 "
            "while it was full, drops it below the floor and zeroes it for the "
            "epoch — and twice in a row trips gate_out_after_misses into a "
            "cooldown. Probe quality is already priced continuously by "
            "reliability_bps; the gate is a sample-size floor, not a second "
            "penalty for the same misses."
        )
    return config


# --- deterministic integer arithmetic ---------------------------------------


def percentile(values: Sequence[int], p: int) -> int:
    """Nearest-rank percentile: ``sorted(values)[ceil(p*n/100) - 1]``.

    No interpolation, deliberately. The interpolating definition averages
    two observations, which reintroduces a division whose rounding two
    validators could disagree about — and the averaged value is not a
    latency anything ever measured. This one always returns a number a
    miner actually produced.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 < p <= 100:
        raise ValueError(f"percentile p={p} must be in (0, 100]")
    ordered = sorted(values)
    rank = (p * len(ordered) + 99) // 100  # ceil(p*n/100), in integers
    return ordered[rank - 1]


def log2_fp(x: int) -> int:
    """``log2(x)`` scaled by ``2**LOG2_FRAC_BITS``, exactly, in integers.

    Integer part from ``bit_length``; fractional part by repeated squaring —
    square the mantissa, and if it crossed 2 emit a 1 bit and halve. One bit
    of the answer per iteration, no lookup tables, no libm, and identical on
    every machine that can multiply.

    Used only through :func:`capacity_bps`, where the ratio of two logs makes
    the base irrelevant; log2 is chosen because it is the one base whose
    fixed-point representation is exact for powers of two.
    """
    if x < 1:
        raise ValueError(f"log2_fp({x}) is undefined; callers must pass 1 + count")

    integer_part = x.bit_length() - 1
    result = integer_part << LOG2_FRAC_BITS
    if x == 1 << integer_part:
        return result  # exact power of two; the mantissa is exactly 1

    one = 1 << LOG2_FRAC_BITS
    mantissa = (x << LOG2_FRAC_BITS) >> integer_part  # in [1, 2), scaled
    for bit in range(1, LOG2_FRAC_BITS + 1):
        mantissa = (mantissa * mantissa) >> LOG2_FRAC_BITS
        if mantissa >= one << 1:
            mantissa >>= 1
            result += one >> bit
    return result


def ema(previous: int | None, observed: int, alpha_bps: int) -> int:
    """Exponential moving average in basis points.

    ``previous is None`` means a miner we have never scored, which takes the
    observation whole. Starting a newcomer at zero and letting it climb over
    several epochs would penalise registration itself.

    Rounds half up rather than truncating. Truncation biases every score
    downward by up to one basis point per epoch, in the same direction every
    time, and §8.3 cubes the result — a systematic shave is not something to
    leave in the one function every score passes through fifty times.
    """
    if previous is None:
        return observed
    weighted = alpha_bps * observed + (BPS_ONE - alpha_bps) * previous
    return (weighted + BPS_ONE // 2) // BPS_ONE


def blend(values: Mapping[str, int | None], mix: SourceMix) -> int | None:
    """Combine per-source component values by the §8.2 mix.

    A source that observed nothing is ``None`` and is dropped, with the
    remaining weights renormalised. This matters on day one: a miner that
    has served no user traffic has no telemetry, and treating "no telemetry"
    as "zero telemetry" would score a perfectly healthy miner at 80% of what
    it earns. Absence of evidence is not evidence of slowness.

    Returns ``None`` if no source observed anything at all.
    """
    numerator = 0
    denominator = 0
    for source in SOURCES:
        value = values.get(source)
        if value is None:
            continue
        weight = mix.weight_of(source)
        if weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator // denominator


def _or_zero(value: int | None) -> int:
    return 0 if value is None else value


# --- observations -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSample:
    """What one source saw of one miner over one epoch.

    ``clean_rejects`` are explicit 429/503 responses — the miner said "I am
    full" instead of timing out. §8.1 charges half a failure for those,
    because honest backpressure is better for users than a hung connection
    and pretending otherwise teaches miners to hang.
    """

    attempts: int = 0
    successes: int = 0
    clean_rejects: int = 0
    ttft_ms: tuple[int, ...] = ()
    tps_milli: tuple[int, ...] = ()
    served: int = 0

    def __post_init__(self) -> None:
        if self.successes + self.clean_rejects > self.attempts:
            raise ValueError(
                f"sample claims {self.successes} successes and "
                f"{self.clean_rejects} clean rejects out of {self.attempts} "
                "attempts"
            )


EMPTY = SourceSample()


@dataclass(frozen=True, slots=True)
class MinerObservations:
    """One miner's epoch, from every source, plus the gate inputs.

    Assembled by ``state.py`` from the probe log, the attestation table and
    the platform's telemetry. Keyed by hotkey rather than uid wherever it
    crosses an epoch boundary — see :func:`score_epoch`.
    """

    uid: int
    hotkey: str
    shadow: SourceSample = EMPTY
    direct: SourceSample = EMPTY
    telemetry: SourceSample = EMPTY

    attested: bool = False
    receipts_mismatched: bool = False
    capacity_overclaimed: bool = False
    cooldown_epochs_left: int = 0

    def sample(self, source: str) -> SourceSample:
        if source == "shadow":
            return self.shadow
        if source == "direct":
            return self.direct
        if source == "telemetry":
            return self.telemetry
        raise KeyError(source)

    @property
    def probe_successes(self) -> int:
        """Successful *probes* — telemetry is self-reported and does not count."""
        return self.shadow.successes + self.direct.successes


# --- the four components ----------------------------------------------------


def latency_bps(ttft_p95_ms: int, target_ms: int) -> int:
    """``clamp(target / p95, 0, 1)`` in basis points.

    A miner at the target earns 10000. One at twice the target earns 5000.
    There is no bonus for beating the target, because the user cannot
    perceive it and paying for it would push miners to optimise the probe
    rather than the product.
    """
    if ttft_p95_ms <= 0:
        # Sub-millisecond TTFT is a broken measurement, not a fast miner.
        # Clamping up rather than down keeps a rounding artefact from
        # looking like a failure; the gate catches an actually-dead miner.
        return BPS_ONE
    return min(BPS_ONE, BPS_ONE * target_ms // ttft_p95_ms)


def throughput_bps(tps_p50_milli: int, target_milli: int) -> int:
    """``clamp(p50 / target, 0, 1)`` in basis points."""
    if tps_p50_milli <= 0:
        return 0
    return min(BPS_ONE, BPS_ONE * tps_p50_milli // target_milli)


def reliability_bps(sample: SourceSample) -> int | None:
    """``(successes + 0.5 * clean_rejects) / attempts`` in basis points.

    §8.1 states this as ``(successes - 0.5*clean_rejects)/attempts``, which
    is the same number *only* if ``successes`` is read as "every response
    that was not a timeout", clean rejects included. Read the natural way —
    successes are 2xx, clean rejects are 429/503, the two are disjoint —
    that formula charges a clean reject *more* than a timeout, which is
    backwards from the stated intent one paragraph above it.

    So the fields here are disjoint and the sign is flipped, which is
    algebraically identical to the design under its own intended reading and
    cannot be misread. A miner that is up and honestly full earns half
    credit: better than one that hangs, worse than one that serves.

    The half is applied by doubling both sides rather than by dividing by
    two, so there is no intermediate rounding. ``None`` when the source made
    no attempts — see :func:`blend`.
    """
    if sample.attempts <= 0:
        return None
    numerator = (2 * sample.successes + sample.clean_rejects) * BPS_ONE
    return min(BPS_ONE, numerator // (2 * sample.attempts))


def capacity_bps(served: int, max_served: int) -> int:
    """``log(1 + served) / log(1 + max_served)`` in basis points.

    Log-scaled because the alternative rewards raw size linearly, and a
    subnet whose capacity term is linear is a subnet where one large
    operator earns proportionally to spend. Log means doubling your fleet is
    worth progressively less, which is the shape we want.

    The base cancels, so :func:`log2_fp` is used and the whole thing stays
    in integers.
    """
    if max_served <= 0 or served <= 0:
        return 0
    denominator = log2_fp(1 + max_served)
    if denominator == 0:
        return 0
    numerator = log2_fp(1 + min(served, max_served))
    return min(BPS_ONE, numerator * BPS_ONE // denominator)


def credited_served(obs: MinerObservations, mix: SourceMix) -> int:
    """The served-request count that feeds capacity, per the §8.2 mix.

    Under the shipped config this is telemetry alone: only real user traffic
    demonstrates concurrency, and a probe fleet we control cannot. Written
    as a weighted sum anyway so that reweighting is a config change rather
    than a code change.
    """
    total = 0
    for source in SOURCES:
        total += mix.weight_of(source) * obs.sample(source).served
    return total // BPS_ONE


# --- the gate ---------------------------------------------------------------


def gate_bps(
    obs: MinerObservations,
    config: ScoringConfig,
    *,
    attestation_mode: AttestationMode,
) -> tuple[int, tuple[str, ...]]:
    """Resolve GATE for one miner. Multiplicative, mostly binary.

    Order matters for the reason string, not the number: cooldown is checked
    before probe count because a miner in cooldown is not being probed and
    would otherwise be told it failed for a reason it cannot act on.
    """
    if obs.cooldown_epochs_left > 0:
        return config.gate_fail_bps, (
            f"cooldown: {obs.cooldown_epochs_left} epoch(s) remaining",
        )

    if obs.probe_successes < config.min_probe_successes:
        return config.gate_fail_bps, (
            f"insufficient probes: {obs.probe_successes} < "
            f"{config.min_probe_successes}",
        )

    if not obs.attested:
        if attestation_mode == "hard":
            return config.gate_fail_bps, ("no valid attestation",)
        if attestation_mode == "warn":
            return config.gate_warn_bps, (
                "no valid attestation (attestation_mode=warn)",
            )
        return config.gate_pass_bps, ("attestation not checked (mode=off)",)

    return config.gate_pass_bps, ()


@dataclass(frozen=True, slots=True)
class GateState:
    """How close a miner is to being dropped from rotation.

    Persisted per hotkey across epochs by ``state.py``. Kept out of that
    module because it is policy, not storage, and policy that lives next to
    a SQL statement is policy nobody unit-tests.
    """

    consecutive_misses: int = 0
    cooldown_epochs_left: int = 0

    @property
    def is_out(self) -> bool:
        return self.cooldown_epochs_left > 0


def advance_gate_state(
    state: GateState, *, gated_out: bool, config: ScoringConfig
) -> GateState:
    """The end-of-epoch transition for one miner's rotation status.

    Failing the gate twice in a row costs a cooldown epoch on top, so that
    flapping — up long enough to be scored, down whenever probes get
    expensive — is strictly worse than staying up. A miner already serving
    cooldown does not accumulate further misses; it is not being probed, and
    charging it for probes it was never sent would make the cooldown
    self-extending.
    """
    if state.cooldown_epochs_left > 0:
        return GateState(0, state.cooldown_epochs_left - 1)
    if not gated_out:
        return GateState(0, 0)
    misses = state.consecutive_misses + 1
    if misses >= config.gate_out_after_misses:
        return GateState(0, config.gate_out_cooldown_epochs)
    return GateState(misses, 0)


def penalty_bps(
    obs: MinerObservations, config: ScoringConfig
) -> tuple[int, tuple[str, ...]]:
    """Multiplicative penalties for behaviour that is wrong but not fatal.

    Composed multiplicatively rather than summed, so two penalties cannot
    add up past zero and turn into a bonus.
    """
    multiplier = BPS_ONE
    reasons: list[str] = []
    if obs.receipts_mismatched:
        multiplier = multiplier * config.penalty_receipt_mismatch // BPS_ONE
        reasons.append("receipt mismatch: signed for traffic it did not serve")
    if obs.capacity_overclaimed:
        multiplier = multiplier * config.penalty_capacity_overclaim // BPS_ONE
        reasons.append("capacity overclaim: advertised more than it accepted")
    return multiplier, tuple(reasons)


# --- results ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Components:
    """The §8.1 breakdown for one miner, all in basis points.

    Kept as a value rather than collapsed into the score because ``/scores``
    serves it verbatim: a miner asking "why am I earning less this week"
    should be able to read the answer off the endpoint instead of guessing.
    """

    latency: int
    throughput: int
    reliability: int
    capacity: int
    quality: int

    def as_dict(self) -> dict[str, int]:
        return {
            "latency_bps": self.latency,
            "throughput_bps": self.throughput,
            "reliability_bps": self.reliability,
            "capacity_bps": self.capacity,
            "quality_bps": self.quality,
        }


@dataclass(frozen=True, slots=True)
class MinerScore:
    uid: int
    hotkey: str
    components: Components
    smoothed_quality_bps: int
    gate_bps: int
    penalty_bps: int
    score_bps: int
    weight_u16: int
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "hotkey": self.hotkey,
            **self.components.as_dict(),
            "smoothed_quality_bps": self.smoothed_quality_bps,
            "gate_bps": self.gate_bps,
            "penalty_bps": self.penalty_bps,
            "score_bps": self.score_bps,
            "weight_u16": self.weight_u16,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Everything one scoring pass produced.

    ``carry_forward`` is what ``state.py`` persists for the next epoch. It is
    keyed by *hotkey*, not uid, and that is load-bearing — see
    :func:`score_epoch`.
    """

    scores: tuple[MinerScore, ...]
    weights: dict[int, int]
    carry_forward: dict[str, int]
    config_version: int
    slo_version: int

    @property
    def is_empty(self) -> bool:
        """True when nobody earned anything.

        The caller must not set an all-zero weight vector — depending on the
        SDK version that is either rejected or read as an abstention, and
        neither is what we mean. ``weights.py`` burns instead.
        """
        return all(w == 0 for w in self.weights.values())


# --- normalisation ----------------------------------------------------------


def cube_normalise(
    scores: Mapping[int, int], *, exponent: int = 3, total: int = MAX_WEIGHT_U16
) -> dict[int, int]:
    """``w_i = s_i^n / sum(s_j^n)``, apportioned to integers that sum exactly.

    Floor division alone loses up to one unit per miner, and a validator that
    emits 65,283 where another emits 65,284 has a different vector even
    though it did the same arithmetic. So the remainder is handed out by
    largest fractional part, ties broken by uid ascending — total order,
    no clock, no iteration-order dependency, exact sum.
    """
    if not scores:
        return {}

    uids = sorted(scores)
    cubes = {uid: scores[uid] ** exponent for uid in uids}
    denominator = sum(cubes.values())
    if denominator == 0:
        return {uid: 0 for uid in uids}

    shares = {uid: cubes[uid] * total // denominator for uid in uids}
    shortfall = total - sum(shares.values())
    if shortfall > 0:
        # Largest remainder first; uid ascending breaks the tie.
        order = sorted(uids, key=lambda uid: (-((cubes[uid] * total) % denominator), uid))
        for uid in order[:shortfall]:
            shares[uid] += 1
    return shares


# --- the epoch --------------------------------------------------------------


def score_epoch(
    observations: Sequence[MinerObservations],
    config: ScoringConfig,
    *,
    attestation_mode: AttestationMode = "hard",
    carry_forward: Mapping[str, int] | None = None,
) -> EpochResult:
    """Score every miner and produce the weight vector. DESIGN.md §8.1–§8.3.

    The order of operations is the part worth reading twice::

        quality   = 0.45L + 0.20T + 0.25R + 0.10C     (smoothed across epochs)
        score     = quality x GATE x penalties        (this epoch only)
        weight    = score^3 / sum(score^3)

    The EMA is applied to *quality*, before the gate — not to the final
    score. If it were applied after, a miner that failed the gate would still
    collect 70% of last epoch's weight, which makes the gate a suggestion.
    Smoothing belongs on the noisy measurement, not on the binary verdict.

    ``carry_forward`` is keyed by hotkey because the metagraph recycles uids.
    Keying the EMA by uid would hand a freshly registered miner the reputation
    of whichever miner it deregistered, which is both unfair and exploitable:
    deregister a well-scored neighbour, take its slot, inherit its score.
    """
    if carry_forward is None:
        carry_forward = {}

    max_served = 0
    served_by_uid: dict[int, int] = {}
    for obs in observations:
        served = credited_served(obs, config.capacity_sources)
        served_by_uid[obs.uid] = served
        max_served = max(max_served, served)

    scored: list[MinerScore] = []
    raw: dict[int, int] = {}
    next_carry: dict[str, int] = {}

    for obs in observations:
        latency_by_source: dict[str, int | None] = {}
        throughput_by_source: dict[str, int | None] = {}
        reliability_by_source: dict[str, int | None] = {}

        for source in SOURCES:
            sample = obs.sample(source)
            latency_by_source[source] = (
                latency_bps(
                    percentile(sample.ttft_ms, 95), config.slo.ttft_p95_target_ms
                )
                if sample.ttft_ms
                else None
            )
            throughput_by_source[source] = (
                throughput_bps(
                    percentile(sample.tps_milli, 50), config.slo.tps_p50_target_milli
                )
                if sample.tps_milli
                else None
            )
            reliability_by_source[source] = reliability_bps(sample)

        # A component nothing observed scores zero. That is only reachable
        # for a miner the gate is about to fail anyway (min_probe_successes), so it
        # never silently costs a working miner anything.
        latency = _or_zero(blend(latency_by_source, config.latency_sources))
        throughput = _or_zero(blend(throughput_by_source, config.throughput_sources))
        reliability = _or_zero(blend(reliability_by_source, config.reliability_sources))
        capacity = capacity_bps(served_by_uid[obs.uid], max_served)

        quality = (
            config.w_latency * latency
            + config.w_throughput * throughput
            + config.w_reliability * reliability
            + config.w_capacity * capacity
        ) // BPS_ONE

        smoothed = ema(carry_forward.get(obs.hotkey), quality, config.ema_alpha_bps)
        next_carry[obs.hotkey] = smoothed

        gate, gate_reasons = gate_bps(obs, config, attestation_mode=attestation_mode)
        penalty, penalty_reasons = penalty_bps(obs, config)
        score = smoothed * gate // BPS_ONE * penalty // BPS_ONE

        raw[obs.uid] = score
        scored.append(
            MinerScore(
                uid=obs.uid,
                hotkey=obs.hotkey,
                components=Components(
                    latency=latency,
                    throughput=throughput,
                    reliability=reliability,
                    capacity=capacity,
                    quality=quality,
                ),
                smoothed_quality_bps=smoothed,
                gate_bps=gate,
                penalty_bps=penalty,
                score_bps=score,
                weight_u16=0,  # filled in below, once the whole set is known
                reasons=gate_reasons + penalty_reasons,
            )
        )

    weights = cube_normalise(raw, exponent=config.exponent)
    scored = [
        MinerScore(
            uid=s.uid,
            hotkey=s.hotkey,
            components=s.components,
            smoothed_quality_bps=s.smoothed_quality_bps,
            gate_bps=s.gate_bps,
            penalty_bps=s.penalty_bps,
            score_bps=s.score_bps,
            weight_u16=weights.get(s.uid, 0),
            reasons=s.reasons,
        )
        for s in scored
    ]
    scored.sort(key=lambda s: s.uid)

    return EpochResult(
        scores=tuple(scored),
        weights=weights,
        carry_forward=next_carry,
        config_version=config.version,
        slo_version=config.slo.version,
    )
