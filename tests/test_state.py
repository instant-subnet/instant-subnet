"""Tests for the validator's sqlite state.

Two things can go wrong with persistence and they fail in opposite ways.

The loud failure is a schema or SQL mistake: the validator will not start, or
a query raises. Those are cheap to find and cheap to fix.

The quiet failure is the one worth writing tests for. State that round-trips
*almost* correctly — an EMA that advances twice for one epoch, a cooldown
that is forgotten across a restart, a miner that vanishes from the roster
because it was never probed — produces a validator that keeps running and
sets slightly wrong weights forever. Nobody notices until vtrust drifts and
somebody spends a week bisecting.

So the tests below spend most of their effort on the boundary between what
was written and what comes back out: that ``observations_for_epoch`` is
driven by the roster and not by the probe table, that reputation is keyed by
hotkey so a recycled uid inherits nothing, and that ``commit_epoch`` is
genuinely one transaction.
"""

from __future__ import annotations

import sqlite3

import pytest

from instant.validator.score import (
    GateState,
    SourceSample,
    load_scoring_config,
    score_epoch,
)
from instant.validator.state import (
    SCHEMA_VERSION,
    ProbeResult,
    StateError,
    TelemetryRow,
    ValidatorState,
    open_state,
)

HK_A = "5AAA"
HK_B = "5BBB"
HK_C = "5CCC"


@pytest.fixture
def state(tmp_path):
    with open_state(tmp_path / "state.sqlite3") as st:
        yield st


@pytest.fixture(scope="module")
def config():
    return load_scoring_config()


def probe(
    epoch: int,
    hotkey: str,
    *,
    uid: int = 0,
    source: str = "direct",
    outcome: str = "success",
    ttft_ms: int | None = 200,
    tps_milli: int | None = 100_000,
    observed_ms: int = 400,
) -> ProbeResult:
    return ProbeResult(
        epoch=epoch,
        uid=uid,
        hotkey=hotkey,
        source=source,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        observed_ms=observed_ms,
        ttft_ms=ttft_ms,
        tps_milli=tps_milli,
    )


# --------------------------------------------------------------------------
# Opening the file
# --------------------------------------------------------------------------


def test_a_fresh_file_is_migrated_to_the_current_schema(tmp_path):
    with open_state(tmp_path / "fresh.sqlite3") as st:
        version = st._db.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_opening_an_existing_file_does_not_run_migrations_again(tmp_path):
    """Reopening must be a no-op, not a second CREATE TABLE.

    The failure mode this catches is a migration loop that re-runs on every
    boot: it would raise "table already exists" and take the validator down
    on restart rather than on first install, which is the worst time to find
    out.
    """
    path = tmp_path / "twice.sqlite3"
    with open_state(path) as st:
        st.record_probes([probe(1, HK_A)])

    with open_state(path) as st:
        assert st._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert st._db.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 1


def test_a_file_from_a_newer_validator_is_refused(tmp_path):
    """Refuse rather than downgrade.

    An older build opening a newer file would run happily against columns it
    does not know exist, write rows missing them, and corrupt the audit trail
    of the newer build that wrote it. Refusing costs an operator one confusing
    error message; the alternative costs them their history.
    """
    path = tmp_path / "future.sqlite3"
    open_state(path).close()
    db = sqlite3.connect(str(path))
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    db.commit()
    db.close()

    with pytest.raises(StateError, match="newer validator"):
        open_state(path)


def test_the_parent_directory_is_created(tmp_path):
    target = tmp_path / "nested" / "deeper" / "state.sqlite3"
    with open_state(target):
        pass
    assert target.exists()


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def test_recording_no_probes_is_not_an_error(state):
    assert state.record_probes([]) == 0
    assert state.record_telemetry([]) == 0


def test_probes_accumulate_rather_than_replace(state):
    """Every probe is a separate observation and all of them count.

    Probes are the sample the percentiles are computed over. If a second
    probe replaced the first, an epoch's p95 would be the last probe's TTFT,
    which is not a p95 of anything.
    """
    state.record_probes([probe(1, HK_A, ttft_ms=100), probe(1, HK_A, ttft_ms=900)])
    rows = state._db.execute("SELECT ttft_ms FROM probes ORDER BY id").fetchall()
    assert [r["ttft_ms"] for r in rows] == [100, 900]


def test_a_later_telemetry_poll_supersedes_the_earlier_one(state):
    """The platform reports a window, not a delta.

    Each poll restates the whole window's aggregate, so accumulating them
    would double-count every request the platform has already told us about.
    """
    state.record_telemetry([
        TelemetryRow(epoch=1, uid=0, hotkey=HK_A, requests=10, successes=10),
    ])
    state.record_telemetry([
        TelemetryRow(epoch=1, uid=0, hotkey=HK_A, requests=25, successes=24),
    ])
    rows = state._db.execute("SELECT requests, successes FROM telemetry").fetchall()
    assert len(rows) == 1
    assert (rows[0]["requests"], rows[0]["successes"]) == (25, 24)


def test_telemetry_for_different_epochs_coexists(state):
    state.record_telemetry([
        TelemetryRow(epoch=1, uid=0, hotkey=HK_A, requests=10, successes=10),
        TelemetryRow(epoch=2, uid=0, hotkey=HK_A, requests=20, successes=20),
    ])
    assert state._db.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0] == 2


def test_an_attestation_verdict_can_be_revised_within_an_epoch(state):
    """Re-verification within an epoch replaces, it does not append.

    A miner that re-attests after a transient NRAS failure should end the
    epoch attested. Appending would leave both verdicts and make the
    aggregate depend on which one a query happened to read first.
    """
    state.record_attestation(epoch=1, hotkey=HK_A, ok=False, verified_ms=1, detail="nras timeout")
    state.record_attestation(epoch=1, hotkey=HK_A, ok=True, verified_ms=2, attestation_id="abc")

    row = state._db.execute("SELECT * FROM attestations").fetchone()
    assert row["ok"] == 1
    assert row["attestation_id"] == "abc"


# --------------------------------------------------------------------------
# Assembling observations
# --------------------------------------------------------------------------


def test_the_roster_decides_who_is_scored_not_the_probe_table(state):
    """A registered miner that never answered must still appear.

    Leaving it out would make "never answered a single probe" and "not
    registered" indistinguishable in the weight vector — and the first of
    those needs to be visibly scored zero, not silently absent.
    """
    state.record_probes([probe(1, HK_A, uid=0)])

    observations = state.observations_for_epoch(1, [(0, HK_A), (1, HK_B)])

    assert [o.hotkey for o in observations] == [HK_A, HK_B]
    silent = observations[1]
    assert silent.direct == SourceSample()
    assert silent.shadow == SourceSample()
    assert silent.probe_successes == 0


def test_a_miner_probed_but_not_on_the_roster_is_dropped(state):
    """Deregistered between probing and scoring: it earns nothing.

    Scoring a hotkey that is no longer on the metagraph would mean emitting a
    weight for a uid that belongs to somebody else now.
    """
    state.record_probes([probe(1, HK_A), probe(1, HK_C)])
    observations = state.observations_for_epoch(1, [(0, HK_A)])
    assert [o.hotkey for o in observations] == [HK_A]


def test_observations_are_ordered_by_uid(state):
    """Stable order in, stable order out.

    ``score_epoch`` sorts anyway, but a roster arriving in metagraph order
    and coming back in dict-insertion order is the kind of thing that makes
    two validators' logs disagree while their weights agree, which wastes an
    afternoon.
    """
    observations = state.observations_for_epoch(1, [(7, HK_C), (2, HK_A), (5, HK_B)])
    assert [o.uid for o in observations] == [2, 5, 7]


def test_probe_outcomes_land_in_the_right_counters(state):
    state.record_probes([
        probe(1, HK_A, outcome="success"),
        probe(1, HK_A, outcome="success"),
        probe(1, HK_A, outcome="clean_reject", ttft_ms=None, tps_milli=None),
        probe(1, HK_A, outcome="failure", ttft_ms=None, tps_milli=None),
    ])
    direct = state.observations_for_epoch(1, [(0, HK_A)])[0].direct
    assert direct.attempts == 4
    assert direct.successes == 2
    assert direct.clean_rejects == 1
    # Only successes contribute timings — a failed probe has no TTFT to report,
    # and counting a timeout as a slow success would flatter the miner.
    assert len(direct.ttft_ms) == 2


def test_shadow_and_direct_probes_stay_in_separate_samples(state):
    """They are weighted differently in §8.2, so mixing them changes scores."""
    state.record_probes([
        probe(1, HK_A, source="shadow", ttft_ms=100),
        probe(1, HK_A, source="direct", ttft_ms=800),
    ])
    obs = state.observations_for_epoch(1, [(0, HK_A)])[0]
    assert obs.shadow.ttft_ms == (100,)
    assert obs.direct.ttft_ms == (800,)


def test_samples_are_sorted_so_they_do_not_depend_on_arrival_order(state):
    """The sample must be a function of the epoch, not of network jitter.

    Two validators probing the same miner see the same values in different
    orders. ``percentile`` sorts internally, so this is belt and braces — but
    it makes the stored sample directly comparable between validators when
    somebody is debugging a disagreement.
    """
    state.record_probes([
        probe(1, HK_A, ttft_ms=500),
        probe(1, HK_A, ttft_ms=100),
        probe(1, HK_A, ttft_ms=300),
    ])
    assert state.observations_for_epoch(1, [(0, HK_A)])[0].direct.ttft_ms == (100, 300, 500)


def test_only_this_epochs_probes_are_used(state):
    state.record_probes([probe(1, HK_A, ttft_ms=100), probe(2, HK_A, ttft_ms=900)])
    assert state.observations_for_epoch(2, [(0, HK_A)])[0].direct.ttft_ms == (900,)


def test_attestation_carries_into_the_observation(state):
    state.record_attestation(epoch=1, hotkey=HK_A, ok=True, verified_ms=1)
    state.record_attestation(epoch=1, hotkey=HK_B, ok=False, verified_ms=1)

    by_hotkey = {o.hotkey: o for o in state.observations_for_epoch(1, [(0, HK_A), (1, HK_B)])}
    assert by_hotkey[HK_A].attested is True
    assert by_hotkey[HK_B].attested is False


def test_an_attestation_from_a_previous_epoch_does_not_count(state):
    """§2.4: a bundle older than the window is a memory, not an attestation."""
    state.record_attestation(epoch=1, hotkey=HK_A, ok=True, verified_ms=1)
    assert state.observations_for_epoch(2, [(0, HK_A)])[0].attested is False


# --------------------------------------------------------------------------
# Pre-aggregated telemetry
# --------------------------------------------------------------------------


def test_platform_percentiles_round_trip_through_one_element_samples(state):
    """The platform hands us p95/p50; we must not re-percentile them.

    A one-element sample is the honest representation: nearest-rank p95 of a
    single observation is that observation, so the number the platform
    computed is the number that reaches scoring, unmodified.
    """
    state.record_telemetry([
        TelemetryRow(
            epoch=1, uid=0, hotkey=HK_A,
            requests=100, successes=99, clean_rejects=1,
            ttft_p95_ms=311, tokens_per_s_p50=87, served=42,
        )
    ])
    telemetry = state.observations_for_epoch(1, [(0, HK_A)])[0].telemetry

    assert telemetry.ttft_ms == (311,)
    # tokens/sec becomes tokens/sec x 1000 — the tps_milli unit used everywhere
    # downstream. Getting this conversion wrong scores every miner at 1/1000th
    # of its throughput, which the gate then reads as a dead miner.
    assert telemetry.tps_milli == (87_000,)
    assert (telemetry.attempts, telemetry.successes, telemetry.clean_rejects) == (100, 99, 1)
    assert telemetry.served == 42


def test_missing_platform_percentiles_stay_missing(state):
    """A window with no successful requests has no p95 to report.

    Empty rather than zero: a zero TTFT would score as infinitely fast. The
    blend in §8.2 drops absent sources and renormalises over what is present,
    which is the correct treatment of "not observed".
    """
    state.record_telemetry([
        TelemetryRow(epoch=1, uid=0, hotkey=HK_A, requests=5, successes=0)
    ])
    telemetry = state.observations_for_epoch(1, [(0, HK_A)])[0].telemetry
    assert telemetry.ttft_ms == ()
    assert telemetry.tps_milli == ()


def test_unverified_receipts_flag_a_mismatch(state):
    """Fewer verified than seen means the platform could not prove the traffic."""
    state.record_telemetry([
        TelemetryRow(epoch=1, uid=0, hotkey=HK_A, requests=10, successes=10,
                     receipts_seen=10, receipts_verified=7),
        TelemetryRow(epoch=1, uid=1, hotkey=HK_B, requests=10, successes=10,
                     receipts_seen=10, receipts_verified=10),
    ])
    by_hotkey = {o.hotkey: o for o in state.observations_for_epoch(1, [(0, HK_A), (1, HK_B)])}
    assert by_hotkey[HK_A].receipts_mismatched is True
    assert by_hotkey[HK_B].receipts_mismatched is False


# --------------------------------------------------------------------------
# Reputation across epochs
# --------------------------------------------------------------------------


def test_carry_forward_is_empty_before_anything_is_committed(state):
    assert state.carry_forward() == {}


def test_carry_forward_survives_a_restart(tmp_path):
    """The point of the file.

    A validator restarting mid-epoch must not reset every miner's EMA to that
    epoch's raw quality — that is a free reputation reset for anyone who
    times a bad epoch against a deploy.
    """
    path = tmp_path / "restart.sqlite3"
    with open_state(path) as st:
        _commit(st, epoch=1, hotkeys=[HK_A, HK_B])
        before = st.carry_forward()

    with open_state(path) as st:
        assert st.carry_forward() == before
        assert set(before) == {HK_A, HK_B}


def test_carry_forward_is_keyed_by_hotkey_not_uid(tmp_path):
    """A recycled uid must inherit nothing.

    The metagraph reuses uids on deregistration. Keyed by uid, an attacker
    could deregister a well-scored neighbour, take its slot, and start the
    epoch holding its EMA. Keyed by hotkey, the new miner starts from
    nothing, which is what it has earned.
    """
    path = tmp_path / "recycle.sqlite3"
    with open_state(path) as st:
        _commit(st, epoch=1, hotkeys=[HK_A])
        assert st.carry_forward()[HK_A] > 0

        # Same uid, different hotkey: the newcomer gets no history.
        carried = st.carry_forward()
        assert HK_C not in carried


def test_gate_state_defaults_to_clean_for_an_unknown_hotkey(state):
    assert state.gate_state("5NeverSeen") == GateState(0, 0)


def test_a_cooldown_survives_a_restart(tmp_path):
    """Otherwise a restart is a way to serve out a cooldown instantly."""
    path = tmp_path / "cooldown.sqlite3"
    with open_state(path) as st:
        _commit(st, epoch=1, hotkeys=[HK_A], gate_states={HK_A: GateState(0, 2)})

    with open_state(path) as st:
        assert st.gate_state(HK_A) == GateState(0, 2)
        # And it reaches the observation, where score_epoch's gate reads it.
        assert st.observations_for_epoch(2, [(0, HK_A)])[0].cooldown_epochs_left == 2


def test_committing_an_epoch_writes_scores_and_reputation_together(state):
    result = _commit(state, epoch=4, hotkeys=[HK_A, HK_B])

    scores = state.scores_for_epoch(4)
    assert {row["hotkey"] for row in scores} == {HK_A, HK_B}
    assert set(state.carry_forward()) == {HK_A, HK_B}
    assert result.scores  # sanity: the fixture actually produced something


def test_a_commit_missing_a_gate_state_is_refused(state):
    """Better to write no epoch than a plausible-looking wrong one.

    Skipping the miner would drop its EMA; defaulting it to a clean state
    would forgive a cooldown. Both leave a running validator quietly wrong,
    which is the failure mode this whole module exists to avoid.
    """
    result = _commit_result(state, epoch=5, hotkeys=[HK_A, HK_B])

    with pytest.raises(StateError, match="no gate state"):
        state.commit_epoch(5, result, {HK_A: GateState()})  # HK_B missing

    assert state.scores_for_epoch(5) == []
    assert state.carry_forward() == {}


class _FailsOnReputation(sqlite3.Connection):
    """A connection whose second write in ``commit_epoch`` fails.

    Injected as a connection factory rather than by patching an attribute:
    ``sqlite3.Connection`` methods are read-only, and a subclass is closer to
    what the real failure looks like anyway — the disk goes away between two
    statements, not between two Python objects.
    """

    attempts = 0

    def executemany(self, sql, rows):  # type: ignore[override]
        type(self).attempts += 1
        if "reputation" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return super().executemany(sql, rows)


def test_a_crash_between_the_two_writes_rolls_back_both(tmp_path):
    """The transaction is the whole point of ``commit_epoch``.

    Half-written — scores in, reputation not — the next run would apply the
    same observations to the same starting EMA, counting one epoch twice.
    Fault-injected rather than argued about, because "it is inside a with
    block" is exactly the kind of claim that stops being true during a
    refactor.
    """
    path = tmp_path / "crash.sqlite3"
    with open_state(path) as st:
        result = _commit_result(st, epoch=5, hotkeys=[HK_A])

    _FailsOnReputation.attempts = 0
    db = sqlite3.connect(str(path), isolation_level="DEFERRED", factory=_FailsOnReputation)
    with ValidatorState(db) as broken:
        with pytest.raises(StateError, match="state write failed"):
            broken.commit_epoch(5, result, {HK_A: GateState()})

    assert _FailsOnReputation.attempts == 2  # scores attempted, then reputation failed

    with open_state(path) as st:
        assert st.scores_for_epoch(5) == []
        assert st.carry_forward() == {}


def test_recommitting_an_epoch_replaces_rather_than_duplicates(state):
    """Replaying an epoch after a crash must be idempotent."""
    result = _commit_result(state, epoch=6, hotkeys=[HK_A])
    gates = {HK_A: GateState()}
    state.commit_epoch(6, result, gates)
    state.commit_epoch(6, result, gates)

    assert len(state.scores_for_epoch(6)) == 1


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------


def test_scores_come_back_with_their_reasons_parsed(state):
    """``reasons`` is JSON in the column and a list at the boundary.

    It is the field an operator reads when a miner asks why it earned
    nothing, so handing back a JSON string for the caller to parse again is
    an invitation to forget.
    """
    _commit(state, epoch=7, hotkeys=[HK_A], attested=False)
    row = state.scores_for_epoch(7)[0]
    assert isinstance(row["reasons"], list)
    assert any("attestation" in reason for reason in row["reasons"])


def test_scores_are_ordered_by_uid(state):
    _commit(state, epoch=8, hotkeys=[HK_C, HK_A, HK_B])
    uids = [row["uid"] for row in state.scores_for_epoch(8)]
    assert uids == sorted(uids)


def test_latest_epoch_is_none_before_anything_is_scored(state):
    assert state.latest_epoch() is None


def test_latest_epoch_tracks_the_highest_committed(state):
    _commit(state, epoch=3, hotkeys=[HK_A])
    _commit(state, epoch=9, hotkeys=[HK_A])
    _commit(state, epoch=5, hotkeys=[HK_A])
    assert state.latest_epoch() == 9


def test_weight_sets_are_recorded_and_the_last_one_is_readable(state):
    state.record_weight_set(
        epoch=1, block=100, submitted_ms=1, ok=True, attempts=1,
        config_version=1, vector_digest="sha256:aaa",
    )
    state.record_weight_set(
        epoch=2, block=460, submitted_ms=2, ok=False, attempts=3,
        config_version=1, vector_digest="sha256:bbb", error="timeout",
    )
    last = state.last_weight_set()
    assert last["epoch"] == 2
    assert last["ok"] == 0
    assert last["error"] == "timeout"


def test_last_weight_set_is_none_on_a_fresh_validator(state):
    assert state.last_weight_set() is None


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------


def test_prune_drops_old_probes_only(state):
    """Probes are the table that grows; everything else is the audit trail.

    If pruning took scores with it, the answer to "why did this miner earn
    that six weeks ago" would be gone — which is precisely when somebody
    asks.
    """
    _commit(state, epoch=1, hotkeys=[HK_A])          # writes epoch 1's probes too
    state.record_probes([probe(2, HK_A), probe(3, HK_A)])
    state.record_telemetry([TelemetryRow(epoch=1, uid=0, hotkey=HK_A, requests=1, successes=1)])
    before = state._db.execute("SELECT COUNT(*) FROM probes").fetchone()[0]

    removed = state.prune(before_epoch=3)

    # Everything below epoch 3 goes; the single epoch-3 probe stays.
    assert removed == before - 1
    assert state._db.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 1
    assert state._db.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0] == 1
    assert len(state.scores_for_epoch(1)) == 1


def test_pruning_an_empty_range_removes_nothing(state):
    state.record_probes([probe(5, HK_A)])
    assert state.prune(before_epoch=5) == 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _commit_result(state: ValidatorState, *, epoch: int, hotkeys, attested: bool = True):
    """Probe every hotkey enough to clear the gate, then score the epoch."""
    config = load_scoring_config()
    rows = []
    for uid, hotkey in enumerate(hotkeys):
        for source in ("shadow", "direct"):
            for _ in range(config.min_probes):
                rows.append(probe(epoch, hotkey, uid=uid, source=source))
        if attested:
            state.record_attestation(epoch=epoch, hotkey=hotkey, ok=True, verified_ms=0)
    state.record_probes(rows)

    roster = [(uid, hotkey) for uid, hotkey in enumerate(hotkeys)]
    return score_epoch(
        state.observations_for_epoch(epoch, roster),
        config,
        attestation_mode="hard",
        carry_forward=state.carry_forward(),
    )


def _commit(state: ValidatorState, *, epoch: int, hotkeys, gate_states=None, attested=True):
    result = _commit_result(state, epoch=epoch, hotkeys=hotkeys, attested=attested)
    gates = gate_states or {hotkey: GateState() for hotkey in hotkeys}
    state.commit_epoch(epoch, result, gates)
    return result
