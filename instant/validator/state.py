"""Validator state: one sqlite file, no server, no ORM. DESIGN.md §9.

A validator has to survive a restart without forgetting who was fast, who
was in cooldown, and what weights it last set — and it has to be able to
answer "why did this miner earn that" weeks later, when the answer matters
because somebody is arguing about it. Both of those are a database, and both
of them are satisfied by a file.

sqlite specifically, and synchronously, for three reasons:

* An epoch is 360 blocks — roughly 72 minutes. Nothing here is on a hot
  path, so the complexity budget for persistence is approximately zero.
* A validator that cannot reach its database must stop, not continue with
  stale reputation. A local file fails in exactly the way that produces
  that behaviour, with no partition semantics to reason about.
* An operator debugging their own vtrust can open the file with the
  ``sqlite3`` binary that is already on the box. That is worth more than
  any query-planner feature we would otherwise be buying.

Raw probe rows are kept rather than incremental aggregates. It costs about
55,000 rows an epoch and buys the ability to recompute any past epoch's
scores from the observations after changing the formula — which is the only
honest way to evaluate a scoring change before shipping it. :meth:`prune`
bounds the growth.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from instant.validator.score import (
    EpochResult,
    GateState,
    MinerObservations,
    SourceSample,
)

#: Bumped whenever the statements in ``_MIGRATIONS`` grow an entry.
SCHEMA_VERSION = 1

ProbeSource = Literal["shadow", "direct"]
ProbeOutcome = Literal["success", "clean_reject", "failure"]

_MIGRATIONS: tuple[str, ...] = (
    # 1 — initial schema.
    """
    CREATE TABLE probes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch        INTEGER NOT NULL,
        hotkey       TEXT    NOT NULL,
        uid          INTEGER NOT NULL,
        source       TEXT    NOT NULL,
        outcome      TEXT    NOT NULL,
        observed_ms  INTEGER NOT NULL,
        ttft_ms      INTEGER,
        tps_milli    INTEGER,
        tokens       INTEGER,
        error        TEXT
    );
    CREATE INDEX probes_by_epoch ON probes (epoch, hotkey);

    CREATE TABLE telemetry (
        epoch             INTEGER NOT NULL,
        hotkey            TEXT    NOT NULL,
        uid               INTEGER NOT NULL,
        requests          INTEGER NOT NULL,
        successes         INTEGER NOT NULL,
        clean_rejects     INTEGER NOT NULL DEFAULT 0,
        ttft_p95_ms       INTEGER,
        tokens_per_s_p50  INTEGER,
        served            INTEGER NOT NULL DEFAULT 0,
        receipts_seen     INTEGER NOT NULL DEFAULT 0,
        receipts_verified INTEGER NOT NULL DEFAULT 0,
        recorded_ms       INTEGER NOT NULL,
        PRIMARY KEY (epoch, hotkey)
    );

    CREATE TABLE attestations (
        epoch          INTEGER NOT NULL,
        hotkey         TEXT    NOT NULL,
        ok             INTEGER NOT NULL,
        attestation_id TEXT,
        detail         TEXT,
        verified_ms    INTEGER NOT NULL,
        PRIMARY KEY (epoch, hotkey)
    );

    CREATE TABLE reputation (
        hotkey                TEXT    PRIMARY KEY,
        smoothed_quality_bps  INTEGER NOT NULL,
        consecutive_misses    INTEGER NOT NULL DEFAULT 0,
        cooldown_epochs_left  INTEGER NOT NULL DEFAULT 0,
        updated_epoch         INTEGER NOT NULL
    );

    CREATE TABLE scores (
        epoch                INTEGER NOT NULL,
        hotkey               TEXT    NOT NULL,
        uid                  INTEGER NOT NULL,
        latency_bps          INTEGER NOT NULL,
        throughput_bps       INTEGER NOT NULL,
        reliability_bps      INTEGER NOT NULL,
        capacity_bps         INTEGER NOT NULL,
        quality_bps          INTEGER NOT NULL,
        smoothed_quality_bps INTEGER NOT NULL,
        gate_bps             INTEGER NOT NULL,
        penalty_bps          INTEGER NOT NULL,
        score_bps            INTEGER NOT NULL,
        weight_u16           INTEGER NOT NULL,
        reasons              TEXT    NOT NULL,
        PRIMARY KEY (epoch, hotkey)
    );

    CREATE TABLE weight_sets (
        epoch          INTEGER PRIMARY KEY,
        block          INTEGER,
        submitted_ms   INTEGER NOT NULL,
        ok             INTEGER NOT NULL,
        attempts       INTEGER NOT NULL,
        config_version INTEGER NOT NULL,
        vector_digest  TEXT    NOT NULL,
        error          TEXT
    );
    """,
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One probe, as recorded. Built by ``probe.py``, consumed here."""

    epoch: int
    uid: int
    hotkey: str
    source: ProbeSource
    outcome: ProbeOutcome
    observed_ms: int
    ttft_ms: int | None = None
    tps_milli: int | None = None
    tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    """One miner's window as the platform reported it. §7.5.

    ``ttft_p95_ms`` and ``tokens_per_s_p50`` arrive pre-aggregated — the
    platform sees every user request and we see a sample of them, so it
    computes the percentiles and we take them. :meth:`ValidatorState.
    observations_for_epoch` turns each into a one-element sample, whose p95
    and p50 are that element, which is the honest way to feed a
    pre-aggregated number into a function that expects observations.
    """

    epoch: int
    uid: int
    hotkey: str
    requests: int
    successes: int
    clean_rejects: int = 0
    ttft_p95_ms: int | None = None
    tokens_per_s_p50: int | None = None
    served: int = 0
    receipts_seen: int = 0
    receipts_verified: int = 0
    recorded_ms: int = 0


class StateError(RuntimeError):
    """The state file is unusable. Always fatal — see the module docstring."""


class ValidatorState:
    """A handle on the sqlite file. Not thread-safe; the epoch loop is one task."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection
        self._db.row_factory = sqlite3.Row

    # --- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ValidatorState:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One transaction, committed or rolled back as a unit.

        Every writer goes through this. A half-written epoch — scores
        persisted, reputation not — would silently double-count an EMA step
        on the next restart.
        """
        try:
            with self._db:
                yield self._db
        except sqlite3.Error as exc:  # pragma: no cover - disk-level failure
            raise StateError(f"state write failed: {exc}") from exc

    # --- recording ----------------------------------------------------------

    def record_probes(self, results: Iterable[ProbeResult]) -> int:
        rows = [
            (
                r.epoch, r.hotkey, r.uid, r.source, r.outcome, r.observed_ms,
                r.ttft_ms, r.tps_milli, r.tokens, r.error,
            )
            for r in results
        ]
        if not rows:
            return 0
        with self._write() as db:
            db.executemany(
                "INSERT INTO probes (epoch, hotkey, uid, source, outcome, "
                "observed_ms, ttft_ms, tps_milli, tokens, error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def record_telemetry(self, rows: Iterable[TelemetryRow]) -> int:
        payload = [
            (
                r.epoch, r.hotkey, r.uid, r.requests, r.successes, r.clean_rejects,
                r.ttft_p95_ms, r.tokens_per_s_p50, r.served, r.receipts_seen,
                r.receipts_verified, r.recorded_ms,
            )
            for r in rows
        ]
        if not payload:
            return 0
        with self._write() as db:
            # Replace rather than insert: the platform is polled repeatedly
            # within an epoch and each poll supersedes the last.
            db.executemany(
                "INSERT OR REPLACE INTO telemetry (epoch, hotkey, uid, requests, "
                "successes, clean_rejects, ttft_p95_ms, tokens_per_s_p50, served, "
                "receipts_seen, receipts_verified, recorded_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def record_attestation(
        self,
        *,
        epoch: int,
        hotkey: str,
        ok: bool,
        verified_ms: int,
        attestation_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        with self._write() as db:
            db.execute(
                "INSERT OR REPLACE INTO attestations "
                "(epoch, hotkey, ok, attestation_id, detail, verified_ms) "
                "VALUES (?,?,?,?,?,?)",
                (epoch, hotkey, int(ok), attestation_id, detail, verified_ms),
            )

    # --- reading ------------------------------------------------------------

    def carry_forward(self) -> dict[str, int]:
        """The EMA input for the next epoch, keyed by hotkey.

        Keyed by hotkey and not uid because the metagraph recycles uids —
        see :func:`instant.validator.score.score_epoch`.
        """
        return {
            row["hotkey"]: row["smoothed_quality_bps"]
            for row in self._db.execute(
                "SELECT hotkey, smoothed_quality_bps FROM reputation"
            )
        }

    def gate_state(self, hotkey: str) -> GateState:
        row = self._db.execute(
            "SELECT consecutive_misses, cooldown_epochs_left FROM reputation "
            "WHERE hotkey = ?",
            (hotkey,),
        ).fetchone()
        if row is None:
            return GateState()
        return GateState(row["consecutive_misses"], row["cooldown_epochs_left"])

    def observations_for_epoch(
        self, epoch: int, roster: Sequence[tuple[int, str]]
    ) -> list[MinerObservations]:
        """Assemble the epoch's observations for every miner on the roster.

        ``roster`` is ``(uid, hotkey)`` from the metagraph, and it — not the
        probe table — decides who gets scored. A miner that was registered
        but never successfully probed must appear with an empty sample so
        that the gate can fail it explicitly; leaving it out entirely would
        make "never answered" and "not registered" indistinguishable in the
        weight vector.
        """
        probes = self._probe_samples(epoch)
        telemetry = self._telemetry_samples(epoch)
        attested = {
            row["hotkey"]
            for row in self._db.execute(
                "SELECT hotkey FROM attestations WHERE epoch = ? AND ok = 1",
                (epoch,),
            )
        }
        mismatched = {
            row["hotkey"]
            for row in self._db.execute(
                "SELECT hotkey FROM telemetry WHERE epoch = ? "
                "AND receipts_verified < receipts_seen",
                (epoch,),
            )
        }

        observations = []
        for uid, hotkey in sorted(roster):
            shadow, direct = probes.get(hotkey, (SourceSample(), SourceSample()))
            gate = self.gate_state(hotkey)
            observations.append(
                MinerObservations(
                    uid=uid,
                    hotkey=hotkey,
                    shadow=shadow,
                    direct=direct,
                    telemetry=telemetry.get(hotkey, SourceSample()),
                    attested=hotkey in attested,
                    receipts_mismatched=hotkey in mismatched,
                    cooldown_epochs_left=gate.cooldown_epochs_left,
                )
            )
        return observations

    def _probe_samples(
        self, epoch: int
    ) -> dict[str, tuple[SourceSample, SourceSample]]:
        buckets: dict[tuple[str, str], dict[str, Any]] = {}
        rows = self._db.execute(
            "SELECT hotkey, source, outcome, ttft_ms, tps_milli FROM probes "
            "WHERE epoch = ? ORDER BY id",
            (epoch,),
        )
        for row in rows:
            key = (row["hotkey"], row["source"])
            bucket = buckets.setdefault(
                key,
                {"attempts": 0, "successes": 0, "clean_rejects": 0,
                 "ttft": [], "tps": []},
            )
            bucket["attempts"] += 1
            if row["outcome"] == "success":
                bucket["successes"] += 1
                if row["ttft_ms"] is not None:
                    bucket["ttft"].append(row["ttft_ms"])
                if row["tps_milli"] is not None:
                    bucket["tps"].append(row["tps_milli"])
            elif row["outcome"] == "clean_reject":
                bucket["clean_rejects"] += 1

        out: dict[str, tuple[SourceSample, SourceSample]] = {}
        for hotkey in {k[0] for k in buckets}:
            samples = []
            for source in ("shadow", "direct"):
                bucket = buckets.get((hotkey, source))
                if bucket is None:
                    samples.append(SourceSample())
                    continue
                samples.append(
                    SourceSample(
                        attempts=bucket["attempts"],
                        successes=bucket["successes"],
                        clean_rejects=bucket["clean_rejects"],
                        # Sorted so the sample is a function of the epoch's
                        # observations and not of the order they arrived in.
                        ttft_ms=tuple(sorted(bucket["ttft"])),
                        tps_milli=tuple(sorted(bucket["tps"])),
                    )
                )
            out[hotkey] = (samples[0], samples[1])
        return out

    def _telemetry_samples(self, epoch: int) -> dict[str, SourceSample]:
        out: dict[str, SourceSample] = {}
        for row in self._db.execute(
            "SELECT * FROM telemetry WHERE epoch = ?", (epoch,)
        ):
            # One-element samples: the platform already computed the
            # percentile over traffic we cannot see, and p95 of one
            # observation is that observation. See TelemetryRow.
            ttft = () if row["ttft_p95_ms"] is None else (row["ttft_p95_ms"],)
            tps = (
                ()
                if row["tokens_per_s_p50"] is None
                else (row["tokens_per_s_p50"] * 1000,)
            )
            out[row["hotkey"]] = SourceSample(
                attempts=row["requests"],
                successes=row["successes"],
                clean_rejects=row["clean_rejects"],
                ttft_ms=ttft,
                tps_milli=tps,
                served=row["served"],
            )
        return out

    # --- committing an epoch ------------------------------------------------

    def commit_epoch(
        self, epoch: int, result: EpochResult, gate_states: dict[str, GateState]
    ) -> None:
        """Persist scores and reputation together, in one transaction.

        Together because they are one fact. A crash between them would leave
        an epoch's scores recorded with the EMA never advanced, and the next
        run would apply the same epoch's observations to the same starting
        value — a miner's bad epoch counted once, its good epoch counted
        twice.

        ``gate_states`` must cover every scored miner. Defaulting a missing
        one to a clean :class:`GateState` would silently forgive a cooldown;
        skipping it would silently drop that miner's EMA. Both are worse than
        refusing to write the epoch at all.
        """
        missing = sorted({s.hotkey for s in result.scores} - set(gate_states))
        if missing:
            raise StateError(
                f"refusing to commit epoch {epoch}: no gate state for {missing}. "
                "Every scored miner needs one, or its cooldown and EMA are lost."
            )

        with self._write() as db:
            db.executemany(
                "INSERT OR REPLACE INTO scores (epoch, hotkey, uid, latency_bps, "
                "throughput_bps, reliability_bps, capacity_bps, quality_bps, "
                "smoothed_quality_bps, gate_bps, penalty_bps, score_bps, "
                "weight_u16, reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        epoch, s.hotkey, s.uid,
                        s.components.latency, s.components.throughput,
                        s.components.reliability, s.components.capacity,
                        s.components.quality, s.smoothed_quality_bps,
                        s.gate_bps, s.penalty_bps, s.score_bps, s.weight_u16,
                        json.dumps(list(s.reasons)),
                    )
                    for s in result.scores
                ],
            )
            db.executemany(
                "INSERT OR REPLACE INTO reputation (hotkey, smoothed_quality_bps, "
                "consecutive_misses, cooldown_epochs_left, updated_epoch) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        hotkey,
                        result.carry_forward.get(hotkey, 0),
                        gate_states[hotkey].consecutive_misses,
                        gate_states[hotkey].cooldown_epochs_left,
                        epoch,
                    )
                    for hotkey in sorted(gate_states)
                ],
            )

    def record_weight_set(
        self,
        *,
        epoch: int,
        block: int | None,
        submitted_ms: int,
        ok: bool,
        attempts: int,
        config_version: int,
        vector_digest: str,
        error: str | None = None,
    ) -> None:
        with self._write() as db:
            db.execute(
                "INSERT OR REPLACE INTO weight_sets (epoch, block, submitted_ms, "
                "ok, attempts, config_version, vector_digest, error) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (epoch, block, submitted_ms, int(ok), attempts, config_version,
                 vector_digest, error),
            )

    def last_weight_set(self) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM weight_sets ORDER BY epoch DESC LIMIT 1"
        ).fetchone()

    def weight_set_for_epoch(self, epoch: int) -> sqlite3.Row | None:
        """Return an epoch's attempt record, including failed attempts.

        A failed chain response is still an attempt.  The one-shot writer
        uses this lookup to refuse a duplicate submission after restart.
        """
        return self._db.execute(
            "SELECT * FROM weight_sets WHERE epoch = ?", (epoch,)
        ).fetchone()

    def scores_for_epoch(self, epoch: int) -> list[dict[str, Any]]:
        """What ``/scores`` serves. Ordered by uid so the output is stable."""
        return [
            {**dict(row), "reasons": json.loads(row["reasons"])}
            for row in self._db.execute(
                "SELECT * FROM scores WHERE epoch = ? ORDER BY uid", (epoch,)
            )
        ]

    def latest_epoch(self) -> int | None:
        row = self._db.execute("SELECT MAX(epoch) AS e FROM scores").fetchone()
        return None if row is None else row["e"]

    def latest_telemetry_epoch(self) -> int | None:
        row = self._db.execute("SELECT MAX(epoch) AS e FROM telemetry").fetchone()
        return None if row is None else row["e"]

    def telemetry_for_epoch(self, epoch: int) -> list[dict[str, Any]]:
        """Return persisted platform rows ordered by UID for operations APIs."""
        return [
            dict(row)
            for row in self._db.execute(
                "SELECT * FROM telemetry WHERE epoch = ? ORDER BY uid", (epoch,)
            )
        ]

    def probe_summary(self, epoch: int, *, source: ProbeSource) -> dict[str, int]:
        """Return bounded-batch counts for operations status and idempotent runs."""
        row = self._db.execute(
            "SELECT COUNT(*) AS attempts, "
            "SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successes "
            "FROM probes WHERE epoch = ? AND source = ?",
            (epoch, source),
        ).fetchone()
        return {
            "attempts": int(row["attempts"] or 0),
            "successes": int(row["successes"] or 0),
        }

    # --- housekeeping -------------------------------------------------------

    def prune(self, *, before_epoch: int) -> int:
        """Drop raw probes older than ``before_epoch``.

        Only ``probes`` — it is the table that grows by tens of thousands of
        rows an epoch. Scores and weight sets are one row per miner per epoch
        and are the audit trail, so they stay. Reputation is current state
        and never grows.
        """
        with self._write() as db:
            cursor = db.execute("DELETE FROM probes WHERE epoch < ?", (before_epoch,))
            return cursor.rowcount


def open_state(path: str | Path) -> ValidatorState:
    """Open (creating if needed) the state file and run migrations.

    WAL so that an operator can query the file with the ``sqlite3`` CLI
    while the validator is running — the case this is actually for is
    someone asking "what is it doing right now" during an incident, and a
    locked database at that moment is the worst possible answer.
    """
    target = Path(path).expanduser()
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)

    try:
        db = sqlite3.connect(str(target), isolation_level="DEFERRED")
    except sqlite3.Error as exc:
        raise StateError(f"cannot open validator state at {target}: {exc}") from exc

    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")

    current = db.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise StateError(
            f"{target} was written by a newer validator (schema {current}, "
            f"this build understands {SCHEMA_VERSION}). Downgrading would "
            "silently drop columns; move the file aside instead."
        )
    for version in range(current, SCHEMA_VERSION):
        with db:
            db.executescript(_MIGRATIONS[version])
            db.execute(f"PRAGMA user_version = {version + 1}")

    return ValidatorState(db)
