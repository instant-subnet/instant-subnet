"""One-shot telemetry-to-score orchestration.

There is deliberately no timer in this module.  PM2 runs the read-only
health process continuously; scoring and its bounded direct-probe batch are
an explicit ``--score-once`` action until shadow probes and a production
epoch scheduler exist.  The one-shot boundary also makes restart/idempotency
behaviour straightforward to test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from instant.validator.score import (
    AttestationMode,
    ScoringConfig,
    advance_gate_state,
    score_epoch,
)

from .probe import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    MAX_CONCURRENCY,
    MAX_PROBE_COUNT,
    ProbeTarget,
    probe_batch,
)
from .state import ValidatorState
from .telemetry import PlatformTelemetryClient, TelemetryBatch, TelemetryError


def _now_ms() -> int:
    return int(time.time() * 1000)


class ScoreRunError(RuntimeError):
    """A scoring pass could not safely produce a persisted epoch."""


@dataclass(frozen=True, slots=True)
class ScoreRun:
    epoch: int
    scored_at_ms: int
    roster_size: int
    telemetry_miners: int
    direct_probe_attempts: int
    direct_probe_successes: int
    total_weight: int
    empty: bool
    reused: bool
    config_version: int
    slo_version: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "scored_at_ms": self.scored_at_ms,
            "roster_size": self.roster_size,
            "telemetry_miners": self.telemetry_miners,
            "direct_probe_attempts": self.direct_probe_attempts,
            "direct_probe_successes": self.direct_probe_successes,
            "total_weight": self.total_weight,
            "empty": self.empty,
            "reused": self.reused,
            "config_version": self.config_version,
            "slo_version": self.slo_version,
        }


class ScoringCoordinator:
    """Fetch telemetry and advance exactly one deterministic score epoch."""

    def __init__(
        self,
        *,
        state: ValidatorState,
        telemetry: PlatformTelemetryClient,
        config: ScoringConfig,
        attestation_mode: AttestationMode,
        probe_http: httpx.AsyncClient | None = None,
        probe_signer: Any | None = None,
        probe_model: str | None = None,
    ) -> None:
        self.state = state
        self.telemetry = telemetry
        self.config = config
        self.attestation_mode = attestation_mode
        self.probe_http = probe_http
        self.probe_signer = probe_signer
        self.probe_model = probe_model

    async def run_once(self, snapshot: Any) -> tuple[ScoreRun, TelemetryBatch]:
        if not snapshot.chain_connected:
            raise ScoreRunError("cannot score without a current chain snapshot")

        # Serving axons are the scoring roster.  A validator may also announce
        # an axon; it must never assign weight to itself.
        roster = sorted(
            (miner.uid, miner.hotkey)
            for miner in snapshot.miners
            if miner.hotkey != snapshot.own_hotkey
        )
        if not roster:
            raise ScoreRunError("current metagraph has no serving miner roster")

        try:
            batch = await self.telemetry.fetch(roster)
        except TelemetryError as exc:
            raise ScoreRunError(str(exc)) from exc

        latest = self.state.latest_epoch()
        if latest is not None and batch.epoch < latest:
            raise ScoreRunError(
                f"telemetry epoch {batch.epoch} is older than scored epoch {latest}"
            )

        existing = self.state.scores_for_epoch(batch.epoch)
        if existing:
            # Do not replace the telemetry beneath an already-committed score.
            # Recomputing against the current reputation would apply EMA and
            # cooldown twice; returning the stored epoch makes retries safe.
            probes = self.state.probe_summary(batch.epoch, source="direct")
            run = ScoreRun(
                epoch=batch.epoch,
                scored_at_ms=_now_ms(),
                roster_size=len(existing),
                telemetry_miners=len(batch.rows),
                direct_probe_attempts=probes["attempts"],
                direct_probe_successes=probes["successes"],
                total_weight=sum(row["weight_u16"] for row in existing),
                empty=all(row["weight_u16"] == 0 for row in existing),
                reused=True,
                config_version=self.config.version,
                slo_version=self.config.slo.version,
            )
            return run, batch

        self.state.record_telemetry(batch.rows)
        await self._run_direct_probes(batch.epoch, snapshot)
        probes = self.state.probe_summary(batch.epoch, source="direct")
        observations = self.state.observations_for_epoch(batch.epoch, roster)
        result = score_epoch(
            observations,
            self.config,
            attestation_mode=self.attestation_mode,
            carry_forward=self.state.carry_forward(),
        )
        scores_by_hotkey = {score.hotkey: score for score in result.scores}
        gate_states = {
            observation.hotkey: advance_gate_state(
                self.state.gate_state(observation.hotkey),
                gated_out=(
                    scores_by_hotkey[observation.hotkey].gate_bps
                    == self.config.gate_fail_bps
                ),
                config=self.config,
            )
            for observation in observations
        }
        self.state.commit_epoch(batch.epoch, result, gate_states)

        run = ScoreRun(
            epoch=batch.epoch,
            scored_at_ms=_now_ms(),
            roster_size=len(roster),
            telemetry_miners=len(batch.rows),
            direct_probe_attempts=probes["attempts"],
            direct_probe_successes=probes["successes"],
            total_weight=sum(result.weights.values()),
            empty=result.is_empty,
            reused=False,
            config_version=result.config_version,
            slo_version=result.slo_version,
        )
        return run, batch

    async def aclose(self) -> None:
        if self.probe_http is not None and self.probe_http is not self.telemetry.http:
            await self.probe_http.aclose()
        await self.telemetry.aclose()

    async def _run_direct_probes(self, epoch: int, snapshot: Any) -> None:
        """Run exactly the configured minimum per miner, with hard global bounds."""
        configured = (
            self.probe_http is not None
            and self.probe_signer is not None
            and self.probe_model is not None
        )
        if not configured:
            return
        if self.config.min_probes > MAX_PROBE_COUNT:
            raise ScoreRunError(
                f"scoring requires {self.config.min_probes} probes per miner, but "
                f"the bounded prober permits at most {MAX_PROBE_COUNT}"
            )

        results = []
        for miner in sorted(snapshot.miners, key=lambda item: item.uid):
            if miner.hotkey == snapshot.own_hotkey:
                continue
            target = ProbeTarget(
                uid=miner.uid,
                hotkey=miner.hotkey,
                url=miner.url,
            )
            batch = await probe_batch(
                self.probe_http,  # type: ignore[arg-type]
                signer=self.probe_signer,
                targets=(target,),
                epoch=epoch,
                model=self.probe_model,  # type: ignore[arg-type]
                count=self.config.min_probes,
                concurrency=min(
                    self.config.probe.max_concurrent_probes, MAX_CONCURRENCY
                ),
                max_completion_tokens=min(
                    self.config.probe.max_tokens, DEFAULT_MAX_COMPLETION_TOKENS
                ),
                timeout_s=float(self.config.probe.timeout_s),
            )
            results.extend(batch)
        self.state.record_probes(results)
