"""PM2-managed wait -> fetch -> score -> set_weights -> wait service."""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .config import ConfigError, Settings
from .platform import PlatformClient
from .scoring import SCORING_VERSION, normalize_weights, score_miners
from .state import StateStore
from .weights import BittensorWeightWriter

log = logging.getLogger("instant.validator")


class WeightWriter(Protocol):
    def set_weights(self, weights: dict[int, int]) -> str: ...


@dataclass(frozen=True, slots=True)
class RunOutcome:
    status: str
    report_id: str
    period_end_block: int
    weights: dict[int, int]
    chain_message: str | None = None


class ValidatorService:
    def __init__(
        self,
        settings: Settings,
        client: PlatformClient,
        state: StateStore,
        writer: WeightWriter | None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.state = state
        self.writer = writer

    def run_once(self, *, now_ms: int | None = None) -> RunOutcome:
        """Apply the latest completed report at most once."""

        timestamp = int(time.time() * 1000) if now_ms is None else now_ms
        report = self.client.fetch_latest(now_ms=timestamp)
        if self.state.is_applied(report):
            return RunOutcome(
                status="already_applied",
                report_id=report.report_id,
                period_end_block=report.period_end_block,
                weights={},
            )
        weights = normalize_weights(score_miners(report.miners))
        if not weights:
            raise RuntimeError("latest report contains no miner with a positive score")
        if not self.settings.enable_weight_writes:
            return RunOutcome(
                status="dry_run",
                report_id=report.report_id,
                period_end_block=report.period_end_block,
                weights=weights,
            )
        if self.writer is None:
            raise RuntimeError("weight writes are enabled but no writer is configured")
        message = self.writer.set_weights(weights)
        self.state.mark_applied(report, applied_at_ms=timestamp)
        return RunOutcome(
            status="applied",
            report_id=report.report_id,
            period_end_block=report.period_end_block,
            weights=weights,
            chain_message=message,
        )

    def run_forever(self, stop: threading.Event) -> None:
        """Run immediately, then poll until PM2 asks the process to stop."""

        while not stop.is_set():
            try:
                outcome = self.run_once()
                log.info(
                    "report=%s period_end=%d status=%s scoring_version=%d weights=%s chain=%s",
                    outcome.report_id,
                    outcome.period_end_block,
                    outcome.status,
                    SCORING_VERSION,
                    outcome.weights,
                    outcome.chain_message or "-",
                )
            except Exception:  # noqa: BLE001 - keep the long-running service alive
                log.exception("validator cycle failed; no weights were recorded")
            stop.wait(self.settings.poll_interval_seconds)


def main() -> int:
    """Start the only validator runtime mode."""

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"configuration error: {exc}")
        return 2
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    log.info(
        "starting network=%s netuid=%d report=%s writes=%s",
        settings.network,
        settings.netuid,
        settings.platform_report_url,
        settings.enable_weight_writes,
    )

    stop = threading.Event()
    for watched in (signal.SIGINT, signal.SIGTERM):
        signal.signal(watched, lambda *_: stop.set())

    client = PlatformClient(settings)
    try:
        writer = BittensorWeightWriter(settings) if settings.enable_weight_writes else None
        ValidatorService(
            settings=settings,
            client=client,
            state=StateStore(settings.state_path),
            writer=writer,
        ).run_forever(stop)
    except Exception:  # noqa: BLE001 - initialization failure should let PM2 restart
        log.exception("validator initialization failed")
        return 3
    finally:
        client.close()
    return 0
