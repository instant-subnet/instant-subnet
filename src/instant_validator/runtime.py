"""One-shot Validator report fetch, validation, scoring, and logging."""

from __future__ import annotations

import argparse
import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .report import MAX_REPORT_BYTES, ReportError, parse_report
from .scoring import log_score_records, score_report

LOG = logging.getLogger("instant.validator")
DEFAULT_REPORT_URL = "https://api.instantsubnet.com/validator/v1/reports/latest"


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    finalized_block: int
    tempo: int
    last_step: int
    hotkeys: tuple[str, ...]


class ChainError(RuntimeError):
    """Finalized chain state is unavailable or inconsistent."""


class BittensorChain:
    def __init__(self, network: str, endpoint: str, *, sdk: Any = None) -> None:
        if sdk is None:
            import bittensor as sdk

        self._subtensor = sdk.subtensor(network=endpoint or network)

    def snapshot(self, netuid: int) -> ChainSnapshot:
        try:
            head = self._subtensor.substrate.get_chain_finalised_head()
            block = self._subtensor.substrate.get_block_number(head)
            info = self._subtensor.get_metagraph_info(netuid, block=block)
        except Exception as exc:
            raise ChainError(f"finalized chain lookup failed: {exc}") from exc
        if info is None:
            raise ChainError("finalized metagraph is unavailable")
        values = (block, info.tempo, info.last_step, info.blocks_since_last_step)
        if any(type(value) is not int or value < 0 for value in values):
            raise ChainError("finalized epoch state is invalid")
        if info.tempo < 1 or info.last_step + info.blocks_since_last_step != block:
            raise ChainError("finalized epoch state is inconsistent")
        hotkeys = tuple(str(value) for value in info.hotkeys)
        if any(not value for value in hotkeys):
            raise ChainError("finalized hotkey roster is invalid")
        return ChainSnapshot(block, info.tempo, info.last_step, hotkeys)


def fetch_report(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "instant-validator/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_REPORT_BYTES + 1)
    except Exception as exc:
        raise RuntimeError(f"Platform report fetch failed: {exc}") from exc
    if not raw or len(raw) > MAX_REPORT_BYTES:
        raise RuntimeError("Platform report size is invalid")
    return raw


def validate_chain(report: dict[str, Any], snapshot: ChainSnapshot) -> None:
    if report["tempo"] != snapshot.tempo:
        raise ChainError("report tempo does not match finalized chain state")
    if report["epoch_end_block"] != snapshot.last_step:
        raise ChainError("report is not the latest finalized chain epoch")
    if report["finalized_block"] > snapshot.finalized_block:
        raise ChainError("report finalized block is ahead of chain state")
    if report["epoch_start_block"] != snapshot.last_step - snapshot.tempo + 1:
        raise ChainError("report epoch start does not match finalized chain state")
    for row in report["miners"]:
        uid = int(row["uid"])
        if uid >= len(snapshot.hotkeys) or snapshot.hotkeys[uid] != row["hotkey"]:
            raise ChainError(f"report UID/hotkey mapping is invalid for UID {uid}")


def run_once(
    *,
    network: str,
    netuid: int,
    report_url: str,
    platform_signer: str,
    chain: BittensorChain,
    timeout: int = 15,
    fetch: Callable[[str, int], bytes] = fetch_report,
) -> tuple[dict[str, Any], ...]:
    snapshot = chain.snapshot(netuid)
    report = parse_report(
        fetch(report_url, timeout),
        expected_network=network,
        expected_netuid=netuid,
        expected_signer=platform_signer,
    )
    validate_chain(report, snapshot)
    records = score_report(report)
    log_score_records(records, LOG)
    LOG.info(
        "report_processed report_id=%s finalized_block=%d miners=%d",
        report["report_id"],
        snapshot.finalized_block,
        len(records),
    )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instant-validator")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-once")
    run.add_argument("--network", default="finney")
    run.add_argument("--netuid", type=int, default=46)
    run.add_argument("--chain-endpoint", default="")
    run.add_argument("--report-url", default=DEFAULT_REPORT_URL)
    run.add_argument(
        "--platform-signer", default=os.environ.get("INSTANT_PLATFORM_SIGNER", "")
    )
    run.add_argument("--timeout", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.platform_signer:
        LOG.error("--platform-signer or INSTANT_PLATFORM_SIGNER is required")
        return 2
    if args.netuid < 1 or args.timeout < 1:
        LOG.error("netuid and timeout must be positive")
        return 2
    try:
        run_once(
            network=args.network,
            netuid=args.netuid,
            report_url=args.report_url,
            platform_signer=args.platform_signer,
            chain=BittensorChain(args.network, args.chain_endpoint),
            timeout=args.timeout,
        )
    except (ChainError, ReportError, RuntimeError):
        LOG.exception("Validator run failed")
        return 1
    return 0
