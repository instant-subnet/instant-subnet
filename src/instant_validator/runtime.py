"""One-shot Validator report fetch, validation, scoring, and logging."""

from __future__ import annotations

import argparse
import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .burn import BittensorBurnWriter, BurnError
from .report import MAX_REPORT_BYTES, ReportError, parse_report
from .scoring import log_score_records, score_report
from .state import StateError, StateStore

LOG = logging.getLogger("instant.validator")
SS58_FORMAT = 42
DEFAULT_REPORT_URL = "https://api.instantsubnet.com/validator/v1/reports/latest"


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    finalized_block: int
    tempo: int
    last_step: int
    hotkeys: tuple[str, ...]
    last_updates: tuple[int, ...]


class ChainError(RuntimeError):
    """Finalized chain state is unavailable or inconsistent."""


def _account_ss58(value: Any) -> str:
    """Decode a storage account value to its SS58 address."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    from scalecodec.utils.ss58 import ss58_encode

    return ss58_encode(bytes(value), SS58_FORMAT)


class BittensorChain:
    def __init__(self, network: str, endpoint: str, *, sdk: Any = None) -> None:
        if sdk is None:
            import bittensor as sdk

        self._subtensor = sdk.subtensor(network=endpoint or network)

    def snapshot(self, netuid: int) -> ChainSnapshot:
        try:
            substrate = self._subtensor.substrate
            head = substrate.get_chain_finalised_head()
            block = substrate.get_block_number(head)

            def query(storage: str, params: list[Any]) -> Any:
                result = substrate.query(
                    "SubtensorModule", storage, params, block_hash=head
                )
                return getattr(result, "value", result)

            size = query("SubnetworkN", [netuid])
            tempo = query("Tempo", [netuid])
            last_step = query("LastMechansimStepBlock", [netuid])
            blocks_since_last_step = query("BlocksSinceLastStep", [netuid])
            last_updates = tuple(query("LastUpdate", [netuid]))
            roster: dict[int, Any] = {}
            for uid, hotkey in substrate.query_map(
                "SubtensorModule", "Keys", [netuid], block_hash=head, page_size=512
            ):
                roster[getattr(uid, "value", uid)] = getattr(hotkey, "value", hotkey)
            hotkeys = tuple(_account_ss58(roster[uid]) for uid in range(size))
        except Exception as exc:
            raise ChainError(f"finalized chain lookup failed: {exc}") from exc
        if type(size) is not int or size < 1:
            raise ChainError("finalized subnet is empty or unknown")
        values = (block, tempo, last_step, blocks_since_last_step)
        if any(type(value) is not int or value < 0 for value in values):
            raise ChainError("finalized epoch state is invalid")
        if tempo < 1 or last_step + blocks_since_last_step != block:
            raise ChainError("finalized epoch state is inconsistent")
        if (
            any(not value for value in hotkeys)
            or len(hotkeys) != len(last_updates)
            or any(type(value) is not int or value < 0 for value in last_updates)
        ):
            raise ChainError("finalized hotkey roster is invalid")
        return ChainSnapshot(block, tempo, last_step, hotkeys, last_updates)


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
    state: StateStore,
    burner: BittensorBurnWriter | None = None,
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
    saved = state.load()
    report_end = report["epoch_end_block"]
    if (
        saved.report_epoch_end_block is not None
        and saved.report_epoch_end_block > report_end
    ):
        raise StateError("Platform returned a report older than local state")
    already_processed = saved.report_epoch_end_block == report_end
    if already_processed and (
        saved.report_id != report["report_id"] or saved.report_digest != report["digest"]
    ):
        raise StateError("A processed report changed")

    records: tuple[dict[str, Any], ...] = ()
    if already_processed:
        LOG.info("report_already_processed report_id=%s", report["report_id"])
    else:
        records = score_report(report)
        saved = replace(
            saved,
            report_id=report["report_id"],
            report_digest=report["digest"],
            report_epoch_end_block=report_end,
        )
        state.save(saved)
        log_score_records(records, LOG)
        LOG.info(
            "report_processed report_id=%s finalized_block=%d miners=%d",
            report["report_id"],
            snapshot.finalized_block,
            len(records),
        )

    if burner is not None and saved.burn_epoch_end_block != report_end:
        try:
            validator_uid = snapshot.hotkeys.index(burner.hotkey)
        except ValueError as exc:
            raise BurnError("Validator hotkey is not registered") from exc
        if snapshot.last_updates[validator_uid] >= report_end:
            LOG.info("burn_already_on_chain epoch_end_block=%d", report_end)
        else:
            message = burner.submit(
                netuid=report["netuid"], finalized_block=snapshot.finalized_block
            )
            LOG.info("burn_submitted epoch_end_block=%d result=%s", report_end, message)
        state.save(replace(saved, burn_epoch_end_block=report_end))
    return records


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instant-validator")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-once")
    run.add_argument("--network", default=os.environ.get("INSTANT_NETWORK", "finney"))
    run.add_argument(
        "--netuid", type=int, default=int(os.environ.get("INSTANT_NETUID", "46"))
    )
    run.add_argument(
        "--chain-endpoint", default=os.environ.get("INSTANT_CHAIN_ENDPOINT", "")
    )
    run.add_argument(
        "--report-url",
        default=os.environ.get("INSTANT_PLATFORM_REPORT_URL", DEFAULT_REPORT_URL),
    )
    run.add_argument(
        "--platform-signer", default=os.environ.get("INSTANT_PLATFORM_SIGNER", "")
    )
    run.add_argument("--timeout", type=int, default=15)
    run.add_argument(
        "--state-path",
        default=os.environ.get(
            "INSTANT_VALIDATOR_STATE_PATH", "/var/lib/instant-validator/state.json"
        ),
    )
    run.add_argument("--burn", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument(
        "--wallet-name", default=os.environ.get("INSTANT_WALLET_NAME", "validator")
    )
    run.add_argument(
        "--wallet-hotkey", default=os.environ.get("INSTANT_WALLET_HOTKEY", "default")
    )
    run.add_argument(
        "--wallet-path",
        default=os.environ.get("INSTANT_WALLET_PATH", "~/.bittensor/wallets"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # The lazy bittensor import replaces the root handlers, which silences
    # LOG.exception below. Give this logger its own handler so failures stay
    # visible in the scheduled-run log.
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        LOG.addHandler(handler)
    LOG.propagate = False
    try:
        burn_enabled = (
            _environment_flag("INSTANT_BURN_ENABLED") if args.burn is None else args.burn
        )
    except ValueError as exc:
        LOG.error("%s", exc)
        return 2
    if not args.platform_signer:
        LOG.error("--platform-signer or INSTANT_PLATFORM_SIGNER is required")
        return 2
    if args.netuid < 1 or args.timeout < 1:
        LOG.error("netuid and timeout must be positive")
        return 2
    try:
        burner = (
            BittensorBurnWriter(
                network=args.network,
                endpoint=args.chain_endpoint,
                wallet_name=args.wallet_name,
                wallet_hotkey=args.wallet_hotkey,
                wallet_path=str(Path(args.wallet_path).expanduser()),
            )
            if burn_enabled
            else None
        )
        run_once(
            network=args.network,
            netuid=args.netuid,
            report_url=args.report_url,
            platform_signer=args.platform_signer,
            chain=BittensorChain(args.network, args.chain_endpoint),
            state=StateStore(Path(args.state_path)),
            burner=burner,
            timeout=args.timeout,
        )
    except (BurnError, ChainError, ReportError, StateError, RuntimeError):
        LOG.exception("Validator run failed")
        return 1
    return 0
