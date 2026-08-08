"""Validator entrypoint, designed to run as one PM2 process."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys

import httpx
import uvicorn

from ..common.config import load_settings
from ..common.guards import describe, enforce
from .app import create_app
from .coordinator import ScoringCoordinator
from .runtime import ValidatorRuntime
from .score import load_scoring_config
from .state import open_state
from .telemetry import PlatformTelemetryClient
from .weights import WeightSafetyError, WeightWriter

log = logging.getLogger("instant.validator")


async def _sync_and_close(runtime: ValidatorRuntime):
    snapshot = await runtime.sync_once()
    await runtime.aclose()
    return snapshot


async def _score_and_close(runtime: ValidatorRuntime) -> dict:
    try:
        return await runtime.score_once()
    finally:
        await runtime.aclose()


async def _set_weights_and_close(
    runtime: ValidatorRuntime, writer: WeightWriter
) -> dict:
    try:
        snapshot = await runtime.sync_once()
        if not snapshot.chain_connected:
            raise WeightSafetyError(
                snapshot.error or "cannot submit without a current chain snapshot"
            )
        # This CLI owns the event loop and has no concurrent work. Keep the
        # call on the creating thread because ValidatorState's sqlite handle
        # is intentionally thread-bound; the writer itself is one finite SDK
        # call with no retry loop.
        result = writer.submit_latest_once()
        return result.to_payload()
    finally:
        await runtime.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="instant-validator")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and exit without loading a wallet",
    )
    actions.add_argument(
        "--once",
        action="store_true",
        help="sync the localnet and platform once, print JSON, and exit",
    )
    actions.add_argument(
        "--score-once",
        action="store_true",
        help=(
            "fetch signed platform telemetry, persist and score one logical "
            "epoch, print JSON, and exit without setting weights"
        ),
    )
    actions.add_argument(
        "--set-weights-once",
        action="store_true",
        help=(
            "submit the latest persisted score vector once; localnet-only and "
            "requires INSTANT_ENABLE_WEIGHT_WRITES=true"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    try:
        settings = load_settings()
        warnings = enforce(settings, role="validator")
    except Exception as exc:  # noqa: BLE001
        log.error("configuration error: %s", exc)
        return 2

    logging.getLogger().setLevel(settings.log_level)
    log.info(describe(settings))
    for warning in warnings:
        log.warning(warning)
    if args.check:
        log.info("configuration is valid")
        return 0

    import bittensor as bt

    try:
        wallet = bt.wallet(
            name=settings.wallet_name,
            hotkey=settings.wallet_hotkey,
            path=settings.wallet_path,
        )
        subtensor = bt.subtensor(network=settings.chain_endpoint)
        state = open_state(settings.state_db)
        scoring_config = load_scoring_config()
    except Exception as exc:  # noqa: BLE001
        log.error("validator initialization failed: %s", exc)
        return 3

    telemetry = PlatformTelemetryClient(
        base_url=settings.platform_url,
        platform_hotkey=settings.platform_ss58,
        validator_signer=wallet.hotkey,
        max_age_s=settings.telemetry_max_age_s,
        max_window_s=settings.telemetry_max_window_s,
        timeout_s=settings.request_timeout_s,
        http=None,
    )
    coordinator = ScoringCoordinator(
        state=state,
        telemetry=telemetry,
        config=scoring_config,
        attestation_mode=settings.attestation_mode,  # type: ignore[arg-type]
        probe_http=httpx.AsyncClient(timeout=settings.request_timeout_s),
        probe_signer=wallet.hotkey,
        probe_model=settings.tier.model_id,
    )
    runtime = ValidatorRuntime(
        subtensor=subtensor,
        wallet=wallet,
        netuid=settings.netuid,
        state=state,
        platform_url=settings.platform_url,
        refresh_s=settings.metagraph_refresh_s,
        request_timeout_s=settings.request_timeout_s,
        coordinator=coordinator,
    )
    if args.once:
        snapshot = asyncio.run(_sync_and_close(runtime))
        print(json.dumps(snapshot.to_payload(), sort_keys=True))
        return 0 if snapshot.chain_connected else 4
    if args.score_once:
        try:
            result = asyncio.run(_score_and_close(runtime))
        except Exception as exc:  # noqa: BLE001 - CLI reports the boundary
            log.error("score-once failed: %s", exc)
            return 5
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.set_weights_once:
        writer = WeightWriter(
            subtensor=subtensor,
            wallet=wallet,
            state=state,
            network=settings.network,
            netuid=settings.netuid,
            enabled=settings.enable_weight_writes,
            expected_spec_version=settings.expected_spec_version,
            mechanism_id=settings.weight_mechanism_id,
            version_key=settings.weight_version_key,
            period_blocks=settings.weight_period_blocks,
            config_version=scoring_config.version,
        )
        try:
            result = asyncio.run(_set_weights_and_close(runtime, writer))
        except WeightSafetyError as exc:
            log.error("weight submission refused: %s", exc)
            return 6
        except Exception as exc:  # noqa: BLE001 - CLI reports the boundary
            log.error("weight submission failed: %s", exc)
            return 7
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 7

    log.info(
        "starting validator operations API on %s:%d; first chain sync runs at boot",
        settings.validator_host,
        settings.validator_port,
    )

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(runtime),
            host=settings.validator_host,
            port=settings.validator_port,
            log_level=settings.log_level.lower(),
            access_log=False,
        )
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(server, "should_exit", True))
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
