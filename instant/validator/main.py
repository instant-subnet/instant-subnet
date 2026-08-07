"""Validator entrypoint, designed to run as one PM2 process."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys

import uvicorn

from ..common.config import load_settings
from ..common.guards import describe, enforce
from .app import create_app
from .runtime import ValidatorRuntime
from .state import open_state

log = logging.getLogger("instant.validator")


async def _sync_and_close(runtime: ValidatorRuntime):
    snapshot = await runtime.sync_once()
    await runtime.aclose()
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="instant-validator")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and exit without loading a wallet",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="sync the localnet and platform once, print JSON, and exit",
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
    except Exception as exc:  # noqa: BLE001
        log.error("validator initialization failed: %s", exc)
        return 3

    runtime = ValidatorRuntime(
        subtensor=subtensor,
        wallet=wallet,
        netuid=settings.netuid,
        state=state,
        platform_url=settings.platform_url,
        refresh_s=settings.metagraph_refresh_s,
        request_timeout_s=settings.request_timeout_s,
    )
    if args.once:
        snapshot = asyncio.run(_sync_and_close(runtime))
        print(json.dumps(snapshot.to_payload(), sort_keys=True))
        return 0 if snapshot.chain_connected else 4

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
