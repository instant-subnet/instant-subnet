"""Platform gateway entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys

import httpx
import uvicorn

from ..common.config import load_settings
from ..common.guards import describe, enforce
from .app import PlatformContext, create_app
from .state import open_state

log = logging.getLogger("instant.platform")


async def _check_miner(ctx: PlatformContext) -> tuple[bool, dict]:
    try:
        health_response = await ctx.http.get(f"{ctx.miner_url}/health")
        health_response.raise_for_status()
        health = health_response.json()
        if not isinstance(health, dict):
            return False, {"error": "miner health response is not a JSON object"}

        manifest_response = await ctx.http.get(f"{ctx.miner_url}/manifest")
        manifest_response.raise_for_status()
        manifest = manifest_response.json()
        if not isinstance(manifest, dict):
            return False, {"error": "miner manifest is not a JSON object"}
        actual_hotkey = manifest.get("hotkey")
        if actual_hotkey != ctx.miner_ss58:
            return False, {
                "error": "miner identity mismatch",
                "expected_hotkey": ctx.miner_ss58,
                "actual_hotkey": actual_hotkey,
            }
        if not health.get("ready"):
            return False, {
                "error": "miner is reachable but not ready",
                "health": health,
                "manifest": manifest,
            }
        return True, {"health": health, "manifest": manifest}
    except (httpx.HTTPError, ValueError) as exc:
        return False, {"error": str(exc)}
    finally:
        await ctx.http.aclose()
        ctx.state.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="instant-platform")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and exit without loading a wallet",
    )
    parser.add_argument(
        "--check-miner",
        action="store_true",
        help="verify miner readiness and identity, print JSON, and exit",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    try:
        settings = load_settings()
        warnings = enforce(settings, role="platform")
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
        ctx = PlatformContext(
            signer=wallet.hotkey,
            miner_url=settings.platform_miner_url,
            miner_ss58=settings.platform_miner_ss58,
            http=httpx.AsyncClient(timeout=settings.request_timeout_s),
            state=open_state(settings.platform_state_db),
            api_key_sha256=settings.platform_api_key_sha256,
            # Read straight from the environment, never through Settings:
            # Settings is logged by describe(), and these are credentials.
            api_key_pepper=os.environ.get("INSTANT_PLATFORM_API_KEY_PEPPER", ""),
            admin_token=os.environ.get("INSTANT_PLATFORM_ADMIN_TOKEN", ""),
            validator_hotkeys=frozenset({settings.platform_validator_ss58}),
            miner_uid=settings.platform_miner_uid,
            stats_window_s=settings.platform_stats_window_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("platform initialization failed: %s", exc)
        return 3

    if args.check_miner:
        ok, payload = asyncio.run(_check_miner(ctx))
        print(json.dumps(payload, sort_keys=True))
        return 0 if ok else 4

    log.info(
        "routing %s:%d -> %s (%s)",
        settings.platform_host,
        settings.platform_port,
        settings.platform_miner_url,
        settings.platform_miner_ss58,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(ctx),
            host=settings.platform_host,
            port=settings.platform_port,
            log_level=settings.log_level.lower(),
            access_log=False,
            timeout_keep_alive=75,
        )
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(server, "should_exit", True))
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
