"""Entrypoint for the local/test-only fixture worker."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

import uvicorn

from ..common.config import load_settings
from ..common.guards import describe, enforce
from .app import create_app

log = logging.getLogger("instant.mock_vllm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="instant-mock-vllm")
    parser.add_argument(
        "--check", action="store_true", help="validate configuration and exit"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    try:
        settings = load_settings()
        warnings = enforce(settings, role="mock")
    except Exception as exc:  # noqa: BLE001 - a boot refusal is an exit code
        log.error("configuration error: %s", exc)
        return 2

    logging.getLogger().setLevel(settings.log_level)
    log.info(describe(settings))
    for warning in warnings:
        log.warning(warning)
    if args.check:
        log.info("configuration is valid")
        return 0

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(
                model_id=settings.tier.model_id,
                first_token_delay_ms=settings.mock_first_token_delay_ms,
                token_delay_ms=settings.mock_token_delay_ms,
            ),
            host=settings.mock_vllm_host,
            port=settings.mock_vllm_port,
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
