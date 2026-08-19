#!/usr/bin/env python3
"""RESI-style five-minute updater for the Validator checkout."""

from __future__ import annotations

import argparse
import fcntl
import logging
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger("instant.validator.updater")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = Path("/var/lock/instant-validator.lock")
CHECK_SECONDS = 300
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def update_once() -> bool:
    """Fast-forward to GitHub main under the same lock used by scoring."""

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("update_skipped reason=validator_busy")
            return False
        current = _run(["git", "rev-parse", "HEAD"])
        _run(["git", "fetch", "origin", "main"])
        available = _run(["git", "rev-parse", "origin/main"])
        if current == available:
            LOG.info("update_current revision=%s", current[:8])
            return False
        _run(["git", "merge", "--ff-only", "origin/main"])
        _run([str(PYTHON), "-m", "pip", "install", "-e", "."])
        LOG.info("update_applied previous=%s current=%s", current[:8], available[:8])
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="update-validator")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if args.once:
            update_once()
            return 0
        while True:
            if update_once():
                return 0  # PM2 restarts the updater from the new checkout.
            time.sleep(CHECK_SECONDS)
    except subprocess.CalledProcessError:
        LOG.exception("update_failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
