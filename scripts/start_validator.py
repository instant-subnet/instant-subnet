#!/usr/bin/env python3
"""Install the cron scorer and PM2 source updater."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPDATER = PROJECT_ROOT / "scripts" / "update_validator.py"
RUNNER = PROJECT_ROOT / "scripts" / "run_validator.sh"
PM2_NAME = "instant-validator-updater"
CRON_MARKER = "# instant-validator-run-once"
CRON_LINE = (
    f"*/5 * * * * flock -n /var/lock/instant-validator.lock {RUNNER} "
    f">> {PROJECT_ROOT / 'logs' / 'validator.log'} 2>&1"
)


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, cwd=PROJECT_ROOT, **kwargs)


def install_cron() -> None:
    current = subprocess.run(
        ["crontab", "-l"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    kept = [line for line in current if line != CRON_MARKER and line != CRON_LINE]
    content = "\n".join([*kept, CRON_MARKER, CRON_LINE]) + "\n"
    _run(["crontab", "-"], input=content, text=True)


def main() -> int:
    try:
        _run(["pm2", "--version"], capture_output=True)
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not python.exists():
            _run([sys.executable, "-m", "venv", ".venv"])
        _run([str(python), "-m", "pip", "install", "-e", "."])
        install_cron()
        subprocess.run(["pm2", "delete", PM2_NAME], cwd=PROJECT_ROOT, check=False)
        _run(
            [
                "pm2",
                "start",
                str(UPDATER),
                "--name",
                PM2_NAME,
                "--interpreter",
                sys.executable,
                "--cwd",
                str(PROJECT_ROOT),
            ]
        )
        _run(["pm2", "save"])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Validator setup failed: {exc}", file=sys.stderr)
        return 1
    print("Validator cron and updater installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
