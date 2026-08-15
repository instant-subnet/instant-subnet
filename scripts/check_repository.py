"""CI guard for the public subnet boundary and line ceiling."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE_LIMIT = 12_500
ALLOWED_ROOTS = {
    ".env.example",
    ".github",
    ".gitignore",
    "README.md",
    "ecosystem.config.cjs",
    "pyproject.toml",
    "scripts",
    "src",
    "tests",
}
FORBIDDEN_PATH_PARTS = {
    ".env.miner.example",
    ".env.platform.example",
    ".env.worker.example",
    "localnet",
    "mock_vllm",
    "toploc_verifier",
}
FORBIDDEN_CONTENT = {
    "local network default": re.compile(r"INSTANT_NETWORK\s*=\s*local\b"),
    "netuid 5 default": re.compile(r"INSTANT_NETUID\s*=\s*5\b"),
    "literal websocket IP": re.compile(r"wss?://(?:\d{1,3}\.){3}\d{1,3}"),
    "retired miner entrypoint": re.compile(r"instant-miner\s*="),
    "retired platform entrypoint": re.compile(r"instant-platform\s*="),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [Path(item) for item in output.split("\0") if item and (ROOT / item).is_file()]


def main() -> int:
    failures: list[str] = []
    line_count = 0
    for relative in tracked_files():
        if relative.parts[0] not in ALLOWED_ROOTS:
            failures.append(f"unexpected top-level path: {relative}")
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            failures.append(f"forbidden public path: {relative}")
        data = (ROOT / relative).read_bytes()
        if b"\0" in data:
            failures.append(f"tracked binary file: {relative}")
            continue
        line_count += data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))
        text = data.decode("utf-8")
        if relative != Path("scripts/check_repository.py"):
            for label, pattern in FORBIDDEN_CONTENT.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")
    if line_count >= LINE_LIMIT:
        failures.append(f"tracked line count {line_count} is not below {LINE_LIMIT}")
    if failures:
        print("repository guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"repository guard passed: {line_count} tracked lines (< {LINE_LIMIT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
