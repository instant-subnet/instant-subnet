"""Check the public repository boundary, content, and line ceiling."""

from __future__ import annotations

import re
import subprocess
from ipaddress import ip_address
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE_LIMIT = 4_000
TEXT_MIME_TYPES = {
    "application/javascript",
    "application/json",
    "application/x-empty",
    "image/svg+xml",
}
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
    "internal planning language": re.compile(
        r"\b(?:closeout|cutover|go-live|goal(?:s)?|handoff|milestone(?:s)?|mvp|"
        r"phase(?:s)?|progress|roadmap)\b",
        re.I,
    ),
    "local network default": re.compile(r"INSTANT_NETWORK\s*=\s*local\b"),
    "netuid 5 default": re.compile(r"INSTANT_NETUID\s*=\s*5\b"),
    "literal websocket IP": re.compile(r"wss?://(?:\d{1,3}\.){3}\d{1,3}"),
    "private hostname": re.compile(
        r"(?:https?|wss?)://[^\s/]+\.(?:internal|lan|local)(?::\d+)?(?:[/\s]|$)", re.I
    ),
    "retired miner entrypoint": re.compile(r"instant-miner\s*="),
    "retired platform entrypoint": re.compile(r"instant-platform\s*="),
}
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [Path(item) for item in output.split("\0") if item and (ROOT / item).is_file()]


def is_text(path: Path) -> bool:
    mime = subprocess.check_output(["file", "-b", "--mime-type", path], text=True).strip()
    return mime.startswith("text/") or mime in TEXT_MIME_TYPES


def main() -> int:
    failures: list[str] = []
    line_count = 0
    for relative in tracked_files():
        if relative.parts[0] not in ALLOWED_ROOTS:
            failures.append(f"unexpected top-level path: {relative}")
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            failures.append(f"forbidden public path: {relative}")
        path = ROOT / relative
        if not is_text(path):
            continue
        data = path.read_bytes()
        line_count += data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))
        text = data.decode("utf-8")
        if relative != Path("scripts/check_repository.py"):
            for label, pattern in FORBIDDEN_CONTENT.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")
            for match in IPV4.finditer(text):
                try:
                    address = ip_address(match.group())
                except ValueError:
                    continue
                if not address.is_loopback:
                    failures.append(f"non-loopback IP address: {relative}")
                    break
    if line_count > LINE_LIMIT:
        failures.append(f"tracked line count {line_count} exceeds {LINE_LIMIT}")
    if failures:
        print("repository guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"repository guard passed: {line_count} tracked lines (<= {LINE_LIMIT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
