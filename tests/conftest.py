from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bittensor as bt
import pytest

from instant_validator.report import canonical_json, parse_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "report-v1.json"
FIXTURE_NOW_MS = 1_786_708_810_000


@pytest.fixture
def report_dict() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def report_raw() -> bytes:
    return FIXTURE_PATH.read_bytes()


@pytest.fixture
def parsed_report(report_raw):
    return parse_report(
        report_raw,
        expected_network="finney",
        expected_netuid=46,
        expected_signer="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        max_age_seconds=86_400,
        future_skew_seconds=60,
        now_ms=FIXTURE_NOW_MS,
    )


def sign_fixture(value: dict) -> bytes:
    payload = {
        key: item for key, item in value.items() if key not in {"digest", "signature"}
    }
    message = canonical_json(payload)
    signer = bt.Keypair.create_from_uri("//Alice")
    value["signer"] = signer.ss58_address
    payload["signer"] = signer.ss58_address
    message = canonical_json(payload)
    value["digest"] = "sha256:" + hashlib.sha256(message).hexdigest()
    value["signature"] = signer.sign(message).hex()
    return json.dumps(value, separators=(",", ":")).encode()
