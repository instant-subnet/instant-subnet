"""Strict parsing and authentication for Validator report v1."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

from bittensor_wallet import Keypair

MAX_REPORT_BYTES = 1_000_000
REPORT_KEYS = {
    "created_at_ms",
    "digest",
    "epoch_end_block",
    "epoch_start_block",
    "finalized_block",
    "miners",
    "netuid",
    "network",
    "report_id",
    "schema_version",
    "signature",
    "signer",
    "tempo",
}
MINER_KEYS = {
    "completion_tokens",
    "failed_requests",
    "generation_tps_p50",
    "hotkey",
    "proof_failed_requests",
    "proof_not_available_requests",
    "proof_timed_out_requests",
    "proof_verified_requests",
    "routed_requests",
    "successful_requests",
    "uid",
    "verified_completion_tokens",
}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SIGNATURE = re.compile(r"[0-9a-f]{128}")


class ReportError(ValueError):
    """A Platform report is malformed or unauthenticated."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_float(_: str) -> None:
    raise ReportError("floats forbidden")


def _reject_constant(_: str) -> None:
    raise ReportError("constants forbidden")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReportError(f"{name} must be an integer >= {minimum}")
    return value


def _keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ReportError(f"{name} fields are invalid")


def parse_report(
    raw: bytes,
    *,
    expected_network: str,
    expected_netuid: int,
    expected_signer: str,
) -> dict[str, Any]:
    """Return one exact, signature-verified report."""

    if not raw or len(raw) > MAX_REPORT_BYTES:
        raise ReportError("report size is invalid")
    try:
        report = json.loads(
            raw,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique,
        )
    except ReportError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError("report is not valid JSON") from exc
    if not isinstance(report, dict):
        raise ReportError("report must be an object")
    _keys(report, REPORT_KEYS, "report")
    if raw != canonical_json(report):
        raise ReportError("report is not canonical JSON")

    if _integer(report["schema_version"], "schema_version", minimum=1) != 1:
        raise ReportError("unsupported schema_version")
    network = report["network"]
    signer = report["signer"]
    if not isinstance(network, str) or not network:
        raise ReportError("network is invalid")
    if not isinstance(signer, str) or not signer:
        raise ReportError("signer is invalid")
    netuid = _integer(report["netuid"], "netuid", minimum=1)
    if network != expected_network or netuid != expected_netuid:
        raise ReportError("report network or netuid does not match configuration")
    if not hmac.compare_digest(signer, expected_signer):
        raise ReportError("report signer does not match configuration")
    start = _integer(report["epoch_start_block"], "epoch_start_block")
    end = _integer(report["epoch_end_block"], "epoch_end_block")
    finalized = _integer(report["finalized_block"], "finalized_block")
    tempo = _integer(report["tempo"], "tempo", minimum=1)
    _integer(report["created_at_ms"], "created_at_ms", minimum=1)
    if end - start + 1 != tempo or finalized < end:
        raise ReportError("report epoch bounds are invalid")
    expected_id = f"{network}-{netuid}-{start}-{end}"
    if report["report_id"] != expected_id:
        raise ReportError("report_id does not match its epoch")

    miners = report["miners"]
    if not isinstance(miners, list):
        raise ReportError("miners must be an array")
    uids: list[int] = []
    hotkeys: list[str] = []
    for index, row in enumerate(miners):
        if not isinstance(row, dict):
            raise ReportError(f"miners[{index}] must be an object")
        _keys(row, MINER_KEYS, f"miners[{index}]")
        values = {
            key: _integer(row[key], f"miners[{index}].{key}")
            for key in MINER_KEYS - {"hotkey"}
        }
        hotkey = row["hotkey"]
        if not isinstance(hotkey, str) or not hotkey:
            raise ReportError(f"miners[{index}].hotkey is invalid")
        if (
            values["successful_requests"] + values["failed_requests"]
            != values["routed_requests"]
        ):
            raise ReportError(f"miners[{index}] request counts do not balance")
        proof_total = sum(
            values[key]
            for key in (
                "proof_verified_requests",
                "proof_failed_requests",
                "proof_timed_out_requests",
                "proof_not_available_requests",
            )
        )
        if proof_total != values["routed_requests"]:
            raise ReportError(f"miners[{index}] proof counts do not balance")
        if values["proof_verified_requests"] > values["successful_requests"]:
            raise ReportError(f"miners[{index}] verified count is invalid")
        if values["proof_not_available_requests"] > values["failed_requests"]:
            raise ReportError(f"miners[{index}] missing proof count is invalid")
        if values["verified_completion_tokens"] > values["completion_tokens"]:
            raise ReportError(f"miners[{index}] verified token count is invalid")
        if values["proof_verified_requests"] == 0 and (
            values["verified_completion_tokens"] != 0 or values["generation_tps_p50"] != 0
        ):
            raise ReportError(f"miners[{index}] verified metrics are invalid")
        if values["proof_verified_requests"] > 0 and (
            values["verified_completion_tokens"] == 0 or values["generation_tps_p50"] == 0
        ):
            raise ReportError(f"miners[{index}] verified metrics are invalid")
        uids.append(values["uid"])
        hotkeys.append(hotkey)
    if uids != sorted(uids) or len(set(uids)) != len(uids):
        raise ReportError("miner UIDs must be sorted and unique")
    if len(set(hotkeys)) != len(hotkeys):
        raise ReportError("miner hotkeys must be unique")

    digest = report["digest"]
    signature = report["signature"]
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ReportError("report digest is invalid")
    if not isinstance(signature, str) or not _SIGNATURE.fullmatch(signature):
        raise ReportError("report signature is invalid")
    payload = canonical_json(
        {key: value for key, value in report.items() if key not in {"digest", "signature"}}
    )
    calculated = "sha256:" + hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, calculated):
        raise ReportError("report digest does not match its payload")
    try:
        verified = Keypair(ss58_address=signer).verify(payload, bytes.fromhex(signature))
    except Exception as exc:
        raise ReportError("report signature is invalid") from exc
    if not verified:
        raise ReportError("report signature is invalid")
    return report
