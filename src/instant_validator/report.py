"""The one signed platform-to-validator report contract."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 1_000_000
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")

_REPORT_KEYS = {
    "schema_version",
    "report_id",
    "network",
    "netuid",
    "period_start_block",
    "period_end_block",
    "created_at_ms",
    "signer",
    "miners",
    "digest",
    "signature",
}
_MINER_KEYS = {
    "uid",
    "hotkey",
    "requests",
    "successes",
    "failures",
    "prompt_tokens",
    "completion_tokens",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tokens_per_second_p50",
    "toploc_verified",
    "toploc_failed",
    "toploc_timed_out",
}


class ReportError(ValueError):
    """A platform report is malformed, untrusted, or stale."""


def _reject_float(_: str) -> None:
    raise ReportError("floating-point values are not allowed")


def _reject_constant(_: str) -> None:
    raise ReportError("non-finite numbers are not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_json(payload: Any) -> bytes:
    """Return the only byte representation used for digesting and signing."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReportError(f"{context} keys differ: missing={missing}, extra={extra}")


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReportError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ReportError(f"{field} must be a non-empty string <= {maximum} characters")
    return value


@dataclass(frozen=True, slots=True)
class MinerMetrics:
    uid: int
    hotkey: str
    requests: int
    successes: int
    failures: int
    prompt_tokens: int
    completion_tokens: int
    ttft_p50_ms: int
    ttft_p95_ms: int
    tokens_per_second_p50: int
    toploc_verified: int
    toploc_failed: int
    toploc_timed_out: int

    @classmethod
    def from_dict(cls, value: Any, index: int) -> MinerMetrics:
        if not isinstance(value, dict):
            raise ReportError(f"miners[{index}] must be an object")
        _exact_keys(value, _MINER_KEYS, f"miners[{index}]")
        prefix = f"miners[{index}]"
        row = cls(
            uid=_integer(value["uid"], f"{prefix}.uid"),
            hotkey=_text(value["hotkey"], f"{prefix}.hotkey", maximum=128),
            requests=_integer(value["requests"], f"{prefix}.requests", minimum=1),
            successes=_integer(value["successes"], f"{prefix}.successes"),
            failures=_integer(value["failures"], f"{prefix}.failures"),
            prompt_tokens=_integer(value["prompt_tokens"], f"{prefix}.prompt_tokens"),
            completion_tokens=_integer(
                value["completion_tokens"], f"{prefix}.completion_tokens"
            ),
            ttft_p50_ms=_integer(value["ttft_p50_ms"], f"{prefix}.ttft_p50_ms"),
            ttft_p95_ms=_integer(value["ttft_p95_ms"], f"{prefix}.ttft_p95_ms"),
            tokens_per_second_p50=_integer(
                value["tokens_per_second_p50"],
                f"{prefix}.tokens_per_second_p50",
            ),
            toploc_verified=_integer(value["toploc_verified"], f"{prefix}.toploc_verified"),
            toploc_failed=_integer(value["toploc_failed"], f"{prefix}.toploc_failed"),
            toploc_timed_out=_integer(
                value["toploc_timed_out"], f"{prefix}.toploc_timed_out"
            ),
        )
        if row.successes + row.failures != row.requests:
            raise ReportError(f"{prefix} request counts do not balance")
        proof_total = row.toploc_verified + row.toploc_failed + row.toploc_timed_out
        if proof_total != row.requests:
            raise ReportError(f"{prefix} TOPLOC counts do not balance")
        if row.successes and (
            row.ttft_p50_ms < 1
            or row.ttft_p95_ms < row.ttft_p50_ms
            or row.tokens_per_second_p50 < 1
        ):
            raise ReportError(f"{prefix} successful metrics are invalid")
        return row


@dataclass(frozen=True, slots=True)
class PlatformReport:
    schema_version: int
    report_id: str
    network: str
    netuid: int
    period_start_block: int
    period_end_block: int
    created_at_ms: int
    signer: str
    miners: tuple[MinerMetrics, ...]
    digest: str
    signature: str


def _verify_sr25519(signer: str, message: bytes, signature: bytes) -> bool:
    try:
        import bittensor as bt

        return bool(bt.Keypair(ss58_address=signer).verify(message, signature))
    except Exception:  # noqa: BLE001 - invalid key/signature is an untrusted boundary
        return False


def parse_report(
    raw: bytes,
    *,
    expected_network: str,
    expected_netuid: int,
    expected_signer: str,
    max_age_seconds: int,
    future_skew_seconds: int,
    now_ms: int | None = None,
) -> PlatformReport:
    """Parse, validate, digest-check, and signature-check one report."""

    if not raw or len(raw) > MAX_REPORT_BYTES:
        raise ReportError("report size is invalid")
    try:
        value = json.loads(
            raw,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ReportError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError("report is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ReportError("report must be an object")
    _exact_keys(value, _REPORT_KEYS, "report")

    schema_version = _integer(value["schema_version"], "schema_version", minimum=1)
    if schema_version != SCHEMA_VERSION:
        raise ReportError(f"unsupported schema_version {schema_version}")
    report_id = _text(value["report_id"], "report_id", maximum=128)
    network = _text(value["network"], "network", maximum=32)
    netuid = _integer(value["netuid"], "netuid", minimum=1)
    start = _integer(value["period_start_block"], "period_start_block")
    end = _integer(value["period_end_block"], "period_end_block")
    if end < start:
        raise ReportError("period_end_block precedes period_start_block")
    created_at_ms = _integer(value["created_at_ms"], "created_at_ms", minimum=1)
    signer = _text(value["signer"], "signer", maximum=128)
    digest_value = _text(value["digest"], "digest", maximum=71)
    signature = _text(value["signature"], "signature", maximum=128)
    if not _DIGEST_RE.fullmatch(digest_value):
        raise ReportError("digest must be lowercase sha256 hex")
    if not _SIGNATURE_RE.fullmatch(signature):
        raise ReportError("signature must be lowercase SR25519 hex")
    miners_value = value["miners"]
    if not isinstance(miners_value, list) or not miners_value:
        raise ReportError("miners must be a non-empty array")
    miners = tuple(
        MinerMetrics.from_dict(row, index) for index, row in enumerate(miners_value)
    )
    uids = [row.uid for row in miners]
    hotkeys = [row.hotkey for row in miners]
    if len(set(uids)) != len(uids):
        raise ReportError("miner UIDs must be unique")
    if len(set(hotkeys)) != len(hotkeys):
        raise ReportError("miner hotkeys must be unique")

    if network != expected_network or netuid != expected_netuid:
        raise ReportError(
            f"report targets {network}/{netuid}, expected {expected_network}/{expected_netuid}"
        )
    if not hmac.compare_digest(signer, expected_signer):
        raise ReportError("report signer does not match configured platform signer")

    timestamp = int(time.time() * 1000) if now_ms is None else now_ms
    if created_at_ms > timestamp + future_skew_seconds * 1000:
        raise ReportError("report creation time is in the future")
    if timestamp - created_at_ms > max_age_seconds * 1000:
        raise ReportError("report is stale")

    signed_payload = {
        key: item for key, item in value.items() if key not in {"digest", "signature"}
    }
    message = canonical_json(signed_payload)
    calculated = "sha256:" + hashlib.sha256(message).hexdigest()
    if not hmac.compare_digest(digest_value, calculated):
        raise ReportError("report digest does not match its payload")
    if not _verify_sr25519(signer, message, bytes.fromhex(signature)):
        raise ReportError("report signature is invalid")

    return PlatformReport(
        schema_version=schema_version,
        report_id=report_id,
        network=network,
        netuid=netuid,
        period_start_block=start,
        period_end_block=end,
        created_at_ms=created_at_ms,
        signer=signer,
        miners=miners,
        digest=digest_value,
        signature=signature,
    )
