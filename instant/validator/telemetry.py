"""Authenticated platform telemetry ingestion for the validator.

The platform is an observation source, not an authority.  Its report is
accepted only when the exact response bytes are Epistula-signed by the
configured platform hotkey, addressed to this validator, fresh, internally
consistent, and bound to the current metagraph ``(uid, hotkey)`` roster.

Receipt verification is performed by the platform in the first localnet
slice.  We preserve its ``seen`` and ``verified`` counters in SQLite so the
existing scoring penalty fails a miner closed when they disagree.  A later
receipt-audit client can independently verify the committed receipt set
without changing the scoring/state boundary implemented here.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from instant.protocol.epistula import (
    ALLOWED_FUTURE_MS,
    EpistulaError,
    ReplayGuard,
    generate_headers,
    verify_headers,
)
from instant.protocol.schemas import StatsResponse

from .state import TelemetryRow

_MAX_STATS_BYTES = 1_000_000
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _now_ms() -> int:
    return int(time.time() * 1000)


class TelemetryError(RuntimeError):
    """A platform report was unavailable, unauthenticated, or inconsistent."""


@dataclass(frozen=True, slots=True)
class TelemetryBatch:
    """A verified platform window mapped to the current miner roster."""

    epoch: int
    window_start_ms: int
    window_end_ms: int
    block_start: int
    block_end: int
    generated_at_ms: int
    rows: tuple[TelemetryRow, ...]

    def to_status(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "generated_at_ms": self.generated_at_ms,
            "miners_ingested": len(self.rows),
            "requests_ingested": sum(row.requests for row in self.rows),
        }


def logical_epoch(stats: StatsResponse) -> int:
    """Derive an idempotent logical epoch from one platform window.

    Chain-bounded reports use their closing block.  The initial platform has
    no chain clock and reports a fixed-width rolling window with block bounds
    set to zero; bucketing its end time by that width prevents two manual
    ``--score-once`` invocations in the same window from advancing EMA and
    cooldown twice.
    """

    if stats.block_end > 0:
        return stats.block_end
    width_ms = stats.window_end_ms - stats.window_start_ms
    if width_ms <= 0:
        raise TelemetryError("platform telemetry window has no positive width")
    return stats.window_end_ms // width_ms


class PlatformTelemetryClient:
    """Fetch and authenticate ``GET /validator/v1/stats`` exactly once."""

    def __init__(
        self,
        *,
        base_url: str,
        platform_hotkey: str,
        validator_signer: Any,
        max_age_s: int = 120,
        max_window_s: int = 7_200,
        timeout_s: float = 30,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.platform_hotkey = platform_hotkey
        self.validator_signer = validator_signer
        self.max_age_ms = max_age_s * 1000
        self.max_window_ms = max_window_s * 1000
        self._replay = ReplayGuard()
        self._owns_http = http is None
        self.http = http or httpx.AsyncClient(timeout=timeout_s)

    async def fetch(
        self, roster: Sequence[tuple[int, str]], *, now_ms: int | None = None
    ) -> TelemetryBatch:
        """Return one verified report, or raise without persisting anything."""

        if not self.base_url:
            raise TelemetryError("INSTANT_PLATFORM_URL is required for telemetry")
        if not self.platform_hotkey:
            raise TelemetryError(
                "INSTANT_PLATFORM_SS58 is required to authenticate telemetry"
            )

        body = b""
        headers = generate_headers(
            self.validator_signer, body, signed_for=self.platform_hotkey
        )
        try:
            response = await self.http.get(
                f"{self.base_url}/validator/v1/stats", headers=headers
            )
        except httpx.HTTPError as exc:
            raise TelemetryError(f"platform telemetry request failed: {exc}") from exc

        if response.status_code != 200:
            raise TelemetryError(
                f"platform telemetry returned HTTP {response.status_code}"
            )
        raw = response.content
        if len(raw) > _MAX_STATS_BYTES:
            raise TelemetryError(
                f"platform telemetry response is {len(raw)} bytes; limit is "
                f"{_MAX_STATS_BYTES}"
            )

        checked_at_ms = _now_ms() if now_ms is None else now_ms
        try:
            verified = verify_headers(
                response.headers,
                raw,
                allowed_signers=[self.platform_hotkey],
                expected_signed_for=self.validator_signer.ss58_address,
                replay_guard=self._replay,
                now_ms=checked_at_ms,
            )
        except EpistulaError as exc:
            raise TelemetryError(
                f"platform telemetry signature rejected: {exc.reason}"
            ) from exc
        # ``verify_headers`` permits an absent Signed-For for public endpoints.
        # Stats are private validator data, so require the recipient binding.
        if verified.signed_for != self.validator_signer.ss58_address:
            raise TelemetryError(
                "platform telemetry response is not signed for this validator"
            )

        try:
            stats = StatsResponse.model_validate_json(raw)
        except ValidationError as exc:
            raise TelemetryError(f"invalid platform telemetry schema: {exc}") from exc

        self._validate(stats, now_ms=checked_at_ms)
        rows = self._map_roster(stats, roster)
        epoch = logical_epoch(stats)
        return TelemetryBatch(
            epoch=epoch,
            window_start_ms=stats.window_start_ms,
            window_end_ms=stats.window_end_ms,
            block_start=stats.block_start,
            block_end=stats.block_end,
            generated_at_ms=stats.generated_at_ms,
            rows=rows,
        )

    def _validate(self, stats: StatsResponse, *, now_ms: int) -> None:
        if stats.window_start_ms < 0 or stats.window_end_ms <= stats.window_start_ms:
            raise TelemetryError("platform telemetry has invalid time bounds")
        if stats.window_end_ms > stats.generated_at_ms + ALLOWED_FUTURE_MS:
            raise TelemetryError("platform telemetry window ends after it was generated")
        if stats.generated_at_ms > now_ms + ALLOWED_FUTURE_MS:
            raise TelemetryError("platform telemetry generated_at_ms is in the future")
        if now_ms - stats.generated_at_ms > self.max_age_ms:
            raise TelemetryError("platform telemetry payload is stale")
        if stats.window_end_ms - stats.window_start_ms > self.max_window_ms:
            raise TelemetryError("platform telemetry window exceeds configured maximum")

        zero_blocks = stats.block_start == 0 and stats.block_end == 0
        valid_blocks = 0 < stats.block_start <= stats.block_end
        if not (zero_blocks or valid_blocks):
            raise TelemetryError("platform telemetry has invalid block bounds")
        if not _SHA256_RE.fullmatch(stats.receipt_merkle_root):
            raise TelemetryError("platform telemetry has an invalid receipt Merkle root")
        if stats.total_requests != sum(miner.requests for miner in stats.miners):
            raise TelemetryError("platform total_requests does not match miner rows")

        hotkeys: set[str] = set()
        uids: set[int] = set()
        for miner in stats.miners:
            if miner.hotkey in hotkeys or miner.uid in uids:
                raise TelemetryError("platform telemetry contains duplicate miner identity")
            hotkeys.add(miner.hotkey)
            uids.add(miner.uid)

            clean_rejects = int(getattr(miner, "clean_rejects", 0))
            served = int(getattr(miner, "served", miner.successes))
            counters = (
                miner.requests,
                miner.successes,
                miner.failures,
                clean_rejects,
                served,
                miner.receipts_seen,
                miner.receipts_verified,
            )
            if any(value < 0 for value in counters):
                raise TelemetryError("platform telemetry contains a negative counter")
            if miner.successes + miner.failures != miner.requests:
                raise TelemetryError(
                    f"platform counts do not balance for uid {miner.uid}"
                )
            if clean_rejects > miner.failures:
                raise TelemetryError(
                    f"clean rejects exceed failures for uid {miner.uid}"
                )
            if served > miner.requests:
                raise TelemetryError(f"served exceeds requests for uid {miner.uid}")
            if not (
                miner.receipts_verified <= miner.receipts_seen <= miner.requests
            ):
                raise TelemetryError(
                    f"receipt counts do not balance for uid {miner.uid}"
                )
            expected_bps = (
                0
                if miner.requests == 0
                else miner.successes * 10_000 // miner.requests
            )
            if miner.success_rate_bps != expected_bps:
                raise TelemetryError(
                    f"success_rate_bps does not match counts for uid {miner.uid}"
                )

    def _map_roster(
        self, stats: StatsResponse, roster: Sequence[tuple[int, str]]
    ) -> tuple[TelemetryRow, ...]:
        current = {int(uid): str(hotkey) for uid, hotkey in roster}
        current_hotkeys = {hotkey: uid for uid, hotkey in current.items()}
        rows: list[TelemetryRow] = []
        epoch = logical_epoch(stats)
        for miner in stats.miners:
            if current.get(miner.uid) != miner.hotkey:
                old_uid = current_hotkeys.get(miner.hotkey)
                detail = (
                    "not on the serving roster"
                    if old_uid is None
                    else f"is now uid {old_uid}, not uid {miner.uid}"
                )
                raise TelemetryError(
                    f"platform miner {miner.hotkey} {detail}; refusing stale UID data"
                )
            clean_rejects = int(getattr(miner, "clean_rejects", 0))
            served = int(getattr(miner, "served", miner.successes))
            rows.append(
                TelemetryRow(
                    epoch=epoch,
                    uid=miner.uid,
                    hotkey=miner.hotkey,
                    requests=miner.requests,
                    successes=miner.successes,
                    clean_rejects=clean_rejects,
                    ttft_p95_ms=(
                        miner.ttft_p95_ms if miner.successes > 0 else None
                    ),
                    tokens_per_s_p50=(
                        miner.tokens_per_s_p50 if miner.successes > 0 else None
                    ),
                    served=served,
                    receipts_seen=miner.receipts_seen,
                    receipts_verified=miner.receipts_verified,
                    recorded_ms=stats.generated_at_ms,
                )
            )
        return tuple(sorted(rows, key=lambda row: row.uid))

    async def aclose(self) -> None:
        if self._owns_http:
            await self.http.aclose()
