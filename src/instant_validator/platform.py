"""Fetch the latest immutable metrics report from the platform."""

from __future__ import annotations

import httpx

from .config import Settings
from .report import MAX_REPORT_BYTES, PlatformReport, ReportError, parse_report


class PlatformError(RuntimeError):
    """The platform report could not be fetched or trusted."""


class PlatformClient:
    def __init__(self, settings: Settings, *, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self._owns_http = http is None
        self.http = http or httpx.Client(
            timeout=settings.request_timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": "instant-validator/0.1",
            },
        )

    def fetch_latest(self, *, now_ms: int | None = None) -> PlatformReport:
        try:
            response = self.http.get(self.settings.platform_report_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PlatformError(f"platform report request failed: {exc}") from exc
        raw = response.content
        if len(raw) > MAX_REPORT_BYTES:
            raise PlatformError("platform report exceeds the size limit")
        try:
            return parse_report(
                raw,
                expected_network=self.settings.network,
                expected_netuid=self.settings.netuid,
                expected_signer=self.settings.platform_signer,
                max_age_seconds=self.settings.report_max_age_seconds,
                future_skew_seconds=self.settings.report_future_skew_seconds,
                now_ms=now_ms,
            )
        except ReportError as exc:
            raise PlatformError(f"platform report rejected: {exc}") from exc

    def close(self) -> None:
        if self._owns_http:
            self.http.close()
