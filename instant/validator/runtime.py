"""The validator's first operational loop.

This module is intentionally read-only with respect to the chain.  It proves
the plumbing we need before registration and stake enter the picture:

* connect to the configured websocket endpoint;
* refresh netuid's metagraph;
* report whether our hotkey is registered and has a validator permit;
* discover serving miner axons; and
* report whether the platform gateway is reachable.

Probes, scoring epochs, and bounded weight submission build on this snapshot.
Keeping them out of the first loop means a process can be deployed and
observed before it has authority to mutate anything.
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import httpx

from .state import ValidatorState

if False:  # pragma: no cover - typing-only without a runtime import cycle
    from .coordinator import ScoringCoordinator


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class MinerEndpoint:
    """One serving axon from a metagraph snapshot."""

    uid: int
    hotkey: str
    ip: str
    port: int

    @property
    def url(self) -> str:
        host = f"[{self.ip}]" if ":" in self.ip else self.ip
        return f"http://{host}:{self.port}"

    def to_payload(self) -> dict[str, Any]:
        return {**asdict(self), "url": self.url}


@dataclass(frozen=True, slots=True)
class ValidatorSnapshot:
    """Everything the health API can say without performing a chain write."""

    chain_connected: bool
    block: int | None
    netuid: int
    neuron_count: int
    own_hotkey: str
    registered: bool
    validator_permit: bool
    miners: tuple[MinerEndpoint, ...]
    platform_reachable: bool
    last_sync_ms: int | None
    stale_after_ms: int
    error: str | None = None

    @property
    def ready(self) -> bool:
        # Platform loss must not stop direct validation.  It is surfaced as
        # degraded health, but readiness is chain authority and freshness.
        fresh = (
            self.last_sync_ms is not None
            and _now_ms() - self.last_sync_ms <= self.stale_after_ms
        )
        return (
            self.chain_connected
            and self.registered
            and self.validator_permit
            and fresh
        )

    @property
    def status(self) -> str:
        if not self.chain_connected:
            return "down"
        if not self.ready:
            return "degraded"
        return "ok" if self.platform_reachable else "degraded"

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        payload["ready"] = self.ready
        payload["stale"] = (
            self.last_sync_ms is None
            or _now_ms() - self.last_sync_ms > self.stale_after_ms
        )
        payload["miners"] = [miner.to_payload() for miner in self.miners]
        return payload


def serving_miners(metagraph: Any) -> tuple[MinerEndpoint, ...]:
    """Return stable, validated HTTP endpoints from a metagraph.

    Bittensor represents an unannounced axon as ``0.0.0.0:0``.  It is a
    neuron, but not a miner endpoint, so it must not enter the routing set.
    """

    found: list[MinerEndpoint] = []
    hotkeys = list(getattr(metagraph, "hotkeys", []))
    axons = list(getattr(metagraph, "axons", []))
    for uid, (hotkey, axon) in enumerate(zip(hotkeys, axons, strict=False)):
        ip = str(getattr(axon, "ip", "") or "").strip()
        try:
            port = int(getattr(axon, "port", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not ip or port < 1 or port > 65_535:
            continue
        try:
            if ipaddress.ip_address(ip).is_unspecified:
                continue
        except ValueError:
            # Axon metadata should be an IP, but accepting a valid hostname
            # keeps tests and private-network deployments straightforward.
            if any(char.isspace() for char in ip):
                continue
        if getattr(axon, "is_serving", True) is False:
            continue
        found.append(MinerEndpoint(uid=uid, hotkey=str(hotkey), ip=ip, port=port))
    return tuple(found)


class ValidatorRuntime:
    """Mutable owner of the current read-only chain snapshot."""

    def __init__(
        self,
        *,
        subtensor: Any,
        wallet: Any,
        netuid: int,
        state: ValidatorState,
        platform_url: str,
        refresh_s: int = 120,
        request_timeout_s: int = 30,
        http: httpx.AsyncClient | None = None,
        coordinator: ScoringCoordinator | None = None,
    ) -> None:
        self.subtensor = subtensor
        self.wallet = wallet
        self.netuid = netuid
        self.state = state
        self.platform_url = platform_url.rstrip("/")
        self.refresh_s = refresh_s
        self.stale_after_ms = max(30, refresh_s * 3) * 1000
        self._owns_http = http is None
        self.http = http or httpx.AsyncClient(timeout=request_timeout_s)
        self.coordinator = coordinator
        self.telemetry_status: dict[str, Any] = {
            "status": "not_run",
            "last_attempt_ms": None,
            "last_success_ms": None,
            "error": None,
        }
        self.scoring_status: dict[str, Any] = {
            "status": "not_run",
            "last_attempt_ms": None,
            "last_success_ms": None,
            "error": None,
        }
        hotkey = wallet.hotkey.ss58_address
        self.snapshot = ValidatorSnapshot(
            chain_connected=False,
            block=None,
            netuid=netuid,
            neuron_count=0,
            own_hotkey=hotkey,
            registered=False,
            validator_permit=False,
            miners=(),
            platform_reachable=False,
            last_sync_ms=None,
            stale_after_ms=self.stale_after_ms,
            error="not synced yet",
        )

    async def sync_once(self) -> ValidatorSnapshot:
        """Refresh chain and platform state once, retaining errors as health."""

        try:
            metagraph = await asyncio.to_thread(
                self.subtensor.metagraph, self.netuid, True
            )
            block = await asyncio.to_thread(self.subtensor.get_current_block)
            hotkeys = [str(value) for value in metagraph.hotkeys]
            own = self.wallet.hotkey.ss58_address
            registered = own in hotkeys
            permit = False
            if registered:
                uid = hotkeys.index(own)
                permits = list(getattr(metagraph, "validator_permit", []))
                permit = uid < len(permits) and bool(permits[uid])
            snapshot = ValidatorSnapshot(
                chain_connected=True,
                block=int(block),
                netuid=self.netuid,
                neuron_count=len(hotkeys),
                own_hotkey=own,
                registered=registered,
                validator_permit=permit,
                miners=serving_miners(metagraph),
                platform_reachable=False,
                last_sync_ms=_now_ms(),
                stale_after_ms=self.stale_after_ms,
            )
        except Exception as exc:  # noqa: BLE001 - health records the boundary
            self.snapshot = replace(
                self.snapshot,
                chain_connected=False,
                platform_reachable=False,
                last_sync_ms=_now_ms(),
                error=f"chain sync failed: {exc}",
            )
            return self.snapshot

        platform_reachable = await self._platform_health()
        self.snapshot = replace(
            snapshot,
            platform_reachable=platform_reachable,
            error=None if platform_reachable else "platform health check failed",
        )
        return self.snapshot

    async def _platform_health(self) -> bool:
        if not self.platform_url:
            return False
        try:
            response = await self.http.get(f"{self.platform_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def score_once(self) -> dict[str, Any]:
        """Fetch authenticated telemetry and persist one score epoch.

        This is called only by the explicit CLI action.  The continuous PM2
        loop remains read-only and never advances scoring state on boot.
        """
        if self.coordinator is None:
            raise RuntimeError("validator scoring coordinator is not configured")

        attempt_ms = _now_ms()
        self.telemetry_status = {
            **self.telemetry_status,
            "status": "fetching",
            "last_attempt_ms": attempt_ms,
            "error": None,
        }
        self.scoring_status = {
            **self.scoring_status,
            "status": "running",
            "last_attempt_ms": attempt_ms,
            "error": None,
        }
        snapshot = await self.sync_once()
        try:
            run, batch = await self.coordinator.run_once(snapshot)
        except Exception as exc:  # noqa: BLE001 - operations status boundary
            message = str(exc)
            self.telemetry_status = {
                **self.telemetry_status,
                "status": "error",
                "error": message,
            }
            self.scoring_status = {
                **self.scoring_status,
                "status": "error",
                "error": message,
            }
            raise

        success_ms = _now_ms()
        self.telemetry_status = {
            "status": "ok",
            "last_attempt_ms": attempt_ms,
            "last_success_ms": success_ms,
            "error": None,
            **batch.to_status(),
        }
        self.scoring_status = {
            "status": "ok",
            "last_attempt_ms": attempt_ms,
            "last_success_ms": success_ms,
            "error": None,
            **run.to_payload(),
        }
        return {
            "telemetry": dict(self.telemetry_status),
            "scoring": dict(self.scoring_status),
            "scores": self.state.scores_for_epoch(run.epoch),
        }

    async def run(self) -> None:
        """Refresh forever; PM2 owns restart policy, this owns transient errors."""

        while True:
            await asyncio.sleep(self.refresh_s)
            await self.sync_once()

    async def aclose(self) -> None:
        if self.coordinator is not None:
            await self.coordinator.aclose()
        if self._owns_http:
            await self.http.aclose()
        close = getattr(self.subtensor, "close", None)
        if close is not None:
            await asyncio.to_thread(close)
        self.state.close()
