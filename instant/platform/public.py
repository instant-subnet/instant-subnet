"""The stats projection served to the platform's control plane.

This exists so the dashboard and metrics page have a data source.
``GET /validator/v1/stats`` cannot be that source: it requires an Epistula
sr25519 signature from a known validator hotkey, so the control plane could
only call it by holding the validator's private key.

**This route is not for browsers, and must not be exposed to one.** The
handoff's architecture keeps this service as the inference data plane and puts
everything browser-facing in the separate control-plane app, so nginx binds
this to loopback and the control plane is the only caller. What that buys:
the gateway stays the single implementation of the aggregation, while the
shape a page actually renders lives in the platform repository and can change
without releasing the subnet runtime.

Two deliberate choices are worth stating, because both look like duplication
until you know why.

**The public shape is its own model, not a reuse of** :class:`StatsResponse`.
The validator response is a *signed* protocol schema — integers only, so the
signature cannot become runtime-dependent, and carrying a receipt merkle root
so a validator can check the aggregates against sampled receipts. Welding a
browser contract onto it would mean every protocol change is a breaking UI
change and every UI need pushes back on the protocol. They are different
contracts with different audiences and different rates of change, so they are
different models. What they share is the *computation*, which stays in
``PlatformState.stats()`` and is not reimplemented here.

**Responses are cached for a few seconds.** ``stats()`` is not a cheap read:
it holds the state lock while fetching every row in the window, then parses
each stored receipt and computes a merkle root. That is fine for one
validator polling occasionally, and not fine for browsers polling every ten
seconds against the same process that relays inference streams. The cache
bounds the cost to one computation per TTL no matter how many readers there
are.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from ..protocol.schemas import StatsResponse

# Matches the page's poll interval, so a browser almost never triggers a
# recomputation it could have shared with the previous caller.
DEFAULT_TTL_S = 10.0

_STRICT = ConfigDict(extra="forbid")


class PublicMinerStats(BaseModel):
    """One miner's window, reduced to what a browser should see.

    Dropped relative to :class:`MinerStatsWindow`: ``hotkey`` (``uid``
    identifies the miner within the subnet without naming a specific machine
    on an unauthenticated endpoint) and ``served`` (it is ``successes`` under
    another name).

    Units are the ones the aggregation already produced and are easy to get
    wrong downstream: ``tokens_per_s_*`` is whole tokens per second,
    ``ttft_*_ms`` is milliseconds, and ``success_rate_bps`` is basis points,
    so 9941 means 99.41%.
    """

    model_config = _STRICT

    uid: int
    requests: int
    successes: int
    failures: int
    clean_rejects: int
    ttft_p50_ms: int
    ttft_p95_ms: int
    tokens_per_s_p50: int
    tokens_per_s_p95: int
    success_rate_bps: int = Field(ge=0, le=10_000)
    prompt_tokens: int
    completion_tokens: int
    receipts_seen: int
    receipts_verified: int
    # Always false/null while INSTANT_ATTESTATION_MODE is off. Carried anyway
    # so that surfacing attestation in the UI later is a frontend change with
    # no matching backend release.
    #
    # Reviewers: attestation_id is an identifier, not a status. Once
    # attestation is enabled `state.py` starts populating it from the receipts,
    # so whatever the control plane renders would begin carrying an enclave
    # identifier with no further change here. That is arguably the point of a
    # verifiable-inference subnet, but publishing it should be a decision
    # someone made rather than a default that arrived on its own -- and it is
    # the control plane's to make, since this route never reaches a browser.
    attestation_ok: bool
    attestation_id: str | None = None


class PublicStats(BaseModel):
    """``GET /public/v1/stats`` — the same window, for the control plane.

    ``receipt_merkle_root`` is deliberately absent: it is the validator's
    audit anchor over the receipt set, it means nothing to a browser, and
    publishing it invites treating it as a public commitment it was never
    designed to be. ``block_start``/``block_end`` are absent because the
    platform never populates them.
    """

    model_config = _STRICT

    generated_at_ms: int
    window_start_ms: int
    window_end_ms: int
    total_requests: int
    miners: list[PublicMinerStats]


def project(stats: StatsResponse) -> PublicStats:
    """Reduce a validator stats response to the public projection."""
    return PublicStats(
        generated_at_ms=stats.generated_at_ms,
        window_start_ms=stats.window_start_ms,
        window_end_ms=stats.window_end_ms,
        total_requests=stats.total_requests,
        miners=[
            PublicMinerStats(
                uid=miner.uid,
                requests=miner.requests,
                successes=miner.successes,
                failures=miner.failures,
                clean_rejects=miner.clean_rejects,
                ttft_p50_ms=miner.ttft_p50_ms,
                ttft_p95_ms=miner.ttft_p95_ms,
                tokens_per_s_p50=miner.tokens_per_s_p50,
                tokens_per_s_p95=miner.tokens_per_s_p95,
                success_rate_bps=miner.success_rate_bps,
                prompt_tokens=miner.prompt_tokens,
                completion_tokens=miner.completion_tokens,
                receipts_seen=miner.receipts_seen,
                receipts_verified=miner.receipts_verified,
                attestation_ok=miner.attestation_ok,
                attestation_id=miner.attestation_id,
            )
            for miner in stats.miners
        ],
    )


def encode(stats: PublicStats) -> bytes:
    """Serialise the projection the way the route returns it."""
    return json.dumps(stats.model_dump(), separators=(",", ":"), sort_keys=True).encode()


class PublicStatsCache:
    """Serve one computed body per TTL, however many readers arrive.

    The clock is injectable so tests can advance time instead of sleeping; a
    test that sleeps to prove a cache expired is a test that fails on a busy
    CI runner.
    """

    __slots__ = ("_body", "_clock", "_computed_at", "_lock", "_ttl_s")

    def __init__(
        self,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s < 0:
            raise ValueError("ttl_s must not be negative")
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._body: bytes | None = None
        self._computed_at = 0.0

    def get(self, compute: Callable[[], bytes]) -> bytes:
        """Return a cached body, or compute and store a fresh one.

        ``compute`` runs while the lock is held, so a burst of readers arriving
        on an empty cache produces one computation rather than one each. It is
        the expensive call this class exists to avoid multiplying.

        The trade is that concurrent readers block here rather than returning,
        and because the route is synchronous each one occupies a threadpool
        worker while it waits. That is the right way round: the alternative
        computes outside the lock and lets N simultaneous misses run N full
        ``stats()`` calls against the same SQLite connection the inference path
        is writing to. Blocked threads are released as soon as the single
        computation lands, and at one computation per TTL the window in which
        anything can pile up is small.
        """
        with self._lock:
            now = self._clock()
            fresh = self._body is not None and (now - self._computed_at) < self._ttl_s
            if not fresh:
                self._body = compute()
                self._computed_at = now
            return self._body
