"""Who is allowed to send this miner work.

Two questions, kept separate on purpose:

* **Is this request authentic?** — Epistula signature, timestamp window,
  replay cache. Answered by :mod:`instant.protocol.epistula`.
* **Is this signer entitled?** — is the hotkey the platform, or a validator
  holding a permit on our netuid? Answered here, from a metagraph snapshot.

Keeping them separate means the entitlement source can be a stale snapshot,
a chain query, or a test fixture, without any of that reaching the signing
code. The miner never queries the chain on the request path — a miner whose
p99 latency includes a substrate round trip is a miner that loses on the one
axis this subnet sells.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..protocol.epistula import (
    EpistulaError,
    ReplayGuard,
    VerifiedRequest,
    verify_headers,
)

log = logging.getLogger("instant.miner.auth")


@dataclass(slots=True)
class AcceptList:
    """Hotkeys allowed to send inference requests, refreshed periodically.

    Held as a snapshot with an explicit age. If the refresh loop dies, the
    snapshot goes stale but the miner keeps serving whoever was valid at the
    last successful refresh — degraded, not down. A miner that stops serving
    because it could not reach the chain has converted a chain problem into
    its own outage and will be scored for it.
    """

    platform_hotkey: str | None = None
    validator_hotkeys: frozenset[str] = frozenset()
    updated_at: float = 0.0
    #: How stale a snapshot may get before we say so out loud. Not an
    #: expiry -- see above.
    warn_after_s: float = 900.0

    def update(self, validator_hotkeys: frozenset[str]) -> None:
        self.validator_hotkeys = validator_hotkeys
        self.updated_at = time.time()

    @property
    def age_s(self) -> float:
        return time.time() - self.updated_at if self.updated_at else float("inf")

    @property
    def is_stale(self) -> bool:
        return self.age_s > self.warn_after_s

    def all(self) -> frozenset[str]:
        if self.platform_hotkey:
            return self.validator_hotkeys | {self.platform_hotkey}
        return self.validator_hotkeys

    def kind_of(self, hotkey: str) -> str:
        if hotkey == self.platform_hotkey:
            return "platform"
        if hotkey in self.validator_hotkeys:
            return "validator"
        return "unknown"


@dataclass(slots=True)
class Authenticator:
    """Verifies inbound requests against the accept-list."""

    my_hotkey: str
    accept: AcceptList
    replay: ReplayGuard = field(default_factory=ReplayGuard)
    #: When false, any authentic signature is accepted regardless of whether
    #: the signer is on the accept-list. Development only -- it exists so a
    #: laptop miner can be poked with curl and a throwaway key without a
    #: registered hotkey.
    enforce_accept_list: bool = True

    def verify(self, headers: dict[str, str], body: bytes) -> VerifiedRequest:
        """Raises :class:`EpistulaError` if the request may not proceed."""
        allowed = self.accept.all() if self.enforce_accept_list else None
        if allowed is not None and not allowed:
            # An empty accept-list would reject everything, which during a
            # cold start looks exactly like "the miner is broken". Say which
            # it is.
            raise EpistulaError(
                "accept-list is empty — miner has not yet synced the metagraph"
            )
        return verify_headers(
            headers,
            body,
            allowed_signers=allowed,
            expected_signed_for=self.my_hotkey,
            replay_guard=self.replay,
        )
