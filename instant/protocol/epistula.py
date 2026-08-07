"""Epistula v2 request signing and verification.

Epistula is the Manifold-originated signing standard already in use across
several Bittensor subnets. We adopt it rather than inventing our own so that
miners who already run a Targon-family miner recognise our headers, and so
that anyone auditing our auth is auditing something with prior art.

The signed message is a period-joined string:

    sha256(body).hexdigest() . uuid . timestamp . signed_for

``signed_for`` is the empty string when absent -- not omitted, not a literal
"None". Getting that wrong produces a signature that verifies on the signer's
machine and nowhere else, which is exactly the bug this module exists to
prevent anyone from writing twice.

Replay protection has two parts and needs both:

* a timestamp window (``ALLOWED_DELTA_MS``), which bounds how long a captured
  request stays useful, and
* a UUID cache covering at least that window, which stops the same request
  being replayed *inside* it.

A timestamp check alone is not replay protection. :class:`ReplayGuard`
implements the second half.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid as uuidlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .keys import Signer
from .keys import sign as _sign
from .keys import verify as _verify

VERSION = "2"

#: How far in the past a request timestamp may be and still be accepted.
#: The Epistula reference uses 8s; we keep that. Tighter is tempting but
#: real clients sit behind proxies with real clock skew, and the UUID cache
#: is what actually stops replay -- this window only bounds its size.
ALLOWED_DELTA_MS = 8_000

#: How far in the *future* a timestamp may be. Some skew is normal; a lot of
#: it means either a broken clock or someone pre-signing requests.
ALLOWED_FUTURE_MS = 2_000

#: Secret-signature time bucket, per the Epistula spec.
SECRET_INTERVAL_MS = 10_000

H_VERSION = "Epistula-Version"
H_TIMESTAMP = "Epistula-Timestamp"
H_UUID = "Epistula-Uuid"
H_SIGNED_BY = "Epistula-Signed-By"
H_SIGNED_FOR = "Epistula-Signed-For"
H_SIGNATURE = "Epistula-Request-Signature"
H_SECRET = "Epistula-Secret-Signature-{}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def message_for(
    body: bytes, request_uuid: str, timestamp_ms: int, signed_for: str | None
) -> str:
    """Build the exact string that gets signed."""
    return (
        f"{hashlib.sha256(body).hexdigest()}"
        f".{request_uuid}"
        f".{timestamp_ms}"
        f".{signed_for or ''}"
    )


def generate_headers(
    signer: Signer,
    body: bytes,
    signed_for: str | None = None,
    *,
    timestamp_ms: int | None = None,
    request_uuid: str | None = None,
) -> dict[str, str]:
    """Produce the Epistula headers for a request.

    ``signed_for`` names the intended recipient's SS58 address. Setting it
    means a miner cannot take a request the platform addressed to *it* and
    replay it against a different miner -- which matters for us, because the
    platform signs shadow probes and miners must not be able to launder them.
    """
    timestamp_ms = _now_ms() if timestamp_ms is None else timestamp_ms
    request_uuid = str(uuidlib.uuid4()) if request_uuid is None else request_uuid

    headers = {
        H_VERSION: VERSION,
        H_TIMESTAMP: str(timestamp_ms),
        H_UUID: request_uuid,
        H_SIGNED_BY: signer.ss58_address,
        H_SIGNATURE: _sign(
            signer, message_for(body, request_uuid, timestamp_ms, signed_for)
        ),
    }

    if signed_for:
        headers[H_SIGNED_FOR] = signed_for
        bucket = math.ceil(timestamp_ms / SECRET_INTERVAL_MS) * SECRET_INTERVAL_MS
        for i, offset in enumerate((-1, 0, 1)):
            headers[H_SECRET.format(i)] = _sign(
                signer, f"{bucket + offset * SECRET_INTERVAL_MS}.{signed_for}"
            )

    return headers


@dataclass(slots=True)
class VerifiedRequest:
    """A request that passed every Epistula check."""

    signed_by: str
    signed_for: str | None
    uuid: str
    timestamp_ms: int


class EpistulaError(Exception):
    """Verification failed. ``reason`` is safe to return to the caller."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class ReplayGuard:
    """Bounded UUID cache covering the acceptance window.

    Entries older than the window can never be accepted on timestamp grounds
    anyway, so they are dropped. This keeps memory proportional to request
    rate over ~10 seconds rather than to total traffic, and means a miner
    under load does not slowly OOM because of its own auth layer.
    """

    window_ms: int = ALLOWED_DELTA_MS + ALLOWED_FUTURE_MS
    _seen: dict[str, int] = field(default_factory=dict)
    _swept_at_ms: int = 0

    def check_and_record(self, request_uuid: str, timestamp_ms: int, now_ms: int) -> bool:
        """True if this UUID is fresh. Records it. False means replay."""
        if now_ms - self._swept_at_ms > self.window_ms:
            self._sweep(now_ms)
        if request_uuid in self._seen:
            return False
        self._seen[request_uuid] = timestamp_ms
        return True

    def _sweep(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}
        self._swept_at_ms = now_ms

    def __len__(self) -> int:
        return len(self._seen)


def verify_headers(
    headers: Mapping[str, str],
    body: bytes,
    *,
    allowed_signers: Iterable[str] | None = None,
    expected_signed_for: str | None = None,
    replay_guard: ReplayGuard | None = None,
    now_ms: int | None = None,
) -> VerifiedRequest:
    """Verify an inbound Epistula-signed request.

    Raises :class:`EpistulaError` on any failure.

    ``allowed_signers`` is the accept-list -- for a miner that is the platform
    hotkey plus every hotkey holding a validator permit, refreshed from the
    metagraph. Passing ``None`` skips the check, which is only appropriate in
    tests and in the ``/attest`` path where the caller has already done it.

    ``expected_signed_for`` should be the verifier's own SS58 address. When
    the request carries ``Epistula-Signed-For``, it must match; a request
    addressed to someone else is rejected even though its signature is
    perfectly valid. That is the whole point of the field.
    """
    now_ms = _now_ms() if now_ms is None else now_ms

    # Header lookup is case-insensitive: Starlette lowercases, httpx preserves,
    # and nginx does its own thing. Normalising here is cheaper than being
    # surprised in production.
    lower = {k.lower(): v for k, v in headers.items()}

    def get(name: str) -> str | None:
        return lower.get(name.lower())

    version = get(H_VERSION)
    if version is not None and version != VERSION:
        raise EpistulaError(f"unsupported Epistula version {version!r}")

    signed_by = get(H_SIGNED_BY)
    signature = get(H_SIGNATURE)
    request_uuid = get(H_UUID)
    raw_timestamp = get(H_TIMESTAMP)

    if not signed_by:
        raise EpistulaError("missing Epistula-Signed-By")
    if not signature:
        raise EpistulaError("missing Epistula-Request-Signature")
    if not request_uuid:
        raise EpistulaError("missing Epistula-Uuid")
    if not raw_timestamp:
        raise EpistulaError("missing Epistula-Timestamp")

    try:
        timestamp_ms = int(raw_timestamp)
    except ValueError as exc:
        raise EpistulaError("Epistula-Timestamp is not an integer") from exc

    if timestamp_ms + ALLOWED_DELTA_MS < now_ms:
        raise EpistulaError(
            f"request is stale by {now_ms - timestamp_ms - ALLOWED_DELTA_MS}ms"
        )
    if timestamp_ms > now_ms + ALLOWED_FUTURE_MS:
        raise EpistulaError("Epistula-Timestamp is in the future")

    signed_for = get(H_SIGNED_FOR)
    if expected_signed_for is not None and signed_for is not None:
        if signed_for != expected_signed_for:
            raise EpistulaError("request is signed for a different recipient")

    # Accept-list check happens before signature verification: it is a dict
    # lookup versus a ~100us curve operation, so an unknown signer costs us
    # nothing. That ordering is the difference between an unauthenticated
    # flood being cheap and being expensive.
    if allowed_signers is not None and signed_by not in set(allowed_signers):
        raise EpistulaError("signer is not on the accept-list")

    message = message_for(body, request_uuid, timestamp_ms, signed_for)
    if not _verify(signed_by, message, signature):
        raise EpistulaError("signature mismatch")

    if replay_guard is not None:
        if not replay_guard.check_and_record(request_uuid, timestamp_ms, now_ms):
            raise EpistulaError("replayed Epistula-Uuid")

    return VerifiedRequest(
        signed_by=signed_by,
        signed_for=signed_for,
        uuid=request_uuid,
        timestamp_ms=timestamp_ms,
    )
