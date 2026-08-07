"""Signing and verification primitives.

Two things live here:

``Signer`` -- a structural protocol describing everything the rest of the
codebase needs from a key. The Bittensor SDK's ``Keypair`` already satisfies
it, so the miner and validator can hand their real wallet hotkey straight to
``instant.protocol`` without an adapter, and tests can hand it a
``LocalKeypair`` without importing the SDK at all.

``LocalKeypair`` -- a dependency-light sr25519 keypair used by tests, by
``scripts/``, and by anything that needs to sign without a wallet on disk.
It is deliberately *not* used by the miner or validator at runtime: those
load keys through the Bittensor wallet so that key handling, permissions,
and the coldkey/hotkey split stay in one well-audited place.
"""

from __future__ import annotations

import secrets
from typing import Protocol, runtime_checkable

import sr25519

from . import ss58


@runtime_checkable
class Signer(Protocol):
    """The subset of a keypair this codebase uses.

    ``bittensor.Keypair`` satisfies this structurally: it exposes
    ``ss58_address``, ``public_key`` and ``sign(data) -> bytes``.
    """

    ss58_address: str

    def sign(self, data: bytes) -> bytes:  # pragma: no cover - protocol
        ...


class LocalKeypair:
    """An sr25519 keypair backed by a 32-byte seed."""

    __slots__ = ("_pair", "public_key", "ss58_address")

    def __init__(self, seed: bytes, ss58_format: int = ss58.BITTENSOR_SS58_FORMAT):
        if len(seed) != 32:
            raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
        self._pair = sr25519.pair_from_seed(seed)
        self.public_key: bytes = self._pair[0]
        self.ss58_address: str = ss58.encode(self.public_key, ss58_format)

    @classmethod
    def generate(cls) -> LocalKeypair:
        return cls(secrets.token_bytes(32))

    @classmethod
    def from_hex_seed(cls, seed_hex: str) -> LocalKeypair:
        return cls(bytes.fromhex(seed_hex.removeprefix("0x")))

    def sign(self, data: bytes) -> bytes:
        return sr25519.sign(self._pair, data)

    def __repr__(self) -> str:
        return f"LocalKeypair({self.ss58_address})"


def sign(signer: Signer, message: str | bytes) -> str:
    """Sign ``message`` and return a ``0x``-prefixed hex signature."""
    data = message.encode("utf-8") if isinstance(message, str) else message
    return "0x" + signer.sign(data).hex()


def verify(address: str, message: str | bytes, signature: str | bytes) -> bool:
    """Verify a signature against an SS58 address.

    Returns ``False`` for every failure mode -- bad address, malformed
    signature, wrong length, mismatch -- rather than raising. Callers are
    request handlers on a public endpoint; an exception path there is a
    denial-of-service shape we do not need.
    """
    try:
        public_key = ss58.decode(address)
    except (ss58.InvalidSS58Address, ValueError):
        return False

    if isinstance(signature, str):
        try:
            sig = bytes.fromhex(signature.removeprefix("0x"))
        except ValueError:
            return False
    else:
        sig = signature
    if len(sig) != 64:
        return False

    data = message.encode("utf-8") if isinstance(message, str) else message
    try:
        return sr25519.verify(sig, data, public_key)
    except Exception:  # noqa: BLE001 - bindings raise on malformed input
        return False
