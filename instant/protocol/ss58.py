"""Minimal SS58 address encoding/decoding.

Implemented here rather than pulled from ``substrate-interface`` so that the
protocol layer -- the part every component depends on and the part we want
covered by fast, hermetic tests -- has no heavy dependencies. The miner and
validator entrypoints depend on the Bittensor SDK; ``instant.protocol`` does
not, and should not start to.

Format: ``base58( prefix || pubkey || checksum )`` where the checksum is the
first 2 bytes of ``blake2b-512(b"SS58PRE" || prefix || pubkey)``.
Bittensor uses network prefix 42 (the generic Substrate prefix).
"""

from __future__ import annotations

import hashlib

import base58

SS58_PREFIX = b"SS58PRE"
BITTENSOR_SS58_FORMAT = 42


class InvalidSS58Address(ValueError):
    """The string is not a well-formed SS58 address for the expected network."""


def _checksum(payload: bytes) -> bytes:
    return hashlib.blake2b(SS58_PREFIX + payload, digest_size=64).digest()[:2]


def _encode_prefix(ss58_format: int) -> bytes:
    if 0 <= ss58_format <= 63:
        return bytes([ss58_format])
    if 64 <= ss58_format <= 16383:
        # Two-byte form: 0b01aaaaaa_bbcccccc reordering per the SS58 spec.
        low = ((ss58_format & 0b0000_0000_1111_1100) >> 2) | 0b0100_0000
        high = (ss58_format >> 8) | ((ss58_format & 0b0000_0000_0000_0011) << 6)
        return bytes([low, high])
    raise ValueError(f"unsupported ss58 format {ss58_format}")


def encode(public_key: bytes, ss58_format: int = BITTENSOR_SS58_FORMAT) -> str:
    """Encode a 32-byte sr25519/ed25519 public key as an SS58 address."""
    if len(public_key) != 32:
        raise ValueError(f"public key must be 32 bytes, got {len(public_key)}")
    body = _encode_prefix(ss58_format) + public_key
    return base58.b58encode(body + _checksum(body)).decode("ascii")


def decode(address: str, ss58_format: int = BITTENSOR_SS58_FORMAT) -> bytes:
    """Decode an SS58 address to its 32-byte public key.

    Verifies the checksum and the network prefix. A wrong-network address is
    rejected rather than silently accepted, because on Bittensor a
    coldkey-vs-hotkey or wrong-chain mix-up should fail loudly at the edge
    rather than three layers down as a signature mismatch.
    """
    try:
        raw = base58.b58decode(address)
    except Exception as exc:  # noqa: BLE001 - base58 raises bare ValueError
        raise InvalidSS58Address(f"not valid base58: {address!r}") from exc

    prefix = _encode_prefix(ss58_format)
    if not raw.startswith(prefix):
        raise InvalidSS58Address(
            f"address {address!r} is not for ss58 format {ss58_format}"
        )
    body, checksum = raw[:-2], raw[-2:]
    if len(body) != len(prefix) + 32:
        raise InvalidSS58Address(
            f"address {address!r} has {len(body) - len(prefix)} payload bytes, expected 32"
        )
    if _checksum(body) != checksum:
        raise InvalidSS58Address(f"checksum mismatch for {address!r}")
    return body[len(prefix):]


def is_valid(address: str, ss58_format: int = BITTENSOR_SS58_FORMAT) -> bool:
    """True if ``address`` decodes cleanly for the given network."""
    try:
        decode(address, ss58_format)
    except (InvalidSS58Address, ValueError):
        return False
    return True
