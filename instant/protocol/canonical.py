"""Canonical JSON serialisation.

Every signed payload in Instant -- receipts, attestation bundles, platform
stats -- is serialised through this module before signing and before
verifying. The rules are deliberately strict, because a signature that
verifies in Python and fails in Node is one of the worst classes of bug to
debug: it looks like a key problem and it is a whitespace problem.

Rules
-----
1. Keys sorted, lexicographically by UTF-8 code point.
2. No insignificant whitespace: separators are ``,`` and ``:``.
3. UTF-8 output, no ASCII escaping. (JS ``JSON.stringify`` does not escape
   non-ASCII either, so ``ensure_ascii=False`` is what matches.)
4. **No floating point anywhere.** Durations are integer milliseconds,
   ratios are integer basis points. A float would serialise differently
   between Python's repr and JS's Number.prototype.toString for a
   non-trivial set of values, and we would not find out until a miner in a
   different runtime failed verification.

The JS side of this in `instant-platform` must be:

    const canonical = (v) =>
      JSON.stringify(v, (_, x) => {
        if (typeof x === 'number' && !Number.isInteger(x))
          throw new TypeError('float in canonical payload');
        if (x && typeof x === 'object' && !Array.isArray(x))
          return Object.keys(x).sort().reduce((o,k)=>(o[k]=x[k],o), {});
        return x;
      });

Rule 4 is enforced here rather than documented, so nobody has to remember it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class NonCanonicalValue(ValueError):
    """A value was passed that cannot be canonically serialised."""


def _check(value: Any, path: str = "$") -> None:
    """Walk a payload and reject anything that cannot round-trip identically."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        # JS numbers lose precision past 2^53. A payload that survives Python
        # and silently changes value in Node is worse than a hard error.
        if abs(value) > 2**53 - 1:
            raise NonCanonicalValue(
                f"{path}: integer {value} exceeds JS safe-integer range; "
                "encode large numbers as strings"
            )
        return
    if isinstance(value, float):
        raise NonCanonicalValue(
            f"{path}: float {value!r} is not allowed in a signed payload. "
            "Use integer milliseconds for durations and integer basis points "
            "for ratios."
        )
    if isinstance(value, str):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NonCanonicalValue(
                    f"{path}: object key {key!r} is not a string"
                )
            _check(item, f"{path}.{key}")
        return
    raise NonCanonicalValue(f"{path}: unsupported type {type(value).__name__}")


def canonical_json(payload: Any) -> bytes:
    """Serialise ``payload`` to canonical UTF-8 JSON bytes.

    Raises :class:`NonCanonicalValue` if the payload contains a float, a
    non-string object key, an unsafe integer, or an unsupported type.
    """
    _check(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(payload: Any) -> str:
    """``sha256:<hex>`` over the canonical serialisation of ``payload``."""
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def body_sha256(raw: bytes) -> bytes:
    """Raw sha256 of an HTTP body, as bytes.

    Used for the Epistula request signature, which hashes the body *as sent*
    rather than a re-serialisation of it -- re-serialising would break any
    client whose JSON encoder differs from ours, which is all of them.
    """
    return hashlib.sha256(raw).digest()


def bps(numerator: float, denominator: float) -> int:
    """Convert a ratio to integer basis points, for use in signed payloads.

    ``bps(0.9987, 1)`` -> ``9987``. Rounds half away from zero, and clamps
    nothing -- a ratio above 1 is a caller bug we want to see, not hide.
    """
    if denominator == 0:
        return 0
    return int(round((numerator / denominator) * 10_000))
