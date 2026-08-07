"""Miner-signed receipts.

A receipt is the miner's signed statement that it served a specific request
and produced a specific response. It is the primitive that makes the routing
decision in DESIGN.md §3 defensible: because the platform sits in the request
path, it *could* misreport miner performance to validators -- but it cannot
fabricate a receipt, because it does not hold miner hotkeys. The platform can
withhold or delay; it cannot invent. Validators audit platform aggregates
against raw receipts and the gap is bounded.

Two design points worth keeping in mind when editing this:

**Self-reported timings are never scored.** ``ttft_ms_self`` and
``total_ms_self`` are in the receipt because they are useful for a miner
debugging its own stack and for spotting network-versus-compute problems.
They are informational only. Every number that touches ``score`` is measured
by the observer. A miner that lies about its own latency gains nothing, so
it has no reason to, so the field stays honest and useful.

**The response hash covers the assembled body.** For a streaming response
that is the concatenation of the SSE ``data:`` payloads, not the raw wire
bytes -- proxies legitimately rewrite chunk boundaries and keep-alive
comments, and hashing the wire form would make honest receipts fail
verification through no fault of the miner.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import canonical_json
from .keys import Signer
from .keys import sign as _sign
from .keys import verify as _verify

RECEIPT_VERSION = "1"

#: Header names used when a receipt rides along with a non-streaming response.
H_RECEIPT = "X-Instant-Receipt"
H_RECEIPT_SIG = "X-Instant-Receipt-Sig"

#: SSE event name used when a receipt is delivered as the final stream event.
SSE_RECEIPT_EVENT = "receipt"


@dataclass(slots=True)
class Receipt:
    """What the miner asserts about one served request.

    All durations are integer milliseconds and all timestamps are integer
    milliseconds since the epoch -- see :mod:`instant.protocol.canonical` for
    why there are no floats anywhere in a signed payload.
    """

    version: str
    request_id: str
    miner_hotkey: str
    signer_of_request: str
    request_body_sha256: str
    response_body_sha256: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms_self: int
    total_ms_self: int
    attestation_id: str | None
    started_at_ms: int
    finished_at_ms: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Receipt:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            # Strict rather than lenient: an unrecognised field means the
            # sender is running a newer receipt version than we verify, and
            # silently ignoring it would mean we sign off on content we did
            # not check.
            raise ValueError(f"unknown receipt fields: {sorted(unknown)}")
        missing = known - set(payload)
        if missing:
            raise ValueError(f"missing receipt fields: {sorted(missing)}")
        return cls(**payload)


@dataclass(slots=True)
class SignedReceipt:
    receipt: Receipt
    signature: str

    def to_payload(self) -> dict[str, Any]:
        return {"receipt": self.receipt.to_payload(), "signature": self.signature}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SignedReceipt:
        return cls(
            receipt=Receipt.from_payload(payload["receipt"]),
            signature=payload["signature"],
        )


def hash_body(raw: bytes) -> str:
    """``sha256:<hex>`` over raw request or assembled response bytes."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build(
    signer: Signer,
    *,
    request_id: str,
    signer_of_request: str,
    request_body: bytes,
    response_body: bytes,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms_self: int,
    total_ms_self: int,
    started_at_ms: int,
    finished_at_ms: int,
    attestation_id: str | None = None,
) -> SignedReceipt:
    """Build and sign a receipt for one served request."""
    receipt = Receipt(
        version=RECEIPT_VERSION,
        request_id=request_id,
        miner_hotkey=signer.ss58_address,
        signer_of_request=signer_of_request,
        request_body_sha256=hash_body(request_body),
        response_body_sha256=hash_body(response_body),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        ttft_ms_self=int(ttft_ms_self),
        total_ms_self=int(total_ms_self),
        attestation_id=attestation_id,
        started_at_ms=int(started_at_ms),
        finished_at_ms=int(finished_at_ms),
    )
    signature = _sign(signer, canonical_json(receipt.to_payload()))
    return SignedReceipt(receipt=receipt, signature=signature)


def verify(
    signed: SignedReceipt,
    *,
    expected_miner_hotkey: str | None = None,
    request_body: bytes | None = None,
    response_body: bytes | None = None,
) -> None:
    """Verify a receipt, raising :class:`ReceiptError` on any failure.

    Pass ``request_body`` and ``response_body`` when you have them -- the
    signature alone proves the miner asserted these hashes, not that the
    hashes describe the traffic you actually saw. Checking both is what turns
    a receipt from a claim into evidence.
    """
    receipt = signed.receipt

    if receipt.version != RECEIPT_VERSION:
        raise ReceiptError(f"unsupported receipt version {receipt.version!r}")

    if expected_miner_hotkey is not None and receipt.miner_hotkey != expected_miner_hotkey:
        raise ReceiptError(
            f"receipt is from {receipt.miner_hotkey}, expected {expected_miner_hotkey}"
        )

    if not _verify(
        receipt.miner_hotkey, canonical_json(receipt.to_payload()), signed.signature
    ):
        raise ReceiptError("receipt signature does not verify")

    if request_body is not None and receipt.request_body_sha256 != hash_body(request_body):
        raise ReceiptError("receipt request hash does not match the request sent")

    if response_body is not None and receipt.response_body_sha256 != hash_body(response_body):
        raise ReceiptError("receipt response hash does not match the response received")

    if receipt.finished_at_ms < receipt.started_at_ms:
        raise ReceiptError("receipt finished before it started")


class ReceiptError(Exception):
    """A receipt failed verification."""


def merkle_root(receipts: list[SignedReceipt]) -> str:
    """Merkle root over a set of receipts, for ``/validator/v1/stats``.

    Lets a validator pull any subset of receipts and prove membership without
    downloading a whole window. Odd nodes are promoted rather than duplicated
    -- duplicating the last leaf is the classic CVE-2012-2459 shape and there
    is no reason to reimplement it.
    """
    if not receipts:
        return "sha256:" + hashlib.sha256(b"").hexdigest()

    level = [
        hashlib.sha256(canonical_json(r.to_payload())).digest() for r in receipts
    ]
    level.sort()  # order-independent, so platform and validator agree

    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.sha256(level[i] + level[i + 1]).digest())
        if len(level) % 2:
            nxt.append(level[-1])  # promote, do not duplicate
        level = nxt

    return "sha256:" + level[0].hex()
