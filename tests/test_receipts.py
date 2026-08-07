"""Receipts, and the merkle root that makes a set of them auditable."""

import pytest

from instant.protocol import receipts
from instant.protocol.receipts import (
    Receipt,
    ReceiptError,
    SignedReceipt,
    hash_body,
    merkle_root,
)

REQ = b'{"model":"openai/gpt-oss-120b"}'
RESP = b'{"choices":[{"message":{"content":"hi"}}]}'


def _build(miner_key, platform_key, **over):
    kw = dict(
        request_id="req-1",
        signer_of_request=platform_key.ss58_address,
        request_body=REQ,
        response_body=RESP,
        prompt_tokens=12,
        completion_tokens=34,
        ttft_ms_self=41,
        total_ms_self=310,
        started_at_ms=1_780_000_000_000,
        finished_at_ms=1_780_000_000_310,
        attestation_id="sha256:" + "ab" * 32,
    )
    kw.update(over)
    return receipts.build(miner_key, **kw)


def test_round_trip(miner_key, platform_key):
    signed = _build(miner_key, platform_key)
    receipts.verify(
        signed,
        expected_miner_hotkey=miner_key.ss58_address,
        request_body=REQ,
        response_body=RESP,
    )


def test_payload_round_trip(miner_key, platform_key):
    signed = _build(miner_key, platform_key)
    again = SignedReceipt.from_payload(signed.to_payload())
    assert again == signed
    receipts.verify(again, expected_miner_hotkey=miner_key.ss58_address)


def test_hotkey_is_taken_from_the_signer(miner_key, platform_key):
    signed = _build(miner_key, platform_key)
    assert signed.receipt.miner_hotkey == miner_key.ss58_address


def test_wrong_miner_is_rejected(miner_key, platform_key, stranger_key):
    signed = _build(miner_key, platform_key)
    with pytest.raises(ReceiptError, match="expected"):
        receipts.verify(signed, expected_miner_hotkey=stranger_key.ss58_address)


def test_tampered_field_breaks_the_signature(miner_key, platform_key):
    signed = _build(miner_key, platform_key)
    signed.receipt.completion_tokens = 9_999
    with pytest.raises(ReceiptError, match="signature"):
        receipts.verify(signed, expected_miner_hotkey=miner_key.ss58_address)


def test_forged_signature_is_rejected(miner_key, platform_key, stranger_key):
    # Someone else signs a receipt claiming to be the miner. The hotkey in
    # the receipt is what the signature is checked against, so this fails.
    signed = _build(miner_key, platform_key)
    other = _build(stranger_key, platform_key)
    signed.signature = other.signature
    with pytest.raises(ReceiptError, match="signature"):
        receipts.verify(signed, expected_miner_hotkey=miner_key.ss58_address)


def test_request_hash_mismatch(miner_key, platform_key):
    signed = _build(miner_key, platform_key)
    with pytest.raises(ReceiptError, match="request hash"):
        receipts.verify(
            signed,
            expected_miner_hotkey=miner_key.ss58_address,
            request_body=b'{"model":"something-else"}',
        )


def test_response_hash_mismatch(miner_key, platform_key):
    # The attack this stops: miner serves garbage, signs a receipt for the
    # good response. Signature is valid; the evidence still fails.
    signed = _build(miner_key, platform_key)
    with pytest.raises(ReceiptError, match="response hash"):
        receipts.verify(
            signed,
            expected_miner_hotkey=miner_key.ss58_address,
            response_body=b"garbage",
        )


def test_negative_duration_is_rejected(miner_key, platform_key):
    signed = _build(
        miner_key,
        platform_key,
        started_at_ms=1_780_000_000_500,
        finished_at_ms=1_780_000_000_100,
    )
    with pytest.raises(ReceiptError, match="finished before"):
        receipts.verify(signed, expected_miner_hotkey=miner_key.ss58_address)


def test_unknown_version_is_rejected(miner_key, platform_key):
    signed = _build(miner_key, platform_key)
    signed.receipt.version = "99"
    with pytest.raises(ReceiptError, match="version"):
        receipts.verify(signed, expected_miner_hotkey=miner_key.ss58_address)


def test_unknown_fields_are_rejected():
    payload = {"version": "1", "surprise": 1}
    with pytest.raises(ValueError, match="unknown receipt fields"):
        Receipt.from_payload(payload)


def test_missing_fields_are_rejected():
    with pytest.raises(ValueError, match="missing receipt fields"):
        Receipt.from_payload({"version": "1"})


def test_hash_body_shape():
    h = hash_body(b"")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_attestation_id_may_be_absent(miner_key, platform_key):
    signed = _build(miner_key, platform_key, attestation_id=None)
    receipts.verify(signed, expected_miner_hotkey=miner_key.ss58_address)


# --- merkle ---------------------------------------------------------------


def test_merkle_root_is_order_independent(miner_key, platform_key):
    rs = [_build(miner_key, platform_key, request_id=f"req-{i}") for i in range(5)]
    assert merkle_root(rs) == merkle_root(list(reversed(rs)))


def test_merkle_root_changes_when_content_changes(miner_key, platform_key):
    rs = [_build(miner_key, platform_key, request_id=f"req-{i}") for i in range(5)]
    before = merkle_root(rs)
    rs.append(_build(miner_key, platform_key, request_id="req-5"))
    assert merkle_root(rs) != before


def test_merkle_root_of_empty_set():
    assert merkle_root([]).startswith("sha256:")


def test_merkle_root_single_receipt(miner_key, platform_key):
    assert merkle_root([_build(miner_key, platform_key)]).startswith("sha256:")


def test_odd_node_is_promoted_not_duplicated(miner_key, platform_key):
    # CVE-2012-2459: duplicating the last leaf makes distinct sets collide.
    # Three receipts and "three receipts where the last is repeated" must not
    # produce the same root.
    a = [_build(miner_key, platform_key, request_id=f"req-{i}") for i in range(3)]
    b = a + [a[-1]]
    assert merkle_root(a) != merkle_root(b)


def test_merkle_root_is_stable_across_calls(miner_key, platform_key):
    rs = [_build(miner_key, platform_key, request_id=f"req-{i}") for i in range(7)]
    assert merkle_root(rs) == merkle_root(rs)
