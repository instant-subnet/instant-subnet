"""Signing primitives, including every way verification is allowed to fail."""

import pytest

from instant.protocol.keys import LocalKeypair, Signer, sign, verify


def test_seed_must_be_32_bytes():
    with pytest.raises(ValueError, match="32 bytes"):
        LocalKeypair(b"\x00" * 16)


def test_deterministic_from_seed():
    a = LocalKeypair(b"\x07" * 32)
    b = LocalKeypair(b"\x07" * 32)
    assert a.ss58_address == b.ss58_address
    assert a.public_key == b.public_key


def test_from_hex_seed_accepts_0x_prefix():
    seed = "07" * 32
    assert (
        LocalKeypair.from_hex_seed(seed).ss58_address
        == LocalKeypair.from_hex_seed("0x" + seed).ss58_address
    )


def test_generate_produces_distinct_keys():
    assert LocalKeypair.generate().ss58_address != LocalKeypair.generate().ss58_address


def test_sign_and_verify(miner_key):
    sig = sign(miner_key, "hello")
    assert sig.startswith("0x")
    assert verify(miner_key.ss58_address, "hello", sig) is True


def test_bytes_and_str_messages_agree(miner_key):
    assert sign(miner_key, "hello") != ""
    sig = sign(miner_key, b"hello")
    assert verify(miner_key.ss58_address, "hello", sig) is True


def test_wrong_signer_fails(miner_key, stranger_key):
    sig = sign(stranger_key, "hello")
    assert verify(miner_key.ss58_address, "hello", sig) is False


def test_tampered_message_fails(miner_key):
    sig = sign(miner_key, "hello")
    assert verify(miner_key.ss58_address, "hello!", sig) is False


def test_verify_returns_false_never_raises(miner_key):
    # Every one of these is something a hostile client can send to a public
    # endpoint. None may raise -- an exception path here is a DoS shape.
    good = sign(miner_key, "hello")
    for addr, msg, sig in [
        ("not-an-address", "hello", good),
        ("", "hello", good),
        (miner_key.ss58_address, "hello", "0xzzzz"),
        (miner_key.ss58_address, "hello", "0x00"),
        (miner_key.ss58_address, "hello", ""),
        (miner_key.ss58_address, "hello", b"\x00" * 63),
        (miner_key.ss58_address, "hello", "0x" + "00" * 64),
    ]:
        assert verify(addr, msg, sig) is False


def test_local_keypair_satisfies_signer_protocol(miner_key):
    assert isinstance(miner_key, Signer)


def test_repr_shows_address(miner_key):
    assert miner_key.ss58_address in repr(miner_key)
