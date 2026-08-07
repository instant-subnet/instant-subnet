"""SS58 is hand-rolled here, so it is pinned against published vectors."""

import pytest

from instant.protocol import ss58

# The canonical Substrate development key. If this ever fails, the
# implementation is wrong -- not the vector.
ALICE_PUBKEY = bytes.fromhex(
    "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"
)
ALICE_SS58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def test_alice_vector():
    assert ss58.encode(ALICE_PUBKEY) == ALICE_SS58
    assert ss58.decode(ALICE_SS58) == ALICE_PUBKEY


def test_round_trip_random_keys():
    import secrets

    for _ in range(50):
        pk = secrets.token_bytes(32)
        assert ss58.decode(ss58.encode(pk)) == pk


def test_wrong_length_public_key():
    with pytest.raises(ValueError, match="32 bytes"):
        ss58.encode(b"\x00" * 31)


def test_checksum_tampering_is_caught():
    bad = ALICE_SS58[:-1] + ("A" if ALICE_SS58[-1] != "A" else "B")
    with pytest.raises(ss58.InvalidSS58Address):
        ss58.decode(bad)
    assert ss58.is_valid(bad) is False


def test_not_base58():
    with pytest.raises(ss58.InvalidSS58Address, match="base58"):
        ss58.decode("not-an-address-0OIl")


def test_wrong_network_prefix_is_rejected():
    # Encode for Kusama (2) and try to decode as Bittensor (42).
    kusama = ss58.encode(ALICE_PUBKEY, ss58_format=2)
    with pytest.raises(ss58.InvalidSS58Address, match="ss58 format"):
        ss58.decode(kusama, ss58_format=42)
    assert ss58.decode(kusama, ss58_format=2) == ALICE_PUBKEY


def test_is_valid_never_raises():
    for junk in ["", "x", "0" * 100, ALICE_SS58 + "z"]:
        assert ss58.is_valid(junk) is False
    assert ss58.is_valid(ALICE_SS58) is True


def test_bittensor_format_is_42():
    assert ss58.BITTENSOR_SS58_FORMAT == 42
