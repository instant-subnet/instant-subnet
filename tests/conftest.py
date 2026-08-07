"""Shared fixtures.

Deterministic keys throughout: a test that fails intermittently because it
generated a different key is a test people learn to ignore.
"""

import pytest

from instant.protocol.keys import LocalKeypair


def _kp(byte: int) -> LocalKeypair:
    return LocalKeypair(bytes([byte]) * 32)


@pytest.fixture
def miner_key() -> LocalKeypair:
    return _kp(0x11)


@pytest.fixture
def platform_key() -> LocalKeypair:
    return _kp(0x22)


@pytest.fixture
def validator_key() -> LocalKeypair:
    return _kp(0x33)


@pytest.fixture
def stranger_key() -> LocalKeypair:
    return _kp(0x44)


@pytest.fixture
def now_ms() -> int:
    # A fixed instant, so nothing in the suite depends on the wall clock.
    return 1_780_000_000_000
