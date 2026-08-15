from pathlib import Path
from types import SimpleNamespace

import pytest

from instant_validator.config import Settings
from instant_validator.weights import BittensorWeightWriter, WeightWriteError


class FakeSubtensor:
    pass


class FakeSdk:
    def __init__(self, result=(True, "finalized")):
        self.chain = FakeSubtensor()
        self.result = result
        self.submit_calls = []
        self.wallet_calls = []
        self.chain_calls = []

    def wallet(self, **kwargs):
        self.wallet_calls.append(kwargs)
        return SimpleNamespace(name="wallet")

    def subtensor(self, **kwargs):
        self.chain_calls.append(kwargs)
        return self.chain

    def submit(self, **kwargs):
        self.submit_calls.append(kwargs)
        return self.result


def settings():
    return Settings.from_env(
        {
            "INSTANT_PLATFORM_SIGNER": "5Signer",
            "INSTANT_ENABLE_WEIGHT_WRITES": "true",
            "INSTANT_WALLET_NAME": "owner",
            "INSTANT_WALLET_HOTKEY": "validator",
            "INSTANT_WALLET_PATH": "/wallets",
        },
        load_env_file=False,
    )


def test_writer_makes_one_direct_finney_set_weights_call():
    sdk = FakeSdk()
    writer = BittensorWeightWriter(settings(), sdk=sdk, submitter=sdk.submit)

    message = writer.set_weights({37: 58_296, 12: 65_535})

    assert message == "finalized"
    assert sdk.chain_calls == [{"network": "finney"}]
    assert sdk.wallet_calls == [
        {"name": "owner", "hotkey": "validator", "path": str(Path("/wallets"))}
    ]
    assert sdk.submit_calls == [
        {
            "subtensor": sdk.chain,
            "wallet": writer.wallet,
            "netuid": 46,
            "mechid": 0,
            "uids": [12, 37],
            "weights": [65_535, 58_296],
            "version_key": 0,
            "wait_for_inclusion": True,
            "wait_for_finalization": True,
            "raise_error": False,
        }
    ]


def test_failed_write_is_reported_without_retry():
    sdk = FakeSdk((False, "rate limited"))
    writer = BittensorWeightWriter(settings(), sdk=sdk, submitter=sdk.submit)
    with pytest.raises(WeightWriteError, match="rate limited"):
        writer.set_weights({12: 65_535})
    assert len(sdk.submit_calls) == 1


def test_empty_vector_never_calls_the_chain():
    sdk = FakeSdk()
    writer = BittensorWeightWriter(settings(), sdk=sdk, submitter=sdk.submit)
    with pytest.raises(WeightWriteError, match="empty"):
        writer.set_weights({})
    assert sdk.submit_calls == []
