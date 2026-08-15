from pathlib import Path
from types import SimpleNamespace

import pytest

from instant_validator.config import Settings
from instant_validator.weights import BittensorWeightWriter, WeightWriteError


class FakeSubtensor:
    def __init__(
        self,
        *,
        mode=None,
        commit_reveal=False,
        owner="5Owner",
        owner_uid=238,
        version_key=62,
    ):
        self.substrate = self
        self.mode = {"Burn": ()} if mode is None else mode
        self.commit_reveal = commit_reveal
        self.owner = owner
        self.owner_uid = owner_uid
        self.version_key = version_key
        self.calls = []

    def get_chain_finalised_head(self):
        return "0x2710"

    def get_block_number(self, block_hash):
        return int(block_hash, 16)

    def query_subtensor(self, *, name, block, params):
        self.calls.append((name, block, params))
        return {
            "RecycleOrBurn": self.mode,
            "CommitRevealWeightsEnabled": self.commit_reveal,
            "WeightsVersionKey": self.version_key,
        }[name]

    def get_subnet_owner_hotkey(self, netuid, *, block):
        return self.owner

    def get_uid_for_hotkey_on_subnet(self, hotkey, netuid, *, block):
        return self.owner_uid if hotkey == self.owner else None


class FakeSdk:
    def __init__(self, result=(True, "finalized"), **chain_options):
        self.chain = FakeSubtensor(**chain_options)
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
            "version_key": 62,
            "wait_for_inclusion": True,
            "wait_for_finalization": True,
            "raise_error": False,
        }
    ]


def test_burn_plan_is_exact_at_one_finalized_block():
    sdk = FakeSdk()
    writer = BittensorWeightWriter(settings(), sdk=sdk, submitter=sdk.submit)

    plan = writer.full_burn_plan()

    assert plan == ({238: 65_535}, 62, 10_000)
    assert {call[1] for call in sdk.chain.calls} == {10_000}


@pytest.mark.parametrize(
    "options, message",
    [
        ({"mode": {"Recycle": ()}}, "Burn mode"),
        ({"commit_reveal": True}, "commit/reveal"),
        ({"owner_uid": None}, "registered UID"),
    ],
)
def test_burn_plan_fails_closed(options, message):
    sdk = FakeSdk(**options)
    writer = BittensorWeightWriter(settings(), sdk=sdk, submitter=sdk.submit)
    with pytest.raises(WeightWriteError, match=message):
        writer.full_burn_plan()
    assert sdk.submit_calls == []


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
