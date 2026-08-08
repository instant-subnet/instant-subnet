from __future__ import annotations

import sys
from types import SimpleNamespace

from instant.common.config import load_settings
from instant.miner import main as miner_main


def settings_for(miner_key, platform_key):
    return load_settings(
        env={
            "INSTANT_NETWORK": "local",
            "INSTANT_CHAIN_ENDPOINT": "ws://68.183.141.180:80",
            "INSTANT_MODEL_TIER": "mock",
            "INSTANT_ATTESTATION_MODE": "off",
            "INSTANT_MINER_EXTERNAL_IP": "165.227.197.158",
            "INSTANT_PLATFORM_SS58": platform_key.ss58_address,
            "INSTANT_WALLET_HOTKEY": "mock1",
        }
    )


def fake_bt(
    miner_key,
    *,
    registered=True,
    announced=True,
    coldkeypub_exists=True,
    coldkeypub_readable=True,
    coldkey_ss58=None,
    registered_owner=None,
):
    coldkey_ss58 = coldkey_ss58 or miner_key.ss58_address
    registered_owner = registered_owner or coldkey_ss58
    hotkeys = [miner_key.ss58_address] if registered else []
    metagraph = SimpleNamespace(
        hotkeys=hotkeys,
        coldkeys=[registered_owner] if registered else [],
        validator_permit=[False] * len(hotkeys),
    )

    class PublicKeyfile:
        path = "/home/instant/.bittensor/wallets/miner/coldkeypub.txt"

        @staticmethod
        def exists_on_device():
            return coldkeypub_exists

        @staticmethod
        def is_readable():
            return coldkeypub_readable

    class Wallet:
        hotkey = miner_key
        coldkeypub_file = PublicKeyfile()
        coldkeypub = SimpleNamespace(ss58_address=coldkey_ss58)

        @property
        def coldkey(self):
            raise AssertionError("the miner must never load a private coldkey")

    wallet = Wallet()

    class Subtensor:
        def __init__(self):
            self.serve_calls = []

        def metagraph(self, netuid):
            assert netuid == 5
            return metagraph

        def serve_axon(self, **kwargs):
            self.serve_calls.append(kwargs)
            return announced

    subtensor = Subtensor()
    module = SimpleNamespace()
    module.wallet = lambda **_: wallet
    module.subtensor = lambda **_: subtensor

    def axon(**kwargs):
        module.axon_kwargs = kwargs
        return SimpleNamespace(**kwargs)

    module.axon = axon
    return module, subtensor


def patch_runtime(monkeypatch, settings, bt):
    monkeypatch.setattr(miner_main, "load_settings", lambda: settings)
    monkeypatch.setattr(miner_main, "enforce", lambda *_args, **_kwargs: [])
    monkeypatch.setitem(sys.modules, "bittensor", bt)
    monkeypatch.setattr(miner_main.signal, "signal", lambda *_: None)
    ran = SimpleNamespace(value=False)

    class Server:
        def __init__(self, _config):
            self.should_exit = False

        def run(self):
            ran.value = True

    monkeypatch.setattr(miner_main.uvicorn, "Config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(miner_main.uvicorn, "Server", Server)
    return ran


def test_unregistered_hotkey_never_announces_or_binds(
    monkeypatch, miner_key, platform_key
):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(miner_key, registered=False)
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main([]) == 3
    assert subtensor.serve_calls == []
    assert ran.value is False


def test_chain_check_does_not_announce_or_bind(monkeypatch, miner_key, platform_key):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(miner_key)
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main(["--check-chain"]) == 0
    assert subtensor.serve_calls == []
    assert ran.value is False


def test_false_axon_announcement_is_fatal(monkeypatch, miner_key, platform_key):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(miner_key, announced=False)
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main([]) == 4
    assert len(subtensor.serve_calls) == 1
    assert ran.value is False


def test_missing_coldkeypub_fails_before_chain_check_or_axon(
    monkeypatch, caplog, miner_key, platform_key
):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(miner_key, coldkeypub_exists=False)
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main(["--check-chain"]) == 3
    assert subtensor.serve_calls == []
    assert ran.value is False
    message = caplog.text
    assert "coldkeypub.txt" in message
    assert "Deploy ONLY" in message
    assert "never deploy the private coldkey" in message


def test_unreadable_coldkeypub_is_actionable(
    monkeypatch, caplog, miner_key, platform_key
):
    settings = settings_for(miner_key, platform_key)
    bt, _ = fake_bt(miner_key, coldkeypub_readable=False)
    patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main(["--check-chain"]) == 3
    assert "not readable by the service user" in caplog.text


def test_coldkeypub_must_match_the_registered_owner(
    monkeypatch, caplog, miner_key, platform_key
):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(
        miner_key,
        coldkey_ss58=miner_key.ss58_address,
        registered_owner=platform_key.ss58_address,
    )
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main([]) == 3
    assert subtensor.serve_calls == []
    assert ran.value is False
    assert "does not own registered hotkey" in caplog.text
    assert platform_key.ss58_address in caplog.text


def test_coldkeypub_must_contain_a_valid_public_identity(
    monkeypatch, caplog, miner_key, platform_key
):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(miner_key, coldkey_ss58="not-an-ss58-address")
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main([]) == 3
    assert subtensor.serve_calls == []
    assert ran.value is False
    assert "not a valid Bittensor SS58 address" in caplog.text
    assert "never deploy the private coldkey" in caplog.text


def test_axon_uses_explicit_external_ip_before_server_starts(
    monkeypatch, miner_key, platform_key
):
    settings = settings_for(miner_key, platform_key)
    bt, subtensor = fake_bt(miner_key)
    ran = patch_runtime(monkeypatch, settings, bt)
    assert miner_main.main([]) == 0
    assert bt.axon_kwargs["external_ip"] == "165.227.197.158"
    assert bt.axon_kwargs["external_port"] == 8091
    assert subtensor.serve_calls[0]["netuid"] == 5
    assert ran.value is True


def test_validator_hotkeys_tolerates_a_partial_metagraph_snapshot():
    graph = SimpleNamespace(
        hotkeys=["validator-a"], validator_permit=[True, True, False]
    )
    assert miner_main._validator_hotkeys(graph) == frozenset({"validator-a"})
