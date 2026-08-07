"""Settings loading.

Most of these tests are about failure. ``load_settings`` is the only place
in the codebase that reads the environment, so it is the only place that can
turn a typo into a silently wrong runtime — and the two cases that matter
most are a boolean that doesn't parse (a safety flag reading as false) and a
missing chain endpoint (connecting to the wrong chain). Both are asserted on
here with the error text, not just the exception type, because the message
is the whole point of failing at boot.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from instant.common.config import (
    DEFAULT_ENDPOINTS,
    ConfigError,
    ModelTier,
    Settings,
    load_settings,
    load_tier,
    load_toml,
)

LOCAL_WS = "ws://10.0.0.5:9944"


def env(**overrides: str) -> dict[str, str]:
    """A minimally valid local environment, plus overrides.

    ``INSTANT_CHAIN_ENDPOINT`` is in the base because local has no default
    and every test that isn't about that would otherwise fail for the wrong
    reason.
    """
    base = {"INSTANT_NETWORK": "local", "INSTANT_CHAIN_ENDPOINT": LOCAL_WS}
    base.update(overrides)
    return base


# --- the environment swap ---------------------------------------------------


def test_env_argument_does_not_leak_into_the_process():
    os.environ["INSTANT_TEST_CANARY"] = "still here"
    try:
        load_settings(env=env())
        assert os.environ.get("INSTANT_TEST_CANARY") == "still here"
        assert "INSTANT_CHAIN_ENDPOINT" not in os.environ
    finally:
        os.environ.pop("INSTANT_TEST_CANARY", None)


def test_env_is_restored_even_when_loading_raises():
    os.environ["INSTANT_TEST_CANARY"] = "still here"
    try:
        with pytest.raises(ConfigError):
            load_settings(env={"INSTANT_NETWORK": "mainnet"})
        assert os.environ.get("INSTANT_TEST_CANARY") == "still here"
    finally:
        os.environ.pop("INSTANT_TEST_CANARY", None)


# --- network and endpoint ---------------------------------------------------


def test_local_requires_an_explicit_endpoint():
    with pytest.raises(ConfigError) as exc:
        load_settings(env={"INSTANT_NETWORK": "local"})
    message = str(exc.value)
    assert "INSTANT_CHAIN_ENDPOINT is required" in message
    # The message has to say what the value looks like; "required" alone
    # sends the reader to the source.
    assert "ws://" in message


def test_local_endpoint_is_taken_verbatim():
    settings = load_settings(env=env())
    assert settings.chain_endpoint == LOCAL_WS
    assert settings.network == "local"
    assert settings.is_mainnet is False


@pytest.mark.parametrize("network", ["test", "finney"])
def test_public_networks_have_defaults(network):
    settings = load_settings(
        env={
            "INSTANT_NETWORK": network,
            "INSTANT_MODEL_TIER": "prod",
            "INSTANT_ATTESTATION_MODE": "hard",
        }
    )
    assert settings.chain_endpoint == DEFAULT_ENDPOINTS[network]


def test_explicit_endpoint_beats_the_default():
    settings = load_settings(
        env={"INSTANT_NETWORK": "test", "INSTANT_CHAIN_ENDPOINT": LOCAL_WS}
    )
    assert settings.chain_endpoint == LOCAL_WS


def test_local_has_no_default_endpoint():
    # Asserted directly, because the absence is deliberate: a default that
    # silently points somewhere is worse than one that fails loudly.
    assert "local" not in DEFAULT_ENDPOINTS


def test_unknown_network_is_rejected():
    with pytest.raises(ConfigError, match="local, test, finney"):
        load_settings(env={"INSTANT_NETWORK": "mainnet"})


def test_network_defaults_to_local():
    with pytest.raises(ConfigError, match="INSTANT_NETWORK=local"):
        load_settings(env={})


# --- scalars ----------------------------------------------------------------


def test_defaults_are_the_documented_ones():
    s = load_settings(env=env())
    assert s.netuid == 5
    assert s.tier.tier == "dev0"
    assert s.model_max_len == 8192
    assert s.wallet_name == "default"
    assert s.wallet_hotkey == "default"
    assert s.miner_host == "0.0.0.0"
    assert s.miner_port == 8091
    assert s.validator_host == "127.0.0.1"
    assert s.validator_port == 8092
    assert s.platform_port == 8090
    assert s.vllm_url == "http://127.0.0.1:8000"
    assert s.max_concurrent == 16
    assert s.log_level == "INFO"
    assert s.allow_gpu_reuse is False
    assert s.allow_unpinned_weights is False


def test_empty_string_reads_as_unset():
    # Shell scripts export empty variables constantly. Treating "" as a value
    # would give a miner an empty wallet name and a confusing failure three
    # steps later.
    s = load_settings(env=env(INSTANT_WALLET_NAME="", INSTANT_MINER_PORT=""))
    assert s.wallet_name == "default"
    assert s.miner_port == 8091


def test_integers_parse():
    s = load_settings(
        env=env(
            INSTANT_NETUID=" 42 ",
            INSTANT_MAX_CONCURRENT="1",
            INSTANT_MAX_MODEL_LEN="4096",
        )
    )
    assert s.netuid == 42
    assert s.max_concurrent == 1
    assert s.model_max_len == 4096


def test_non_integer_is_rejected_by_name():
    with pytest.raises(ConfigError, match="INSTANT_NETUID='two'"):
        load_settings(env=env(INSTANT_NETUID="two"))


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_truthy_booleans(raw):
    assert load_settings(env=env(INSTANT_ALLOW_GPU_REUSE=raw)).allow_gpu_reuse is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "No", "off", ""])
def test_falsy_booleans(raw):
    assert load_settings(env=env(INSTANT_ALLOW_GPU_REUSE=raw)).allow_gpu_reuse is False


@pytest.mark.parametrize("raw", ["truee", "y", "2", "enabled", "null"])
def test_a_boolean_typo_refuses_rather_than_reading_as_false(raw):
    # The failure mode this exists for: INSTANT_ALLOW_UNPINNED_WEIGHTS=yess
    # quietly disabling a safety flag nobody notices for a month.
    with pytest.raises(ConfigError) as exc:
        load_settings(env=env(INSTANT_ALLOW_UNPINNED_WEIGHTS=raw))
    assert "is not a boolean" in str(exc.value)
    assert "INSTANT_ALLOW_UNPINNED_WEIGHTS" in str(exc.value)


# --- tiers ------------------------------------------------------------------


def test_tier_selects_the_model():
    s = load_settings(env=env(INSTANT_MODEL_TIER="dev1"))
    assert s.tier.model_id == "openai/gpt-oss-20b"
    assert s.tier.max_model_len == 131072
    assert s.tier.cpu_ok is False
    assert s.tier.is_production is False


def test_runtime_context_can_be_lower_than_the_tier_ceiling():
    s = load_settings(
        env=env(INSTANT_MODEL_TIER="dev1", INSTANT_MAX_MODEL_LEN="8192")
    )
    assert s.model_max_len == 8192
    assert s.tier.max_model_len == 131072


@pytest.mark.parametrize("value", ["0", "131073"])
def test_runtime_context_must_fit_the_tier(value):
    with pytest.raises(ConfigError, match="tier ceiling"):
        load_settings(
            env=env(INSTANT_MODEL_TIER="dev1", INSTANT_MAX_MODEL_LEN=value)
        )


def test_prod_tier_is_the_production_model():
    tier = load_tier("prod")
    assert tier.model_id == "openai/gpt-oss-120b"
    assert tier.is_production is True
    assert tier.min_vram_gb == 80


def test_unknown_tier_lists_the_real_ones():
    with pytest.raises(ConfigError) as exc:
        load_tier("dev2")
    message = str(exc.value)
    assert "unknown model tier 'dev2'" in message
    for known in ("dev0", "dev1", "prod"):
        assert known in message


def test_every_tier_in_the_shipped_file_loads():
    # Guards the file itself: a tier added to models.toml without every key
    # fails here rather than at a miner's boot.
    for name in load_toml("models.toml"):
        assert isinstance(load_tier(name), ModelTier)


def test_tier_missing_a_key_is_named(tmp_path, monkeypatch):
    import instant.common.config as config

    (tmp_path / "models.toml").write_text(
        '[dev0]\nmodel_id = "x"\nmax_model_len = 1\n'
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    with pytest.raises(ConfigError) as exc:
        config.load_tier("dev0")
    message = str(exc.value)
    for missing in ("attestation", "cpu_ok", "min_vram_gb", "served_by"):
        assert missing in message


def test_missing_config_file_names_the_path(tmp_path, monkeypatch):
    import instant.common.config as config

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    with pytest.raises(ConfigError, match="missing config file"):
        config.load_toml("models.toml")


# --- attestation mode -------------------------------------------------------


def test_attestation_mode_defaults_to_the_tier():
    assert load_settings(env=env()).attestation_mode == "off"
    assert load_settings(env=env(INSTANT_MODEL_TIER="dev1")).attestation_mode == "hard"


def test_attestation_mode_can_be_overridden():
    s = load_settings(env=env(INSTANT_MODEL_TIER="dev1", INSTANT_ATTESTATION_MODE="warn"))
    assert s.attestation_mode == "warn"


def test_unknown_attestation_mode_is_rejected():
    with pytest.raises(ConfigError, match="hard, warn or off"):
        load_settings(env=env(INSTANT_ATTESTATION_MODE="soft"))


# --- the object itself ------------------------------------------------------


def test_settings_are_frozen():
    # Config that mutates at runtime is config a log line cannot describe.
    s = load_settings(env=env())
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.network = "finney"  # type: ignore[misc]


def test_settings_carries_no_secrets():
    # Nothing in Settings should be a key or a token; the wallet is loaded
    # from disk by name. If this ever fails, a log line just leaked.
    fields = set(Settings.__dataclass_fields__)
    assert not {f for f in fields if "secret" in f or "seed" in f or "key" in f} - {
        "wallet_name", "wallet_hotkey",
    }
