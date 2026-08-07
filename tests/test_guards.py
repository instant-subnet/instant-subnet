"""Boot-time refusals.

Each test here corresponds to one way a development affordance could reach
mainnet. They assert that the process *raises* — not that it warns — because
the entire argument for this module (guards.py's own docstring) is that a
warning scrolls past and a refusal does not.

The paired assertion in most tests is that the same configuration is fine on
local. A guard that refuses everywhere is a guard someone will delete the
first time it blocks a dev box.
"""

from __future__ import annotations

import dataclasses

import pytest

from instant.common.config import ConfigError, Settings, load_settings, load_tier
from instant.common.guards import UnsafeConfiguration, describe, enforce
from instant.protocol.ss58 import encode

VALID_SS58 = encode(bytes(range(32)))


def settings(**overrides) -> Settings:
    """A safe local dev config, plus overrides.

    Built through ``load_settings`` rather than by hand so that a field added
    to Settings cannot leave this helper silently constructing something the
    loader would never produce.
    """
    base = load_settings(
        env={"INSTANT_NETWORK": "local", "INSTANT_CHAIN_ENDPOINT": "ws://10.0.0.5:9944"}
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def mainnet(**overrides) -> Settings:
    """A safe *mainnet* config, plus overrides — the baseline that must pass."""
    base = load_settings(
        env={
            "INSTANT_NETWORK": "finney",
            "INSTANT_MODEL_TIER": "prod",
            "INSTANT_NETUID": "77",
            "INSTANT_PLATFORM_URL": "https://platform.example",
        }
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- the baselines ----------------------------------------------------------


def test_a_correct_mainnet_config_passes_silently():
    # If this ever starts warning, the warnings below stop meaning anything.
    assert enforce(mainnet(), role="miner") == []
    assert enforce(mainnet(), role="validator") == []


def test_the_default_dev_config_warns_but_runs():
    warnings = enforce(settings(), role="miner")
    assert warnings, "dev0 runs with attestation off and must say so"
    assert any("attestation_mode=off" in w for w in warnings)


def test_every_refusal_is_catchable_as_a_config_error():
    # main.py catches UnsafeConfiguration and ConfigError separately; this is
    # the relationship that makes both paths exit 2 rather than traceback.
    assert issubclass(UnsafeConfiguration, ConfigError)


# --- attestation ------------------------------------------------------------

@pytest.mark.parametrize("mode", ["off", "warn"])
def test_soft_attestation_is_refused_on_mainnet(mode):
    with pytest.raises(UnsafeConfiguration) as exc:
        enforce(mainnet(attestation_mode=mode))
    message = str(exc.value)
    assert "INSTANT_ATTESTATION_MODE" in message
    assert "§6.1" in message  # the reader gets sent somewhere, not just refused


@pytest.mark.parametrize("mode", ["off", "warn"])
def test_soft_attestation_only_warns_off_mainnet(mode):
    warnings = enforce(settings(attestation_mode=mode))
    assert any(f"attestation_mode={mode}" in w for w in warnings)
    assert any("Development only" in w for w in warnings)


# --- Sybil affordances ------------------------------------------------------


def test_gpu_reuse_is_refused_on_mainnet():
    with pytest.raises(UnsafeConfiguration) as exc:
        enforce(mainnet(allow_gpu_reuse=True))
    assert "§8.4" in str(exc.value)


def test_gpu_reuse_warns_on_local():
    assert any("allow_gpu_reuse" in w for w in enforce(settings(allow_gpu_reuse=True)))


def test_unpinned_weights_is_refused_on_mainnet():
    with pytest.raises(UnsafeConfiguration) as exc:
        enforce(mainnet(allow_unpinned_weights=True))
    assert "attestation prove the hardware" in str(exc.value)


def test_unpinned_weights_warns_on_local():
    warnings = enforce(settings(allow_unpinned_weights=True))
    assert any("WEIGHTS.lock" in w for w in warnings)


# --- model tier -------------------------------------------------------------


@pytest.mark.parametrize("tier", ["dev0", "dev1"])
def test_a_dev_tier_cannot_run_on_mainnet(tier):
    with pytest.raises(UnsafeConfiguration) as exc:
        enforce(mainnet(tier=load_tier(tier), attestation_mode="hard"))
    message = str(exc.value)
    assert "INSTANT_MODEL_TIER=prod" in message
    assert load_tier(tier).model_id in message


def test_cpu_tier_with_hard_attestation_is_refused_everywhere():
    # Not a mainnet rule: the combination is simply impossible. Left alone it
    # produces a miner that boots, serves nothing, and rejects every request
    # with an attestation error.
    with pytest.raises(UnsafeConfiguration) as exc:
        enforce(settings(tier=load_tier("dev0"), attestation_mode="hard"))
    assert "cannot produce a GPU attestation" in str(exc.value)


def test_cpu_tier_with_attestation_off_is_the_supported_dev_shape():
    warnings = enforce(settings(tier=load_tier("dev0"), attestation_mode="off"))
    assert all("cannot produce" not in w for w in warnings)


# --- chain endpoint ---------------------------------------------------------


def test_plaintext_chain_endpoint_is_refused_on_mainnet():
    with pytest.raises(UnsafeConfiguration, match="unencrypted"):
        enforce(mainnet(chain_endpoint="ws://1.2.3.4:9944"))


def test_a_finney_endpoint_on_a_non_finney_network_is_refused():
    # The expensive accident: INSTANT_NETWORK=test left over from a script,
    # endpoint pointing at mainnet, weights set on somebody's live subnet.
    with pytest.raises(UnsafeConfiguration) as exc:
        enforce(
            settings(chain_endpoint="wss://entrypoint-finney.opentensor.ai:443")
        )
    assert "guessing which" in str(exc.value)


def test_our_own_localnet_websocket_is_fine():
    # ws:// to a droplet we run is the whole point of the local network.
    enforce(settings(chain_endpoint="ws://165.227.0.1:9944"))


# --- roles ------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1])
def test_miner_concurrency_below_one_is_rejected(value):
    with pytest.raises(ConfigError, match="INSTANT_MAX_CONCURRENT"):
        enforce(settings(max_concurrent=value), role="miner")


def test_concurrency_is_not_a_validator_concern():
    # The validator never serves inference, so the miner's limit must not
    # block it from starting.
    enforce(settings(max_concurrent=0), role="validator")


def test_plaintext_platform_url_is_refused_on_mainnet():
    with pytest.raises(UnsafeConfiguration, match="plaintext http"):
        enforce(mainnet(platform_url="http://api.instantsubnet.com"), role="miner")


def test_plaintext_platform_url_is_fine_locally():
    enforce(settings(platform_url="http://localhost:3000"), role="miner")


def test_miner_rejects_an_invalid_platform_address():
    with pytest.raises(ConfigError, match="INSTANT_PLATFORM_SS58"):
        enforce(settings(platform_ss58="not-an-ss58-address"), role="miner")


def test_validator_on_the_localnet_netuid_over_mainnet_is_flagged():
    warnings = enforce(mainnet(netuid=5), role="validator")
    assert any("netuid=5" in w for w in warnings)


def test_that_flag_is_a_warning_not_a_refusal():
    # It is genuinely ambiguous — netuid 5 could one day be ours. Refusing
    # would be a guess; warning is not.
    assert enforce(mainnet(netuid=5), role="validator") != []


def test_platform_requires_the_target_miner_hotkey():
    with pytest.raises(ConfigError, match="INSTANT_PLATFORM_MINER_SS58"):
        enforce(settings(), role="platform")


def test_initial_platform_must_bind_to_loopback():
    with pytest.raises(UnsafeConfiguration, match="bind to loopback"):
        enforce(
            settings(
                platform_host="0.0.0.0",
                platform_miner_ss58=VALID_SS58,
            ),
            role="platform",
        )


def test_platform_rejects_an_invalid_target_miner_hotkey():
    with pytest.raises(ConfigError, match="INSTANT_PLATFORM_MINER_SS58"):
        enforce(
            settings(platform_miner_ss58="not-an-ss58-address"), role="platform"
        )


def test_initial_platform_is_localnet_only():
    with pytest.raises(UnsafeConfiguration, match="localnet plumbing"):
        enforce(mainnet(platform_miner_ss58=VALID_SS58), role="platform")


def test_unknown_role_is_a_programming_error():
    with pytest.raises(ConfigError, match="unknown role"):
        enforce(settings(), role="observer")


# --- the banner -------------------------------------------------------------


def test_describe_names_everything_that_decides_behaviour():
    line = describe(mainnet())
    for expected in ("network=finney", "netuid=77", "tier=prod", "attestation=hard"):
        assert expected in line
    assert "openai/gpt-oss-120b" in line


def test_describe_leaks_no_wallet_material():
    line = describe(settings())
    assert "wallet" not in line.lower()
