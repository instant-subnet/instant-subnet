from pathlib import Path

import pytest

from instant_validator.config import ConfigError, Settings


def values(**overrides):
    base = {"INSTANT_PLATFORM_SIGNER": "5Signer"}
    base.update(overrides)
    return base


def test_production_defaults_are_finney_46_and_writes_are_off():
    settings = Settings.from_env(values(), load_env_file=False)

    assert settings.network == "finney"
    assert settings.netuid == 46
    assert settings.chain_endpoint == ""
    assert settings.chain_target == "finney"
    assert settings.enable_weight_writes is False


def test_private_environment_can_explicitly_select_another_chain():
    settings = Settings.from_env(
        values(
            INSTANT_NETWORK="private-test",
            INSTANT_NETUID="71",
            INSTANT_CHAIN_ENDPOINT="ws://chain.example:9944",
            INSTANT_ENABLE_WEIGHT_WRITES="true",
            INSTANT_STATE_PATH="/tmp/instant-validator-test.json",
        ),
        load_env_file=False,
    )

    assert settings.chain_target == "ws://chain.example:9944"
    assert settings.netuid == 71
    assert settings.enable_weight_writes is True
    assert settings.state_path == Path("/tmp/instant-validator-test.json")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"INSTANT_PLATFORM_SIGNER": ""}, "SIGNER is required"),
        ({"INSTANT_NETUID": "zero"}, "must be an integer"),
        ({"INSTANT_CHAIN_ENDPOINT": "http://chain.example"}, "must use ws"),
        ({"INSTANT_PLATFORM_REPORT_URL": "not-a-url"}, "HTTP"),
        ({"INSTANT_ENABLE_WEIGHT_WRITES": "sometimes"}, "true or false"),
        ({"INSTANT_POLL_INTERVAL_SECONDS": "1"}, "between 10"),
    ],
)
def test_invalid_configuration_fails_at_startup(override, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(values(**override), load_env_file=False)
