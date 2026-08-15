"""Environment-only configuration for the validator service."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


class ConfigError(ValueError):
    """The validator configuration is missing or unsafe."""


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _boolean(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name, "true" if default else "false").strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise ConfigError(f"{name} must be true or false")


def _bounded(value: int, name: str, minimum: int, maximum: int) -> int:
    if value < minimum or value > maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """The complete configuration for one validator process."""

    network: str
    netuid: int
    chain_endpoint: str
    wallet_name: str
    wallet_hotkey: str
    wallet_path: Path
    platform_report_url: str
    platform_signer: str
    poll_interval_seconds: int
    report_max_age_seconds: int
    report_future_skew_seconds: int
    request_timeout_seconds: int
    state_path: Path
    enable_weight_writes: bool
    burn_miner_emissions: bool
    log_level: str

    @property
    def chain_target(self) -> str:
        """Use an explicit endpoint only when the operator supplied one."""

        return self.chain_endpoint or self.network

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
    ) -> Settings:
        """Load one strict configuration.

        Production defaults are Finney and subnet 46. A private ``.env`` may
        explicitly select a different network, netuid, or chain endpoint.
        """

        if values is None:
            if load_env_file:
                load_dotenv(".env", override=False)
            values = os.environ

        network = values.get("INSTANT_NETWORK", "finney").strip()
        if not network:
            raise ConfigError("INSTANT_NETWORK cannot be empty")

        netuid = _bounded(
            _integer(values, "INSTANT_NETUID", 46),
            "INSTANT_NETUID",
            1,
            65_535,
        )
        endpoint = values.get("INSTANT_CHAIN_ENDPOINT", "").strip()
        if endpoint and urlparse(endpoint).scheme not in {"ws", "wss"}:
            raise ConfigError("INSTANT_CHAIN_ENDPOINT must use ws:// or wss://")

        burn_miner_emissions = _boolean(values, "INSTANT_BURN_MINER_EMISSIONS", True)

        report_url = values.get(
            "INSTANT_PLATFORM_REPORT_URL",
            "https://api.instantsubnet.com/validator/v1/reports/latest",
        ).strip()
        parsed_report_url = urlparse(report_url)
        if not burn_miner_emissions and (
            parsed_report_url.scheme not in {"http", "https"}
            or not parsed_report_url.netloc
        ):
            raise ConfigError("INSTANT_PLATFORM_REPORT_URL must be an HTTP(S) URL")

        signer = values.get("INSTANT_PLATFORM_SIGNER", "").strip()
        if not burn_miner_emissions and not signer:
            raise ConfigError("INSTANT_PLATFORM_SIGNER is required")

        poll_interval = _bounded(
            _integer(values, "INSTANT_POLL_INTERVAL_SECONDS", 60),
            "INSTANT_POLL_INTERVAL_SECONDS",
            10,
            86_400,
        )
        max_age = _bounded(
            _integer(values, "INSTANT_REPORT_MAX_AGE_SECONDS", 86_400),
            "INSTANT_REPORT_MAX_AGE_SECONDS",
            60,
            604_800,
        )
        future_skew = _bounded(
            _integer(values, "INSTANT_REPORT_FUTURE_SKEW_SECONDS", 60),
            "INSTANT_REPORT_FUTURE_SKEW_SECONDS",
            0,
            300,
        )
        timeout = _bounded(
            _integer(values, "INSTANT_REQUEST_TIMEOUT_SECONDS", 15),
            "INSTANT_REQUEST_TIMEOUT_SECONDS",
            1,
            120,
        )
        level = values.get("INSTANT_LOG_LEVEL", "INFO").strip().upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("INSTANT_LOG_LEVEL is invalid")

        return cls(
            network=network,
            netuid=netuid,
            chain_endpoint=endpoint,
            wallet_name=values.get("INSTANT_WALLET_NAME", "validator").strip(),
            wallet_hotkey=values.get("INSTANT_WALLET_HOTKEY", "default").strip(),
            wallet_path=Path(
                values.get("INSTANT_WALLET_PATH", "~/.bittensor/wallets").strip()
            ).expanduser(),
            platform_report_url=report_url,
            platform_signer=signer,
            poll_interval_seconds=poll_interval,
            report_max_age_seconds=max_age,
            report_future_skew_seconds=future_skew,
            request_timeout_seconds=timeout,
            state_path=Path(
                values.get("INSTANT_STATE_PATH", "var/validator-state.json").strip()
            ).expanduser(),
            enable_weight_writes=_boolean(values, "INSTANT_ENABLE_WEIGHT_WRITES", False),
            burn_miner_emissions=burn_miner_emissions,
            log_level=level,
        )
