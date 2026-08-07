"""Settings, loaded from the environment and ``config/*.toml``.

One object, built once at boot, validated once at boot. Nothing in the
codebase reads ``os.environ`` outside this module — a config value that can
be read from three places is a config value that will eventually disagree
with itself.

The tier indirection matters more than it looks. ``INSTANT_MODEL_TIER``
selects a row from ``config/models.toml``, and everything downstream — the
model id, the context length, the minimum VRAM, whether attestation is even
possible — comes from that row. Changing what we run is one env var, and
there is no model id hardcoded anywhere else in the repo. See DESIGN.md
§2.2.1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:  # pragma: no cover - stdlib on 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("INSTANT_CONFIG_DIR", REPO_ROOT / "config"))

Network = Literal["local", "test", "finney"]

#: Chain endpoints per network. ``local`` has no default on purpose: it is a
#: box we run, its address is deployment-specific, and a wrong default that
#: silently connects somewhere is worse than a missing one that fails loudly.
DEFAULT_ENDPOINTS: dict[str, str] = {
    "test": "wss://test.finney.opentensor.ai:443",
    "finney": "wss://entrypoint-finney.opentensor.ai:443",
}


class ConfigError(Exception):
    """Configuration is invalid. Raised at boot, never mid-request."""


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(
        f"{name}={raw!r} is not a boolean. Use true/false — a typo that "
        f"silently reads as false is how a safety flag gets disabled."
    )


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc


def load_toml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"missing config file {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


@dataclass(frozen=True, slots=True)
class ModelTier:
    """One row of ``config/models.toml``."""

    tier: str
    model_id: str
    max_model_len: int
    min_vram_gb: int
    cpu_ok: bool
    attestation: str
    served_by: str

    @property
    def is_production(self) -> bool:
        return self.tier == "prod"


def load_tier(tier: str) -> ModelTier:
    table = load_toml("models.toml")
    if tier not in table:
        raise ConfigError(
            f"unknown model tier {tier!r}; config/models.toml has "
            f"{sorted(table)}"
        )
    row = table[tier]
    missing = {
        "model_id", "max_model_len", "min_vram_gb", "cpu_ok", "attestation",
        "served_by",
    } - set(row)
    if missing:
        raise ConfigError(f"models.toml [{tier}] is missing {sorted(missing)}")
    return ModelTier(tier=tier, **row)


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything a miner or validator needs to start.

    Built by :func:`load_settings`, validated by
    :func:`instant.common.guards.enforce`. Frozen because config that changes
    at runtime is config you cannot reason about from a log line.
    """

    network: Network
    chain_endpoint: str
    netuid: int
    tier: ModelTier
    model_max_len: int

    wallet_name: str
    wallet_hotkey: str
    wallet_path: str

    # Miner
    miner_host: str
    miner_port: int
    vllm_url: str
    max_concurrent: int
    platform_url: str
    platform_ss58: str

    # Validator
    validator_host: str
    validator_port: int
    state_db: str
    metagraph_refresh_s: int

    # Platform gateway. The first plumbing release uses one explicitly
    # configured miner; metagraph-based routing is the next layer.
    platform_host: str
    platform_port: int
    platform_miner_url: str
    platform_miner_ss58: str
    request_timeout_s: int

    # Safety flags. Every one of these is checked against ``network`` in
    # guards.enforce, and every one of them exits the process rather than
    # warning when the combination is wrong.
    attestation_mode: str
    allow_gpu_reuse: bool
    allow_unpinned_weights: bool

    log_level: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mainnet(self) -> bool:
        return self.network == "finney"


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Read settings from the process environment.

    ``env`` is for tests. Production always uses ``os.environ``, so that
    there is exactly one path through this function that anyone has to
    reason about.
    """
    if env is not None:
        old = dict(os.environ)
        os.environ.clear()
        os.environ.update(env)
        try:
            return load_settings()
        finally:
            os.environ.clear()
            os.environ.update(old)

    network = _env("INSTANT_NETWORK", "local")
    if network not in ("local", "test", "finney"):
        raise ConfigError(
            f"INSTANT_NETWORK={network!r} must be one of local, test, finney"
        )

    endpoint = _env("INSTANT_CHAIN_ENDPOINT") or DEFAULT_ENDPOINTS.get(network)
    if not endpoint:
        raise ConfigError(
            "INSTANT_CHAIN_ENDPOINT is required when INSTANT_NETWORK=local. "
            "It is the websocket address of our own subtensor box, e.g. "
            "ws://<droplet-ip>:80 — there is no sensible default."
        )

    tier_name = _env("INSTANT_MODEL_TIER", "dev0")
    tier = load_tier(tier_name)
    model_max_len = _env_int("INSTANT_MAX_MODEL_LEN", tier.max_model_len)
    if not 1 <= model_max_len <= tier.max_model_len:
        raise ConfigError(
            f"INSTANT_MAX_MODEL_LEN={model_max_len} must be between 1 and "
            f"the {tier_name} tier ceiling ({tier.max_model_len})"
        )

    attestation_mode = _env("INSTANT_ATTESTATION_MODE") or tier.attestation
    if attestation_mode not in ("hard", "warn", "off"):
        raise ConfigError(
            f"INSTANT_ATTESTATION_MODE={attestation_mode!r} must be hard, warn or off"
        )

    return Settings(
        network=network,  # type: ignore[arg-type]
        chain_endpoint=endpoint,
        netuid=_env_int("INSTANT_NETUID", 5),
        tier=tier,
        model_max_len=model_max_len,
        wallet_name=_env("INSTANT_WALLET_NAME", "default"),
        wallet_hotkey=_env("INSTANT_WALLET_HOTKEY", "default"),
        wallet_path=_env("INSTANT_WALLET_PATH", "~/.bittensor/wallets"),
        miner_host=_env("INSTANT_MINER_HOST", "0.0.0.0"),
        miner_port=_env_int("INSTANT_MINER_PORT", 8091),
        vllm_url=_env("INSTANT_VLLM_URL", "http://127.0.0.1:8000"),
        max_concurrent=_env_int("INSTANT_MAX_CONCURRENT", 16),
        platform_url=_env("INSTANT_PLATFORM_URL", "http://127.0.0.1:8090"),
        platform_ss58=_env("INSTANT_PLATFORM_SS58", ""),
        validator_host=_env("INSTANT_VALIDATOR_HOST", "127.0.0.1"),
        validator_port=_env_int("INSTANT_VALIDATOR_PORT", 8092),
        state_db=_env("INSTANT_STATE_DB", "./validator_state.sqlite3"),
        metagraph_refresh_s=_env_int("INSTANT_METAGRAPH_REFRESH_S", 120),
        platform_host=_env("INSTANT_PLATFORM_HOST", "127.0.0.1"),
        platform_port=_env_int("INSTANT_PLATFORM_PORT", 8090),
        platform_miner_url=_env(
            "INSTANT_PLATFORM_MINER_URL", "http://127.0.0.1:8091"
        ),
        platform_miner_ss58=_env("INSTANT_PLATFORM_MINER_SS58", ""),
        request_timeout_s=_env_int("INSTANT_REQUEST_TIMEOUT_S", 30),
        attestation_mode=attestation_mode,
        allow_gpu_reuse=_env_bool("INSTANT_ALLOW_GPU_REUSE", False),
        allow_unpinned_weights=_env_bool("INSTANT_ALLOW_UNPINNED_WEIGHTS", False),
        log_level=_env("INSTANT_LOG_LEVEL", "INFO"),
    )
