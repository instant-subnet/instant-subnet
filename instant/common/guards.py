"""Boot-time refusals.

Every development affordance in this codebase is a thing that must never run
on mainnet: stubbed attestation, reused GPUs, unpinned weights, a CPU-only
tier. The mechanism that stops them is not code review and not a warning
log — it is this module, which raises before the process finishes starting.

The reasoning, stated once so it does not have to be re-argued in each
review: a dev flag that merely logs on mainnet is a dev flag that will
eventually run on mainnet. Warnings scroll. Somebody sets an env var to
debug a Tuesday incident, the incident ends, the var stays. Refusing to boot
is the only guard that survives contact with a real on-call rotation, and
the cost of it is one clear error message at deploy time.

Each rule below names what is wrong, what to set instead, and why it exists.
An error message that says "invalid configuration" teaches nobody anything
at 3am.
"""

from __future__ import annotations

from ..protocol.ss58 import is_valid
from .config import ConfigError, Settings


class UnsafeConfiguration(ConfigError):
    """A configuration that is legal in development and forbidden on mainnet."""


def enforce(settings: Settings, *, role: str = "miner") -> list[str]:
    """Validate ``settings`` for ``role``. Raises, or returns warnings.

    Called from both miner and validator entrypoints before any socket is
    opened and before any key is loaded. Returns the list of non-fatal
    warnings so the caller can log them at WARNING rather than swallowing
    them.
    """
    warnings: list[str] = []
    mainnet = settings.is_mainnet

    # --- attestation ----------------------------------------------------
    if settings.attestation_mode != "hard":
        if mainnet:
            raise UnsafeConfiguration(
                f"INSTANT_ATTESTATION_MODE={settings.attestation_mode!r} with "
                f"INSTANT_NETWORK=finney. Attestation is a hard gate on mainnet "
                f"(DESIGN.md §6.1) — it is the only thing that makes 'served "
                f"from a TEE' a claim rather than a marketing line. Unset the "
                f"variable, or run against local/test."
            )
        warnings.append(
            f"attestation_mode={settings.attestation_mode} — miners are not "
            f"being held to a TEE proof. Development only."
        )

    # --- GPU reuse ------------------------------------------------------
    if settings.allow_gpu_reuse:
        if mainnet:
            raise UnsafeConfiguration(
                "INSTANT_ALLOW_GPU_REUSE=true with INSTANT_NETWORK=finney. "
                "One GPU serving several registered hotkeys is the Sybil shape "
                "the uniqueness rule in DESIGN.md §8.4 exists to catch. The "
                "flag exists so that several dev miners can share one CC box; "
                "it has no legitimate use on mainnet."
            )
        warnings.append(
            "allow_gpu_reuse=true — duplicate GPU UUIDs will be logged, not "
            "zeroed. Development only."
        )

    # --- weight pinning -------------------------------------------------
    if settings.allow_unpinned_weights:
        if mainnet:
            raise UnsafeConfiguration(
                "INSTANT_ALLOW_UNPINNED_WEIGHTS=true with "
                "INSTANT_NETWORK=finney. Unpinned weights means a miner can "
                "serve any model it likes and still attest successfully, "
                "which makes the attestation prove the hardware and nothing "
                "about the model."
            )
        warnings.append("allow_unpinned_weights=true — WEIGHTS.lock is advisory.")

    # --- model tier -----------------------------------------------------
    if mainnet and not settings.tier.is_production:
        raise UnsafeConfiguration(
            f"INSTANT_MODEL_TIER={settings.tier.tier!r} "
            f"({settings.tier.model_id}) with INSTANT_NETWORK=finney. The "
            f"ladder in DESIGN.md §2.2.1 is for building; mainnet serves the "
            f"production model. Set INSTANT_MODEL_TIER=prod."
        )

    if settings.tier.cpu_ok and settings.attestation_mode == "hard":
        raise UnsafeConfiguration(
            f"model tier {settings.tier.tier!r} is CPU-only but "
            f"attestation_mode is 'hard'. A CPU-only tier cannot produce a "
            f"GPU attestation, so every request would be rejected. Set "
            f"INSTANT_ATTESTATION_MODE=off for tier {settings.tier.tier!r}."
        )

    # --- chain endpoint -------------------------------------------------
    if mainnet and settings.chain_endpoint.startswith("ws://"):
        raise UnsafeConfiguration(
            f"INSTANT_CHAIN_ENDPOINT={settings.chain_endpoint} is unencrypted "
            f"while INSTANT_NETWORK=finney. Use wss://."
        )

    if not mainnet and settings.chain_endpoint.startswith("wss://entrypoint-finney"):
        raise UnsafeConfiguration(
            f"INSTANT_NETWORK={settings.network} but the endpoint points at "
            f"finney. One of the two is wrong, and guessing which would be a "
            f"good way to set weights on mainnet by accident."
        )

    # --- role-specific --------------------------------------------------
    if role == "miner":
        if settings.max_concurrent < 1:
            raise ConfigError("INSTANT_MAX_CONCURRENT must be at least 1")
        if settings.platform_ss58 and not is_valid(settings.platform_ss58):
            raise ConfigError(
                "INSTANT_PLATFORM_SS58 must be a valid Bittensor SS58 address"
            )
        if mainnet and settings.platform_url.startswith("http://"):
            raise UnsafeConfiguration(
                "INSTANT_PLATFORM_URL is plaintext http on mainnet."
            )
    elif role == "validator":
        if settings.metagraph_refresh_s < 1:
            raise ConfigError("INSTANT_METAGRAPH_REFRESH_S must be at least 1")
        if mainnet and settings.netuid == 5:
            # netuid 5 is our localnet convention. On finney it is somebody
            # else's subnet, and setting weights there would be both useless
            # and rude.
            warnings.append(
                "netuid=5 on finney — confirm this is the Instant netuid and "
                "not the localnet default left in place."
            )
    elif role == "platform":
        if settings.request_timeout_s < 1:
            raise ConfigError("INSTANT_REQUEST_TIMEOUT_S must be at least 1")
        if settings.platform_host not in {"127.0.0.1", "::1", "localhost"}:
            raise UnsafeConfiguration(
                "The unauthenticated localnet platform must bind to loopback. "
                "Set INSTANT_PLATFORM_HOST=127.0.0.1 and expose only a protected "
                "nginx route or SSH tunnel."
            )
        if not settings.platform_miner_ss58:
            raise ConfigError(
                "INSTANT_PLATFORM_MINER_SS58 is required for the platform. "
                "Epistula binds each request to the target miner hotkey."
            )
        if not is_valid(settings.platform_miner_ss58):
            raise ConfigError(
                "INSTANT_PLATFORM_MINER_SS58 must be a valid Bittensor SS58 address"
            )
        if mainnet:
            raise UnsafeConfiguration(
                "The initial platform gateway intentionally has no user API-key "
                "or quota layer. It is a localnet plumbing service and refuses "
                "to start on finney."
            )
    else:
        raise ConfigError(f"unknown role {role!r}")

    return warnings


def describe(settings: Settings) -> str:
    """One-line startup banner. What is running, where, and how gated."""
    return (
        f"instant network={settings.network} netuid={settings.netuid} "
        f"tier={settings.tier.tier} model={settings.tier.model_id} "
        f"attestation={settings.attestation_mode} "
        f"chain={settings.chain_endpoint}"
    )
