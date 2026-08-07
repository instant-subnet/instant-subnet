"""Miner entrypoint: ``python -m instant.miner``.

This is the only file in the miner that imports the Bittensor SDK. Everything
else takes a :class:`~instant.protocol.keys.Signer` and a set of hotkeys, so
the whole request path is testable without a wallet, a chain, or a network.

Startup order is deliberate:

1. Load and **validate** settings. If a development flag is set on mainnet,
   the process dies here — before a key is loaded, before a socket is opened,
   before anything is announced on chain.
2. Load the wallet and confirm the hotkey is registered on our netuid.
3. Sync the metagraph once, so the accept-list is populated before the
   listener starts. A miner that starts serving with an empty accept-list
   rejects every request and looks broken.
4. Announce the axon so validators can find us.
5. Serve, refreshing the metagraph in the background.

Steps 1–3 fail loudly. Step 5's refresh loop fails quietly and keeps serving
from the last good snapshot, because a chain hiccup should not take a
working miner offline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import uvicorn

from ..common.config import Settings, load_settings
from ..common.guards import UnsafeConfiguration, describe, enforce
from ..protocol.epistula import ReplayGuard
from .app import MinerContext, create_app
from .attest import (
    AttestationCache,
    HardwareAttestor,
    StubAttestor,
    image_digest,
    weights_digest,
)
from .auth import AcceptList, Authenticator
from .upstream import VllmClient

log = logging.getLogger("instant.miner")

def build_context(settings: Settings, wallet, metagraph) -> MinerContext:
    """Assemble the miner from validated settings and chain objects."""
    hotkey = wallet.hotkey
    repo = Path(__file__).resolve().parents[2]

    if settings.attestation_mode == "off":
        attestor = StubAttestor(
            hotkey_ss58=hotkey.ss58_address, model_id=settings.tier.model_id
        )
        wd = idg = ""
    else:
        attestor = HardwareAttestor(
            hotkey_ss58=hotkey.ss58_address,
            model_id=settings.tier.model_id,
            weights_lock=repo / "WEIGHTS.lock",
            images_lock=repo / "IMAGES.lock",
        )
        wd = weights_digest(repo / "WEIGHTS.lock")
        idg = image_digest(repo / "IMAGES.lock")

    accept = AcceptList(platform_hotkey=settings.platform_ss58 or None)
    accept.update(_validator_hotkeys(metagraph))

    return MinerContext(
        signer=hotkey,
        auth=Authenticator(
            my_hotkey=hotkey.ss58_address,
            accept=accept,
            replay=ReplayGuard(),
            enforce_accept_list=settings.network != "local"
            or bool(accept.all()),
        ),
        vllm=VllmClient(settings.vllm_url),
        attestation=AttestationCache(attestor),
        model_id=settings.tier.model_id,
        max_model_len=settings.model_max_len,
        max_concurrent=settings.max_concurrent,
        attestation_mode=settings.attestation_mode,
        weights_digest=wd,
        image_digest=idg,
    )

def _validator_hotkeys(metagraph) -> frozenset[str]:
    """Hotkeys holding a validator permit.

    Permit rather than stake threshold: the permit is the chain's own answer
    to "is this a validator", and reimplementing it with a stake cutoff means
    maintaining our own copy of a rule that already exists and can change.
    """
    out = set()
    for uid, permit in enumerate(metagraph.validator_permit):
        if permit:
            out.add(metagraph.hotkeys[uid])
    return frozenset(out)


async def _refresh_loop(
    ctx: MinerContext,
    subtensor,
    metagraph,
    netuid: int,
    refresh_s: int,
) -> None:
    while True:
        await asyncio.sleep(refresh_s)
        try:
            await asyncio.to_thread(metagraph.sync, subtensor=subtensor, lite=True)
            ctx.auth.accept.update(_validator_hotkeys(metagraph))
            log.debug(
                "metagraph refreshed: %d validators", len(ctx.auth.accept.validator_hotkeys)
            )
        except Exception as exc:  # noqa: BLE001 - never let this kill the miner
            # Deliberately broad. The refresh loop dying takes the accept-list
            # stale, which /health reports as degraded; the refresh loop
            # raising takes the whole process down, which is worse.
            log.warning("metagraph refresh failed (serving from last snapshot): %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="instant-miner")
    parser.add_argument(
        "--check", action="store_true",
        help="validate configuration and exit without starting the server",
    )
    parser.add_argument(
        "--check-chain",
        action="store_true",
        help="connect to the chain, verify registration, and exit",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    try:
        settings = load_settings()
        warnings = enforce(settings, role="miner")
    except UnsafeConfiguration as exc:
        log.error("refusing to start: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        log.error("configuration error: %s", exc)
        return 2

    logging.getLogger().setLevel(settings.log_level)
    log.info(describe(settings))
    for warning in warnings:
        log.warning(warning)

    if args.check:
        log.info("configuration is valid")
        return 0

    import bittensor as bt

    try:
        wallet = bt.wallet(
            name=settings.wallet_name,
            hotkey=settings.wallet_hotkey,
            path=settings.wallet_path,
        )
        subtensor = bt.subtensor(network=settings.chain_endpoint)
        metagraph = subtensor.metagraph(settings.netuid)
    except Exception as exc:  # noqa: BLE001
        log.error("chain connection failed: %s", exc)
        return 3

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        log.error(
            "hotkey %s is not registered on netuid %d (%s). Register it with "
            "`btcli subnet register` before starting the miner.",
            wallet.hotkey.ss58_address, settings.netuid, settings.chain_endpoint,
        )
        return 3

    if args.check_chain:
        log.info(
            "chain check passed: hotkey=%s netuid=%d neurons=%d",
            wallet.hotkey.ss58_address,
            settings.netuid,
            len(metagraph.hotkeys),
        )
        return 0

    ctx = build_context(settings, wallet, metagraph)
    log.info(
        "miner %s ready: %d validators on the accept-list",
        wallet.hotkey.ss58_address, len(ctx.auth.accept.validator_hotkeys),
    )

    # Announce where we are. Validators and the platform read this from the
    # metagraph rather than being told out of band.
    try:
        axon = bt.axon(
            wallet=wallet, port=settings.miner_port, external_port=settings.miner_port
        )
        axon.serve(netuid=settings.netuid, subtensor=subtensor)
        log.info("axon announced on port %d", settings.miner_port)
    except Exception as exc:  # noqa: BLE001
        log.error("failed to announce axon: %s", exc)
        return 4

    app = create_app(
        ctx,
        background=lambda: _refresh_loop(
            ctx,
            subtensor,
            metagraph,
            settings.netuid,
            settings.metagraph_refresh_s,
        ),
    )

    config = uvicorn.Config(
        app,
        host=settings.miner_host,
        port=settings.miner_port,
        log_level=settings.log_level.lower(),
        access_log=False,     # one log line per token stream is not useful
        timeout_keep_alive=75,
    )
    server = uvicorn.Server(config)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(server, "should_exit", True))

    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
