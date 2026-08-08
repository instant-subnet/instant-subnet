#!/usr/bin/env python3
"""Stake one registered validator hotkey on the Instant development localnet.

The command is intentionally pinned to the known spec-393 endpoint and netuid 5.
It performs one bounded ``add_stake`` call, waits for finalization, and never
accepts a wallet password, mnemonic, or seed as an argument or environment
variable. Bittensor Wallet reads the encrypted coldkey password interactively.
"""

from __future__ import annotations

import argparse
import math

LOCAL_ENDPOINT = "ws://68.183.141.180:80"
LOCAL_NETUID = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet-name", required=True)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets")
    parser.add_argument("--amount-tao", required=True, type=float)
    args = parser.parse_args()

    if not math.isfinite(args.amount_tao) or args.amount_tao <= 0:
        parser.error("--amount-tao must be a finite value greater than zero")

    # Bittensor 9.x still inspects argv at import time, so import only after
    # this narrow CLI has consumed its arguments.
    import bittensor as bt

    wallet = bt.wallet(
        name=args.wallet_name,
        hotkey=args.hotkey,
        path=args.wallet_path,
    )
    subtensor = bt.subtensor(network=LOCAL_ENDPOINT)
    hotkey = wallet.hotkey.ss58_address

    if not subtensor.is_hotkey_registered_on_subnet(hotkey, LOCAL_NETUID):
        print(f"refusing: hotkey={hotkey} is not registered on netuid={LOCAL_NETUID}")
        return 1

    graph = subtensor.metagraph(LOCAL_NETUID, lite=True)
    uid = list(graph.hotkeys).index(hotkey)
    stake_before = float(graph.S[uid])
    print(
        f"staking {args.amount_tao:g} local TAO to hotkey={hotkey} "
        f"uid={uid} netuid={LOCAL_NETUID}; stake_before={stake_before:g}"
    )
    ok = subtensor.add_stake(
        wallet=wallet,
        hotkey_ss58=hotkey,
        netuid=LOCAL_NETUID,
        amount=bt.Balance.from_tao(args.amount_tao),
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=False,
    )
    if not ok:
        print("stake transaction failed")
        return 1

    refreshed = subtensor.metagraph(LOCAL_NETUID, lite=True)
    refreshed_uid = list(refreshed.hotkeys).index(hotkey)
    stake_after = float(refreshed.S[refreshed_uid])
    permit = bool(refreshed.validator_permit[refreshed_uid])
    print(
        f"stake confirmed: hotkey={hotkey} uid={refreshed_uid} "
        f"stake_after={stake_after:g} validator_permit={permit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
