#!/usr/bin/env python3
"""Register one hotkey on the Instant development localnet.

This deliberately supports only the known spec-393 development endpoint and
netuid 5.  It exists because newer btcli releases compose calls that this
older runtime does not expose, while the repository-pinned SDK is compatible.
The wallet password is read interactively by bittensor-wallet and is never
accepted as an argument or environment variable.
"""

from __future__ import annotations

import argparse

LOCAL_ENDPOINT = "ws://68.183.141.180:80"
LOCAL_NETUID = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet-name", required=True)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument(
        "--wallet-path",
        default="~/.bittensor/wallets",
    )
    args = parser.parse_args()

    # Bittensor 9.x still inspects argv at import time, so import only after
    # our small CLI has consumed its own arguments.
    import bittensor as bt

    wallet = bt.wallet(
        name=args.wallet_name,
        hotkey=args.hotkey,
        path=args.wallet_path,
    )
    subtensor = bt.subtensor(network=LOCAL_ENDPOINT)
    address = wallet.hotkey.ss58_address

    if subtensor.is_hotkey_registered_on_subnet(address, LOCAL_NETUID):
        uid = subtensor.get_uid_for_hotkey_on_subnet(address, LOCAL_NETUID)
        print(f"already registered: hotkey={address} uid={uid}")
        return 0

    print(
        f"registering hotkey={address} on netuid={LOCAL_NETUID} "
        f"at {LOCAL_ENDPOINT}; recycle={subtensor.recycle(LOCAL_NETUID)}"
    )
    ok = subtensor.burned_register(
        wallet=wallet,
        netuid=LOCAL_NETUID,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    if not ok:
        print("registration failed")
        return 1

    uid = subtensor.get_uid_for_hotkey_on_subnet(address, LOCAL_NETUID)
    print(f"registration confirmed: hotkey={address} uid={uid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
