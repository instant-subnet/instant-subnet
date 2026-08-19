"""The only Phase 4 chain write: 100% to the subnet owner UID."""

from __future__ import annotations

from typing import Any

FULL_WEIGHT = 65_535


class BurnError(RuntimeError):
    """The authorized burn vector could not be submitted safely."""


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


class BittensorBurnWriter:
    def __init__(
        self,
        *,
        network: str,
        endpoint: str,
        wallet_name: str,
        wallet_hotkey: str,
        wallet_path: str,
        sdk: Any = None,
        submitter: Any = None,
    ) -> None:
        if sdk is None:
            import bittensor as sdk

        if submitter is None:
            from bittensor.core.extrinsics.mechanism import (
                set_mechanism_weights_extrinsic,
            )

            submitter = set_mechanism_weights_extrinsic
        self.subtensor = sdk.subtensor(network=endpoint or network)
        self.wallet = sdk.wallet(
            name=wallet_name,
            hotkey=wallet_hotkey,
            path=wallet_path,
        )
        self.submitter = submitter

    @property
    def hotkey(self) -> str:
        value = self.wallet.hotkey.ss58_address
        if not isinstance(value, str) or not value:
            raise BurnError("Validator hotkey is invalid")
        return value

    def _query(self, name: str, netuid: int, block: int) -> Any:
        return _value(
            self.subtensor.query_subtensor(name=name, block=block, params=[netuid])
        )

    def submit(self, *, netuid: int, finalized_block: int) -> str:
        if self._query("RecycleOrBurn", netuid, finalized_block) != {"Burn": ()}:
            raise BurnError("Subnet is not in Burn mode")
        if self._query("CommitRevealWeightsEnabled", netuid, finalized_block) is not False:
            raise BurnError("Commit/reveal is enabled")
        owner = _value(
            self.subtensor.get_subnet_owner_hotkey(netuid, block=finalized_block)
        )
        if not isinstance(owner, str) or not owner:
            raise BurnError("Subnet owner hotkey is invalid")
        uid = _value(
            self.subtensor.get_uid_for_hotkey_on_subnet(
                owner, netuid, block=finalized_block
            )
        )
        version = self._query("WeightsVersionKey", netuid, finalized_block)
        if type(uid) is not int or uid < 0 or type(version) is not int or version < 0:
            raise BurnError("Subnet owner UID or weights version is invalid")
        try:
            ok, message = self.submitter(
                subtensor=self.subtensor,
                wallet=self.wallet,
                netuid=netuid,
                mechid=0,
                uids=[uid],
                weights=[FULL_WEIGHT],
                version_key=version,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                raise_error=False,
            )
        except Exception as exc:
            raise BurnError(f"Burn write raised: {exc}") from exc
        if not ok:
            raise BurnError(f"Burn write failed: {message or 'unknown error'}")
        return str(message or "finalized")
