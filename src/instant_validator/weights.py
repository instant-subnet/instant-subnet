"""The sole Finney write boundary: one direct, finalized weight call."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import Settings

FULL_WEIGHT = 65_535


class WeightWriteError(RuntimeError):
    """Finney rejected or failed a weight submission."""


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


class BittensorWeightWriter:
    """Own the validator wallet, chain connection, and direct weight write."""

    def __init__(
        self,
        settings: Settings,
        *,
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

        self.settings = settings
        self.submitter = submitter
        self.wallet = sdk.wallet(
            name=settings.wallet_name,
            hotkey=settings.wallet_hotkey,
            path=str(settings.wallet_path),
        )
        self.subtensor = sdk.subtensor(network=settings.chain_target)

    def _finalized_block(self) -> int:
        block_hash = self.subtensor.substrate.get_chain_finalised_head()
        block = self.subtensor.substrate.get_block_number(block_hash)
        if type(block) is not int or block < 0:
            raise WeightWriteError("finalized block lookup returned an invalid block")
        return block

    def _query(self, name: str, block: int) -> Any:
        return _value(
            self.subtensor.query_subtensor(
                name=name, block=block, params=[self.settings.netuid]
            )
        )

    def _version_key(self, block: int) -> int:
        value = self._query("WeightsVersionKey", block)
        if type(value) is not int or value < 0:
            raise WeightWriteError("on-chain WeightsVersionKey is invalid")
        return value

    def full_burn_plan(self) -> tuple[dict[int, int], int, int]:
        """Return the fixed 100% owner vote and version at one finalized block."""

        block = self._finalized_block()
        if self._query("RecycleOrBurn", block) != {"Burn": ()}:
            raise WeightWriteError("refusing burn write: subnet is not in Burn mode")
        if self._query("CommitRevealWeightsEnabled", block) is not False:
            raise WeightWriteError("refusing direct burn write: commit/reveal is enabled")
        owner = _value(
            self.subtensor.get_subnet_owner_hotkey(self.settings.netuid, block=block)
        )
        if not isinstance(owner, str) or not owner:
            raise WeightWriteError("refusing burn write: subnet owner hotkey is invalid")
        uid = _value(
            self.subtensor.get_uid_for_hotkey_on_subnet(
                owner, self.settings.netuid, block=block
            )
        )
        if type(uid) is not int or uid < 0:
            raise WeightWriteError(
                "refusing burn write: subnet owner has no registered UID"
            )
        return {uid: FULL_WEIGHT}, self._version_key(block), block

    def set_weights(
        self,
        weights: Mapping[int, int],
        *,
        version_key: int | None = None,
    ) -> str:
        """Submit exactly one vector; never discover or contact miners."""

        if not weights:
            raise WeightWriteError("refusing to submit an empty weight vector")
        ordered = sorted((int(uid), int(weight)) for uid, weight in weights.items())
        if version_key is None:
            version_key = self._version_key(self._finalized_block())
        try:
            ok, message = self.submitter(
                subtensor=self.subtensor,
                wallet=self.wallet,
                netuid=self.settings.netuid,
                mechid=0,
                uids=[uid for uid, _ in ordered],
                weights=[weight for _, weight in ordered],
                version_key=version_key,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                raise_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - chain SDK boundary
            raise WeightWriteError(f"set_weights raised: {exc}") from exc
        if not ok:
            raise WeightWriteError(f"set_weights failed: {message or 'unknown error'}")
        return str(message or "finalized")
