"""The sole Finney write boundary: one direct ``set_weights`` call."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import Settings


class WeightWriteError(RuntimeError):
    """Finney rejected or failed a weight submission."""


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

    def set_weights(self, weights: Mapping[int, int]) -> str:
        """Submit exactly one vector; never discover or contact miners."""

        if not weights:
            raise WeightWriteError("refusing to submit an empty weight vector")
        ordered = sorted((int(uid), int(weight)) for uid, weight in weights.items())
        try:
            ok, message = self.submitter(
                subtensor=self.subtensor,
                wallet=self.wallet,
                netuid=self.settings.netuid,
                mechid=0,
                uids=[uid for uid, _ in ordered],
                weights=[weight for _, weight in ordered],
                version_key=self.settings.weight_version_key,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                raise_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - chain SDK boundary
            raise WeightWriteError(f"set_weights raised: {exc}") from exc
        if not ok:
            raise WeightWriteError(f"set_weights failed: {message or 'unknown error'}")
        return str(message or "finalized")
