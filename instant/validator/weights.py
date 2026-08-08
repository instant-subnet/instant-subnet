"""Fail-closed, one-attempt localnet weight submission.

This module intentionally does not call ``Subtensor.set_weights``.  In the
pinned Bittensor 9.12.2 SDK that helper can loop forever when an extrinsic is
returned as unsuccessful without raising.  We compose the spec-393
``set_mechanism_weights`` call directly and invoke ``sign_and_send_extrinsic``
exactly once, signed by the validator hotkey.

The writer has no scheduler and is never called by validator startup.  An
operator must select ``--set-weights-once`` *and* set the localnet-only
``INSTANT_ENABLE_WEIGHT_WRITES=true`` safety flag.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from instant.protocol.canonical import digest
from instant.validator.score import MAX_WEIGHT_U16

from .state import ValidatorState


def _now_ms() -> int:
    return int(time.time() * 1000)


def _plain(value: Any) -> Any:
    value = getattr(value, "value", value)
    item = getattr(value, "item", None)
    return item() if callable(item) else value


class WeightSafetyError(RuntimeError):
    """A preflight or readback gate refused a chain write."""


@dataclass(frozen=True, slots=True)
class WeightSubmission:
    epoch: int
    block: int
    ok: bool
    attempts: int
    vector_digest: str
    uids: tuple[int, ...]
    weights: tuple[int, ...]
    message: str
    reconciled: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "block": self.block,
            "ok": self.ok,
            "attempts": self.attempts,
            "vector_digest": self.vector_digest,
            "uids": list(self.uids),
            "weights": list(self.weights),
            "message": self.message,
            "reconciled": self.reconciled,
        }


class WeightWriter:
    """Perform all preflights, one extrinsic attempt, and chain readback."""

    def __init__(
        self,
        *,
        subtensor: Any,
        wallet: Any,
        state: ValidatorState,
        network: str,
        netuid: int,
        enabled: bool,
        expected_spec_version: int = 393,
        mechanism_id: int = 0,
        version_key: int = 0,
        period_blocks: int = 8,
        config_version: int = 1,
    ) -> None:
        self.subtensor = subtensor
        self.wallet = wallet
        self.state = state
        self.network = network
        self.netuid = netuid
        self.enabled = enabled
        self.expected_spec_version = expected_spec_version
        self.mechanism_id = mechanism_id
        self.version_key = version_key
        self.period_blocks = period_blocks
        self.config_version = config_version

    def submit_latest_once(self) -> WeightSubmission:
        """Submit the latest persisted score vector, never more than once."""

        if not self.enabled:
            raise WeightSafetyError(
                "weight writes are disabled; set INSTANT_ENABLE_WEIGHT_WRITES=true "
                "only for an operator-reviewed localnet one-shot"
            )
        if self.network != "local":
            raise WeightSafetyError("weight writes in this skeleton are localnet-only")

        epoch = self.state.latest_epoch()
        if epoch is None:
            raise WeightSafetyError("no scored epoch exists; run --score-once first")
        prior = self.state.weight_set_for_epoch(epoch)
        if prior is not None:
            raise WeightSafetyError(
                f"epoch {epoch} already has a recorded weight attempt; refusing a "
                "duplicate submission"
            )

        scores = self.state.scores_for_epoch(epoch)
        vector = sorted(
            (
                (int(row["uid"]), str(row["hotkey"]), int(row["weight_u16"]))
                for row in scores
                if int(row["weight_u16"]) > 0
            ),
            key=lambda item: item[0],
        )
        if not vector:
            raise WeightSafetyError(
                f"epoch {epoch} has an all-zero vector; refusing to submit"
            )
        if sum(weight for _, _, weight in vector) != MAX_WEIGHT_U16:
            raise WeightSafetyError(
                "score vector does not sum to the canonical u16 total 65535"
            )
        uids = tuple(uid for uid, _, _ in vector)
        weights = tuple(weight for _, _, weight in vector)
        if len(set(uids)) != len(uids):
            raise WeightSafetyError("score vector contains duplicate UIDs")
        vector_digest = digest(
            {"netuid": self.netuid, "mecid": self.mechanism_id,
             "uids": list(uids), "weights": list(weights),
             "version_key": self.version_key}
        )

        graph = self.subtensor.metagraph(self.netuid, lite=True)
        hotkeys = [str(value) for value in graph.hotkeys]
        own_hotkey = self.wallet.hotkey.ss58_address
        if own_hotkey not in hotkeys:
            raise WeightSafetyError("validator hotkey is not registered on this netuid")
        own_uid = hotkeys.index(own_hotkey)
        permits = list(getattr(graph, "validator_permit", []))
        if own_uid >= len(permits) or not bool(_plain(permits[own_uid])):
            raise WeightSafetyError("validator does not hold a current validator permit")
        stake = self._stake_for(graph, own_uid)
        if stake <= 0:
            raise WeightSafetyError("validator has no stake on this subnet")
        if own_uid in uids:
            raise WeightSafetyError("score vector assigns weight to the validator itself")
        for uid, expected_hotkey, _ in vector:
            if uid >= len(hotkeys) or hotkeys[uid] != expected_hotkey:
                raise WeightSafetyError(
                    f"uid/hotkey mapping changed since scoring for uid {uid}"
                )

        block = int(self.subtensor.get_current_block())
        spec_version = self._runtime_spec_version()
        if spec_version != self.expected_spec_version:
            raise WeightSafetyError(
                f"runtime specVersion is {spec_version}, expected "
                f"{self.expected_spec_version}; call shape is not trusted"
            )

        commit_reveal = self._hyperparameter("CommitRevealWeightsEnabled")
        if bool(commit_reveal):
            raise WeightSafetyError(
                "commit-reveal weights are enabled; one-shot direct writer is incompatible"
            )
        chain_version_key = int(self._hyperparameter("WeightsVersionKey"))
        if chain_version_key != self.version_key:
            raise WeightSafetyError(
                f"WeightsVersionKey is {chain_version_key}, configured "
                f"{self.version_key}"
            )
        minimum = int(self._hyperparameter("MinAllowedWeights"))
        if len(uids) < minimum:
            raise WeightSafetyError(
                f"vector has {len(uids)} nonzero weights; chain requires {minimum}"
            )
        maximum = self._weight_limit_u16(
            self._hyperparameter("MaxWeightsLimit")
        )
        if max(weights) > maximum:
            raise WeightSafetyError(
                f"vector maximum {max(weights)} exceeds chain limit {maximum}"
            )

        last_updates = list(getattr(graph, "last_update", []))
        if own_uid >= len(last_updates):
            raise WeightSafetyError("metagraph omitted validator LastUpdate")
        last_update = int(_plain(last_updates[own_uid]))

        # Crash reconciliation happens before rate-limit refusal, but only in
        # the mortal-era window of a very recent update. An exact vector can
        # legitimately remain unchanged for many epochs; treating any old
        # match as a completed write would never refresh LastUpdate and would
        # eventually make the validator inactive.
        recently_updated = 0 <= block - last_update <= self.period_blocks
        if recently_updated and self._readback(own_uid) == dict(
            zip(uids, weights, strict=True)
        ):
            result = WeightSubmission(
                epoch=epoch,
                block=block,
                ok=True,
                attempts=0,
                vector_digest=vector_digest,
                uids=uids,
                weights=weights,
                message="exact vector already on chain; reconciled without submission",
                reconciled=True,
            )
            self._record(result, config_version=self.config_version)
            return result

        rate_limit = int(self._hyperparameter("WeightsSetRateLimit"))
        if last_update > 0 and block - last_update < rate_limit:
            raise WeightSafetyError(
                f"weight rate limit: {block - last_update} blocks elapsed, "
                f"{rate_limit} required"
            )

        # Metadata composition is still preflight.  If spec-393 no longer has
        # this exact call/field shape, compose_call raises before any signing
        # or submission occurs.
        try:
            call = self.subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="set_mechanism_weights",
                call_params={
                    "netuid": self.netuid,
                    "mecid": self.mechanism_id,
                    "dests": list(uids),
                    "weights": list(weights),
                    "version_key": self.version_key,
                },
            )
        except Exception as exc:  # noqa: BLE001 - metadata boundary
            raise WeightSafetyError(f"cannot compose trusted weight call: {exc}") from exc

        # Exactly one call; no loop and no SDK high-level set_weights helper.
        try:
            ok, message = self.subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=self.wallet,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                sign_with="hotkey",
                use_nonce=True,
                nonce_key="hotkey",
                period=self.period_blocks,
                raise_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - chain transport boundary
            ok, message = False, f"extrinsic raised: {exc}"

        try:
            final_block = int(self.subtensor.get_current_block())
        except Exception:  # noqa: BLE001 - preserve the known pre-submit block
            final_block = block
        if ok:
            try:
                expected = dict(zip(uids, weights, strict=True))
                actual = self._readback(own_uid)
                if actual != expected:
                    ok = False
                    message = (
                        f"finalized but readback differs: expected {expected}, "
                        f"got {actual}"
                    )
                else:
                    refreshed = self.subtensor.metagraph(self.netuid, lite=True)
                    refreshed_updates = list(getattr(refreshed, "last_update", []))
                    new_last = (
                        int(_plain(refreshed_updates[own_uid]))
                        if own_uid < len(refreshed_updates)
                        else last_update
                    )
                    if new_last <= last_update:
                        ok = False
                        message = (
                            "finalized and weights read back, but LastUpdate did "
                            "not advance"
                        )
            except Exception as exc:  # noqa: BLE001 - record post-write uncertainty
                ok = False
                message = f"finalized but readback failed: {exc}"

        result = WeightSubmission(
            epoch=epoch,
            block=final_block,
            ok=bool(ok),
            attempts=1,
            vector_digest=vector_digest,
            uids=uids,
            weights=weights,
            message=str(message or ("finalized and verified" if ok else "rejected")),
        )
        self._record(result, config_version=self.config_version)
        return result

    def _runtime_spec_version(self) -> int:
        substrate = self.subtensor.substrate
        head = substrate.get_chain_head()
        info = substrate.get_block_runtime_version(head)
        if not isinstance(info, dict) or "specVersion" not in info:
            raise WeightSafetyError("chain did not return a runtime specVersion")
        return int(info["specVersion"])

    def _hyperparameter(self, name: str) -> Any:
        value = self.subtensor.get_hyperparameter(name, self.netuid)
        value = _plain(value)
        if value is None:
            raise WeightSafetyError(f"chain did not return {name}")
        return value

    @staticmethod
    def _weight_limit_u16(value: Any) -> int:
        value = _plain(value)
        if isinstance(value, float):
            if not 0 <= value <= 1:
                raise WeightSafetyError("MaxWeightsLimit float is outside [0,1]")
            return int(value * MAX_WEIGHT_U16)
        return int(value)

    @staticmethod
    def _stake_for(graph: Any, uid: int) -> float:
        for name in ("S", "stake", "total_stake"):
            values = getattr(graph, name, None)
            if values is not None and uid < len(values):
                return float(_plain(values[uid]))
        raise WeightSafetyError("metagraph omitted validator stake")

    def _readback(self, own_uid: int) -> dict[int, int]:
        rows: Sequence[Any] = self.subtensor.weights(
            self.netuid, mechid=self.mechanism_id
        )
        for raw_source, raw_pairs in rows:
            if int(_plain(raw_source)) != own_uid:
                continue
            return {
                int(_plain(uid)): int(_plain(weight))
                for uid, weight in raw_pairs
                if int(_plain(weight)) > 0
            }
        return {}

    def _record(self, result: WeightSubmission, *, config_version: int) -> None:
        self.state.record_weight_set(
            epoch=result.epoch,
            block=result.block,
            submitted_ms=_now_ms(),
            ok=result.ok,
            attempts=result.attempts,
            config_version=config_version,
            vector_digest=result.vector_digest,
            error=None if result.ok else result.message,
        )
