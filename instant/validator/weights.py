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

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from instant.protocol.canonical import digest
from instant.validator.score import BPS_ONE, MAX_WEIGHT_U16, cube_normalise

from .state import ValidatorState

log = logging.getLogger("instant.validator")

#: The burn dial, in basis points. 10000 is 100%: the entire emission goes to
#: the subnet owner and miners are paid nothing.
#:
#: This is a constant rather than configuration, and deliberately so. Changing
#: what miners are paid is a code change, a review and a deploy — not an env
#: var somebody can set differently on one host. Every validator must apply
#: the same rate in the same epoch or they submit different vectors from
#: identical observations and burn each other's vtrust; a constant in the
#: released artifact is the only form of that number which cannot silently
#: differ per operator.
#:
#: 100% is also the one setting where that risk is nil, because the vector is
#: ``{owner: 65535}`` regardless of what any miner scored. That makes it the
#: right place to launch from and dial down out of, rather than up into.
MANUAL_BURN_BPS = 10_000


def _now_ms() -> int:
    return int(time.time() * 1000)


#: How far a read-back weight may differ from the expected normalised value.
#: One unit absorbs the chain's own rounding choice without admitting a
#: materially different vector: 65535 units is 100%, so one unit is 0.0015%.
READBACK_TOLERANCE_U16 = 1


def normalise_u16(weights: Mapping[int, int]) -> dict[int, int]:
    """Scale a vector so its largest weight is ``MAX_WEIGHT_U16``.

    ``set_mechanism_weights`` renormalises on the way in: the chain keeps the
    *ratios* and rescales so the maximum becomes u16::MAX.  A vector submitted
    as ``{1: 31173, 3: 34362}`` reads back as ``{1: 59453, 3: 65535}`` — the
    same proportions, different absolute numbers.

    With one miner this is the identity (65535 stays 65535), which is why
    comparing raw submitted values appeared to work right up until a second
    miner existed and every write started reporting a false mismatch.

    Rounds half up rather than truncating, which matches the value the runtime
    produced when this was checked against the live chain.
    """
    peak = max(weights.values(), default=0)
    if peak <= 0:
        return dict(weights)
    return {
        uid: (value * MAX_WEIGHT_U16 * 2 + peak) // (peak * 2)
        for uid, value in weights.items()
    }


def readback_matches(expected: Mapping[int, int], actual: Mapping[int, int]) -> bool:
    """True if ``actual`` is ``expected`` as the chain would have stored it.

    Compares the normalised forms, so this asks the question that matters — did
    the chain record the proportions we chose — rather than whether it echoed
    our arbitrary scale back verbatim.
    """
    wanted = normalise_u16(expected)
    # ``_readback`` keeps only positive weights, so a uid whose share is small
    # enough to normalise to zero is simply absent from the chain's map. Compare
    # on the same footing rather than calling that a mismatch.
    significant = {uid: value for uid, value in wanted.items() if value > 0}
    if set(significant) != set(actual):
        return False
    return all(
        abs(actual[uid] - value) <= READBACK_TOLERANCE_U16
        for uid, value in significant.items()
    )


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
        burn_rate_bps: int = MANUAL_BURN_BPS,
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
        self.burn_rate_bps = burn_rate_bps

    def _burn_target(self, hotkeys: list[str], own_uid: int) -> tuple[int, str]:
        """Locate the subnet owner in the metagraph, or refuse.

        The recipient is read from the chain rather than configured. A pinned
        uid is a hostage to uid recycling -- if the owner vacates that slot, a
        pinned uid pays whoever registers into it next and every log line still
        reads correctly. ``SubnetOwnerHotkey`` is the chain's own answer to
        "who owns this subnet", and it is the same answer for every validator.

        Every failure here raises. Returning the vector unburned instead would
        pay miners the entire emission -- with a 50% rate that is double what
        was intended, produced by a transient RPC hiccup. Refusing leaves the
        previous vector standing on chain, which is the smaller error.
        """
        try:
            owner = _plain(
                self.subtensor.query_subtensor(
                    "SubnetOwnerHotkey", params=[self.netuid]
                )
            )
        except Exception as exc:  # noqa: BLE001 - boundary is the chain
            raise WeightSafetyError(
                f"cannot read SubnetOwnerHotkey for netuid {self.netuid}: {exc}. "
                "Refusing to submit rather than paying miners the burn share"
            ) from exc

        owner = str(owner) if owner else ""
        if not owner:
            raise WeightSafetyError(
                f"chain reports no owner hotkey for netuid {self.netuid}; "
                "refusing to submit a vector with nowhere to burn to"
            )
        if owner not in hotkeys:
            raise WeightSafetyError(
                f"subnet owner hotkey {owner} is not registered on netuid "
                f"{self.netuid}; there is no uid to assign the burn share to"
            )
        burn_uid = hotkeys.index(owner)
        if burn_uid == own_uid:
            raise WeightSafetyError(
                f"subnet owner uid {burn_uid} is this validator; assigning the "
                "burn share to ourselves is self-dealing, not burning"
            )
        return burn_uid, owner

    def _apply_burn(
        self,
        scored: list[tuple[int, str, int]],
        *,
        hotkeys: list[str],
        own_uid: int,
        epoch: int,
    ) -> list[tuple[int, str, int]]:
        """Reserve the burn share for the owner, apportion the rest to miners.

        At the shipped :data:`MANUAL_BURN_BPS` of 10000 the miner half is empty
        and this collapses to ``{owner: 65535}``. The apportionment below is
        what makes dialling the constant down a one-line change rather than a
        second implementation.

        Integer arithmetic throughout, and the remainder is handed out by the
        same largest-remainder rule the scoring path uses, so two validators
        with identical scores and the same released constant produce
        byte-identical vectors rather than nearly-identical ones.
        """
        if self.burn_rate_bps <= 0:
            return list(scored)

        burn_uid, burn_hotkey = self._burn_target(hotkeys, own_uid)

        # Nobody earned anything, so the whole emission burns. Submitting the
        # all-zero vector instead is either rejected or read as an abstention
        # depending on the SDK version, and neither is what we mean.
        if not scored:
            log.warning(
                "epoch %s scored no miners; burning the full emission to uid %s",
                epoch,
                burn_uid,
            )
            return [(burn_uid, burn_hotkey, MAX_WEIGHT_U16)]

        # rate_bps >= 1 puts this at 6 or more, so the burn share is never
        # empty once the dial is off zero.
        burn_u16 = MAX_WEIGHT_U16 * self.burn_rate_bps // BPS_ONE
        miner_total = MAX_WEIGHT_U16 - burn_u16

        # exponent=1 makes this a plain proportional apportionment. Reusing
        # cube_normalise rather than repeating its remainder handling means
        # there is exactly one implementation of "share integers exactly".
        shares = cube_normalise(
            {uid: weight for uid, _, weight in scored}, exponent=1, total=miner_total
        )
        hotkey_of = {uid: hotkey for uid, hotkey, _ in scored}
        hotkey_of.setdefault(burn_uid, burn_hotkey)

        # A share can round to zero at a high burn rate -- 9999 bps leaves 7
        # units for the whole field -- and a zero weight is not something to
        # submit, so those uids drop out. Dropping them cannot change the total,
        # because they contribute nothing to it.
        totals = {uid: share for uid, share in shares.items() if share > 0}

        # The owner may itself be a scored miner, in which case it earns its
        # miner share *and* the burn. Appending a second entry for the same uid
        # would produce a vector the chain will not accept, and would fail as a
        # confusing "duplicate UIDs" rather than as the thing that happened.
        totals[burn_uid] = totals.get(burn_uid, 0) + burn_u16

        vector = [(uid, hotkey_of[uid], totals[uid]) for uid in sorted(totals)]

        log.info(
            "epoch %s burning %s bps (%s of %s) to owner uid %s; %s miners share %s",
            epoch,
            self.burn_rate_bps,
            burn_u16,
            MAX_WEIGHT_U16,
            burn_uid,
            len(shares),
            miner_total,
        )
        return vector

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
        scored = sorted(
            (
                (int(row["uid"]), str(row["hotkey"]), int(row["weight_u16"]))
                for row in scores
                if int(row["weight_u16"]) > 0
            ),
            key=lambda item: item[0],
        )
        if scored and sum(weight for _, _, weight in scored) != MAX_WEIGHT_U16:
            raise WeightSafetyError(
                "score vector does not sum to the canonical u16 total 65535"
            )

        # The metagraph is read before the vector is final, because the burn
        # share is assigned to a uid resolved from it. Everything downstream --
        # the digest, the chain-limit gates, the readback expectation -- has to
        # see the vector we actually submit, not the one before burn.
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
        for uid, expected_hotkey, _ in scored:
            if uid >= len(hotkeys) or hotkeys[uid] != expected_hotkey:
                raise WeightSafetyError(
                    f"uid/hotkey mapping changed since scoring for uid {uid}"
                )

        vector = self._apply_burn(
            scored, hotkeys=hotkeys, own_uid=own_uid, epoch=epoch
        )
        if not vector:
            raise WeightSafetyError(
                f"epoch {epoch} has an all-zero vector and no burn is "
                "configured; refusing to submit"
            )
        if sum(weight for _, _, weight in vector) != MAX_WEIGHT_U16:
            raise WeightSafetyError(
                "weight vector does not sum to the canonical u16 total 65535"
            )
        uids = tuple(uid for uid, _, _ in vector)
        weights = tuple(weight for _, _, weight in vector)
        if len(set(uids)) != len(uids):
            raise WeightSafetyError("weight vector contains duplicate UIDs")
        if own_uid in uids:
            raise WeightSafetyError("weight vector assigns weight to the validator itself")
        vector_digest = digest(
            {"netuid": self.netuid, "mecid": self.mechanism_id,
             "uids": list(uids), "weights": list(weights),
             "version_key": self.version_key}
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
        if recently_updated and readback_matches(
            dict(zip(uids, weights, strict=True)), self._readback(own_uid)
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
                if not readback_matches(expected, actual):
                    ok = False
                    message = (
                        "finalized but readback differs: expected "
                        f"{normalise_u16(expected)} (normalised from {expected}), "
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
