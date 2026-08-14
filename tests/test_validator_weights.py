"""Safety gates for the explicit, bounded localnet weight writer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from instant.validator import weights
from instant.validator.score import (
    GateState,
    MinerObservations,
    SourceSample,
    load_scoring_config,
    score_epoch,
)
from instant.validator.state import open_state
from instant.validator.weights import WeightSafetyError, WeightWriter


class FakeSubstrate:
    def __init__(self, spec=393):
        self.spec = spec
        self.composed = []

    def get_chain_head(self):
        return "0xhead"

    def get_block_runtime_version(self, head):
        assert head == "0xhead"
        return {"specVersion": self.spec}

    def compose_call(self, **kwargs):
        self.composed.append(kwargs)
        return {"call": kwargs}


class FakeSubtensor:
    def __init__(self, graph, *, block=250, spec=393, success=True):
        self.graph = graph
        self.block = block
        self.substrate = FakeSubstrate(spec)
        self.success = success
        self.sent = []
        self.chain_weights = {}
        self.params = {
            "CommitRevealWeightsEnabled": False,
            "WeightsVersionKey": 0,
            "MinAllowedWeights": 1,
            "MaxWeightsLimit": 65_535,
            "WeightsSetRateLimit": 100,
        }

    def metagraph(self, netuid, lite=True):
        assert (netuid, lite) == (5, True)
        return self.graph

    def get_current_block(self):
        return self.block

    def get_hyperparameter(self, name, netuid):
        assert netuid == 5
        return self.params[name]

    def weights(self, netuid, mechid=0):
        assert (netuid, mechid) == (5, 0)
        return [(0, sorted(self.chain_weights.items()))] if self.chain_weights else []

    def sign_and_send_extrinsic(self, **kwargs):
        self.sent.append(kwargs)
        if not self.success:
            return False, "Priority is too low"
        params = kwargs["call"]["call"]["call_params"]
        self.chain_weights = dict(zip(params["dests"], params["weights"], strict=True))
        self.graph.last_update[0] = self.block
        return True, ""

    def set_weights(self, *args, **kwargs):  # pragma: no cover - regression trap
        raise AssertionError("unbounded high-level helper must never be called")


def graph(validator_key, miner_key, *, permit=True, stake=100.0, last_update=100):
    return SimpleNamespace(
        hotkeys=[validator_key.ss58_address, miner_key.ss58_address],
        validator_permit=[permit, False],
        S=[stake, 0.0],
        last_update=[last_update, 0],
    )


def scored_state(miner_key, *, healthy=True):
    state = open_state(":memory:")
    config = load_scoring_config()
    sample = (
        SourceSample(
            attempts=20,
            successes=20,
            ttft_ms=(100,) * 20,
            tps_milli=(120_000,) * 20,
        )
        if healthy
        else SourceSample()
    )
    observation = MinerObservations(
        uid=1,
        hotkey=miner_key.ss58_address,
        direct=sample,
        attested=False,
    )
    result = score_epoch([observation], config, attestation_mode="off")
    state.commit_epoch(123, result, {miner_key.ss58_address: GateState()})
    return state, config


def writer(chain, state, validator_key, config, **overrides):
    values = dict(
        subtensor=chain,
        wallet=SimpleNamespace(hotkey=validator_key),
        state=state,
        network="local",
        netuid=5,
        enabled=True,
        config_version=config.version,
    )
    values.update(overrides)
    return WeightWriter(**values)


def test_disabled_writer_never_composes_or_submits(validator_key, miner_key):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(graph(validator_key, miner_key))
    target = writer(chain, state, validator_key, config, enabled=False)
    with pytest.raises(WeightSafetyError, match="disabled"):
        target.submit_latest_once()
    assert chain.substrate.composed == []
    assert chain.sent == []


def test_exact_spec393_call_is_hotkey_signed_once_and_read_back(
    validator_key, miner_key
):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(graph(validator_key, miner_key))

    result = writer(chain, state, validator_key, config).submit_latest_once()

    assert result.ok is True
    assert result.attempts == 1
    assert len(chain.sent) == 1
    assert chain.sent[0]["sign_with"] == "hotkey"
    assert chain.sent[0]["wait_for_finalization"] is True
    assert chain.substrate.composed == [
        {
            "call_module": "SubtensorModule",
            "call_function": "set_mechanism_weights",
            "call_params": {
                "netuid": 5,
                "mecid": 0,
                "dests": [1],
                "weights": [65_535],
                "version_key": 0,
            },
        }
    ]
    row = state.weight_set_for_epoch(123)
    assert (row["ok"], row["attempts"], row["config_version"]) == (
        1,
        1,
        config.version,
    )


def test_unsuccessful_extrinsic_is_not_retried_and_blocks_duplicate_epoch(
    validator_key, miner_key
):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(graph(validator_key, miner_key), success=False)
    target = writer(chain, state, validator_key, config)

    result = target.submit_latest_once()

    assert result.ok is False
    assert len(chain.sent) == 1
    assert state.weight_set_for_epoch(123)["attempts"] == 1
    with pytest.raises(WeightSafetyError, match="duplicate"):
        target.submit_latest_once()
    assert len(chain.sent) == 1


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda chain: setattr(chain.substrate, "spec", 394), "specVersion"),
        (lambda chain: chain.params.__setitem__("CommitRevealWeightsEnabled", True), "commit-reveal"),
        (lambda chain: chain.params.__setitem__("WeightsVersionKey", 1), "WeightsVersionKey"),
        (lambda chain: chain.params.__setitem__("MinAllowedWeights", 2), "requires 2"),
        (lambda chain: chain.params.__setitem__("MaxWeightsLimit", 10_000), "exceeds"),
        (lambda chain: setattr(chain.graph, "validator_permit", [False, False]), "permit"),
        (lambda chain: setattr(chain.graph, "S", [0.0, 0.0]), "no stake"),
    ],
)
def test_preflight_failures_happen_before_compose_or_submit(
    validator_key, miner_key, mutation, error
):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(graph(validator_key, miner_key))
    mutation(chain)
    with pytest.raises(WeightSafetyError, match=error):
        writer(chain, state, validator_key, config).submit_latest_once()
    assert chain.substrate.composed == []
    assert chain.sent == []


def test_rate_limit_is_checked_in_blocks(validator_key, miner_key):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(
        graph(validator_key, miner_key, last_update=200), block=250
    )
    with pytest.raises(WeightSafetyError, match="50 blocks elapsed"):
        writer(chain, state, validator_key, config).submit_latest_once()
    assert chain.sent == []


def test_all_zero_score_is_never_submitted(validator_key, miner_key):
    state, config = scored_state(miner_key, healthy=False)
    chain = FakeSubtensor(graph(validator_key, miner_key))
    with pytest.raises(WeightSafetyError, match="all-zero"):
        writer(chain, state, validator_key, config).submit_latest_once()
    assert chain.substrate.composed == []


def test_exact_existing_chain_vector_reconciles_without_extrinsic(
    validator_key, miner_key
):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(graph(validator_key, miner_key, last_update=248))
    chain.chain_weights = {1: 65_535}

    result = writer(chain, state, validator_key, config).submit_latest_once()

    assert result.reconciled is True
    assert result.attempts == 0
    assert chain.substrate.composed == []
    assert chain.sent == []


def test_old_matching_vector_is_resubmitted_to_refresh_last_update(
    validator_key, miner_key
):
    state, config = scored_state(miner_key)
    chain = FakeSubtensor(graph(validator_key, miner_key, last_update=100))
    chain.chain_weights = {1: 65_535}

    result = writer(chain, state, validator_key, config).submit_latest_once()

    assert result.reconciled is False
    assert result.attempts == 1
    assert len(chain.sent) == 1


# --- chain-normalised readback ----------------------------------------------


def test_the_chain_renormalises_to_max_u16():
    # Observed against the live localnet: a two-miner vector submitted as
    # {1: 31173, 3: 34362} read back as {1: 59453, 3: 65535}. Same ratios,
    # rescaled so the largest weight is u16::MAX.
    assert weights.normalise_u16({1: 31173, 3: 34362}) == {1: 59453, 3: 65535}


def test_normalising_a_single_miner_is_the_identity():
    # Why the raw comparison appeared to work until a second miner existed.
    assert weights.normalise_u16({1: 65535}) == {1: 65535}


def test_normalisation_preserves_proportions():
    normalised = weights.normalise_u16({1: 1000, 2: 2000, 3: 4000})
    assert normalised[3] == 65535
    assert normalised[2] == pytest.approx(normalised[3] // 2, abs=1)
    assert normalised[1] == pytest.approx(normalised[3] // 4, abs=1)


def test_an_all_zero_vector_normalises_to_itself():
    assert weights.normalise_u16({1: 0, 2: 0}) == {1: 0, 2: 0}
    assert weights.normalise_u16({}) == {}


def test_readback_accepts_the_chains_rescaled_vector():
    # The regression: this exact pair reported "finalized but readback differs"
    # and marked a perfectly good multi-miner write as failed.
    assert weights.readback_matches({1: 31173, 3: 34362}, {1: 59453, 3: 65535})


def test_readback_tolerates_one_unit_of_chain_rounding():
    assert weights.readback_matches({1: 31173, 3: 34362}, {1: 59452, 3: 65535})
    assert weights.readback_matches({1: 31173, 3: 34362}, {1: 59454, 3: 65535})


def test_readback_rejects_a_different_split():
    # Same UIDs, materially different proportions: must still fail.
    assert not weights.readback_matches({1: 31173, 3: 34362}, {1: 32767, 3: 65535})


def test_readback_rejects_a_different_uid_set():
    assert not weights.readback_matches({1: 65535}, {2: 65535})
    assert not weights.readback_matches({1: 31173, 3: 34362}, {1: 65535})
    assert not weights.readback_matches({1: 65535}, {})


def test_readback_ignores_a_share_too_small_to_store():
    # _readback keeps only positive weights, so a uid whose normalised share
    # rounds to zero is absent from the chain map rather than present as 0.
    expected = {1: 1, 2: 400_000, 3: 400_000}
    assert weights.normalise_u16(expected)[1] == 0
    assert weights.readback_matches(expected, {2: 65535, 3: 65535})
    # ...but a uid with a real share going missing is still a mismatch.
    assert not weights.readback_matches({1: 30000, 2: 35535}, {2: 65535})
