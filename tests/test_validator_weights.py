"""Safety gates for the explicit, bounded localnet weight writer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from instant.validator import weights
from instant.validator.score import (
    MAX_WEIGHT_U16,
    GateState,
    MinerObservations,
    SourceSample,
    load_scoring_config,
    score_epoch,
)
from instant.validator.state import open_state
from instant.validator.weights import (
    MANUAL_BURN_BPS,
    WeightSafetyError,
    WeightWriter,
)


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
    def __init__(self, graph, *, block=250, spec=393, success=True, owner=None):
        self.graph = graph
        self.block = block
        self.substrate = FakeSubstrate(spec)
        self.success = success
        self.sent = []
        self.chain_weights = {}
        #: What ``SubnetOwnerHotkey`` answers. ``owner_query`` overrides it so
        #: a test can make the chain raise instead of reply.
        self.owner_hotkey = owner
        self.owner_query = None
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

    def query_subtensor(self, name, params=None):
        assert params == [5]
        if self.owner_query is not None:
            return self.owner_query()
        return self.owner_hotkey

    def weights(self, netuid, mechid=0):
        assert (netuid, mechid) == (5, 0)
        return [(0, sorted(self.chain_weights.items()))] if self.chain_weights else []

    def sign_and_send_extrinsic(self, **kwargs):
        self.sent.append(kwargs)
        if not self.success:
            return False, "Priority is too low"
        params = kwargs["call"]["call"]["call_params"]
        # The runtime renormalises on the way in -- it keeps the ratios and
        # rescales so the largest weight becomes u16::MAX -- and drops shares
        # too small to store. Modelling that here is what makes the readback
        # gate testable end to end; with a single miner the rescale is the
        # identity, which is why this went unnoticed until a vector had two
        # entries in it.
        submitted_vector = dict(zip(params["dests"], params["weights"], strict=True))
        self.chain_weights = {
            uid: weight
            for uid, weight in weights.normalise_u16(submitted_vector).items()
            if weight > 0
        }
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
        # The gates below are about preflight and readback, not about burn, so
        # the dial is off here and the burn tests set it explicitly. The
        # shipped default is asserted directly in
        # `test_the_shipped_constant_pays_only_the_owner_end_to_end`.
        burn_rate_bps=0,
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


# --- dynamic burn -----------------------------------------------------------
#
# The recipient is read from the chain rather than configured, so these tests
# always state who the chain thinks the owner is. `graph()` puts the validator
# at uid 0, so burn fixtures need a third neuron to burn to -- burning to
# ourselves is refused, and rightly.


def burn_graph(validator_key, miner_key, owner_key, *, second_miner=None):
    hotkeys = [validator_key.ss58_address, miner_key.ss58_address,
               owner_key.ss58_address]
    if second_miner is not None:
        hotkeys.append(second_miner.ss58_address)
    size = len(hotkeys)
    return SimpleNamespace(
        hotkeys=hotkeys,
        validator_permit=[True] + [False] * (size - 1),
        S=[100.0] + [0.0] * (size - 1),
        last_update=[100] + [0] * (size - 1),
    )


def burn_chain(validator_key, miner_key, owner_key, **kwargs):
    graph_ = burn_graph(validator_key, miner_key, owner_key, **kwargs)
    return FakeSubtensor(graph_, owner=owner_key.ss58_address)


def submitted(chain):
    """The (uid -> weight) mapping actually composed into the extrinsic."""
    params = chain.substrate.composed[-1]["call_params"]
    return dict(zip(params["dests"], params["weights"], strict=True))


def test_burn_reserves_the_owners_share_and_miners_split_the_rest(
    validator_key, miner_key, stranger_key
):
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    writer(chain, state, validator_key, config, burn_rate_bps=5000).submit_latest_once()

    vector = submitted(chain)
    assert vector == {1: 32768, 2: 32767}
    assert sum(vector.values()) == 65535


def test_a_zero_rate_submits_exactly_what_scoring_produced(
    validator_key, miner_key, stranger_key
):
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    writer(chain, state, validator_key, config, burn_rate_bps=0).submit_latest_once()

    # Not merely "no burn uid present" -- the vector is byte-identical to the
    # pre-burn one, so turning the dial to zero is genuinely a no-op.
    assert submitted(chain) == {1: 65535}


def test_a_full_rate_pays_only_the_owner(validator_key, miner_key, stranger_key):
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    writer(chain, state, validator_key, config, burn_rate_bps=10_000).submit_latest_once()

    assert submitted(chain) == {2: 65535}


def test_every_rate_and_miner_count_still_sums_to_the_u16_total(
    validator_key, miner_key, stranger_key, platform_key
):
    # The chain rejects a vector that does not apportion the total exactly, so
    # this is the invariant that matters most: it must hold for every dial
    # position, not just the round ones.
    for rate in (0, 1, 3333, 5000, 6667, 9999, 10_000):
        for second in (None, platform_key):
            state, config = scored_state(miner_key)
            chain = burn_chain(
                validator_key, miner_key, stranger_key, second_miner=second
            )
            writer(
                chain, state, validator_key, config, burn_rate_bps=rate
            ).submit_latest_once()
            vector = submitted(chain)
            assert sum(vector.values()) == 65535, (rate, second is not None)
            assert all(weight > 0 for weight in vector.values()), rate


def test_burn_is_deterministic_across_runs(validator_key, miner_key, stranger_key):
    # Two validators with the same scores and the same rate must produce the
    # same bytes, or they disagree on chain and lose vtrust for it.
    vectors = []
    for _ in range(3):
        state, config = scored_state(miner_key)
        chain = burn_chain(validator_key, miner_key, stranger_key)
        writer(
            chain, state, validator_key, config, burn_rate_bps=3333
        ).submit_latest_once()
        vectors.append(submitted(chain))
    assert vectors[0] == vectors[1] == vectors[2]


def test_nothing_scored_burns_the_whole_emission(
    validator_key, miner_key, stranger_key
):
    state, config = scored_state(miner_key, healthy=False)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    writer(chain, state, validator_key, config, burn_rate_bps=5000).submit_latest_once()

    # Not 50% of nothing -- an all-zero vector is either rejected or read as an
    # abstention, so the entire emission goes to the owner instead.
    assert submitted(chain) == {2: 65535}


def test_nothing_scored_and_no_burn_still_refuses_to_submit(
    validator_key, miner_key, stranger_key
):
    state, config = scored_state(miner_key, healthy=False)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    with pytest.raises(WeightSafetyError, match="all-zero"):
        writer(
            chain, state, validator_key, config, burn_rate_bps=0
        ).submit_latest_once()
    assert not chain.sent


def test_an_unreadable_owner_refuses_rather_than_paying_miners_the_burn(
    validator_key, miner_key, stranger_key
):
    # Failing open here would hand miners the entire emission -- with a 50%
    # rate, double what was intended, caused by a transient RPC error.
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)

    def boom():
        raise RuntimeError("websocket closed")

    chain.owner_query = boom
    with pytest.raises(WeightSafetyError, match="cannot read SubnetOwnerHotkey"):
        writer(
            chain, state, validator_key, config, burn_rate_bps=5000
        ).submit_latest_once()
    assert not chain.sent


def test_an_owner_missing_from_the_metagraph_is_refused(
    validator_key, miner_key, stranger_key, platform_key
):
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    chain.owner_hotkey = platform_key.ss58_address  # deregistered owner
    with pytest.raises(WeightSafetyError, match="not registered"):
        writer(
            chain, state, validator_key, config, burn_rate_bps=5000
        ).submit_latest_once()
    assert not chain.sent


def test_an_empty_owner_answer_is_refused(validator_key, miner_key, stranger_key):
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    chain.owner_hotkey = ""
    with pytest.raises(WeightSafetyError, match="no owner hotkey"):
        writer(
            chain, state, validator_key, config, burn_rate_bps=5000
        ).submit_latest_once()
    assert not chain.sent


def test_burning_to_ourselves_is_refused(validator_key, miner_key, stranger_key):
    # If we ever own the subnet we are validating, the burn share would come
    # straight back to us. That is self-dealing, not burning.
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    chain.owner_hotkey = validator_key.ss58_address
    with pytest.raises(WeightSafetyError, match="self-dealing"):
        writer(
            chain, state, validator_key, config, burn_rate_bps=5000
        ).submit_latest_once()
    assert not chain.sent


def test_the_owner_is_never_asked_for_when_the_dial_is_off(
    validator_key, miner_key, stranger_key
):
    # A validator running rate_bps=0 must not fail because the chain cannot
    # answer a question whose answer it does not need.
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)

    def boom():
        raise AssertionError("owner must not be queried when burn is off")

    chain.owner_query = boom
    writer(chain, state, validator_key, config, burn_rate_bps=0).submit_latest_once()
    assert chain.sent


def test_the_readback_gate_compares_against_the_burned_vector(
    validator_key, miner_key, stranger_key
):
    # The chain renormalises to u16::MAX, so the burned vector -- not the
    # scored one -- is what must survive the readback comparison.
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    result = writer(
        chain, state, validator_key, config, burn_rate_bps=5000
    ).submit_latest_once()
    assert result.ok, result.message
    assert result.uids == (1, 2)
    assert weights.readback_matches(
        dict(zip(result.uids, result.weights, strict=True)), chain.chain_weights
    )




def burn_writer(validator_key, owner_hotkey, rate):
    """A writer wired only for `_apply_burn`; no state, no submission."""
    chain = SimpleNamespace(query_subtensor=lambda name, params=None: owner_hotkey)
    return WeightWriter(
        subtensor=chain,
        wallet=SimpleNamespace(hotkey=validator_key),
        state=None,
        network="local",
        netuid=5,
        enabled=False,
        burn_rate_bps=rate,
    )


def test_an_owner_that_also_mines_earns_its_share_plus_the_burn(validator_key):
    # Latent today -- uid 0 announces 0.0.0.0:0, so serving_miners() excludes
    # it -- but it arms the moment the subnet owner runs a miner. Appending a
    # second entry for that uid produced a vector the chain rejects, surfacing
    # as "duplicate UIDs" rather than as the situation that actually occurred.
    hotkeys = ["5Validator", "5Miner", "5Owner"]
    scored = [(1, "5Miner", 32_768), (2, "5Owner", 32_767)]
    vector = burn_writer(validator_key, "5Owner", 5000)._apply_burn(
        scored, hotkeys=hotkeys, own_uid=0, epoch=1
    )

    uids = [uid for uid, _, _ in vector]
    assert len(uids) == len(set(uids)), vector
    assert sum(weight for _, _, weight in vector) == MAX_WEIGHT_U16
    # 16384 of the miner half, plus the whole 32767 burn share.
    assert dict((uid, weight) for uid, _, weight in vector) == {1: 16384, 2: 49151}


def test_shares_that_round_to_nothing_drop_out_without_losing_the_total(
    validator_key,
):
    # 9999 bps leaves 7 units for the entire field, so most of 20 miners round
    # to zero. They must disappear rather than be submitted as zero weights,
    # and the total must survive their removal.
    hotkeys = ["5Validator"] + [f"5Miner{i}" for i in range(1, 21)] + ["5Owner"]
    scored = [(i, f"5Miner{i}", 3276 + i) for i in range(1, 21)]
    vector = burn_writer(validator_key, "5Owner", 9999)._apply_burn(
        scored, hotkeys=hotkeys, own_uid=0, epoch=1
    )

    assert sum(weight for _, _, weight in vector) == MAX_WEIGHT_U16
    assert all(weight > 0 for _, _, weight in vector)
    assert len(vector) < len(scored) + 1  # somebody really did round away
    assert 21 in [uid for uid, _, _ in vector]  # the owner is still paid


@pytest.mark.parametrize("rate", [1, 2500, 5000, 7500, 9999, 10_000])
@pytest.mark.parametrize("miners", [1, 2, 3, 7, 20])
def test_the_total_is_exact_for_every_rate_and_field_size(validator_key, rate, miners):
    hotkeys = (
        ["5Validator"] + [f"5Miner{i}" for i in range(1, miners + 1)] + ["5Owner"]
    )
    owner_uid = miners + 1
    scored = [(i, f"5Miner{i}", 1000 + i * 7) for i in range(1, miners + 1)]
    vector = burn_writer(validator_key, "5Owner", rate)._apply_burn(
        scored, hotkeys=hotkeys, own_uid=0, epoch=1
    )

    uids = [uid for uid, _, _ in vector]
    assert sum(weight for _, _, weight in vector) == MAX_WEIGHT_U16
    assert len(uids) == len(set(uids))
    assert all(weight > 0 for _, _, weight in vector)
    assert owner_uid in uids


# --- the shipped constant ---------------------------------------------------


def test_the_shipped_constant_is_a_full_burn():
    assert MANUAL_BURN_BPS == 10_000


def test_a_writer_built_without_a_rate_uses_the_shipped_constant(
    validator_key, miner_key, stranger_key
):
    # Nothing in production passes burn_rate_bps -- main.py builds the writer
    # without it -- so the default is the live behaviour, not a test detail.
    state, config = scored_state(miner_key)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    built = WeightWriter(
        subtensor=chain,
        wallet=SimpleNamespace(hotkey=validator_key),
        state=state,
        network="local",
        netuid=5,
        enabled=True,
        config_version=config.version,
    )
    assert built.burn_rate_bps == MANUAL_BURN_BPS


def test_the_shipped_constant_pays_only_the_owner_end_to_end(
    validator_key, miner_key, stranger_key
):
    # A healthy, fully scored miner still earns nothing at 100%. That is the
    # launch position: emission is held at the owner while miners onboard.
    state, config = scored_state(miner_key, healthy=True)
    chain = burn_chain(validator_key, miner_key, stranger_key)
    result = WeightWriter(
        subtensor=chain,
        wallet=SimpleNamespace(hotkey=validator_key),
        state=state,
        network="local",
        netuid=5,
        enabled=True,
        config_version=config.version,
    ).submit_latest_once()

    assert result.ok, result.message
    assert submitted(chain) == {2: MAX_WEIGHT_U16}
    assert result.uids == (2,)


def test_a_full_burn_is_identical_whatever_the_miners_scored(validator_key):
    # The property that makes 100% safe to launch on: every validator submits
    # the same vector regardless of what it observed, so no two validators can
    # disagree and there is no vtrust to lose while the fleet is still being
    # onboarded.
    hotkeys = ["5Validator", "5MinerA", "5MinerB", "5Owner"]
    fields = [
        [(1, "5MinerA", 65_535)],
        [(1, "5MinerA", 32_768), (2, "5MinerB", 32_767)],
        [(1, "5MinerA", 60_000), (2, "5MinerB", 5_535)],
        [],
    ]
    vectors = [
        burn_writer(validator_key, "5Owner", MANUAL_BURN_BPS)._apply_burn(
            scored, hotkeys=hotkeys, own_uid=0, epoch=1
        )
        for scored in fields
    ]
    assert all(v == [(3, "5Owner", MAX_WEIGHT_U16)] for v in vectors), vectors
