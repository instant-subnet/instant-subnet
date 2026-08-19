from dataclasses import replace

import pytest

from instant_validator.config import Settings
from instant_validator.service import ValidatorService
from instant_validator.state import StateStore


class FakeClient:
    def __init__(self, report):
        self.report = report
        self.calls = 0

    def fetch_latest(self, *, now_ms=None):
        self.calls += 1
        return self.report


class FakeWriter:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def full_burn_plan(self):
        return {238: 65_535}, 62, 10_000

    def set_weights(self, weights, *, version_key=None):
        self.calls.append((weights, version_key))
        if self.error:
            raise self.error
        return "finalized"


def settings(*, writes, burn=False):
    return Settings.from_env(
        {
            "INSTANT_PLATFORM_SIGNER": "5Signer",
            "INSTANT_ENABLE_WEIGHT_WRITES": "true" if writes else "false",
            "INSTANT_BURN_MINER_EMISSIONS": "true" if burn else "false",
        },
        load_env_file=False,
    )


def proof_clean(report):
    return replace(
        report,
        miners=tuple(
            replace(
                row,
                toploc_verified=row.requests,
                toploc_failed=0,
                toploc_timed_out=0,
            )
            for row in report.miners
        ),
    )


@pytest.mark.parametrize(
    "writes, status, calls",
    [
        (False, "dry_run_burn", []),
        (True, "burn_applied", [({238: 65_535}, 62)]),
    ],
)
def test_burn_is_chain_only(writes, status, calls, parsed_report):
    client = FakeClient(parsed_report)
    writer = FakeWriter()
    service = ValidatorService(settings(writes=writes, burn=True), client, None, writer)

    outcome = service.run_once()

    assert outcome.status == status
    assert outcome.weights == {238: 65_535}
    assert outcome.period_end_block == 10_000
    assert writer.calls == calls
    assert client.calls == 0


def test_scoring_dry_run_logs_vector_but_does_not_advance_state(tmp_path, parsed_report):
    state = StateStore(tmp_path / "state.json")
    service = ValidatorService(
        settings(writes=False), FakeClient(proof_clean(parsed_report)), state, writer=None
    )

    outcome = service.run_once(now_ms=1_786_708_811_000)

    assert outcome.status == "dry_run"
    assert outcome.weights == {12: 65_535, 37: 30_287}
    assert state.load() is None


def test_successful_scoring_write_advances_state_and_is_not_repeated(
    tmp_path, parsed_report
):
    state = StateStore(tmp_path / "state.json")
    writer = FakeWriter()
    service = ValidatorService(
        settings(writes=True), FakeClient(proof_clean(parsed_report)), state, writer
    )

    first = service.run_once(now_ms=1_786_708_811_000)
    second = service.run_once(now_ms=1_786_708_812_000)

    assert first.status == "applied"
    assert second.status == "already_applied"
    assert len(writer.calls) == 1


def test_failed_scoring_write_does_not_mark_report_applied(tmp_path, parsed_report):
    state = StateStore(tmp_path / "state.json")
    writer = FakeWriter(error=RuntimeError("chain unavailable"))
    service = ValidatorService(
        settings(writes=True), FakeClient(proof_clean(parsed_report)), state, writer
    )

    with pytest.raises(RuntimeError, match="chain unavailable"):
        service.run_once(now_ms=1_786_708_811_000)
    assert state.load() is None


def test_report_with_no_positive_score_never_writes(tmp_path, parsed_report):
    rows = tuple(
        replace(
            row,
            successes=0,
            failures=row.requests,
            ttft_p50_ms=0,
            ttft_p95_ms=0,
            tokens_per_second_p50=0,
        )
        for row in parsed_report.miners
    )
    empty = replace(parsed_report, miners=rows)
    writer = FakeWriter()
    service = ValidatorService(
        settings(writes=True),
        FakeClient(empty),
        StateStore(tmp_path / "state.json"),
        writer,
    )
    with pytest.raises(RuntimeError, match="no miner"):
        service.run_once(now_ms=1_786_708_811_000)
    assert writer.calls == []
