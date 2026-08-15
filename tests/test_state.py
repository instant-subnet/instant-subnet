import json

import pytest

from instant_validator.state import StateError, StateStore


def test_state_records_only_the_successfully_applied_period(tmp_path, parsed_report):
    store = StateStore(tmp_path / "validator-state.json")
    assert store.is_applied(parsed_report) is False

    store.mark_applied(parsed_report, applied_at_ms=1_786_708_811_000)

    assert store.is_applied(parsed_report) is True
    assert store.load().digest == parsed_report.digest


def test_corrupt_state_fails_closed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"legacy": True}), encoding="utf-8")
    with pytest.raises(StateError, match="invalid shape"):
        StateStore(path).load()
