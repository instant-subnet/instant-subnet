from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from instant_validator.state import StateError, StateStore, ValidatorState


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "state.json"
        self.store = StateStore(self.path)

    def test_missing_state_is_empty_and_saved_state_is_durable(self) -> None:
        self.assertEqual(self.store.load(), ValidatorState())
        state = ValidatorState("report-1", "sha256:digest", 720, 720)

        self.store.save(state)

        self.assertEqual(StateStore(self.path).load(), state)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_malformed_or_partial_state_fails_closed(self) -> None:
        for value in (b"not-json", b"{}", b'{"report_id":"only"}'):
            with self.subTest(value=value):
                self.path.write_bytes(value)
                with self.assertRaises(StateError):
                    self.store.load()


if __name__ == "__main__":
    unittest.main()
