from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from instant_validator.runtime import ChainSnapshot, run_once
from instant_validator.state import StateStore

FIXTURE = Path(__file__).parent / "fixtures/validator_report_v1.json"


class _Chain:
    def __init__(self, report: dict) -> None:
        hotkeys = [f"unused-{uid}" for uid in range(56)]
        for miner in report["miners"]:
            hotkeys[miner["uid"]] = miner["hotkey"]
        self.value = ChainSnapshot(
            report["finalized_block"],
            report["tempo"],
            report["epoch_end_block"],
            tuple(hotkeys),
            tuple(0 for _ in hotkeys),
        )

    def snapshot(self, netuid: int) -> ChainSnapshot:
        return self.value


class ShadowAcceptanceTests(unittest.TestCase):
    def test_two_validators_and_late_restart_produce_one_identical_result(self) -> None:
        raw = FIXTURE.read_bytes().rstrip(b"\n")
        report = json.loads(raw)
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            arguments = {
                "network": report["network"],
                "netuid": report["netuid"],
                "report_url": "fixture",
                "platform_signer": report["signer"],
                "chain": _Chain(report),
                "fetch": lambda url, timeout: raw,
            }
            first = run_once(**arguments, state=StateStore(Path(first_dir) / "state.json"))
            second_store = StateStore(Path(second_dir) / "state.json")
            late = run_once(**arguments, state=second_store)
            restarted = run_once(**arguments, state=second_store)

        self.assertEqual(first, late)
        self.assertEqual(restarted, ())
        self.assertEqual(
            [record["normalized_weight"] for record in first], [65_535, 30_287, 0]
        )
        self.assertEqual(first[2]["score_bps"], 0)
        self.assertTrue(first[2]["disqualified"])


if __name__ == "__main__":
    unittest.main()
