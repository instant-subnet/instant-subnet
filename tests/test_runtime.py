from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from instant_validator.runtime import (
    BittensorChain,
    ChainError,
    ChainSnapshot,
    main,
    run_once,
    validate_chain,
)

FIXTURES = Path(__file__).parent / "fixtures"


class _Chain:
    def __init__(self, snapshot: ChainSnapshot) -> None:
        self.value = snapshot
        self.netuids: list[int] = []

    def snapshot(self, netuid: int) -> ChainSnapshot:
        self.netuids.append(netuid)
        return self.value


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = (FIXTURES / "validator_report_v1.json").read_bytes().rstrip(b"\n")
        self.report = json.loads(self.raw)
        hotkeys = [f"unused-{uid}" for uid in range(56)]
        for row in self.report["miners"]:
            hotkeys[row["uid"]] = row["hotkey"]
        self.snapshot = ChainSnapshot(
            finalized_block=self.report["finalized_block"],
            tempo=self.report["tempo"],
            last_step=self.report["epoch_end_block"],
            hotkeys=tuple(hotkeys),
        )

    def test_run_once_fetches_authenticates_scores_logs_and_exits(self) -> None:
        chain = _Chain(self.snapshot)
        with self.assertLogs("instant.validator", level=logging.INFO) as captured:
            records = run_once(
                network=self.report["network"],
                netuid=self.report["netuid"],
                report_url="https://platform.invalid/latest",
                platform_signer=self.report["signer"],
                chain=chain,
                fetch=lambda url, timeout: self.raw,
            )

        self.assertEqual(chain.netuids, [46])
        self.assertEqual([record["uid"] for record in records], [12, 37, 55])
        self.assertEqual(
            [record["normalized_weight"] for record in records], [65_535, 30_287, 0]
        )
        self.assertEqual(sum("miner_score " in line for line in captured.output), 3)
        self.assertIn("report_processed report_id=local-46-361-720", captured.output[-1])

    def test_chain_must_match_report_epoch_and_roster(self) -> None:
        invalid = [
            ChainSnapshot(722, 361, 720, self.snapshot.hotkeys),
            ChainSnapshot(722, 360, 719, self.snapshot.hotkeys),
            ChainSnapshot(719, 360, 720, self.snapshot.hotkeys),
            ChainSnapshot(722, 360, 720, self.snapshot.hotkeys[:55]),
        ]

        for snapshot in invalid:
            with self.subTest(snapshot=snapshot), self.assertRaises(ChainError):
                validate_chain(self.report, snapshot)

    def test_bittensor_adapter_reads_finalized_metagraph(self) -> None:
        substrate = mock.Mock()
        substrate.get_chain_finalised_head.return_value = "0xhead"
        substrate.get_block_number.return_value = 722
        subtensor = mock.Mock(substrate=substrate)
        subtensor.get_metagraph_info.return_value = SimpleNamespace(
            tempo=360,
            last_step=720,
            blocks_since_last_step=2,
            hotkeys=["hotkey-0"],
        )
        sdk = mock.Mock()
        sdk.subtensor.return_value = subtensor

        snapshot = BittensorChain("local", "ws://chain", sdk=sdk).snapshot(46)

        sdk.subtensor.assert_called_once_with(network="ws://chain")
        subtensor.get_metagraph_info.assert_called_once_with(46, block=722)
        self.assertEqual(snapshot, ChainSnapshot(722, 360, 720, ("hotkey-0",)))

    def test_cli_requires_platform_signer_before_chain_contact(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(main(["run-once"]), 2)


if __name__ == "__main__":
    unittest.main()
