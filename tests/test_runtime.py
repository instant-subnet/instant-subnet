from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from instant_validator.burn import BurnError
from instant_validator.runtime import (
    BittensorChain,
    ChainError,
    ChainSnapshot,
    main,
    run_once,
    validate_chain,
)
from instant_validator.state import StateStore

FIXTURES = Path(__file__).parent / "fixtures"


class _Chain:
    def __init__(self, snapshot: ChainSnapshot) -> None:
        self.value = snapshot
        self.netuids: list[int] = []

    def snapshot(self, netuid: int) -> ChainSnapshot:
        self.netuids.append(netuid)
        return self.value


class _Burner:
    def __init__(self, hotkey: str, *, fail: bool = False) -> None:
        self.hotkey = hotkey
        self.fail = fail
        self.calls: list[tuple[int, int]] = []

    def submit(self, *, netuid: int, finalized_block: int) -> str:
        self.calls.append((netuid, finalized_block))
        if self.fail:
            raise BurnError("failed")
        return "finalized"


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
            last_updates=tuple(0 for _ in hotkeys),
        )
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = StateStore(Path(self.temporary.name) / "state.json")

    def test_run_once_fetches_authenticates_scores_logs_and_exits(self) -> None:
        chain = _Chain(self.snapshot)
        with self.assertLogs("instant.validator", level=logging.INFO) as captured:
            records = run_once(
                network=self.report["network"],
                netuid=self.report["netuid"],
                report_url="https://platform.invalid/latest",
                platform_signer=self.report["signer"],
                chain=chain,
                state=self.state,
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
            ChainSnapshot(722, 361, 720, self.snapshot.hotkeys, self.snapshot.last_updates),
            ChainSnapshot(722, 360, 719, self.snapshot.hotkeys, self.snapshot.last_updates),
            ChainSnapshot(719, 360, 720, self.snapshot.hotkeys, self.snapshot.last_updates),
            ChainSnapshot(
                722,
                360,
                720,
                self.snapshot.hotkeys[:55],
                self.snapshot.last_updates[:55],
            ),
        ]

        for snapshot in invalid:
            with self.subTest(snapshot=snapshot), self.assertRaises(ChainError):
                validate_chain(self.report, snapshot)

    def test_bittensor_adapter_reads_finalized_storage(self) -> None:
        values = {
            "SubnetworkN": 1,
            "Tempo": 360,
            "LastMechansimStepBlock": 720,
            "BlocksSinceLastStep": 2,
            "LastUpdate": [700],
        }
        substrate = mock.Mock()
        substrate.get_chain_finalised_head.return_value = "0xhead"
        substrate.get_block_number.return_value = 722
        substrate.query.side_effect = lambda module, storage, params, block_hash: (
            SimpleNamespace(value=values[storage])
        )
        substrate.query_map.return_value = [(0, SimpleNamespace(value="hotkey-0"))]
        subtensor = mock.Mock(substrate=substrate)
        sdk = mock.Mock()
        sdk.subtensor.return_value = subtensor

        snapshot = BittensorChain("local", "ws://chain", sdk=sdk).snapshot(46)

        sdk.subtensor.assert_called_once_with(network="ws://chain")
        substrate.query.assert_any_call(
            "SubtensorModule", "Tempo", [46], block_hash="0xhead"
        )
        substrate.query_map.assert_called_once_with(
            "SubtensorModule", "Keys", [46], block_hash="0xhead", page_size=512
        )
        self.assertEqual(snapshot, ChainSnapshot(722, 360, 720, ("hotkey-0",), (700,)))

    def test_bittensor_adapter_decodes_raw_account_bytes(self) -> None:
        raw = (
            176, 116, 52, 235, 88, 16, 239, 83, 225, 14, 3, 243, 79, 222, 208, 242,
            75, 195, 154, 124, 220, 208, 95, 210, 106, 238, 217, 235, 197, 120, 72, 23,
        )
        values = {
            "SubnetworkN": 1,
            "Tempo": 360,
            "LastMechansimStepBlock": 720,
            "BlocksSinceLastStep": 2,
            "LastUpdate": [700],
        }
        substrate = mock.Mock()
        substrate.get_chain_finalised_head.return_value = "0xhead"
        substrate.get_block_number.return_value = 722
        substrate.query.side_effect = lambda module, storage, params, block_hash: (
            SimpleNamespace(value=values[storage])
        )
        substrate.query_map.return_value = [(0, SimpleNamespace(value=[raw]))]
        sdk = mock.Mock()
        sdk.subtensor.return_value = mock.Mock(substrate=substrate)

        snapshot = BittensorChain("local", "ws://chain", sdk=sdk).snapshot(46)

        self.assertEqual(
            snapshot.hotkeys,
            ("5G44ofsENjXcNRNLVPKRjPvbqBaiZWFGkYxKawTaUx4GE8b1",),
        )

    def test_report_and_burn_are_each_handled_once(self) -> None:
        burner = _Burner(self.snapshot.hotkeys[12])
        arguments = {
            "network": self.report["network"],
            "netuid": self.report["netuid"],
            "report_url": "https://platform.invalid/latest",
            "platform_signer": self.report["signer"],
            "chain": _Chain(self.snapshot),
            "state": self.state,
            "burner": burner,
            "fetch": lambda url, timeout: self.raw,
        }

        first = run_once(**arguments)
        second = run_once(**arguments)

        self.assertEqual(len(first), 3)
        self.assertEqual(second, ())
        self.assertEqual(burner.calls, [(46, 722)])
        saved = self.state.load()
        self.assertEqual(saved.report_id, self.report["report_id"])
        self.assertEqual(saved.report_digest, self.report["digest"])
        self.assertEqual(saved.report_epoch_end_block, 720)
        self.assertEqual(saved.burn_epoch_end_block, 720)

    def test_chain_last_update_prevents_burn_after_local_state_loss(self) -> None:
        hotkey = self.snapshot.hotkeys[12]
        last_updates = list(self.snapshot.last_updates)
        last_updates[12] = self.report["epoch_end_block"]
        snapshot = ChainSnapshot(
            self.snapshot.finalized_block,
            self.snapshot.tempo,
            self.snapshot.last_step,
            self.snapshot.hotkeys,
            tuple(last_updates),
        )
        burner = _Burner(hotkey)

        run_once(
            network=self.report["network"],
            netuid=self.report["netuid"],
            report_url="https://platform.invalid/latest",
            platform_signer=self.report["signer"],
            chain=_Chain(snapshot),
            state=self.state,
            burner=burner,
            fetch=lambda url, timeout: self.raw,
        )

        self.assertEqual(burner.calls, [])
        self.assertEqual(self.state.load().burn_epoch_end_block, 720)

    def test_failed_burn_is_retried_without_reprocessing_report(self) -> None:
        burner = _Burner(self.snapshot.hotkeys[12], fail=True)
        arguments = {
            "network": self.report["network"],
            "netuid": self.report["netuid"],
            "report_url": "https://platform.invalid/latest",
            "platform_signer": self.report["signer"],
            "chain": _Chain(self.snapshot),
            "state": self.state,
            "burner": burner,
            "fetch": lambda url, timeout: self.raw,
        }

        with self.assertRaises(BurnError):
            run_once(**arguments)
        self.assertIsNone(self.state.load().burn_epoch_end_block)
        burner.fail = False

        records = run_once(**arguments)

        self.assertEqual(records, ())
        self.assertEqual(burner.calls, [(46, 722), (46, 722)])
        self.assertEqual(self.state.load().burn_epoch_end_block, 720)

    def test_cli_requires_platform_signer_before_chain_contact(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(main(["run-once"]), 2)


if __name__ == "__main__":
    unittest.main()
