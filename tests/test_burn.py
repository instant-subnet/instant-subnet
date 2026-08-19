from __future__ import annotations

import unittest
from types import SimpleNamespace

from instant_validator.burn import FULL_WEIGHT, BittensorBurnWriter, BurnError


class _Subtensor:
    def __init__(self) -> None:
        self.mode = {"Burn": ()}
        self.commit_reveal = False
        self.calls: list[tuple[str, int, list[int]]] = []

    def query_subtensor(self, *, name: str, block: int, params: list[int]):
        self.calls.append((name, block, params))
        return {
            "RecycleOrBurn": self.mode,
            "CommitRevealWeightsEnabled": self.commit_reveal,
            "WeightsVersionKey": 62,
        }[name]

    def get_subnet_owner_hotkey(self, netuid: int, *, block: int) -> str:
        return "5Owner"

    def get_uid_for_hotkey_on_subnet(self, hotkey: str, netuid: int, *, block: int) -> int:
        return 238


class _Sdk:
    def __init__(self) -> None:
        self.chain = _Subtensor()
        self.wallet_value = SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="5Validator")
        )

    def subtensor(self, **kwargs):
        return self.chain

    def wallet(self, **kwargs):
        return self.wallet_value


class BurnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk = _Sdk()
        self.submissions: list[dict] = []
        self.writer = BittensorBurnWriter(
            network="local",
            endpoint="ws://chain",
            wallet_name="validator",
            wallet_hotkey="default",
            wallet_path="/wallets",
            sdk=self.sdk,
            submitter=self._submit,
        )

    def _submit(self, **kwargs):
        self.submissions.append(kwargs)
        return True, "finalized"

    def test_writer_can_only_submit_the_full_owner_burn_vector(self) -> None:
        result = self.writer.submit(netuid=46, finalized_block=722)

        self.assertEqual(result, "finalized")
        self.assertEqual(self.writer.hotkey, "5Validator")
        self.assertEqual(len(self.submissions), 1)
        self.assertEqual(self.submissions[0]["uids"], [238])
        self.assertEqual(self.submissions[0]["weights"], [FULL_WEIGHT])
        self.assertEqual(self.submissions[0]["version_key"], 62)
        self.assertTrue(self.submissions[0]["wait_for_finalization"])

    def test_writer_refuses_non_burn_or_commit_reveal_subnets(self) -> None:
        for mode, commit_reveal in (({"Recycle": ()}, False), ({"Burn": ()}, True)):
            with self.subTest(mode=mode, commit_reveal=commit_reveal):
                self.sdk.chain.mode = mode
                self.sdk.chain.commit_reveal = commit_reveal
                with self.assertRaises(BurnError):
                    self.writer.submit(netuid=46, finalized_block=722)
        self.assertEqual(self.submissions, [])


if __name__ == "__main__":
    unittest.main()
