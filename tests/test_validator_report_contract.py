from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures"
REPORT_KEYS = {
    "created_at_ms",
    "digest",
    "epoch_end_block",
    "epoch_start_block",
    "finalized_block",
    "miners",
    "netuid",
    "network",
    "report_id",
    "schema_version",
    "signature",
    "signer",
    "tempo",
}
MINER_KEYS = {
    "completion_tokens",
    "failed_requests",
    "generation_tps_p50",
    "hotkey",
    "proof_failed_requests",
    "proof_not_available_requests",
    "proof_timed_out_requests",
    "proof_verified_requests",
    "routed_requests",
    "successful_requests",
    "uid",
    "verified_completion_tokens",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


class ValidatorReportContractTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = (FIXTURES / "validator_report_v1.json").read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.wire = raw[:-1]
        self.report = json.loads(self.wire)

    def test_report_fixture_is_canonical_and_digest_bound(self) -> None:
        self.assertEqual(self.wire, canonical_json(self.report))
        self.assertEqual(set(self.report), REPORT_KEYS)
        payload = {
            key: value
            for key, value in self.report.items()
            if key not in {"digest", "signature"}
        }
        self.assertEqual(
            self.report["digest"],
            "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest(),
        )
        self.assertEqual(len(bytes.fromhex(self.report["signature"])), 64)

    def test_report_fixture_is_one_finalized_local_epoch(self) -> None:
        self.assertEqual(self.report["network"], "local")
        self.assertEqual(self.report["netuid"], 46)
        self.assertEqual(self.report["tempo"], 360)
        self.assertEqual(self.report["report_id"], "local-46-361-720")
        self.assertEqual(
            self.report["epoch_end_block"] - self.report["epoch_start_block"] + 1,
            self.report["tempo"],
        )
        self.assertGreaterEqual(
            self.report["finalized_block"], self.report["epoch_end_block"]
        )

    def test_miner_rows_are_sorted_unique_and_balanced(self) -> None:
        miners = self.report["miners"]
        self.assertEqual([row["uid"] for row in miners], [12, 37, 55])
        self.assertEqual(len({row["hotkey"] for row in miners}), len(miners))
        for row in miners:
            self.assertEqual(set(row), MINER_KEYS)
            self.assertEqual(
                row["successful_requests"] + row["failed_requests"],
                row["routed_requests"],
            )
            self.assertEqual(
                row["proof_verified_requests"]
                + row["proof_failed_requests"]
                + row["proof_timed_out_requests"]
                + row["proof_not_available_requests"],
                row["routed_requests"],
            )
            self.assertLessEqual(
                row["verified_completion_tokens"], row["completion_tokens"]
            )
            self.assertLessEqual(
                row["proof_verified_requests"], row["successful_requests"]
            )
            self.assertLessEqual(
                row["proof_not_available_requests"], row["failed_requests"]
            )
            if row["proof_verified_requests"]:
                self.assertGreater(row["verified_completion_tokens"], 0)
                self.assertGreater(row["generation_tps_p50"], 0)
            else:
                self.assertEqual(row["verified_completion_tokens"], 0)
                self.assertEqual(row["generation_tps_p50"], 0)

    def test_scoring_fixture_uses_frozen_relative_integer_math(self) -> None:
        miners = self.report["miners"]
        clean = [
            row
            for row in miners
            if row["proof_failed_requests"] == 0
            and row["proof_timed_out_requests"] == 0
        ]
        maxima = {
            "speed": max(row["generation_tps_p50"] for row in clean),
            "tokens": max(row["verified_completion_tokens"] for row in clean),
            "requests": max(row["successful_requests"] for row in clean),
        }
        calculated = []
        raw_rows = []
        for row in miners:
            speed = row["generation_tps_p50"] * 10_000 // maxima["speed"]
            tokens = row["verified_completion_tokens"] * 10_000 // maxima["tokens"]
            requests = row["successful_requests"] * 10_000 // maxima["requests"]
            success = row["successful_requests"] * 10_000 // row["routed_requests"]
            raw_score = (60 * speed + 25 * tokens + 10 * requests + 5 * success) // 100
            disqualified = (
                row["proof_failed_requests"] > 0 or row["proof_timed_out_requests"] > 0
            )
            score = 0 if disqualified else raw_score
            raw_rows.append(
                (row, speed, tokens, requests, success, raw_score, score, disqualified)
            )
        highest = max(item[-2] for item in raw_rows)
        for (
            row,
            speed,
            tokens,
            requests,
            success,
            raw_score,
            score,
            disqualified,
        ) in raw_rows:
            calculated.append(
                {
                    "disqualified": disqualified,
                    "normalized_weight": score * 65_535 // highest if score else 0,
                    "raw_score_bps": raw_score,
                    "requests_bps": requests,
                    "score_bps": score,
                    "speed_bps": speed,
                    "success_bps": success,
                    "tokens_bps": tokens,
                    "uid": row["uid"],
                }
            )
        expected = json.loads((FIXTURES / "validator_scores_v1.json").read_text())[
            "scores"
        ]
        self.assertEqual(calculated, expected)


if __name__ == "__main__":
    unittest.main()
