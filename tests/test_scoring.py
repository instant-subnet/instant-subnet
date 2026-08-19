from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path

from instant_validator.scoring import log_score_records, score_report

FIXTURES = Path(__file__).parent / "fixtures"


class _Handler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = json.loads((FIXTURES / "validator_report_v1.json").read_text())

    def test_records_include_raw_metrics_components_and_weights(self) -> None:
        records = score_report(self.report)

        self.assertEqual([record["uid"] for record in records], [12, 37, 55])
        self.assertTrue(set(self.report["miners"][0]).issubset(records[0]))
        self.assertEqual(records[0]["report_id"], "local-46-361-720")
        self.assertEqual(records[0]["generation_tps_p50"], 90)
        self.assertEqual(records[0]["verified_completion_tokens"], 24_000)
        self.assertEqual(records[0]["speed_bps"], 10_000)
        self.assertEqual(records[0]["score_bps"], 9_990)
        self.assertEqual(records[0]["normalized_weight"], 65_535)

    def test_failed_proof_disqualifies_entire_epoch_without_hiding_components(
        self,
    ) -> None:
        record = score_report(self.report)[2]

        self.assertTrue(record["disqualified"])
        self.assertEqual(record["proof_failed_requests"], 1)
        self.assertEqual(record["raw_score_bps"], 8_294)
        self.assertGreater(record["speed_bps"], 0)
        self.assertEqual(record["score_bps"], 0)
        self.assertEqual(record["normalized_weight"], 0)

    def test_disqualified_miners_do_not_set_relative_maxima(self) -> None:
        eligible = self.report["miners"][0]
        disqualified = {
            **self.report["miners"][2],
            "generation_tps_p50": eligible["generation_tps_p50"] * 10,
            "verified_completion_tokens": eligible["verified_completion_tokens"] * 10,
            "successful_requests": eligible["successful_requests"] * 10,
            "routed_requests": eligible["routed_requests"] * 10,
            "proof_failed_requests": 0,
            "proof_timed_out_requests": 1,
        }

        records = score_report({**self.report, "miners": [disqualified, eligible]})

        self.assertEqual([record["uid"] for record in records], [12, 55])
        self.assertEqual(records[0]["speed_bps"], 10_000)
        self.assertEqual(records[0]["tokens_bps"], 10_000)
        self.assertEqual(records[0]["requests_bps"], 10_000)
        self.assertTrue(records[1]["disqualified"])
        self.assertGreater(records[1]["raw_score_bps"], 10_000)
        self.assertEqual(records[1]["score_bps"], 0)
        self.assertEqual(records[1]["normalized_weight"], 0)

    def test_zero_maxima_and_empty_reports_produce_zero_weights(self) -> None:
        empty = {**self.report, "report_id": "empty", "miners": []}
        self.assertEqual(score_report(empty), ())

        miner = {
            **self.report["miners"][0],
            "generation_tps_p50": 0,
            "verified_completion_tokens": 0,
            "successful_requests": 0,
            "failed_requests": 1,
            "routed_requests": 1,
            "proof_verified_requests": 0,
            "proof_not_available_requests": 1,
        }
        record = score_report({**self.report, "miners": [miner]})[0]
        self.assertEqual(record["speed_bps"], 0)
        self.assertEqual(record["tokens_bps"], 0)
        self.assertEqual(record["requests_bps"], 0)
        self.assertEqual(record["success_bps"], 0)
        self.assertEqual(record["normalized_weight"], 0)

    def test_logger_emits_one_canonical_complete_record_per_miner(self) -> None:
        records = score_report(self.report)
        logger = logging.getLogger("instant-validator-scoring-test")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _Handler()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)

        log_score_records(records, logger)

        self.assertEqual(len(handler.messages), 3)
        logged = json.loads(handler.messages[2].removeprefix("miner_score "))
        self.assertEqual(logged, records[2])
        self.assertEqual(
            handler.messages[2],
            "miner_score "
            + json.dumps(
                records[2],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
