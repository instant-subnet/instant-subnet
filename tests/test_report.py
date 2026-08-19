from __future__ import annotations

import json
import unittest
from pathlib import Path

from instant_validator.report import ReportError, canonical_json, parse_report

FIXTURES = Path(__file__).parent / "fixtures"


class ReportParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = (FIXTURES / "validator_report_v1.json").read_bytes().rstrip(b"\n")
        self.report = json.loads(self.raw)
        self.arguments = {
            "expected_network": self.report["network"],
            "expected_netuid": self.report["netuid"],
            "expected_signer": self.report["signer"],
        }

    def test_cross_repo_fixture_authenticates(self) -> None:
        parsed = parse_report(self.raw, **self.arguments)

        self.assertEqual(parsed, self.report)

    def test_payload_change_fails_closed(self) -> None:
        changed = {**self.report, "created_at_ms": self.report["created_at_ms"] + 1}

        with self.assertRaisesRegex(ReportError, "digest"):
            parse_report(canonical_json(changed), **self.arguments)

    def test_configuration_binding_fails_closed(self) -> None:
        for key, value in (
            ("expected_network", "finney"),
            ("expected_netuid", 5),
            ("expected_signer", self.report["miners"][0]["hotkey"]),
        ):
            arguments = {**self.arguments, key: value}
            with self.subTest(key=key), self.assertRaises(ReportError):
                parse_report(self.raw, **arguments)

    def test_noncanonical_and_duplicate_json_fail_closed(self) -> None:
        with self.assertRaisesRegex(ReportError, "canonical"):
            parse_report(self.raw + b"\n", **self.arguments)
        with self.assertRaisesRegex(ReportError, "duplicate"):
            parse_report(b'{"schema_version":1,"schema_version":1}', **self.arguments)


if __name__ == "__main__":
    unittest.main()
