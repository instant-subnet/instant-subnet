import json

import pytest
from conftest import FIXTURE_NOW_MS, sign_fixture

from instant_validator.report import ReportError, parse_report

SIGNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def parse(raw, **overrides):
    arguments = {
        "expected_network": "finney",
        "expected_netuid": 46,
        "expected_signer": SIGNER,
        "max_age_seconds": 86_400,
        "future_skew_seconds": 60,
        "now_ms": FIXTURE_NOW_MS,
    }
    arguments.update(overrides)
    return parse_report(raw, **arguments)


def test_golden_report_signature_and_contract(report_raw):
    report = parse(report_raw)

    assert report.report_id == "finney-46-5000000-5000099"
    assert report.period_end_block == 5_000_099
    assert [(row.uid, row.hotkey) for row in report.miners] == [
        (12, "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"),
        (37, "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"),
    ]


def test_wrong_configured_signer_is_rejected(report_raw):
    with pytest.raises(ReportError, match="configured platform signer"):
        parse(report_raw, expected_signer="5WrongSigner")


def test_wrong_network_or_netuid_is_rejected(report_raw):
    with pytest.raises(ReportError, match="expected finney/47"):
        parse(report_raw, expected_netuid=47)


def test_stale_and_future_reports_are_rejected(report_raw):
    with pytest.raises(ReportError, match="stale"):
        parse(report_raw, now_ms=FIXTURE_NOW_MS + 86_401_000)
    with pytest.raises(ReportError, match="future"):
        parse(report_raw, now_ms=FIXTURE_NOW_MS - 71_000)


def test_tampering_breaks_the_digest(report_dict):
    report_dict["miners"][0]["successes"] = 97
    raw = json.dumps(report_dict, separators=(",", ":")).encode()
    with pytest.raises(ReportError, match="request counts do not balance"):
        parse(raw)


def test_valid_digest_with_invalid_signature_is_rejected(report_dict):
    raw = sign_fixture(report_dict)
    value = json.loads(raw)
    value["signature"] = "0" * 128
    with pytest.raises(ReportError, match="signature is invalid"):
        parse(json.dumps(value).encode())


def test_float_and_unbalanced_toploc_counts_are_rejected(report_dict):
    report_dict["miners"][0]["tokens_per_second_p50"] = 90.5
    with pytest.raises(ReportError, match="floating-point"):
        parse(json.dumps(report_dict).encode())

    report_dict["miners"][0]["tokens_per_second_p50"] = 90
    report_dict["miners"][0]["toploc_timed_out"] = 0
    with pytest.raises(ReportError, match="TOPLOC counts do not balance"):
        parse(json.dumps(report_dict).encode())


def test_empty_report_never_reaches_scoring(report_dict):
    report_dict["miners"] = []
    with pytest.raises(ReportError, match="non-empty"):
        parse(sign_fixture(report_dict))


def test_duplicate_json_keys_are_rejected(report_raw):
    raw = report_raw.replace(b'"netuid": 46,', b'"netuid": 46, "netuid": 46,')
    with pytest.raises(ReportError, match="duplicate JSON key: netuid"):
        parse(raw)
