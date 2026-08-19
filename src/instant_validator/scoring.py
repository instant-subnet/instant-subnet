"""Deterministic scoring of one validated Platform report."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .report import MinerMetrics

BPS = 10_000
MAX_WEIGHT = 65_535


def _relative(value: int, maximum: int) -> int:
    return 0 if maximum == 0 else value * BPS // maximum


def score_report(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return complete per-Miner scoring records ordered by UID."""

    report_id = str(report["report_id"])
    miners = sorted(report["miners"], key=lambda row: int(row["uid"]))
    eligible = [
        row
        for row in miners
        if int(row["proof_failed_requests"]) == 0
        and int(row["proof_timed_out_requests"]) == 0
    ]
    maxima = {
        "speed": max((int(row["generation_tps_p50"]) for row in eligible), default=0),
        "tokens": max(
            (int(row["verified_completion_tokens"]) for row in eligible), default=0
        ),
        "requests": max((int(row["successful_requests"]) for row in eligible), default=0),
    }

    records = []
    for row in miners:
        routed = int(row["routed_requests"])
        successful = int(row["successful_requests"])
        speed_bps = _relative(int(row["generation_tps_p50"]), maxima["speed"])
        tokens_bps = _relative(int(row["verified_completion_tokens"]), maxima["tokens"])
        requests_bps = _relative(successful, maxima["requests"])
        success_bps = 0 if routed == 0 else successful * BPS // routed
        raw_score_bps = (
            60 * speed_bps + 25 * tokens_bps + 10 * requests_bps + 5 * success_bps
        ) // 100
        disqualified = (
            int(row["proof_failed_requests"]) > 0
            or int(row["proof_timed_out_requests"]) > 0
        )
        records.append(
            {
                "report_id": report_id,
                **row,
                "speed_bps": speed_bps,
                "tokens_bps": tokens_bps,
                "requests_bps": requests_bps,
                "success_bps": success_bps,
                "raw_score_bps": raw_score_bps,
                "disqualified": disqualified,
                "score_bps": 0 if disqualified else raw_score_bps,
            }
        )

    highest = max((int(record["score_bps"]) for record in records), default=0)
    for record in records:
        score = int(record["score_bps"])
        record["normalized_weight"] = (
            0 if score == 0 or highest == 0 else score * MAX_WEIGHT // highest
        )
    return tuple(records)


def log_score_records(records: Iterable[Mapping[str, Any]], logger: logging.Logger) -> None:
    """Write one canonical operational log entry for every Miner score."""

    for record in records:
        logger.info(
            "miner_score %s",
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )


def score_miners(miners: Iterable[MinerMetrics]) -> dict[int, int]:
    """Apply the current formula to the Validator's parsed report rows."""

    rows = [
        {
            "uid": miner.uid,
            "hotkey": miner.hotkey,
            "routed_requests": miner.requests,
            "successful_requests": miner.successes,
            "generation_tps_p50": miner.tokens_per_second_p50,
            "verified_completion_tokens": miner.completion_tokens,
            "proof_failed_requests": miner.toploc_failed,
            "proof_timed_out_requests": miner.toploc_timed_out,
        }
        for miner in miners
    ]
    return {
        int(record["uid"]): int(record["score_bps"])
        for record in score_report({"report_id": "parsed", "miners": rows})
    }


def normalize_weights(scores: Mapping[int, int]) -> dict[int, int]:
    """Normalize positive scores using the current floor-based formula."""

    positive = {int(uid): int(score) for uid, score in scores.items() if score > 0}
    if not positive:
        return {}
    highest = max(positive.values())
    return {uid: positive[uid] * MAX_WEIGHT // highest for uid in sorted(positive)}
