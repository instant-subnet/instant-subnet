"""Durable platform request telemetry and receipt evidence."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

from ..protocol import receipts
from ..protocol.schemas import MinerStatsWindow, StatsResponse

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE requests (
    request_id       TEXT PRIMARY KEY,
    miner_hotkey     TEXT    NOT NULL,
    started_ms       INTEGER NOT NULL,
    finished_ms      INTEGER,
    stream           INTEGER NOT NULL,
    status_code      INTEGER,
    success          INTEGER NOT NULL DEFAULT 0,
    clean_reject     INTEGER NOT NULL DEFAULT 0,
    ttft_ms          INTEGER,
    total_ms         INTEGER,
    tps_milli        INTEGER,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    receipt_json     TEXT,
    receipt_seen     INTEGER NOT NULL DEFAULT 0,
    receipt_verified INTEGER NOT NULL DEFAULT 0,
    error            TEXT
);
CREATE INDEX requests_by_window ON requests (miner_hotkey, started_ms);
"""


class PlatformState:
    """One SQLite connection protected for FastAPI/TestClient thread handoff."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def begin(
        self,
        *,
        request_id: str,
        miner_hotkey: str,
        started_ms: int,
        stream: bool,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO requests (request_id, miner_hotkey, started_ms, stream) "
                "VALUES (?,?,?,?)",
                (request_id, miner_hotkey, int(started_ms), int(stream)),
            )

    def finish(
        self,
        *,
        request_id: str,
        finished_ms: int,
        status_code: int | None,
        success: bool,
        clean_reject: bool = False,
        ttft_ms: int | None = None,
        total_ms: int | None = None,
        tps_milli: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        signed_receipt: receipts.SignedReceipt | None = None,
        receipt_seen: bool = False,
        receipt_verified: bool = False,
        error: str | None = None,
    ) -> None:
        if success and not receipt_verified:
            raise ValueError("a successful platform request requires a verified receipt")
        if receipt_verified and (not receipt_seen or signed_receipt is None):
            raise ValueError("receipt_verified requires stored receipt evidence")
        if success and clean_reject:
            raise ValueError("a clean reject cannot also be a success")
        receipt_json = (
            json.dumps(
                signed_receipt.to_payload(), separators=(",", ":"), sort_keys=True
            )
            if signed_receipt is not None
            else None
        )
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE requests SET finished_ms=?, status_code=?, success=?, "
                "clean_reject=?, ttft_ms=?, total_ms=?, tps_milli=?, prompt_tokens=?, "
                "completion_tokens=?, receipt_json=?, receipt_seen=?, "
                "receipt_verified=?, error=? WHERE request_id=?",
                (
                    int(finished_ms),
                    status_code,
                    int(success),
                    int(clean_reject),
                    ttft_ms,
                    total_ms,
                    tps_milli,
                    int(prompt_tokens),
                    int(completion_tokens),
                    receipt_json,
                    int(receipt_seen),
                    int(receipt_verified),
                    error,
                    request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"unknown platform request id {request_id}")

    def stats(
        self,
        *,
        miner_hotkey: str,
        miner_uid: int,
        window_start_ms: int,
        window_end_ms: int,
        generated_at_ms: int,
        block_start: int = 0,
        block_end: int = 0,
    ) -> StatsResponse:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM requests WHERE miner_hotkey=? AND started_ms>=? "
                "AND started_ms<=? ORDER BY started_ms, request_id",
                (miner_hotkey, int(window_start_ms), int(window_end_ms)),
            ).fetchall()

        requests_count = len(rows)
        successes = sum(int(row["success"]) for row in rows)
        failures = requests_count - successes
        clean_rejects = sum(int(row["clean_reject"]) for row in rows)
        successful = [row for row in rows if row["success"]]
        ttfts = [int(row["ttft_ms"]) for row in successful if row["ttft_ms"] is not None]
        tps = [int(row["tps_milli"]) for row in successful if row["tps_milli"] is not None]

        signed_receipts: list[receipts.SignedReceipt] = []
        for row in rows:
            if not row["receipt_verified"] or not row["receipt_json"]:
                continue
            try:
                signed_receipts.append(
                    receipts.SignedReceipt.from_payload(json.loads(row["receipt_json"]))
                )
            except (KeyError, TypeError, ValueError):
                # A damaged row cannot be proof. Keep it in requests/seen, but
                # do not count it as verified or commit it into the root.
                continue

        attestation_ids = {
            item.receipt.attestation_id
            for item in signed_receipts
            if item.receipt.attestation_id
        }
        miner = MinerStatsWindow(
            hotkey=miner_hotkey,
            uid=miner_uid,
            requests=requests_count,
            successes=successes,
            failures=failures,
            clean_rejects=clean_rejects,
            served=successes,
            ttft_p50_ms=_percentile(ttfts, 50),
            ttft_p95_ms=_percentile(ttfts, 95),
            tokens_per_s_p50=_percentile(tps, 50) // 1000,
            tokens_per_s_p95=_percentile(tps, 95) // 1000,
            success_rate_bps=(successes * 10_000 // requests_count)
            if requests_count
            else 0,
            prompt_tokens=sum(int(row["prompt_tokens"]) for row in successful),
            completion_tokens=sum(
                int(row["completion_tokens"]) for row in successful
            ),
            receipts_seen=sum(int(row["receipt_seen"]) for row in rows),
            receipts_verified=len(signed_receipts),
            attestation_ok=(
                bool(successes)
                and len(signed_receipts) == successes
                and all(item.receipt.attestation_id for item in signed_receipts)
                and len(attestation_ids) == 1
            ),
            attestation_id=next(iter(attestation_ids))
            if len(attestation_ids) == 1
            else None,
        )
        return StatsResponse(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            block_start=block_start,
            block_end=block_end,
            miners=[miner],
            receipt_merkle_root=receipts.merkle_root(signed_receipts),
            total_requests=requests_count,
            generated_at_ms=generated_at_ms,
        )


def open_state(path: str | Path) -> PlatformState:
    raw = str(path)
    if raw != ":memory:":
        Path(raw).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        raw = str(Path(raw).expanduser())
    db = sqlite3.connect(raw, check_same_thread=False)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            with db:
                db.executescript(_SCHEMA)
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        elif version != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported platform database schema {version}; expected {SCHEMA_VERSION}"
            )
    except Exception:
        db.close()
        raise
    return PlatformState(db)


def _percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent / 100) - 1)
    return ordered[index]
