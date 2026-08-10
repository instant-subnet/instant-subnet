"""Durable platform request telemetry and receipt evidence."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Literal

from ..protocol import receipts
from ..protocol.schemas import MinerStatsWindow, StatsResponse

SCHEMA_VERSION = 2
# An active request is omitted from a scoring snapshot. If the process crashed
# before finishing it, the row becomes a failure after this grace period rather
# than disappearing from telemetry forever. This is twice the deployed 300s
# upstream timeout and still well inside the one-hour stats window.
UNFINISHED_GRACE_MS = 10 * 60 * 1000

#: Keys are 32 random bytes; the prefix is for display and lookup only, never
#: for authentication. ``digest`` is an HMAC-SHA256 under a server-held pepper,
#: so a stolen database alone does not yield usable credentials.
_SCHEMA_KEYS = """
CREATE TABLE api_keys (
    key_id      TEXT PRIMARY KEY,
    prefix      TEXT NOT NULL,
    last4       TEXT NOT NULL,
    digest      TEXT NOT NULL UNIQUE,
    label       TEXT NOT NULL DEFAULT '',
    created_ms  INTEGER NOT NULL,
    revoked_ms  INTEGER
);
CREATE INDEX api_keys_by_digest ON api_keys (digest);
"""

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
""" + _SCHEMA_KEYS

#: Upgrades from an older on-disk schema, applied in order. Without this the
#: gateway refuses to open a database written by a previous version, which
#: would mean discarding the request and receipt history that validator scoring
#: reads. Each step must be additive: a migration that rewrites rows can lose
#: evidence a miner has already been paid for.
_MIGRATIONS: dict[int, str] = {1: _SCHEMA_KEYS}


class KeyConflictError(ValueError):
    """A key id and digest refer to different existing credentials."""


class KeyRevokedError(ValueError):
    """An exact registration retry refers to a permanently revoked key."""


class PlatformState:
    """One SQLite connection protected for FastAPI/TestClient thread handoff."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- API keys ---------------------------------------------------------
    #
    # The gateway stores only digests. A raw key exists in this process for the
    # length of one registration call and is never written anywhere.

    def register_key(
        self,
        *,
        key_id: str,
        prefix: str,
        last4: str,
        digest: str,
        label: str,
        created_ms: int,
    ) -> Literal["registered", "already_registered"]:
        """Record one exact key-id/digest pair, safely retrying lost responses.

        A retry of the same active pair is idempotent. Reusing either the id or
        digest for a different credential is a conflict, and a revoked secret
        can never be registered again.
        """
        with self._lock, self._db:
            by_id = self._db.execute(
                "SELECT key_id, digest, revoked_ms FROM api_keys WHERE key_id=?",
                (key_id,),
            ).fetchone()
            by_digest = self._db.execute(
                "SELECT key_id, digest, revoked_ms FROM api_keys WHERE digest=?",
                (digest,),
            ).fetchone()
            if by_id is not None or by_digest is not None:
                same_pair = (
                    by_id is not None
                    and by_digest is not None
                    and by_id["key_id"] == by_digest["key_id"] == key_id
                    and by_id["digest"] == by_digest["digest"] == digest
                )
                if not same_pair:
                    raise KeyConflictError(
                        "key_id or key digest is already registered to another key"
                    )
                if by_id["revoked_ms"] is not None:
                    raise KeyRevokedError("this key was revoked and cannot be reused")
                return "already_registered"
            self._db.execute(
                "INSERT INTO api_keys "
                "(key_id, prefix, last4, digest, label, created_ms, revoked_ms) "
                "VALUES (?,?,?,?,?,?,NULL)",
                (key_id, prefix, last4, digest, label, int(created_ms)),
            )
            return "registered"

    def revoke_key(self, *, key_id: str, revoked_ms: int) -> bool:
        """Revoke by id. Returns False if the key is unknown or already revoked."""
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE api_keys SET revoked_ms=? "
                "WHERE key_id=? AND revoked_ms IS NULL",
                (int(revoked_ms), key_id),
            )
            return cur.rowcount > 0

    def key_is_active(self, digest: str) -> bool:
        """True when this digest names a key that exists and is not revoked."""
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM api_keys WHERE digest=? AND revoked_ms IS NULL",
                (digest,),
            ).fetchone()
        return row is not None

    def list_keys(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key_id, prefix, last4, label, created_ms, revoked_ms "
                "FROM api_keys ORDER BY created_ms, key_id"
            ).fetchall()
        return [dict(row) for row in rows]

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
                "AND started_ms<=? AND (finished_ms IS NOT NULL OR started_ms<=?) "
                "ORDER BY started_ms, request_id",
                (
                    miner_hotkey,
                    int(window_start_ms),
                    int(window_end_ms),
                    int(generated_at_ms) - UNFINISHED_GRACE_MS,
                ),
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
            # A miner-signed receipt can name any attestation id. Until a
            # separate verifier persists fresh hardware evidence bound to this
            # miner/model, repeating that id proves nothing and must never turn
            # a public TEE indicator green.
            attestation_ok=False,
            attestation_id=None,
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
            _apply_schema(db, _SCHEMA, SCHEMA_VERSION)
        elif version > SCHEMA_VERSION:
            # Forward-dated database: this binary is older than the one that
            # wrote it. Refuse rather than guess at a schema we do not know.
            raise RuntimeError(
                f"platform database schema {version} is newer than this build "
                f"supports ({SCHEMA_VERSION}); deploy the newer release or "
                f"restore a matching backup"
            )
        elif version < SCHEMA_VERSION:
            for step in range(version, SCHEMA_VERSION):
                script = _MIGRATIONS.get(step)
                if script is None:
                    raise RuntimeError(
                        f"no migration from platform database schema {step}; "
                        f"cannot reach {SCHEMA_VERSION}"
                    )
                _apply_schema(db, script, step + 1)
    except Exception:
        db.close()
        raise
    return PlatformState(db)


def _apply_schema(db: sqlite3.Connection, script: str, version: int) -> None:
    """Apply DDL and its version marker in one explicit SQLite transaction.

    ``executescript`` commits before executing its input, even inside a Python
    connection context manager. Putting BEGIN/COMMIT inside the script is what
    prevents a failed migration from leaving half-created tables behind while
    ``user_version`` still names the old schema.
    """
    try:
        db.executescript(
            "BEGIN IMMEDIATE;\n"
            f"{script}\n"
            f"PRAGMA user_version={int(version)};\n"
            "COMMIT;"
        )
    except Exception:
        if db.in_transaction:
            db.rollback()
        raise


def _percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent / 100) - 1)
    return ordered[index]
