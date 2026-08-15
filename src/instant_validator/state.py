"""One-file state for the last successfully applied report."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .report import PlatformReport


class StateError(RuntimeError):
    """The validator state file is malformed or cannot be persisted."""


@dataclass(frozen=True, slots=True)
class AppliedReport:
    schema_version: int
    report_id: str
    period_end_block: int
    digest: str
    applied_at_ms: int


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> AppliedReport | None:
        if not self.path.exists():
            return None
        try:
            value: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError(f"cannot read validator state: {exc}") from exc
        expected = {
            "schema_version",
            "report_id",
            "period_end_block",
            "digest",
            "applied_at_ms",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise StateError("validator state has an invalid shape")
        if value["schema_version"] != 1:
            raise StateError("validator state schema_version must be 1")
        if not isinstance(value["report_id"], str) or not value["report_id"]:
            raise StateError("validator state report_id is invalid")
        for field in ("period_end_block", "applied_at_ms"):
            if type(value[field]) is not int or value[field] < 0:
                raise StateError(f"validator state {field} is invalid")
        if not isinstance(value["digest"], str):
            raise StateError("validator state digest is invalid")
        return AppliedReport(**value)

    def is_applied(self, report: PlatformReport) -> bool:
        current = self.load()
        return current is not None and (
            current.report_id == report.report_id
            or current.period_end_block >= report.period_end_block
        )

    def mark_applied(self, report: PlatformReport, *, applied_at_ms: int) -> None:
        record = AppliedReport(
            schema_version=1,
            report_id=report.report_id,
            period_end_block=report.period_end_block,
            digest=report.digest,
            applied_at_ms=applied_at_ms,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as exc:
            raise StateError(f"cannot persist validator state: {exc}") from exc
