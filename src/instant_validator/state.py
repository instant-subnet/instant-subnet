"""Small durable state for one Validator."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


class StateError(RuntimeError):
    """The local Validator state is invalid or cannot be saved."""


@dataclass(frozen=True, slots=True)
class ValidatorState:
    report_id: str | None = None
    report_digest: str | None = None
    report_epoch_end_block: int | None = None
    burn_epoch_end_block: int | None = None


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> ValidatorState:
        if not self.path.exists():
            return ValidatorState()
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateError("Validator state is unreadable") from exc
        if not isinstance(value, dict) or set(value) != set(ValidatorState.__slots__):
            raise StateError("Validator state fields are invalid")
        report_id = value["report_id"]
        digest = value["report_digest"]
        report_end = value["report_epoch_end_block"]
        burn_end = value["burn_epoch_end_block"]
        if (report_id is None) != (digest is None) or (report_id is None) != (
            report_end is None
        ):
            raise StateError("Validator report state is incomplete")
        if report_id is not None and (not isinstance(report_id, str) or not report_id):
            raise StateError("Validator report ID is invalid")
        if digest is not None and (not isinstance(digest, str) or not digest):
            raise StateError("Validator report digest is invalid")
        for name, item in (("report epoch", report_end), ("burn epoch", burn_end)):
            if item is not None and (type(item) is not int or item < 0):
                raise StateError(f"Validator {name} is invalid")
        if burn_end is not None and (report_end is None or burn_end > report_end):
            raise StateError("Validator burn epoch is inconsistent")
        return ValidatorState(report_id, digest, report_end, burn_end)

    def save(self, state: ValidatorState) -> None:
        data = (
            json.dumps(asdict(state), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, temporary = tempfile.mkstemp(dir=parent, prefix=".validator-state-")
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                directory = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError as exc:
            raise StateError("Validator state could not be saved") from exc
