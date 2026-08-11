"""Server-sent-event parsing for streamed completions.

This module owns one security-critical question: **which bytes does a receipt
commit to?**  A miner signs over the concatenated ``data:`` payloads of the
stream it served (``StreamOutcome.assembled`` in ``miner/upstream.py``), so any
observer that wants to verify that signature has to reassemble exactly the same
bytes.  Two independent implementations of "exactly the same bytes" is how
receipt verification quietly stops working, which is why the platform observer
and the validator's direct prober share this one.

Parsing lives here; *policy* does not.  Deciding whether a receipt is
acceptable — whose hotkey may sign, which request it must bind, what counts as
a probe failure — differs between the platform and a validator, so each caller
keeps its own verification and this module only reports what it saw.
"""

from __future__ import annotations

import json
import time
from typing import Any

from . import receipts


def frame_boundary(buffer: bytearray) -> tuple[int, int] | None:
    """Index and width of the first frame separator, LF or CRLF, whichever is first."""
    lf = buffer.find(b"\n\n")
    crlf = buffer.find(b"\r\n\r\n")
    choices = [(lf, 2), (crlf, 4)]
    valid = [choice for choice in choices if choice[0] >= 0]
    return min(valid, default=None, key=lambda choice: choice[0])


def has_content(raw: bytes) -> bool:
    """True if this data frame carries model output the user would perceive.

    ``reasoning_content`` counts.  On gpt-oss the reasoning stream is what
    arrives first and it is what a waiting user sees moving, so treating it as
    "nothing yet" would report a time-to-first-token several hundred
    milliseconds later than the moment the response actually started.
    """
    try:
        payload = json.loads(raw)
    except ValueError:
        return False
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
    return False


def observed_tps_milli(tokens: int, *, total_ms: int, ttft_ms: int) -> int:
    """Milli-tokens/sec over the *generation* window, excluding time to first token.

    Dividing by total time instead would fold prefill, queueing, and the
    observer's network round trip into the denominator and report a throughput
    the miner never produced — badly understating a miner that is simply far
    away from the validator.

    Known limit, worth knowing before trusting the absolute number: on a link
    with real latency the first content frame tends to arrive coalesced with
    several later ones, so the observed first-token moment is *late*. That makes
    ttft pessimistic and, because it shortens the denominator here, throughput
    optimistic. Measured from Europe against the launch H200 this read 451
    tok/s where the box itself sustains about 335. An observer in the same
    region sees far less of it. Both distortions favour caution in the miner's
    direction for latency and against it for throughput, and neither changes
    scoring while both terms sit clamped at their SLO targets.
    """
    generation_ms = max(1, total_ms - ttft_ms)
    return max(0, int(tokens)) * 1_000_000 // generation_ms


class StreamCommitment:
    """Reassembles a streamed completion into the bytes its receipt commits to.

    Feed it wire chunks; it returns the ordinary data frames (so a relay can
    forward them) and accumulates :attr:`assembled` for receipt verification.
    Named events are handled rather than forwarded: the private receipt event is
    captured, an error event is recorded, and anything else is a protocol error
    — forwarding an arbitrary named event would let a miner show a reader bytes
    it never signed.
    """

    def __init__(self, *, started_perf: float) -> None:
        self.started_perf = started_perf
        self.first_content_perf: float | None = None
        self.assembled = bytearray()
        self.buffer = bytearray()
        self.receipt_payload: dict[str, Any] | None = None
        self.receipt_seen = False
        self.error_event = False
        self.protocol_error: str | None = None

    def feed(self, chunk: bytes, *, now: float | None = None) -> list[bytes]:
        forwarded: list[bytes] = []
        self.buffer.extend(chunk)
        while True:
            boundary = frame_boundary(self.buffer)
            if boundary is None:
                return forwarded
            index, width = boundary
            frame = bytes(self.buffer[:index])
            del self.buffer[: index + width]
            if self._frame(frame, now=now):
                # Normalise the separator only; the frame bytes are otherwise
                # relayed exactly.
                forwarded.append(frame + b"\n\n")

    def _frame(self, raw: bytes, *, now: float | None = None) -> bool:
        lines = raw.replace(b"\r\n", b"\n").split(b"\n")
        event = next(
            (line[6:].strip().decode() for line in lines if line.startswith(b"event:")),
            None,
        )
        data_lines = [line[5:].strip() for line in lines if line.startswith(b"data:")]
        if event == receipts.SSE_RECEIPT_EVENT:
            self.receipt_seen = True
            if not data_lines:
                self.protocol_error = "miner emitted a receipt event without data"
                return False
            try:
                payload = json.loads(b"\n".join(data_lines))
                self.receipt_payload = payload if isinstance(payload, dict) else None
            except ValueError:
                self.receipt_payload = None
            return False
        if event == "error":
            self.error_event = True
            return False
        if event is not None:
            # Only ordinary OpenAI data frames are part of the receipt's
            # response commitment.
            self.protocol_error = f"miner emitted unsupported SSE event {event!r}"
            return False
        if self.protocol_error is not None or self.error_event:
            return False
        if not data_lines:
            # SSE comment heartbeats are transport metadata, not model output,
            # and keep long reasoning requests alive through intermediaries.
            nonblank = [line.strip() for line in lines if line.strip()]
            return bool(nonblank) and all(line.startswith(b":") for line in nonblank)
        data = b"\n".join(data_lines)
        if data == b"[DONE]":
            return False
        try:
            payload = json.loads(data)
        except ValueError:
            self.protocol_error = "miner emitted a non-JSON SSE data frame"
            return False
        if not isinstance(payload, dict):
            self.protocol_error = "miner emitted a non-object SSE data frame"
            return False
        if payload.get("error") is not None:
            self.error_event = True
            return False
        self.assembled.extend(data)
        if self.first_content_perf is None and has_content(data):
            self.first_content_perf = _perf(now)
        return True

    def ttft_ms(self, total_ms: int) -> int:
        if self.first_content_perf is None:
            return total_ms
        return max(0, int((self.first_content_perf - self.started_perf) * 1000))

    def text(self) -> str:
        """The concatenated visible ``content`` deltas, for answer checking.

        Deliberately ignores ``reasoning_content``: reasoning is the model
        talking to itself and is not the answer.  ``has_content`` counts it for
        timing because a user sees it move, which is a different question.
        """
        parts: list[str] = []
        for frame in self._data_frames():
            for choice in frame.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if isinstance(piece, str):
                    parts.append(piece)
        return "".join(parts)

    def _data_frames(self) -> list[dict[str, Any]]:
        """Re-parse assembled bytes as the sequence of JSON objects they are."""
        frames: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        raw = bytes(self.assembled).decode("utf-8", "replace")
        index = 0
        while index < len(raw):
            try:
                value, end = decoder.raw_decode(raw, index)
            except ValueError:
                # A malformed tail truncates the text rather than raising, which
                # fails the answer check closed. That is the right direction: we
                # cannot claim a miner answered correctly on bytes we could not
                # read.
                break
            if end <= index:
                break  # no forward progress; never spin inside a validator
            if isinstance(value, dict):
                frames.append(value)
            index = end
        return frames


def _perf(now: float | None) -> float:
    return time.perf_counter() if now is None else now
