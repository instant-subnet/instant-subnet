"""Shared SSE parsing: which bytes a receipt commits to, and when output began.

This module is used by both the platform observer and the validator's direct
prober, so a bug here breaks receipt verification on both sides at once.
"""

from __future__ import annotations

import json

from instant.protocol import receipts, sse


def frame(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def delta(content: str | None = None, *, reasoning: str | None = None) -> bytes:
    inner: dict[str, object] = {}
    if content is not None:
        inner["content"] = content
    if reasoning is not None:
        inner["reasoning_content"] = reasoning
    return frame({"choices": [{"delta": inner}]})


def wire(*frames: bytes, separator: bytes = b"\n\n") -> bytes:
    return b"".join(b"data: " + f + separator for f in frames)


def test_assembled_is_the_concatenated_data_payloads():
    # This is the contract with miner/upstream.py's StreamOutcome.assembled:
    # the receipt hashes these bytes, not the wire bytes.
    a, b = delta("41 "), delta("42")
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(wire(a, b), now=1.0)
    assert bytes(commitment.assembled) == a + b


def test_done_and_the_receipt_event_are_not_part_of_the_commitment():
    a = delta("41")
    signed = {"receipt": {"x": 1}, "signature": "sig"}
    payload = wire(a) + b"data: [DONE]\n\n"
    payload += (
        b"event: "
        + receipts.SSE_RECEIPT_EVENT.encode()
        + b"\ndata: "
        + json.dumps(signed, separators=(",", ":")).encode()
        + b"\n\n"
    )
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(payload, now=1.0)

    assert bytes(commitment.assembled) == a
    assert commitment.receipt_seen is True
    assert commitment.receipt_payload == signed
    assert commitment.protocol_error is None


def test_crlf_framing_parses_identically():
    a, b = delta("41 "), delta("42")
    lf = sse.StreamCommitment(started_perf=0.0)
    lf.feed(wire(a, b), now=1.0)
    crlf = sse.StreamCommitment(started_perf=0.0)
    crlf.feed(wire(a, b, separator=b"\r\n\r\n"), now=1.0)
    assert bytes(lf.assembled) == bytes(crlf.assembled)


def test_a_chunk_split_mid_frame_still_assembles():
    # Wire chunks do not respect frame boundaries.
    a, b = delta("41 "), delta("42")
    payload = wire(a, b)
    commitment = sse.StreamCommitment(started_perf=0.0)
    for index in range(0, len(payload), 7):
        commitment.feed(payload[index : index + 7], now=1.0)
    assert bytes(commitment.assembled) == a + b


def test_a_heartbeat_comment_is_not_output():
    a = delta("41")
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(b": keep-alive\n\n" + wire(a), now=1.0)
    assert bytes(commitment.assembled) == a
    assert commitment.protocol_error is None


def test_an_unsupported_named_event_is_a_protocol_error():
    # Forwarding an arbitrary named event would let a miner show a reader bytes
    # it never signed.
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(b"event: sneaky\ndata: {}\n\n", now=1.0)
    assert "sneaky" in (commitment.protocol_error or "")


def test_an_error_event_is_recorded():
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(b"event: error\ndata: {}\n\n", now=1.0)
    assert commitment.error_event is True


def test_a_non_json_data_frame_is_a_protocol_error():
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(b"data: not-json\n\n", now=1.0)
    assert commitment.protocol_error == "miner emitted a non-JSON SSE data frame"


def test_ttft_is_the_first_content_frame_not_the_first_byte():
    # A stream can open with metadata frames carrying no output. Timing from the
    # first byte would report a token that had not been generated yet.
    #
    # Offsets are binary-exact quarters so the assertion pins the parsing rather
    # than the sub-millisecond truncation in ttft_ms.
    empty = delta("")
    commitment = sse.StreamCommitment(started_perf=10.0)
    commitment.feed(wire(empty), now=10.125)
    commitment.feed(wire(delta("41")), now=10.25)
    assert commitment.ttft_ms(total_ms=500) == 250


def test_reasoning_counts_as_first_output():
    # On gpt-oss the reasoning stream arrives first and is what a waiting user
    # sees moving, so it is when the response started.
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(wire(delta(reasoning="thinking")), now=0.3)
    assert commitment.ttft_ms(total_ms=900) == 300


def test_ttft_falls_back_to_total_when_nothing_was_produced():
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(b"data: [DONE]\n\n", now=1.0)
    assert commitment.ttft_ms(total_ms=750) == 750


def test_text_is_visible_content_only():
    commitment = sse.StreamCommitment(started_perf=0.0)
    commitment.feed(
        wire(delta(reasoning="the user wants"), delta("41 "), delta("42")), now=1.0
    )
    # Reasoning is the model talking to itself, not the answer.
    assert commitment.text() == "41 42"


def test_throughput_excludes_time_to_first_token():
    # The regression this guards: dividing by total time folds prefill and the
    # observer's network round trip into throughput, so a distant miner looks
    # slow. 100 tokens generated in 500ms is 200 tok/s regardless of a 1s wait.
    assert sse.observed_tps_milli(100, total_ms=1_500, ttft_ms=1_000) == 200_000
    assert sse.observed_tps_milli(100, total_ms=600, ttft_ms=100) == 200_000


def test_throughput_never_divides_by_zero():
    assert sse.observed_tps_milli(5, total_ms=10, ttft_ms=10) > 0
    assert sse.observed_tps_milli(5, total_ms=10, ttft_ms=99) > 0


def test_the_platform_observer_shares_this_commitment():
    # Two implementations of "the bytes the receipt covers" is how receipt
    # verification quietly stops working.
    from instant.platform.app import _StreamObserver

    assert issubclass(_StreamObserver, sse.StreamCommitment)
