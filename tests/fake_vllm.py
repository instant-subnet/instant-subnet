"""A fake vLLM, as an httpx transport.

The miner's whole request path runs through ``httpx`` to a local
OpenAI-compatible server. To test that path honestly we need something that
behaves like vLLM on the wire — including emitting SSE frames *over time*,
because half of what :mod:`instant.miner.upstream` does is measure when
frames arrive.

Why a transport rather than ``httpx.ASGITransport`` pointed at a small ASGI
app: ASGITransport runs the application to completion and only then returns a
response whose body is ``b"".join(body_parts)``. Every frame appears to
arrive at once. That is fine for testing JSON endpoints and useless for
testing time-to-first-token — under ASGITransport, TTFT would always equal
total time and the assertion that we skip vLLM's role-only opening delta
would pass whether or not the code did so. A transport that yields frames
with real (small) delays is the only version of this fixture that can fail
when the measurement code is wrong.

Everything the fake does is driven by the :class:`FakeVllm` state object, so
a test mutates one attribute and calls the miner.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


@dataclass
class FakeVllm:
    """Mutable knobs for the fake upstream."""

    model_id: str = "test-model"

    #: Content deltas the fake will stream, one SSE frame each.
    content: list[str] = field(default_factory=lambda: ["Hello", ",", " world"])
    prompt_tokens: int = 11
    completion_tokens: int = 3
    finish_reason: str = "stop"

    #: ``/health`` returns 200 when true, 503 when false — vLLM's own
    #: behaviour while weights are still loading.
    ready: bool = True

    #: Force an error status from ``/v1/chat/completions``.
    status: int = 200
    error_message: str = "fake upstream error"

    #: Seconds to wait between the role-only opening delta and the first
    #: frame carrying content. This is the gap TTFT must span.
    pre_content_delay_s: float = 0.0

    #: Drop the connection after this many SSE frames have been yielded.
    fail_after_frames: int | None = None

    #: Every payload the miner forwarded, in order. Lets a test assert on
    #: what the miner actually sent rather than what it was given.
    requests: list[dict[str, Any]] = field(default_factory=list)

    def transport(self) -> FakeVllmTransport:
        return FakeVllmTransport(self)

    # --- response bodies ---------------------------------------------------

    def completion_body(self) -> dict[str, Any]:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1_780_000_000,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(self.content)},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }

    def stream_frames(self) -> list[tuple[float, bytes]]:
        """``(delay_before, frame)`` pairs, in the order vLLM emits them."""
        base = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 1_780_000_000,
            "model": self.model_id,
        }
        frames: list[tuple[float, bytes]] = []

        # vLLM's opening delta announces the role and carries no content.
        # Timing to this frame is the bug the TTFT test exists to catch.
        frames.append(
            (0.0, _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        )

        for i, token in enumerate(self.content):
            frames.append(
                (
                    self.pre_content_delay_s if i == 0 else 0.0,
                    _sse({**base, "choices": [{"index": 0, "delta": {"content": token}}]}),
                )
            )

        frames.append(
            (
                0.0,
                _sse(
                    {
                        **base,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": self.finish_reason}
                        ],
                    }
                ),
            )
        )
        # The usage frame that ``stream_options.include_usage`` asks for:
        # empty choices, usage populated.
        frames.append(
            (
                0.0,
                _sse(
                    {
                        **base,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": self.prompt_tokens,
                            "completion_tokens": self.completion_tokens,
                            "total_tokens": self.prompt_tokens + self.completion_tokens,
                        },
                    }
                ),
            )
        )
        frames.append((0.0, b"data: [DONE]\n\n"))
        return frames


class _SseStream(httpx.AsyncByteStream):
    def __init__(self, fake: FakeVllm):
        self._fake = fake

    async def __aiter__(self) -> AsyncIterator[bytes]:
        sent = 0
        for delay, frame in self._fake.stream_frames():
            if (
                self._fake.fail_after_frames is not None
                and sent >= self._fake.fail_after_frames
            ):
                # What a mid-stream upstream death looks like to httpx.
                raise httpx.ReadError("connection reset by peer")
            if delay:
                await asyncio.sleep(delay)
            yield frame
            sent += 1


class FakeVllmTransport(httpx.AsyncBaseTransport):
    """Speaks enough of the vLLM HTTP surface for the miner to be exercised."""

    def __init__(self, fake: FakeVllm):
        self.fake = fake

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        fake = self.fake
        path = request.url.path

        if path == "/health":
            return httpx.Response(200 if fake.ready else 503, json={"status": "ok"})

        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": fake.model_id}]})

        if path != "/v1/chat/completions":
            return httpx.Response(404, json={"error": {"message": "not found"}})

        payload = json.loads(await request.aread())
        fake.requests.append(payload)

        if fake.status >= 400:
            return httpx.Response(
                fake.status, json={"error": {"message": fake.error_message}}
            )

        if not payload.get("stream"):
            return httpx.Response(200, json=fake.completion_body())

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SseStream(fake),
        )
