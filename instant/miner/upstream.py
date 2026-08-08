"""The vLLM client.

Everything that touches the OpenAI-compatible upstream lives here, so that
the request handler in :mod:`instant.miner.app` reads as auth, measure,
forward, sign — and the details of how vLLM streams are one module over.

The measurement discipline is the part worth reading carefully. Time to
first token is measured from *just before the upstream request is sent* to
*the arrival of the first chunk carrying actual content*. Not the first
byte, not the first SSE frame — vLLM emits a role-only delta first, and
counting that as the first token would make every miner look 20ms faster
than it is, uniformly, which is worse than useless: it is a number that
looks meaningful and is not.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

#: vLLM terminates an SSE stream with this sentinel rather than closing.
SSE_DONE = "[DONE]"


class UpstreamError(Exception):
    """vLLM failed. ``status`` is what we should return to the caller."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class StreamOutcome:
    """What actually happened, as measured by us rather than claimed by vLLM."""

    ttft_ms: int = 0
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    chunks: int = 0
    #: The assembled response body — the concatenated ``data:`` payloads,
    #: which is what a receipt hashes. See receipts.py for why not the wire
    #: bytes.
    assembled: bytearray = field(default_factory=bytearray)


class VllmClient:
    """A long-lived HTTP client for the local vLLM server.

    One client for the process lifetime: connection reuse is the difference
    between a 2ms and a 40ms floor on TTFT, and on a subnet that sells speed
    that is not a rounding error.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 300.0,
        connect_timeout_s: float = 2.0,
        max_connections: int = 256,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        #: ``transport`` exists so the test suite can point this client at an
        #: in-process fake vLLM through ``httpx.ASGITransport``. That keeps the
        #: streaming, timeout and error-mapping code below on the real code
        #: path in tests rather than mocked around.
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """True if vLLM is up and has finished loading.

        ``/health`` returns 200 only once the model is resident, which is
        exactly the signal the miner needs to decide whether to report
        ``ready``.
        """
        try:
            r = await self._client.get("/health", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def models(self) -> list[str]:
        try:
            r = await self._client.get("/v1/models", timeout=5.0)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return []

    async def ready(self, expected_model: str) -> bool:
        """The worker is healthy only when it advertises the configured model."""
        if not await self.health():
            return False
        return expected_model in await self.models()

    async def complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], StreamOutcome]:
        """Non-streaming completion."""
        body = {**payload, "stream": False}
        started = time.perf_counter()
        try:
            r = await self._client.post("/v1/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise UpstreamError("upstream timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream unreachable: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if r.status_code >= 400:
            raise UpstreamError(_extract_error(r), _client_or_server(r.status_code))

        try:
            data = r.json()
        except ValueError as exc:
            raise UpstreamError("upstream returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise UpstreamError("upstream returned a non-object completion")
        actual_model = data.get("model")
        expected_model = body.get("model")
        if actual_model != expected_model:
            raise UpstreamError(
                f"upstream served model {actual_model!r}, expected {expected_model!r}"
            )
        usage = data.get("usage") or {}
        outcome = StreamOutcome(
            # Without streaming there is no first token to time, so TTFT is
            # the whole request. Reporting 0 here would quietly flatter every
            # non-streaming call.
            ttft_ms=elapsed_ms,
            total_ms=elapsed_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=(data.get("choices") or [{}])[0].get("finish_reason"),
            chunks=1,
        )
        outcome.assembled.extend(json.dumps(data, separators=(",", ":")).encode())
        return data, outcome

    async def stream(
        self, payload: dict[str, Any], outcome: StreamOutcome
    ) -> AsyncIterator[bytes]:
        """Stream a completion, filling ``outcome`` as it goes.

        Yields raw SSE frames ready to write to the client. The caller owns
        the receipt — this function only measures, because a module that both
        measures and signs is a module where the two can quietly disagree.
        """
        body = {
            **payload,
            "stream": True,
            # Ask vLLM for a usage frame at the end. Without this we would
            # have to count tokens ourselves, and our tokeniser and vLLM's
            # would disagree on exactly the multi-byte cases that matter.
            "stream_options": {"include_usage": True},
        }

        started = time.perf_counter()
        first_content_at: float | None = None

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise UpstreamError(
                        _extract_error(response), _client_or_server(response.status_code)
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        # Comments and keep-alives. Forward them — they keep
                        # intermediaries from timing the connection out — but
                        # do not count them as content.
                        yield (line + "\n\n").encode()
                        continue

                    data = line[5:].strip()
                    if data == SSE_DONE:
                        break

                    actual_model = _model_from_frame(data)
                    expected_model = body.get("model")
                    if actual_model is not None and actual_model != expected_model:
                        raise UpstreamError(
                            f"upstream served model {actual_model!r}, "
                            f"expected {expected_model!r}"
                        )

                    outcome.chunks += 1
                    outcome.assembled.extend(data.encode())

                    if first_content_at is None and _has_content(data):
                        first_content_at = time.perf_counter()

                    _absorb_usage(data, outcome)
                    yield (line + "\n\n").encode()

        except httpx.TimeoutException as exc:
            raise UpstreamError("upstream timed out mid-stream", 504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream stream failed: {exc}") from exc

        now = time.perf_counter()
        outcome.total_ms = int((now - started) * 1000)
        outcome.ttft_ms = int(((first_content_at or now) - started) * 1000)


def _has_content(data: str) -> bool:
    """True if this SSE frame carries generated text.

    vLLM's first delta is ``{"role":"assistant"}`` with no content. Timing
    to that frame instead of to real output would shave a uniform ~20ms off
    every miner's TTFT — a number that looks like a measurement and is an
    artefact.
    """
    try:
        parsed = json.loads(data)
    except ValueError:
        return False
    for choice in parsed.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content"):
            return True
        if delta.get("reasoning_content"):
            return True
        if delta.get("tool_calls"):
            return True
    return False


def _model_from_frame(data: str) -> str | None:
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    model = parsed.get("model")
    return model if isinstance(model, str) else None


def _absorb_usage(data: str, outcome: StreamOutcome) -> None:
    try:
        parsed = json.loads(data)
    except ValueError:
        return
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        outcome.prompt_tokens = int(usage.get("prompt_tokens", outcome.prompt_tokens))
        outcome.completion_tokens = int(
            usage.get("completion_tokens", outcome.completion_tokens)
        )
    for choice in parsed.get("choices") or []:
        if choice.get("finish_reason"):
            outcome.finish_reason = choice["finish_reason"]


def _extract_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"upstream returned {response.status_code}"
    err = payload.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err or f"upstream returned {response.status_code}")


def _client_or_server(status: int) -> int:
    """Map an upstream status to ours.

    A 400 from vLLM means the *caller* sent something invalid, so it stays a
    400 — reporting it as 502 would make a malformed request look like a
    miner failure, and the miner would be scored for someone else's bug.
    """
    if status in (400, 404, 422):
        return status
    if status == 429:
        return 429
    return 502
