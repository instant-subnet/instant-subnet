"""Finite, receipt-verified direct probes from a validator to miners.

Direct probes **stream**, for three reasons that all follow from the same fact:
on a non-streaming request no response byte arrives until generation has
finished.

* ``ttft_ms`` would then be total time, not time to first token.  It grew with
  the answer's length, so asking a miner for more tokens scored it as slower,
  and the number was compared against ``ttft_p95_target_ms`` — a *TTFT* target.
* Throughput had to divide by total time, folding prefill, queueing, and the
  validator's own network round trip into the denominator.  A miner far from the
  validator was reported slower than it is.  Streaming gives a real
  time-to-first-token, so throughput can divide by ``total - ttft``: the
  generation window alone.
* Customers stream.  A probe that does not is trivially distinguishable from
  customer traffic, which is exactly what a miner needs in order to serve
  probes better than it serves users.

The bytes a streamed receipt commits to are the concatenated ``data:`` payloads,
reassembled by :class:`instant.protocol.sse.StreamCommitment` — the same code
the platform observer uses, so the two cannot drift on what was signed.

The module owns no loop and no HTTP client.  A caller supplies an
``httpx.AsyncClient`` and explicitly awaits either :func:`probe_direct` or the
bounded :func:`probe_batch` helper.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import secrets
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from instant.protocol import receipts, ss58, sse
from instant.protocol.epistula import generate_headers
from instant.protocol.keys import Signer
from instant.validator.state import ProbeOutcome, ProbeResult

DEFAULT_PROBE_COUNT = 20
MAX_PROBE_COUNT = 20
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 4
DEFAULT_TIMEOUT_S = 30.0
MAX_TIMEOUT_S = 120.0
DEFAULT_MAX_COMPLETION_TOKENS = 256
MAX_COMPLETION_TOKENS = 256
MAX_PROMPT_BYTES = 1_024
MAX_RESPONSE_BYTES = 1_048_576

#: gpt-oss spends its completion budget on reasoning before it emits a single
#: visible token.  Without this the probe's whole allowance is consumed by a
#: truncated reasoning preamble and the response arrives with ``content: null``
#: and ``finish_reason: "length"`` — measured against the real miner, an 8-token
#: probe returned nothing but ``'The user says: "'``.  "low" keeps reasoning to
#: roughly ten tokens so the rest of the budget is the answer we verify.
PROBE_REASONING_EFFORT = "low"

#: Nonce width, mirroring ``[probe].prompt_nonce_bytes`` in
#: ``config/scoring.toml``.  Eight bytes is the floor because the whole point is
#: that a miner cannot enumerate the space ahead of time.
DEFAULT_PROMPT_NONCE_BYTES = 16
MIN_PROMPT_NONCE_BYTES = 8
MAX_PROMPT_NONCE_BYTES = 64

#: How many integers a challenge asks for.  Measured on the launch H200: a
#: 20-integer answer is 58 completion tokens in 179 ms (324 tok/s observed) and
#: a 120-integer answer is 258 tokens in 769 ms (335 tok/s).  Fitting those
#: puts fixed overhead near 8 ms, so even the short end reads within about 4%
#: of asymptotic throughput while costing a quarter of the GPU time.  Below
#: roughly 20 the overhead starts to dominate and throughput is understated.
MIN_COUNT_SPAN = 20
MAX_COUNT_SPAN = 40

#: Upper bound on the first integer.  Kept to three digits so the token cost of
#: an answer stays predictable, and away from zero so the answer is never a
#: prefix of a shorter run.
MAX_COUNT_START = 900

#: A 429/503 is the miner saying "full, retry later", not "broken".  Retrying
#: the slot keeps a miner that is busy serving customers from being gated out
#: for honest backpressure, which is the outcome ``reliability_bps`` already
#: goes out of its way to avoid charging as a failure.  Every exchange is still
#: recorded, so the rejections themselves remain visible to scoring.
DEFAULT_CLEAN_REJECT_RETRIES = 2
MAX_CLEAN_REJECT_RETRIES = 4
_RETRY_BACKOFF_S: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Challenge:
    """One probe's prompt and the exact answer it must produce.

    Holding both together is what lets a probe assert that inference actually
    happened.  Verifying only the receipt — all this module used to do — proves
    the miner signed the bytes it sent, not that the bytes were an answer, so a
    miner returning ``content: null`` was scored as a success.
    """

    prompt: str
    expected: str


def new_challenge(
    *,
    nonce_bytes: int = DEFAULT_PROMPT_NONCE_BYTES,
    rng: random.Random | None = None,
) -> Challenge:
    """Build one unpredictable, exactly-verifiable probe challenge.

    The nonce goes in *front*.  A shared prefix is what a paged KV cache keys
    on, so a trailing nonce would still let the miner serve most of the prompt
    from cache and report a TTFT that measures its cache rather than its
    hardware — the failure ``prompt_nonce_bytes`` exists to prevent, and which
    it did not prevent while nothing read it.

    Counting is the family because it is the rare question that is long enough
    to measure sustained throughput, cheap to state, and has exactly one
    correct answer the validator can compute without a model.  A single fixed
    word would leave one canned completion correct for every probe forever.

    Unlike :mod:`instant.validator.score`, this is deliberately *not*
    deterministic across validators: each validator scores only the probes it
    sent itself, so shared randomness would buy nothing and a predictable
    schedule is exactly what a miner would exploit.  ``rng`` is injectable so
    tests can pin it; the default draws from the OS.
    """
    if (
        isinstance(nonce_bytes, bool)
        or not isinstance(nonce_bytes, int)
        or not MIN_PROMPT_NONCE_BYTES <= nonce_bytes <= MAX_PROMPT_NONCE_BYTES
    ):
        raise ValueError(
            f"nonce_bytes must be between {MIN_PROMPT_NONCE_BYTES} and "
            f"{MAX_PROMPT_NONCE_BYTES}"
        )

    source = secrets.SystemRandom() if rng is None else rng
    nonce = "".join(source.choice("0123456789abcdef") for _ in range(nonce_bytes * 2))
    start = source.randint(1, MAX_COUNT_START)
    span = source.randint(MIN_COUNT_SPAN, MAX_COUNT_SPAN)
    last = start + span - 1

    prompt = (
        f"Session {nonce}. Output the integers from {start} to {last} "
        f"inclusive, separated by single spaces, and nothing else."
    )
    if len(prompt.encode()) > MAX_PROMPT_BYTES:  # pragma: no cover - bounded above
        raise ValueError("generated prompt exceeded the probe prompt budget")
    return Challenge(
        prompt=prompt,
        expected=" ".join(str(value) for value in range(start, last + 1)),
    )


class _ContentMismatch(RuntimeError):
    """A 2xx whose body was not the answer the challenge demanded."""


def _verify_content(content: str, expected: str) -> None:
    """Assert the streamed completion is the challenge's answer.

    Compared on whitespace-collapsed text: the answer is a space-separated run
    of integers, and a model that emits a newline or a double space has still
    answered correctly.  Everything else must match exactly, because the
    expected string is fully determined by the prompt.
    """
    if not content.strip():
        # gpt-oss emits nothing but reasoning when the budget is too small,
        # which used to be recorded as a successful probe.
        raise _ContentMismatch("completion had no visible content")

    if _WHITESPACE.sub(" ", content).strip() != expected:
        raise _ContentMismatch("completion did not match the challenge answer")


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    """A miner discovered from the metagraph and selected for probing."""

    uid: int
    hotkey: str
    url: str

    def __post_init__(self) -> None:
        if isinstance(self.uid, bool) or not isinstance(self.uid, int) or self.uid < 0:
            raise ValueError("probe target uid must be a non-negative integer")
        if not ss58.is_valid(self.hotkey):
            raise ValueError("probe target hotkey must be a valid Bittensor SS58 address")

        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("probe target url must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("probe target url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("probe target url must not contain a query or fragment")

    @property
    def completions_url(self) -> str:
        return f"{self.url.rstrip('/')}/v1/chat/completions"


class _ResponseTooLarge(RuntimeError):
    pass


class _StreamTooSlow(RuntimeError):
    """A stream stayed open past the probe's total budget."""


def _now_ms() -> int:
    return int(time.time() * 1_000)


def _perf_counter_ns() -> int:
    return time.perf_counter_ns()


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (_perf_counter_ns() - started_ns) // 1_000_000)


def _result(
    *,
    epoch: int,
    target: ProbeTarget,
    outcome: ProbeOutcome,
    ttft_ms: int | None = None,
    tps_milli: int | None = None,
    tokens: int | None = None,
    error: str | None = None,
) -> ProbeResult:
    return ProbeResult(
        epoch=epoch,
        uid=target.uid,
        hotkey=target.hotkey,
        source="direct",
        outcome=outcome,
        observed_ms=_now_ms(),
        ttft_ms=ttft_ms,
        tps_milli=tps_milli,
        tokens=tokens,
        error=error,
    )


def _safe_error(prefix: str, detail: object | None = None) -> str:
    if detail is None:
        return prefix
    clean = " ".join(str(detail).split())[:180]
    return f"{prefix}: {clean}" if clean else prefix


def _validate_call(
    *,
    epoch: int,
    signer: Signer,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    timeout_s: float,
) -> None:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if not ss58.is_valid(signer.ss58_address):
        raise ValueError("validator signer must have a valid Bittensor SS58 address")
    if not isinstance(model, str) or not model.strip() or len(model.encode()) > 200:
        raise ValueError("model must be a non-empty string of at most 200 bytes")
    if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > MAX_PROMPT_BYTES:
        raise ValueError(f"prompt must contain 1 to {MAX_PROMPT_BYTES} UTF-8 bytes")
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or not 1 <= max_completion_tokens <= MAX_COMPLETION_TOKENS
    ):
        raise ValueError(
            f"max_completion_tokens must be between 1 and {MAX_COMPLETION_TOKENS}"
        )
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("timeout_s must be a number")
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be greater than 0 and at most {MAX_TIMEOUT_S}")


def _verify_receipt(
    *,
    payload: dict[str, object] | None,
    target: ProbeTarget,
    signer: Signer,
    request_id: str,
    request_body: bytes,
    response_body: bytes,
) -> receipts.Receipt:
    """Verify the receipt the miner streamed as its private SSE event.

    A streamed receipt arrives *after* ``[DONE]`` rather than in a header,
    because it commits to the response and the response is not complete until
    the last token has been sent.
    """
    if payload is None:
        raise receipts.ReceiptError("miner receipt is missing")

    try:
        signed = receipts.SignedReceipt.from_payload(payload)
        receipts.verify(
            signed,
            expected_miner_hotkey=target.hotkey,
            request_body=request_body,
            response_body=response_body,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise receipts.ReceiptError(str(exc)) from exc

    receipt = signed.receipt
    if receipt.request_id != request_id:
        raise receipts.ReceiptError("receipt request id does not match the probe")
    if receipt.signer_of_request != signer.ss58_address:
        raise receipts.ReceiptError("receipt request signer is not this validator")
    if (
        isinstance(receipt.completion_tokens, bool)
        or not isinstance(receipt.completion_tokens, int)
        or receipt.completion_tokens < 0
    ):
        raise receipts.ReceiptError("receipt completion token count is invalid")
    return receipt


async def probe_direct(
    http: httpx.AsyncClient,
    *,
    signer: Signer,
    target: ProbeTarget,
    epoch: int,
    model: str,
    challenge: Challenge | None = None,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ProbeResult:
    """Run one signed direct probe and return its storage-ready result.

    HTTP 429 and 503 are clean capacity rejections.  A 2xx is successful only
    when its miner-signed receipt binds the exact request bytes, response
    bytes, request UUID, validator hotkey, and expected miner hotkey — *and*
    the completion is the exact answer the challenge demanded.  Every other
    status, transport error, oversized response, receipt error, or wrong answer
    is a probe failure.

    ``challenge`` defaults to a fresh :func:`new_challenge`, so a caller cannot
    accidentally reuse one prompt across a batch and hand the miner a cache key.
    """

    if challenge is None:
        challenge = new_challenge()

    _validate_call(
        epoch=epoch,
        signer=signer,
        model=model,
        prompt=challenge.prompt,
        max_completion_tokens=max_completion_tokens,
        timeout_s=timeout_s,
    )

    request_id = str(uuid.uuid4())
    request_body = json.dumps(
        {
            "max_completion_tokens": max_completion_tokens,
            "messages": [{"content": challenge.prompt, "role": "user"}],
            "model": model,
            "reasoning_effort": PROBE_REASONING_EFFORT,
            "stream": True,
            "temperature": 0,
            "user": f"validator-probe:{request_id}",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    headers = {
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
        "Content-Type": "application/json",
        **generate_headers(
            signer,
            request_body,
            signed_for=target.hotkey,
            request_uuid=request_id,
        ),
    }

    started_ns = _perf_counter_ns()
    total_ms = 0
    received = 0
    # One clock read shared by both, so ttft and total cannot disagree about
    # when the request started.
    commitment = sse.StreamCommitment(started_perf=started_ns / 1e9)

    try:
        async with http.stream(
            "POST",
            target.completions_url,
            content=request_body,
            headers=headers,
            follow_redirects=False,
            timeout=timeout_s,
        ) as response:
            status_code = response.status_code
            if 200 <= status_code < 300:
                deadline_ms = int(timeout_s * 1000)
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > MAX_RESPONSE_BYTES:
                        raise _ResponseTooLarge(
                            f"response exceeded {MAX_RESPONSE_BYTES} bytes"
                        )
                    # One clock read per chunk, used for both the deadline and
                    # the frame timestamp, so the whole timing path stays behind
                    # _perf_counter_ns and tests can drive it deterministically.
                    now_ns = _perf_counter_ns()
                    # httpx applies timeout_s per read, not to the stream as a
                    # whole, so a miner dribbling one token just inside each read
                    # window could hold a probe open indefinitely -- and with a
                    # sequential batch, stall the validator. Bound the total.
                    if (now_ns - started_ns) // 1_000_000 > deadline_ms:
                        raise _StreamTooSlow(
                            f"stream exceeded {deadline_ms} ms in total"
                        )
                    commitment.feed(chunk, now=now_ns / 1e9)
            else:
                # Errors are ordinary JSON, not a stream; drain to free the
                # connection but do not try to parse frames out of it.
                await response.aread()
            total_ms = _elapsed_ms(started_ns)
    except httpx.TimeoutException as exc:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=_safe_error("miner request timed out", exc),
        )
    except httpx.HTTPError as exc:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=_safe_error("miner request failed", exc),
        )
    except (_ResponseTooLarge, _StreamTooSlow) as exc:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=str(exc),
        )

    if status_code in {429, 503}:
        return _result(
            epoch=epoch,
            target=target,
            outcome="clean_reject",
            error=f"miner HTTP {status_code}",
        )
    if not 200 <= status_code < 300:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=f"miner HTTP {status_code}",
        )

    if commitment.protocol_error is not None:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=_safe_error("malformed miner stream", commitment.protocol_error),
        )
    if commitment.error_event:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error="miner emitted an error event mid-stream",
        )

    try:
        receipt = _verify_receipt(
            payload=commitment.receipt_payload,
            target=target,
            signer=signer,
            request_id=request_id,
            request_body=request_body,
            response_body=bytes(commitment.assembled),
        )
    except receipts.ReceiptError as exc:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=_safe_error("invalid miner receipt", exc),
        )

    # Checked after the receipt, so a mismatch is attributable: the miner has
    # already signed for exactly these bytes, making a wrong answer signed
    # evidence rather than an unattributable transport oddity.
    try:
        _verify_content(commitment.text(), challenge.expected)
    except _ContentMismatch as exc:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=_safe_error("content mismatch", exc),
        )

    # Streaming is what makes both of these mean what they are named. On a
    # non-streaming request no body byte arrives until generation has finished,
    # so "time to first byte" was really total time and grew with the token
    # count -- scoring a longer answer as a slower miner -- while dividing by
    # total time folded the validator's network round trip into throughput.
    ttft_ms = commitment.ttft_ms(total_ms)
    return _result(
        epoch=epoch,
        target=target,
        outcome="success",
        ttft_ms=ttft_ms,
        tps_milli=sse.observed_tps_milli(
            receipt.completion_tokens, total_ms=total_ms, ttft_ms=ttft_ms
        ),
        tokens=receipt.completion_tokens,
    )


async def probe_batch(
    http: httpx.AsyncClient,
    *,
    signer: Signer,
    targets: Sequence[ProbeTarget],
    epoch: int,
    model: str,
    count: int = DEFAULT_PROBE_COUNT,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    clean_reject_retries: int = DEFAULT_CLEAN_REJECT_RETRIES,
    rng: random.Random | None = None,
) -> tuple[ProbeResult, ...]:
    """Run a finite round-robin batch of ``count`` slots.

    Each slot draws its own :func:`new_challenge`, so no two probes in a batch
    share a prompt and none of them is predictable.

    A slot whose exchange is a clean 429/503 reject is retried up to
    ``clean_reject_retries`` times with a short backoff, because that status
    means "full, retry later".  Every exchange is returned, so the result count
    is at least ``count`` and at most ``count * (1 + clean_reject_retries)``:
    the rejections stay visible to ``reliability_bps`` while the eventual
    success still counts toward the gate's success floor.
    """

    if not targets:
        raise ValueError("at least one probe target is required")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= MAX_PROBE_COUNT
    ):
        raise ValueError(f"count must be between 1 and {MAX_PROBE_COUNT}")
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or not 1 <= concurrency <= MAX_CONCURRENCY
    ):
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    if (
        isinstance(clean_reject_retries, bool)
        or not isinstance(clean_reject_retries, int)
        or not 0 <= clean_reject_retries <= MAX_CLEAN_REJECT_RETRIES
    ):
        raise ValueError(
            f"clean_reject_retries must be between 0 and {MAX_CLEAN_REJECT_RETRIES}"
        )

    schedule = tuple(targets[index % len(targets)] for index in range(count))
    semaphore = asyncio.Semaphore(concurrency)

    async def run(target: ProbeTarget) -> list[ProbeResult]:
        collected: list[ProbeResult] = []
        for attempt in range(clean_reject_retries + 1):
            async with semaphore:
                result = await probe_direct(
                    http,
                    signer=signer,
                    target=target,
                    epoch=epoch,
                    model=model,
                    challenge=new_challenge(rng=rng),
                    max_completion_tokens=max_completion_tokens,
                    timeout_s=timeout_s,
                )
            collected.append(result)
            if result.outcome != "clean_reject":
                break
            if attempt < clean_reject_retries:
                # Slept outside the semaphore so a full miner does not hold a
                # concurrency slot idle while we wait for it to drain.
                await asyncio.sleep(_RETRY_BACKOFF_S[attempt])
        return collected

    # gather preserves schedule order; every task is finite and bounded by
    # count * (1 + retries), so nothing is left running after this returns.
    batches = await asyncio.gather(*(run(target) for target in schedule))
    return tuple(result for batch in batches for result in batch)
