"""Finite, receipt-verified direct probes from a validator to miners.

Direct probes deliberately use a non-streaming OpenAI-compatible request.  We
still read the response incrementally so ``ttft_ms`` is the observer's time to
the first response-body byte.  A non-streaming response does not expose token
boundaries, so throughput is the signed completion-token count divided by the
observer's *total* request time; using ``total - ttft`` here would measure only
the JSON response transfer and badly overstate generation throughput.

The module owns no loop and no HTTP client.  A caller supplies an
``httpx.AsyncClient`` and explicitly awaits either :func:`probe_direct` or the
bounded :func:`probe_batch` helper.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from instant.protocol import receipts, ss58
from instant.protocol.epistula import generate_headers
from instant.protocol.keys import Signer
from instant.validator.state import ProbeOutcome, ProbeResult

DEFAULT_PROBE_COUNT = 20
MAX_PROBE_COUNT = 20
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 4
DEFAULT_TIMEOUT_S = 30.0
MAX_TIMEOUT_S = 120.0
DEFAULT_MAX_COMPLETION_TOKENS = 8
MAX_COMPLETION_TOKENS = 32
MAX_PROMPT_BYTES = 1_024
MAX_RESPONSE_BYTES = 1_048_576
DEFAULT_PROMPT = "Reply with exactly: pong"


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
    raw_receipt: str | None,
    signature: str | None,
    target: ProbeTarget,
    signer: Signer,
    request_id: str,
    request_body: bytes,
    response_body: bytes,
) -> receipts.Receipt:
    if raw_receipt is None or signature is None:
        raise receipts.ReceiptError("miner receipt is missing")

    try:
        payload = json.loads(raw_receipt)
        if not isinstance(payload, dict):
            raise ValueError("receipt header is not a JSON object")
        signed = receipts.SignedReceipt(
            receipt=receipts.Receipt.from_payload(payload),
            signature=signature,
        )
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
    prompt: str = DEFAULT_PROMPT,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ProbeResult:
    """Run one signed direct probe and return its storage-ready result.

    HTTP 429 and 503 are clean capacity rejections.  A 2xx is successful only
    when its miner-signed receipt binds the exact request bytes, response
    bytes, request UUID, validator hotkey, and expected miner hotkey.  Every
    other status, transport error, oversized response, or receipt error is a
    probe failure.
    """

    _validate_call(
        epoch=epoch,
        signer=signer,
        model=model,
        prompt=prompt,
        max_completion_tokens=max_completion_tokens,
        timeout_s=timeout_s,
    )

    request_id = str(uuid.uuid4())
    request_body = json.dumps(
        {
            "max_completion_tokens": max_completion_tokens,
            "messages": [{"content": prompt, "role": "user"}],
            "model": model,
            "stream": False,
            "temperature": 0,
            "user": f"validator-probe:{request_id}",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    headers = {
        "Accept": "application/json",
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
    first_byte_ms: int | None = None
    total_ms = 0
    response_body = bytearray()

    try:
        async with http.stream(
            "POST",
            target.completions_url,
            content=request_body,
            headers=headers,
            follow_redirects=False,
            timeout=timeout_s,
        ) as response:
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if first_byte_ms is None:
                    first_byte_ms = _elapsed_ms(started_ns)
                response_body.extend(chunk)
                if len(response_body) > MAX_RESPONSE_BYTES:
                    raise _ResponseTooLarge(f"response exceeded {MAX_RESPONSE_BYTES} bytes")

            total_ms = _elapsed_ms(started_ns)
            if first_byte_ms is None:
                first_byte_ms = total_ms
            status_code = response.status_code
            raw_receipt = response.headers.get(receipts.H_RECEIPT)
            receipt_signature = response.headers.get(receipts.H_RECEIPT_SIG)
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
    except _ResponseTooLarge as exc:
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

    try:
        receipt = _verify_receipt(
            raw_receipt=raw_receipt,
            signature=receipt_signature,
            target=target,
            signer=signer,
            request_id=request_id,
            request_body=request_body,
            response_body=bytes(response_body),
        )
    except receipts.ReceiptError as exc:
        return _result(
            epoch=epoch,
            target=target,
            outcome="failure",
            error=_safe_error("invalid miner receipt", exc),
        )

    # Milli-tokens/s keeps the existing integer-only ProbeResult contract.
    tps_milli = receipt.completion_tokens * 1_000_000 // max(1, total_ms)
    return _result(
        epoch=epoch,
        target=target,
        outcome="success",
        ttft_ms=first_byte_ms,
        tps_milli=tps_milli,
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
    prompt: str = DEFAULT_PROMPT,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[ProbeResult, ...]:
    """Run a finite round-robin batch, capped at 20 probes and 4 in flight."""

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

    schedule = tuple(targets[index % len(targets)] for index in range(count))
    semaphore = asyncio.Semaphore(concurrency)

    async def run(target: ProbeTarget) -> ProbeResult:
        async with semaphore:
            return await probe_direct(
                http,
                signer=signer,
                target=target,
                epoch=epoch,
                model=model,
                prompt=prompt,
                max_completion_tokens=max_completion_tokens,
                timeout_s=timeout_s,
            )

    # gather preserves schedule order; every task is finite and at most 20
    # tasks exist, so there is no detached/background work after this returns.
    return tuple(await asyncio.gather(*(run(target) for target in schedule)))
