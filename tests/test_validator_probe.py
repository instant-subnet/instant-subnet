"""Direct validator probes: real signatures, exact bytes, finite batches."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Callable

import httpx
import pytest

from instant.protocol import receipts
from instant.protocol.epistula import verify_headers
from instant.protocol.keys import LocalKeypair
from instant.validator import probe
from instant.validator.probe import ProbeTarget, probe_batch, probe_direct
from instant.validator.state import ProbeResult


def target(miner_key: LocalKeypair, uid: int = 7) -> ProbeTarget:
    return ProbeTarget(
        uid=uid,
        hotkey=miner_key.ss58_address,
        url=f"http://miner-{uid}.test:8091",
    )


def receipt_headers(signed: receipts.SignedReceipt) -> dict[str, str]:
    return {
        receipts.H_RECEIPT: json.dumps(
            signed.receipt.to_payload(), separators=(",", ":"), sort_keys=True
        ),
        receipts.H_RECEIPT_SIG: signed.signature,
    }


def signed_response(
    *,
    miner_key: LocalKeypair,
    request_id: str,
    signer_of_request: str,
    request_body: bytes,
    response_body: bytes,
    receipt_request_body: bytes | None = None,
    receipt_response_body: bytes | None = None,
    receipt_request_id: str | None = None,
    receipt_signer: str | None = None,
    completion_tokens: int = 4,
) -> receipts.SignedReceipt:
    return receipts.build(
        miner_key,
        request_id=receipt_request_id or request_id,
        signer_of_request=receipt_signer or signer_of_request,
        request_body=(
            request_body if receipt_request_body is None else receipt_request_body
        ),
        response_body=(
            response_body if receipt_response_body is None else receipt_response_body
        ),
        prompt_tokens=5,
        completion_tokens=completion_tokens,
        ttft_ms_self=10,
        total_ms_self=40,
        started_at_ms=1_780_000_000_000,
        finished_at_ms=1_780_000_000_040,
    )


#: A fixed challenge for tests that need to know the answer up front. Probes
#: generate their own by default; pinning one here keeps assertions exact.
FIXED_CHALLENGE = probe.Challenge(
    prompt="Session " + "ab" * 16 + ". Output the integers from 41 to 60 "
    "inclusive, separated by single spaces, and nothing else.",
    expected=" ".join(str(value) for value in range(41, 61)),
)


def delta_frames(content: str) -> list[bytes]:
    """Serialize ``content`` as one OpenAI delta frame per token.

    Chunked rather than sent whole so the stream has a real first-content
    moment for ttft to land on, the way a miner's does.
    """
    pieces = re.findall(r"\s*\S+", content) or [content]
    return [
        json.dumps(
            {"id": "cmpl-1", "choices": [{"delta": {"content": piece}}]},
            separators=(",", ":"),
        ).encode()
        for piece in pieces
    ]


def sse_wire(frames: list[bytes], signed: receipts.SignedReceipt | None) -> bytes:
    """Assemble the miner's wire format: data frames, [DONE], then the receipt.

    The receipt follows ``[DONE]`` exactly as ``miner/app.py`` emits it, because
    it commits to a response that is not complete until the last token is sent.
    """
    out = b"".join(b"data: " + frame + b"\n\n" for frame in frames)
    out += b"data: [DONE]\n\n"
    if signed is not None:
        out += (
            b"event: "
            + receipts.SSE_RECEIPT_EVENT.encode()
            + b"\ndata: "
            + json.dumps(signed.to_payload(), separators=(",", ":")).encode()
            + b"\n\n"
        )
    return out


def answer_for(request_body: bytes) -> str:
    """The correct answer to whatever challenge the probe just sent."""
    prompt = json.loads(request_body)["messages"][0]["content"]
    first, last = re.search(r"from (\d+) to (\d+)", prompt).groups()
    return " ".join(str(v) for v in range(int(first), int(last) + 1))


def committed(frames: list[bytes]) -> bytes:
    """The bytes a receipt commits to: the concatenated ``data:`` payloads."""
    return b"".join(frames)


def streaming_miner(
    *,
    validator_key: LocalKeypair,
    miner_key: LocalKeypair,
    content: Callable[[bytes], str] | None = None,
    **receipt_overrides,
) -> Callable[[httpx.Request], httpx.Response]:
    """A handler that answers the challenge over SSE and signs what it sent.

    ``content`` maps the request body to the text the miner will stream; the
    default answers correctly. Tests override it to model a broken miner, and
    pass ``receipt_overrides`` through to model a lying one.
    """

    produce = content or answer_for

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        frames = delta_frames(produce(request_body))
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
            **receipt_overrides,
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    return handler


async def run_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    validator_key: LocalKeypair,
    miner_key: LocalKeypair,
    challenge: probe.Challenge | None = None,
) -> ProbeResult:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await probe_direct(
            client,
            signer=validator_key,
            target=target(miner_key),
            epoch=42,
            model="instant/mock-echo",
            challenge=challenge,
        )


@pytest.mark.asyncio
async def test_success_is_epistula_signed_and_receipt_verified(
    monkeypatch, validator_key, miner_key, now_ms
):
    clock = iter([1_000_000_000, 1_120_000_000, 1_250_000_000])
    monkeypatch.setattr(probe, "_perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(probe, "_now_ms", lambda: now_ms)
    seen: dict[str, object] = {}
    frames = delta_frames(FIXED_CHALLENGE.expected)

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        seen.update(
            request_body=request_body,
            request_id=verified.uuid,
            payload=json.loads(request_body),
        )
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    result = await run_with_handler(
        handler,
        validator_key=validator_key,
        miner_key=miner_key,
        challenge=FIXED_CHALLENGE,
    )

    assert seen["payload"] == {
        "max_completion_tokens": probe.DEFAULT_MAX_COMPLETION_TOKENS,
        "messages": [{"content": FIXED_CHALLENGE.prompt, "role": "user"}],
        "model": "instant/mock-echo",
        # Without this gpt-oss spends the entire budget on reasoning and
        # returns content: null, which used to be scored as a success.
        "reasoning_effort": "low",
        # Streaming is what makes ttft_ms a time to first *token*.
        "stream": True,
        "temperature": 0,
        "user": f"validator-probe:{seen['request_id']}",
    }
    # started 1.00s, first content 1.12s, finished 1.25s.  ttft is 120ms and
    # throughput divides by the 130ms generation window, not the 250ms total.
    assert result == ProbeResult(
        epoch=42,
        uid=7,
        hotkey=miner_key.ss58_address,
        source="direct",
        outcome="success",
        observed_ms=now_ms,
        ttft_ms=120,
        tps_milli=4 * 1_000_000 // 130,
        tokens=4,
        error=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_capacity_status_is_a_clean_reject(status, validator_key, miner_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "at_capacity"})

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "clean_reject"
    assert result.error == f"miner HTTP {status}"
    assert result.ttft_ms is None
    assert result.tps_milli is None
    assert result.tokens is None


@pytest.mark.asyncio
async def test_other_http_status_is_a_failure(validator_key, miner_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream_failed"})

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert result.error == "miner HTTP 500"


@pytest.mark.asyncio
async def test_probe_never_follows_a_miner_redirect(validator_key, miner_key):
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(302, headers={"Location": "http://elsewhere.test/steal"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        result = await probe_direct(
            client,
            signer=validator_key,
            target=target(miner_key),
            epoch=42,
            model="instant/mock-echo",
        )

    assert requests == 1
    assert result.outcome == "failure"
    assert result.error == "miner HTTP 302"


@pytest.mark.asyncio
async def test_transport_timeout_is_a_failure(validator_key, miner_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow miner", request=request)

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert result.error == "miner request timed out: slow miner"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        ("request_body", "request hash"),
        ("response_body", "response hash"),
        ("request_id", "request id"),
        ("request_signer", "request signer"),
    ],
)
async def test_receipt_must_bind_exact_probe(
    mutation,
    error_fragment,
    validator_key,
    miner_key,
    stranger_key,
):
    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        frames = delta_frames(FIXED_CHALLENGE.expected)
        kwargs: dict[str, object] = {}
        if mutation == "request_body":
            kwargs["receipt_request_body"] = request_body + b" "
        elif mutation == "response_body":
            kwargs["receipt_response_body"] = committed(frames) + b" "
        elif mutation == "request_id":
            kwargs["receipt_request_id"] = "not-the-epistula-uuid"
        elif mutation == "request_signer":
            kwargs["receipt_signer"] = stranger_key.ss58_address
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
            **kwargs,
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    result = await run_with_handler(
        handler,
        validator_key=validator_key,
        miner_key=miner_key,
        challenge=FIXED_CHALLENGE,
    )

    assert result.outcome == "failure"
    assert result.error is not None
    assert error_fragment in result.error
    assert result.tokens is None


@pytest.mark.asyncio
async def test_receipt_must_be_from_expected_miner(validator_key, miner_key, stranger_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(request.headers, request_body)
        frames = delta_frames(FIXED_CHALLENGE.expected)
        signed = signed_response(
            miner_key=stranger_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    result = await run_with_handler(
        handler,
        validator_key=validator_key,
        miner_key=miner_key,
        challenge=FIXED_CHALLENGE,
    )

    assert result.outcome == "failure"
    assert result.error is not None
    assert "expected" in result.error


@pytest.mark.asyncio
async def test_missing_receipt_is_a_failure(validator_key, miner_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert result.error == "invalid miner receipt: miner receipt is missing"


@pytest.mark.asyncio
async def test_oversized_response_is_bounded(monkeypatch, validator_key, miner_key):
    monkeypatch.setattr(probe, "MAX_RESPONSE_BYTES", 8)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"123456789")

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert result.error == "response exceeded 8 bytes"


@pytest.mark.asyncio
async def test_batch_runs_exactly_twenty_round_robin_with_bounded_concurrency(
    monkeypatch, validator_key, miner_key, stranger_key
):
    targets = (target(miner_key, uid=3), target(stranger_key, uid=9))
    active = 0
    high_water = 0
    calls: list[int] = []

    async def fake_probe_direct(http, *, target, epoch, **kwargs):
        nonlocal active, high_water
        active += 1
        high_water = max(high_water, active)
        calls.append(target.uid)
        await asyncio.sleep(0)
        active -= 1
        return ProbeResult(
            epoch=epoch,
            uid=target.uid,
            hotkey=target.hotkey,
            source="direct",
            outcome="clean_reject",
            observed_ms=1,
        )

    monkeypatch.setattr(probe, "probe_direct", fake_probe_direct)
    async with httpx.AsyncClient() as client:
        results = await probe_batch(
            client,
            signer=validator_key,
            targets=targets,
            epoch=42,
            model="instant/mock-echo",
            # Retries are exercised separately; without this the fake's
            # clean_reject would be retried and there would be 60 exchanges.
            clean_reject_retries=0,
        )

    assert len(results) == 20
    assert [result.uid for result in results] == [3, 9] * 10
    assert calls == [3, 9] * 10
    assert high_water == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 21, True])
async def test_batch_count_is_hard_bounded(count, validator_key, miner_key):
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="count must be between"):
            await probe_batch(
                client,
                signer=validator_key,
                targets=(target(miner_key),),
                epoch=42,
                model="instant/mock-echo",
                count=count,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, 5, True])
async def test_batch_concurrency_is_hard_bounded(concurrency, validator_key, miner_key):
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="concurrency must be between"):
            await probe_batch(
                client,
                signer=validator_key,
                targets=(target(miner_key),),
                epoch=42,
                model="instant/mock-echo",
                concurrency=concurrency,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uid": -1},
        {"hotkey": "not-an-ss58-address"},
        {"url": "ftp://miner.test"},
        {"url": "http://user:pass@miner.test"},
        {"url": "http://miner.test?redirect=elsewhere"},
    ],
)
def test_probe_target_rejects_malformed_discovery_data(kwargs, miner_key):
    values = {
        "uid": 7,
        "hotkey": miner_key.ss58_address,
        "url": "http://miner.test:8091",
        **kwargs,
    }
    with pytest.raises(ValueError):
        ProbeTarget(**values)


# --- challenge generation ---------------------------------------------------


def test_a_challenge_answer_is_exactly_the_range_it_asks_for():
    challenge = probe.new_challenge(rng=random.Random(1))
    first, last = re.search(r"from (\d+) to (\d+)", challenge.prompt).groups()
    expected = " ".join(str(v) for v in range(int(first), int(last) + 1))
    assert challenge.expected == expected


def test_the_nonce_is_a_prefix_not_a_suffix():
    # A paged KV cache keys on leading tokens, so a trailing nonce would still
    # let the miner serve most of the prompt from cache and report a TTFT that
    # measures the cache rather than the hardware.
    challenge = probe.new_challenge(nonce_bytes=16, rng=random.Random(7))
    assert re.fullmatch(r"Session [0-9a-f]{32}\. Output the integers .*", challenge.prompt)


def test_challenges_do_not_repeat():
    prompts = {probe.new_challenge().prompt for _ in range(200)}
    assert len(prompts) == 200


def test_the_span_stays_inside_the_measured_throughput_window():
    for seed in range(50):
        challenge = probe.new_challenge(rng=random.Random(seed))
        first, last = re.search(r"from (\d+) to (\d+)", challenge.prompt).groups()
        span = int(last) - int(first) + 1
        assert probe.MIN_COUNT_SPAN <= span <= probe.MAX_COUNT_SPAN


@pytest.mark.parametrize("nonce_bytes", [0, 7, 65, True])
def test_a_nonce_too_short_to_be_unguessable_is_refused(nonce_bytes):
    with pytest.raises(ValueError, match="nonce_bytes must be between"):
        probe.new_challenge(nonce_bytes=nonce_bytes)


def test_a_prompt_never_exceeds_the_probe_budget():
    challenge = probe.new_challenge(nonce_bytes=probe.MAX_PROMPT_NONCE_BYTES)
    assert len(challenge.prompt.encode()) <= probe.MAX_PROMPT_BYTES


# --- content verification ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_null_content_completion_is_a_failure(validator_key, miner_key):
    # The exact shape the live miner produced with an 8-token budget: the whole
    # allowance went to reasoning, so no visible content was ever emitted and
    # finish_reason was "length". This used to be scored as a success.
    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        frames = [
            json.dumps(
                {"choices": [{"delta": {"reasoning_content": 'The user says: "'}}]},
                separators=(",", ":"),
            ).encode()
        ]
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert "no visible content" in result.error


@pytest.mark.asyncio
async def test_a_wrong_answer_is_a_failure_even_with_a_valid_receipt(
    validator_key, miner_key
):
    # A correctly signed receipt proves the miner sent these bytes. It says
    # nothing about the bytes being an answer, which is why content is checked.
    result = await run_with_handler(
        streaming_miner(
            validator_key=validator_key,
            miner_key=miner_key,
            content=lambda _: "41 42 43 nope",
        ),
        validator_key=validator_key,
        miner_key=miner_key,
        challenge=FIXED_CHALLENGE,
    )

    assert result.outcome == "failure"
    assert "content mismatch" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        " ".join(str(v) for v in range(41, 61)),           # exact
        "  ".join(str(v) for v in range(41, 61)),          # doubled spaces
        "\n".join(str(v) for v in range(41, 61)),          # newlines
        " " + " ".join(str(v) for v in range(41, 61)) + "\n",  # padded
    ],
)
async def test_harmless_whitespace_variation_still_answers(
    content, validator_key, miner_key
):
    result = await run_with_handler(
        streaming_miner(
            validator_key=validator_key,
            miner_key=miner_key,
            content=lambda _: content,
        ),
        validator_key=validator_key,
        miner_key=miner_key,
        challenge=FIXED_CHALLENGE,
    )

    assert result.outcome == "success"


# --- clean-reject retry -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_busy_miner_is_retried_and_every_exchange_is_recorded(
    monkeypatch, validator_key, miner_key
):
    # 429 means "full, retry later". Retrying is what stops a miner that is
    # busy serving real customers from being gated out for honest backpressure.
    monkeypatch.setattr(probe, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0, 0.0))
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "at_capacity"})
        frames = delta_frames(answer_for(request_body))
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await probe_batch(
            client,
            signer=validator_key,
            targets=(target(miner_key),),
            epoch=42,
            model="instant/mock-echo",
            count=1,
        )

    # Both the rejection and the eventual success survive: reliability_bps
    # still charges the backpressure, and the success counts toward the floor.
    assert [r.outcome for r in results] == ["clean_reject", "success"]


@pytest.mark.asyncio
async def test_a_permanently_full_miner_stops_after_the_retry_budget(
    monkeypatch, validator_key, miner_key
):
    monkeypatch.setattr(probe, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0, 0.0))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "at_capacity"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await probe_batch(
            client,
            signer=validator_key,
            targets=(target(miner_key),),
            epoch=42,
            model="instant/mock-echo",
            count=2,
            clean_reject_retries=2,
        )

    # 2 slots x (1 + 2 retries), and it stops rather than retrying forever.
    assert len(results) == 6
    assert {r.outcome for r in results} == {"clean_reject"}


@pytest.mark.asyncio
async def test_a_failure_is_not_retried(monkeypatch, validator_key, miner_key):
    # Only capacity rejections mean "try again". A 500 is not backpressure.
    monkeypatch.setattr(probe, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0, 0.0))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await probe_batch(
            client,
            signer=validator_key,
            targets=(target(miner_key),),
            epoch=42,
            model="instant/mock-echo",
            count=3,
        )

    assert len(results) == 3
    assert {r.outcome for r in results} == {"failure"}


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [-1, 5, True])
async def test_the_retry_budget_is_hard_bounded(retries, validator_key, miner_key):
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="clean_reject_retries must be between"):
            await probe_batch(
                client,
                signer=validator_key,
                targets=(target(miner_key),),
                epoch=42,
                model="instant/mock-echo",
                clean_reject_retries=retries,
            )


@pytest.mark.asyncio
async def test_a_batch_gives_every_probe_its_own_challenge(validator_key, miner_key):
    # One shared prompt across a batch would hand the miner a cache key and
    # make 19 of the 20 probes measure the cache.
    prompts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        prompts.append(json.loads(request_body)["messages"][0]["content"])
        frames = delta_frames(answer_for(request_body))
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
        )
        return httpx.Response(
            200,
            content=sse_wire(frames, signed),
            headers={"Content-Type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await probe_batch(
            client,
            signer=validator_key,
            targets=(target(miner_key),),
            epoch=42,
            model="instant/mock-echo",
            count=20,
        )

    assert len(results) == 20
    assert {r.outcome for r in results} == {"success"}
    assert len(set(prompts)) == 20


@pytest.mark.asyncio
async def test_a_dribbling_stream_is_cut_off_at_the_total_budget(
    monkeypatch, validator_key, miner_key
):
    # httpx's timeout is per read, so a miner that sends one small chunk just
    # inside each read window would otherwise hold the probe open indefinitely
    # and, because a batch is sequential, stall the whole validator.
    async def slow_body():
        for _ in range(50):
            yield b"data: " + delta_frames("41")[0] + b"\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=slow_body(),
            headers={"Content-Type": "text/event-stream"},
        )

    # Clock jumps past the 30s budget on the second chunk.
    clock = iter([0, 31_000_000_000, 31_000_000_000, 31_000_000_000])
    monkeypatch.setattr(probe, "_perf_counter_ns", lambda: next(clock))

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert "exceeded" in result.error
    assert "in total" in result.error


@pytest.mark.asyncio
async def test_ttft_does_not_grow_with_the_length_of_the_answer(
    monkeypatch, validator_key, miner_key
):
    """The regression that streaming exists to fix.

    Non-streaming probes reported total time as ``ttft_ms``, so it scaled with
    the token count: against the live miner it read 301/348/383 ms for 64/80/92
    tokens. That compared a full-generation time against a *TTFT* target and
    scored a miner as slower for producing a longer answer.
    """
    answer = " ".join(str(v) for v in range(41, 61))
    frames = delta_frames(answer)

    # First content lands 100ms in; the stream then runs on until 2s.
    reads = {"n": 0}

    def clock() -> int:
        reads["n"] += 1
        if reads["n"] == 1:
            return 0
        if reads["n"] == 2:
            return 100_000_000
        return 2_000_000_000

    monkeypatch.setattr(probe, "_perf_counter_ns", clock)

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=committed(frames),
        )

        async def body():
            for chunk in sse_wire(frames, signed).split(b"\n\n"):
                if chunk:
                    yield chunk + b"\n\n"

        return httpx.Response(
            200, content=body(), headers={"Content-Type": "text/event-stream"}
        )

    result = await run_with_handler(
        handler,
        validator_key=validator_key,
        miner_key=miner_key,
        challenge=FIXED_CHALLENGE,
    )

    assert result.outcome == "success"
    # 100ms, not the 2000ms the whole stream took.
    assert result.ttft_ms == 100
    # ...and throughput is over the 1900ms generation window, not the 2000ms total.
    assert result.tps_milli == 4 * 1_000_000 // 1_900
