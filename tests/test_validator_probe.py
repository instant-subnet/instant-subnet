"""Direct validator probes: real signatures, exact bytes, finite batches."""

from __future__ import annotations

import asyncio
import json
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


async def run_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    validator_key: LocalKeypair,
    miner_key: LocalKeypair,
) -> ProbeResult:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await probe_direct(
            client,
            signer=validator_key,
            target=target(miner_key),
            epoch=42,
            model="instant/mock-echo",
        )


@pytest.mark.asyncio
async def test_success_is_epistula_signed_and_receipt_verified(
    monkeypatch, validator_key, miner_key, now_ms
):
    clock = iter([1_000_000_000, 1_120_000_000, 1_250_000_000])
    monkeypatch.setattr(probe, "_perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(probe, "_now_ms", lambda: now_ms)
    seen: dict[str, object] = {}
    response_body = b'{"id":"cmpl-1","choices":[{"message":{"content":"pong"}}]}'

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        payload = json.loads(request_body)
        seen.update(
            request_body=request_body,
            request_id=verified.uuid,
            payload=payload,
        )
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=response_body,
        )
        return httpx.Response(
            200,
            content=response_body,
            headers=receipt_headers(signed),
        )

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert seen["payload"] == {
        "max_completion_tokens": 8,
        "messages": [{"content": "Reply with exactly: pong", "role": "user"}],
        "model": "instant/mock-echo",
        "stream": False,
        "temperature": 0,
        "user": f"validator-probe:{seen['request_id']}",
    }
    assert result == ProbeResult(
        epoch=42,
        uid=7,
        hotkey=miner_key.ss58_address,
        source="direct",
        outcome="success",
        observed_ms=now_ms,
        ttft_ms=120,
        tps_milli=16_000,
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
    response_body = b'{"id":"cmpl-1","choices":[]}'

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(
            request.headers,
            request_body,
            allowed_signers={validator_key.ss58_address},
            expected_signed_for=miner_key.ss58_address,
        )
        kwargs: dict[str, object] = {}
        if mutation == "request_body":
            kwargs["receipt_request_body"] = request_body + b" "
        elif mutation == "response_body":
            kwargs["receipt_response_body"] = response_body + b" "
        elif mutation == "request_id":
            kwargs["receipt_request_id"] = "not-the-epistula-uuid"
        elif mutation == "request_signer":
            kwargs["receipt_signer"] = stranger_key.ss58_address
        signed = signed_response(
            miner_key=miner_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=response_body,
            **kwargs,
        )
        return httpx.Response(
            200,
            content=response_body,
            headers=receipt_headers(signed),
        )

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
    )

    assert result.outcome == "failure"
    assert result.error is not None
    assert error_fragment in result.error
    assert result.tokens is None


@pytest.mark.asyncio
async def test_receipt_must_be_from_expected_miner(validator_key, miner_key, stranger_key):
    response_body = b'{"id":"cmpl-1","choices":[]}'

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        verified = verify_headers(request.headers, request_body)
        signed = signed_response(
            miner_key=stranger_key,
            request_id=verified.uuid,
            signer_of_request=verified.signed_by,
            request_body=request_body,
            response_body=response_body,
        )
        return httpx.Response(
            200,
            content=response_body,
            headers=receipt_headers(signed),
        )

    result = await run_with_handler(
        handler, validator_key=validator_key, miner_key=miner_key
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
