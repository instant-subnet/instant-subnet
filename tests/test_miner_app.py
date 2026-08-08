"""The miner's HTTP surface, end to end.

These tests drive real HTTP through the real app against a fake vLLM
(``tests/fake_vllm.py``) and a :class:`StubAttestor`. Nothing is monkey-
patched and no handler is called directly: a request is signed with a real
sr25519 key, sent as bytes, verified by the real Epistula code, forwarded to
the fake upstream over the real streaming client, and the receipt that comes
back is verified with the real receipt verifier.

That matters because most of the bugs this file is here to catch live in the
seams — a signature computed over re-serialised JSON instead of the bytes
that arrived, a receipt hashing the wire form instead of the assembled body,
a TTFT measured to the wrong frame. Testing the handlers in isolation would
step over every one of them.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fake_vllm import FakeVllm
from fastapi.testclient import TestClient

from instant.miner.app import MinerContext, create_app
from instant.miner.attest import AttestationCache, AttestationUnavailable, StubAttestor
from instant.miner.auth import AcceptList, Authenticator
from instant.miner.upstream import VllmClient
from instant.protocol import receipts
from instant.protocol.attestation import (
    AttestationBundle,
    AttestationError,
    AttestationPolicy,
    VendorChecks,
    compute_binding,
    new_nonce,
    verify_bundle,
)
from instant.protocol.epistula import generate_headers

MODEL = "test-model"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def fake() -> FakeVllm:
    return FakeVllm(model_id=MODEL)


@pytest.fixture
def accept(platform_key, validator_key) -> AcceptList:
    lst = AcceptList(platform_hotkey=platform_key.ss58_address)
    lst.update(frozenset({validator_key.ss58_address}))
    return lst


@pytest.fixture
def ctx(fake, miner_key, accept) -> MinerContext:
    return MinerContext(
        signer=miner_key,
        auth=Authenticator(my_hotkey=miner_key.ss58_address, accept=accept),
        vllm=VllmClient("http://vllm.test", transport=fake.transport()),
        attestation=AttestationCache(
            StubAttestor(hotkey_ss58=miner_key.ss58_address, model_id=MODEL)
        ),
        model_id=MODEL,
        max_model_len=8192,
        max_concurrent=4,
        attestation_mode="off",
    )


@pytest.fixture
def client(ctx):
    with TestClient(create_app(ctx)) as c:
        yield c


def body_for(**overrides) -> bytes:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return json.dumps(payload, separators=(",", ":")).encode()


def signed(key, body: bytes, *, to: str) -> dict[str, str]:
    headers = generate_headers(key, body, signed_for=to)
    headers["content-type"] = "application/json"
    return headers


def assembled_from_sse(text: str) -> bytes:
    """Rebuild what the miner hashed: the upstream ``data:`` payloads.

    Excludes ``[DONE]`` and any frame carrying an ``event:`` line — the
    receipt and error events are ours, not the upstream's, and the miner did
    not hash them either.
    """
    out = bytearray()
    for frame in text.split("\n\n"):
        lines = [ln for ln in frame.split("\n") if ln]
        if not lines or any(ln.startswith("event:") for ln in lines):
            continue
        for line in lines:
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            out.extend(data.encode())
    return bytes(out)


def sse_events(text: str) -> list[tuple[str, dict]]:
    """``(event_name, payload)`` for frames that carry an ``event:`` line."""
    events = []
    for frame in text.split("\n\n"):
        lines = [ln for ln in frame.split("\n") if ln]
        name = next((ln[6:].strip() for ln in lines if ln.startswith("event:")), None)
        if name is None:
            continue
        data = next((ln[5:].strip() for ln in lines if ln.startswith("data:")), "{}")
        events.append((name, json.loads(data)))
    return events


# --------------------------------------------------------------------------
# Unauthenticated endpoints
# --------------------------------------------------------------------------


def test_health_is_ok_when_upstream_is_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["model_id"] == MODEL
    assert body["attestation_mode"] == "off"


def test_health_reports_loading_before_weights_are_resident(client, fake):
    fake.ready = False
    body = client.get("/health").json()
    assert body["ready"] is False
    # Young process, upstream not up yet: still loading, not broken.
    assert body["status"] == "loading"


def test_health_is_not_ready_when_upstream_advertises_a_different_model(client, fake):
    fake.model_id = "wrong-model"
    body = client.get("/health").json()
    assert body["ready"] is False
    assert body["model_id"] == MODEL


def test_health_is_degraded_when_the_accept_list_is_stale(client, ctx):
    ctx.auth.accept.updated_at = time.time() - 10_000
    body = client.get("/health").json()
    # Serving fine, but about to start rejecting everyone. Saying "ok" here
    # would hide the failure.
    assert body["ready"] is True
    assert body["status"] == "degraded"


def test_capacity_reports_admission_state(client, ctx):
    body = client.get("/capacity").json()
    assert body == {
        "max_concurrent": 4,
        "in_flight": 0,
        "queue_depth": 0,
        "max_model_len": 8192,
        "accepting": True,
        "tokens_per_s_hint": None,
    }

    ctx.in_flight = 6
    body = client.get("/capacity").json()
    assert body["accepting"] is False
    assert body["queue_depth"] == 2


def test_manifest_has_no_attestation_id_before_a_challenge(client, miner_key):
    body = client.get("/manifest").json()
    assert body["hotkey"] == miner_key.ss58_address
    assert body["model_id"] == MODEL
    assert body["attestation_id"] is None
    assert body["attested_at_ms"] is None


def test_unauthenticated_endpoints_need_no_signature(client):
    for path in ("/health", "/capacity", "/manifest"):
        assert client.get(path).status_code == 200


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def test_unsigned_inference_is_rejected(client):
    r = client.post("/v1/chat/completions", content=body_for())
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_signer_off_the_accept_list_is_rejected(client, stranger_key, miner_key):
    body = body_for()
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(stranger_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 401
    assert "accept-list" in r.json()["detail"]


def test_request_addressed_to_another_miner_is_rejected(
    client, platform_key, stranger_key
):
    """A perfectly valid signature, for someone else. Rejected anyway.

    This is what stops a miner laundering the platform's probe traffic
    through a different miner and claiming the result.
    """
    body = body_for()
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=stranger_key.ss58_address),
    )
    assert r.status_code == 401
    assert "different recipient" in r.json()["detail"]


def test_tampering_with_the_body_invalidates_the_signature(
    client, platform_key, miner_key
):
    body = body_for()
    headers = signed(platform_key, body, to=miner_key.ss58_address)
    r = client.post(
        "/v1/chat/completions",
        content=body_for(messages=[{"role": "user", "content": "something else"}]),
        headers=headers,
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "signature mismatch"


def test_replaying_a_request_is_rejected(client, platform_key, miner_key):
    body = body_for()
    headers = signed(platform_key, body, to=miner_key.ss58_address)

    first = client.post("/v1/chat/completions", content=body, headers=headers)
    assert first.status_code == 200

    second = client.post("/v1/chat/completions", content=body, headers=headers)
    assert second.status_code == 401
    assert second.json()["detail"] == "replayed Epistula-Uuid"


def test_empty_accept_list_says_so_rather_than_looking_broken(
    fake, miner_key, platform_key
):
    ctx = MinerContext(
        signer=miner_key,
        auth=Authenticator(my_hotkey=miner_key.ss58_address, accept=AcceptList()),
        vllm=VllmClient("http://vllm.test", transport=fake.transport()),
        attestation=AttestationCache(
            StubAttestor(hotkey_ss58=miner_key.ss58_address, model_id=MODEL)
        ),
        model_id=MODEL,
        max_model_len=8192,
        max_concurrent=4,
        attestation_mode="off",
    )
    with TestClient(create_app(ctx)) as client:
        body = body_for()
        r = client.post(
            "/v1/chat/completions",
            content=body,
            headers=signed(platform_key, body, to=miner_key.ss58_address),
        )
    assert r.status_code == 401
    assert "has not yet synced the metagraph" in r.json()["detail"]


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------


def test_a_model_we_do_not_serve_is_a_400_not_a_silent_substitution(
    client, platform_key, miner_key
):
    body = body_for(model="openai/gpt-oss-120b")
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 400
    assert r.json()["error"] == "model_mismatch"


def test_unknown_fields_are_rejected(client, platform_key, miner_key):
    body = body_for(temperture=0.7)  # the typo, not the field
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_at_capacity_is_429(client, ctx, platform_key, miner_key):
    ctx.in_flight = ctx.max_concurrent
    body = body_for()
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 429
    assert r.json()["error"] == "at_capacity"


# --------------------------------------------------------------------------
# Non-streaming inference
# --------------------------------------------------------------------------


def test_completion_returns_a_verifiable_receipt(
    client, platform_key, miner_key, fake
):
    body = body_for()
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello, world"

    signed_receipt = receipts.SignedReceipt(
        receipt=receipts.Receipt.from_payload(json.loads(r.headers[receipts.H_RECEIPT])),
        signature=r.headers[receipts.H_RECEIPT_SIG],
    )

    # The whole point: the hashes in the receipt describe the bytes that
    # actually crossed the wire, in both directions.
    receipts.verify(
        signed_receipt,
        expected_miner_hotkey=miner_key.ss58_address,
        request_body=body,
        response_body=r.content,
    )
    assert signed_receipt.receipt.signer_of_request == platform_key.ss58_address
    assert signed_receipt.receipt.prompt_tokens == fake.prompt_tokens
    assert signed_receipt.receipt.completion_tokens == fake.completion_tokens


def test_the_miner_forwards_stream_false_to_the_upstream(
    client, platform_key, miner_key, fake
):
    body = body_for()
    client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert fake.requests[-1]["stream"] is False


def test_an_upstream_5xx_becomes_502(client, platform_key, miner_key, fake):
    fake.status = 500
    body = body_for()
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 502
    assert r.json()["error"] == "upstream_error"


def test_an_upstream_400_stays_a_400(client, platform_key, miner_key, fake):
    """A malformed request is the caller's fault, not the miner's.

    Reporting it as 502 would record a miner failure for someone else's bug
    and the miner would be scored for it.
    """
    fake.status = 400
    body = body_for()
    r = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 400


def test_in_flight_returns_to_zero_after_a_failure(
    client, ctx, platform_key, miner_key, fake
):
    fake.status = 500
    body = body_for()
    client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert ctx.in_flight == 0


def test_upstream_model_substitution_is_a_502_without_a_receipt(
    client, platform_key, miner_key, fake
):
    fake.model_id = "wrong-model"
    body = body_for()
    response = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    assert response.status_code == 502
    assert "served model" in response.json()["detail"]
    assert receipts.H_RECEIPT not in response.headers


# --------------------------------------------------------------------------
# Streaming inference
# --------------------------------------------------------------------------


def test_stream_ends_with_a_verifiable_receipt(client, platform_key, miner_key, fake):
    body = body_for(stream=True)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    ) as r:
        assert r.status_code == 200
        assert r.headers["x-accel-buffering"] == "no"
        text = "".join(r.iter_text())

    assert "data: [DONE]" in text
    events = sse_events(text)
    assert [name for name, _ in events] == [receipts.SSE_RECEIPT_EVENT]

    signed_receipt = receipts.SignedReceipt.from_payload(events[0][1])
    receipts.verify(
        signed_receipt,
        expected_miner_hotkey=miner_key.ss58_address,
        request_body=body,
        response_body=assembled_from_sse(text),
    )
    assert signed_receipt.receipt.completion_tokens == fake.completion_tokens


def test_stream_model_substitution_emits_error_and_no_receipt(
    client, platform_key, miner_key, fake
):
    fake.model_id = "wrong-model"
    body = body_for(stream=True)
    response = client.post(
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    )
    names = [name for name, _ in sse_events(response.text)]
    assert names == ["error"]
    assert receipts.SSE_RECEIPT_EVENT not in names


def test_the_miner_asks_the_upstream_for_a_usage_frame(
    client, platform_key, miner_key, fake
):
    body = body_for(stream=True)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    ) as r:
        list(r.iter_text())
    assert fake.requests[-1]["stream_options"] == {"include_usage": True}


def test_ttft_skips_the_role_only_opening_delta(
    client, platform_key, miner_key, fake
):
    """The measurement discipline, asserted.

    vLLM emits ``{"role":"assistant"}`` immediately and the first real token
    some time later. If TTFT were timed to the first frame it would come back
    near zero here regardless of how long generation actually took — a number
    that looks like a measurement and is an artefact.
    """
    fake.pre_content_delay_s = 0.15
    body = body_for(stream=True)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    ) as r:
        text = "".join(r.iter_text())

    receipt = receipts.SignedReceipt.from_payload(sse_events(text)[0][1]).receipt
    assert receipt.ttft_ms_self >= 140
    assert receipt.total_ms_self >= receipt.ttft_ms_self


def test_a_mid_stream_failure_produces_no_receipt(
    client, platform_key, miner_key, fake
):
    """A miner that did not finish the work must not sign as though it had.

    The honest outcome is an error event and no receipt: the observer records
    a failure, which is what happened.
    """
    fake.fail_after_frames = 2
    body = body_for(stream=True)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    ) as r:
        text = "".join(r.iter_text())

    names = [name for name, _ in sse_events(text)]
    assert names == ["error"]
    assert receipts.SSE_RECEIPT_EVENT not in names
    assert "data: [DONE]" not in text


def test_in_flight_returns_to_zero_after_a_stream(client, ctx, platform_key, miner_key):
    body = body_for(stream=True)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers=signed(platform_key, body, to=miner_key.ss58_address),
    ) as r:
        list(r.iter_text())
    assert ctx.in_flight == 0


@pytest.mark.asyncio
async def test_stream_capacity_is_reserved_before_the_iterator_runs(
    ctx, fake, platform_key, miner_key
):
    ctx.max_concurrent = 1
    fake.pre_content_delay_s = 0.1
    app = create_app(ctx)
    body_one = body_for(stream=True, seed=1)
    body_two = body_for(stream=True, seed=2)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    ) as http:
        first = asyncio.create_task(
            http.post(
                "/v1/chat/completions",
                content=body_one,
                headers=signed(platform_key, body_one, to=miner_key.ss58_address),
            )
        )
        await asyncio.sleep(0.01)
        second = await http.post(
            "/v1/chat/completions",
            content=body_two,
            headers=signed(platform_key, body_two, to=miner_key.ss58_address),
        )
        first_response = await first
    await ctx.vllm.aclose()
    assert first_response.status_code == 200
    assert second.status_code == 429
    assert ctx.in_flight == 0


# --------------------------------------------------------------------------
# Attestation
# --------------------------------------------------------------------------


def test_attest_returns_a_bundle_bound_to_the_challenge(
    client, validator_key, miner_key
):
    nonce = new_nonce()
    body = json.dumps({"nonce": nonce}, separators=(",", ":")).encode()
    r = client.post(
        "/attest",
        content=body,
        headers=signed(validator_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 200

    bundle = AttestationBundle.from_payload(r.json()["bundle"])
    assert bundle.hotkey == miner_key.ss58_address
    assert bundle.nonce == nonce
    assert bundle.model_id == MODEL

    binding = compute_binding(
        hotkey_ss58=bundle.hotkey,
        nonce_hex=bundle.nonce,
        weights_digest=bundle.weights_digest,
        image_digest=bundle.image_digest,
    )
    assert bundle.gpu.eat_nonce == binding.hex()
    assert bytes.fromhex(bundle.cpu.report_data_hex)[:32] == binding


def test_a_stub_bundle_is_rejected_by_a_real_policy(client, validator_key, miner_key):
    """Fail closed. A stub that escapes development does not pass.

    Every vendor check is handed back as ``True`` here — the strongest case a
    stub could ever make — and it is still rejected, on the CC-mode step.
    """
    nonce = new_nonce()
    body = json.dumps({"nonce": nonce}, separators=(",", ":")).encode()
    r = client.post(
        "/attest",
        content=body,
        headers=signed(validator_key, body, to=miner_key.ss58_address),
    )
    bundle = AttestationBundle.from_payload(r.json()["bundle"])

    vendor = VendorChecks(
        cpu_chain_valid=True,
        gpu_token_valid=True,
        cpu_report_data=bytes.fromhex(bundle.cpu.report_data_hex),
        gpu_eat_nonce=bundle.gpu.eat_nonce,
    )
    with pytest.raises(AttestationError, match="confidential-compute mode"):
        verify_bundle(
            bundle,
            expected_hotkey=miner_key.ss58_address,
            expected_nonce=nonce,
            vendor=vendor,
            policy=AttestationPolicy(),
            now_ms=int(time.time() * 1000),
        )


def test_manifest_exposes_the_attestation_after_a_challenge(
    client, validator_key, miner_key
):
    assert client.get("/manifest").json()["attestation_id"] is None

    nonce = new_nonce()
    body = json.dumps({"nonce": nonce}, separators=(",", ":")).encode()
    client.post(
        "/attest",
        content=body,
        headers=signed(validator_key, body, to=miner_key.ss58_address),
    )

    manifest = client.get("/manifest").json()
    assert manifest["attestation_id"].startswith("sha256:")
    assert manifest["attested_at_ms"] > 0


@pytest.mark.parametrize(
    "nonce, detail",
    [
        ("zz" * 32, "not hex"),
        ("ab" * 33, "must be 32 bytes"),
    ],
)
def test_a_bad_nonce_is_a_400(client, validator_key, miner_key, nonce, detail):
    body = json.dumps({"nonce": nonce}, separators=(",", ":")).encode()
    r = client.post(
        "/attest",
        content=body,
        headers=signed(validator_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 400
    assert detail in r.json()["detail"]


def test_attest_is_authenticated(client, stranger_key, miner_key):
    body = json.dumps({"nonce": new_nonce()}, separators=(",", ":")).encode()
    r = client.post(
        "/attest",
        content=body,
        headers=signed(stranger_key, body, to=miner_key.ss58_address),
    )
    assert r.status_code == 401


def test_an_unavailable_attestation_is_503_not_500(fake, miner_key, validator_key, accept):
    """"The attestation path is broken" is a retry, not a dead miner."""

    class BrokenAttestor:
        def generate(self, nonce_hex: str):
            raise AttestationUnavailable("snpguest not found — is this a SEV-SNP guest?")

    ctx = MinerContext(
        signer=miner_key,
        auth=Authenticator(my_hotkey=miner_key.ss58_address, accept=accept),
        vllm=VllmClient("http://vllm.test", transport=fake.transport()),
        attestation=AttestationCache(BrokenAttestor()),
        model_id=MODEL,
        max_model_len=8192,
        max_concurrent=4,
        attestation_mode="hard",
    )
    with TestClient(create_app(ctx)) as client:
        body = json.dumps({"nonce": new_nonce()}, separators=(",", ":")).encode()
        r = client.post(
            "/attest",
            content=body,
            headers=signed(validator_key, body, to=miner_key.ss58_address),
        )
    assert r.status_code == 503
    assert r.json()["error"] == "attestation_unavailable"


def test_two_dev_miners_do_not_share_a_stub_gpu_uuid(miner_key, validator_key):
    """The GPU-uniqueness rule must not zero co-located dev miners."""
    a = StubAttestor(hotkey_ss58=miner_key.ss58_address, model_id=MODEL)
    b = StubAttestor(hotkey_ss58=validator_key.ss58_address, model_id=MODEL)
    assert a.gpu_uuid != b.gpu_uuid
