"""The six-step attestation check, exercised without a GPU or a CVM.

This is the point of splitting the vendor signature checks out into
``VendorChecks``: every rejection path below is reachable in a unit test on a
laptop. An attestation verifier whose failure branches are only exercisable
on real confidential hardware is a verifier whose failure branches are never
exercised.
"""

import pytest

from instant.protocol import attestation as att
from instant.protocol.attestation import (
    ALLOWED_CLOCK_SKEW_MS,
    BUNDLE_VERSION,
    AttestationBundle,
    AttestationError,
    AttestationPolicy,
    CpuAttestation,
    GpuAttestation,
    VendorChecks,
    compute_binding,
    new_nonce,
    report_data_matches,
    verify_bundle,
)

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "sha256:" + "11" * 32
IMAGE = "sha256:" + "22" * 32
NOW = 1_780_000_000_000


def make_bundle(miner_key, nonce, **over):
    binding = compute_binding(
        hotkey_ss58=over.get("hotkey", miner_key.ss58_address),
        nonce_hex=nonce,
        weights_digest=over.get("weights_digest", WEIGHTS),
        image_digest=over.get("image_digest", IMAGE),
    )
    report_data = binding + b"\x00" * 32
    return AttestationBundle(
        version=over.get("version", BUNDLE_VERSION),
        hotkey=over.get("hotkey", miner_key.ss58_address),
        nonce=nonce,
        weights_digest=over.get("weights_digest", WEIGHTS),
        image_digest=over.get("image_digest", IMAGE),
        model_id=over.get("model_id", MODEL),
        cpu=CpuAttestation(
            kind=over.get("cpu_kind", "sev-snp"),
            report_b64="ZmFrZQ==",
            measurement="ab" * 48,
            report_data_hex=over.get("report_data_hex", report_data.hex()),
        ),
        gpu=GpuAttestation(
            nras_token="header.payload.sig",
            eat_nonce=over.get("eat_nonce", binding.hex()),
            gpu_uuid=over.get("gpu_uuid", "GPU-0f6c1e9a-0000-0000-0000-000000000001"),
            cc_mode=over.get("cc_mode", "ON"),
            driver_version="550.90.07",
        ),
        generated_at_ms=over.get("generated_at_ms", NOW - 1_000),
    ), report_data


def vendor_for(report_data, binding, **over):
    return VendorChecks(
        cpu_chain_valid=over.get("cpu_chain_valid", True),
        gpu_token_valid=over.get("gpu_token_valid", True),
        cpu_report_data=over.get("cpu_report_data", report_data),
        gpu_eat_nonce=over.get("gpu_eat_nonce", binding.hex()),
    )


@pytest.fixture
def nonce():
    return new_nonce()


@pytest.fixture
def policy():
    return AttestationPolicy(
        allowed_model_ids=frozenset({MODEL}),
        expected_weights={MODEL: WEIGHTS},
        allowed_image_digests=frozenset({IMAGE}),
    )


def ok_case(miner_key, nonce, **over):
    bundle, report_data = make_bundle(miner_key, nonce, **over)
    binding = compute_binding(
        hotkey_ss58=bundle.hotkey,
        nonce_hex=bundle.nonce,
        weights_digest=bundle.weights_digest,
        image_digest=bundle.image_digest,
    )
    return bundle, vendor_for(report_data, binding), binding


# --- happy path -----------------------------------------------------------


def test_valid_bundle_passes(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce)
    result = verify_bundle(
        bundle,
        expected_hotkey=miner_key.ss58_address,
        expected_nonce=nonce,
        vendor=vendor,
        policy=policy,
        now_ms=NOW,
    )
    assert result.ok
    assert result.gpu_uuid == bundle.gpu.gpu_uuid
    assert result.model_id == MODEL
    assert result.warnings == []
    assert result.attestation_id.startswith("sha256:")


def test_attestation_id_is_stable_and_content_bound(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    first = bundle.digest()
    assert bundle.digest() == first
    bundle.model_id = "openai/gpt-oss-20b"
    assert bundle.digest() != first


# --- step 1: structure and identity --------------------------------------


def test_wrong_bundle_version(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce, version="0")
    with pytest.raises(AttestationError, match="bundle version"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_bundle_for_another_hotkey(miner_key, stranger_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce)
    with pytest.raises(AttestationError, match="bundle is for"):
        verify_bundle(bundle, expected_hotkey=stranger_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_replayed_bundle_with_a_stale_nonce(miner_key, nonce, policy):
    # The bundle is internally consistent and every vendor check passes --
    # it is simply an answer to a question nobody just asked.
    bundle, vendor, _ = ok_case(miner_key, nonce)
    with pytest.raises(AttestationError, match="nonce does not match"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=new_nonce(), vendor=vendor, policy=policy,
                      now_ms=NOW)


# --- step 2: freshness ----------------------------------------------------


def test_bundle_from_the_future(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(
        miner_key, nonce, generated_at_ms=NOW + ALLOWED_CLOCK_SKEW_MS + 1_000
    )
    with pytest.raises(AttestationError, match="future"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_small_clock_skew_is_tolerated(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(
        miner_key, nonce, generated_at_ms=NOW + ALLOWED_CLOCK_SKEW_MS - 1
    )
    verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                  expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_stale_bundle(miner_key, nonce, policy):
    old = NOW - policy.max_age_ms - 60_000
    bundle, vendor, _ = ok_case(miner_key, nonce, generated_at_ms=old)
    with pytest.raises(AttestationError, match="stale"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


# --- step 3: vendor trust -------------------------------------------------


def test_unaccepted_cpu_tee_kind(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce, cpu_kind="sgx")
    with pytest.raises(AttestationError, match="TEE kind"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_bad_cpu_chain(miner_key, nonce, policy):
    bundle, report_data = make_bundle(miner_key, nonce)
    binding = compute_binding(hotkey_ss58=bundle.hotkey, nonce_hex=nonce,
                              weights_digest=WEIGHTS, image_digest=IMAGE)
    vendor = vendor_for(report_data, binding, cpu_chain_valid=False)
    with pytest.raises(AttestationError, match="CPU attestation chain"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_bad_gpu_token(miner_key, nonce, policy):
    bundle, report_data = make_bundle(miner_key, nonce)
    binding = compute_binding(hotkey_ss58=bundle.hotkey, nonce_hex=nonce,
                              weights_digest=WEIGHTS, image_digest=IMAGE)
    vendor = vendor_for(report_data, binding, gpu_token_valid=False)
    with pytest.raises(AttestationError, match="NVIDIA attestation token"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


# --- step 4: binding ------------------------------------------------------


def test_binding_depends_on_every_input(miner_key, stranger_key, nonce):
    base = compute_binding(hotkey_ss58=miner_key.ss58_address, nonce_hex=nonce,
                           weights_digest=WEIGHTS, image_digest=IMAGE)
    variants = [
        compute_binding(hotkey_ss58=stranger_key.ss58_address, nonce_hex=nonce,
                        weights_digest=WEIGHTS, image_digest=IMAGE),
        compute_binding(hotkey_ss58=miner_key.ss58_address, nonce_hex=new_nonce(),
                        weights_digest=WEIGHTS, image_digest=IMAGE),
        compute_binding(hotkey_ss58=miner_key.ss58_address, nonce_hex=nonce,
                        weights_digest="sha256:" + "33" * 32, image_digest=IMAGE),
        compute_binding(hotkey_ss58=miner_key.ss58_address, nonce_hex=nonce,
                        weights_digest=WEIGHTS, image_digest="sha256:" + "44" * 32),
    ]
    assert all(v != base for v in variants)
    assert len(set(variants)) == len(variants)


def test_binding_rejects_bad_inputs(miner_key):
    with pytest.raises(AttestationError, match="bad hotkey"):
        compute_binding(hotkey_ss58="nope", nonce_hex=new_nonce(),
                        weights_digest=WEIGHTS, image_digest=IMAGE)
    with pytest.raises(AttestationError, match="valid hex"):
        compute_binding(hotkey_ss58=miner_key.ss58_address, nonce_hex="zz",
                        weights_digest=WEIGHTS, image_digest=IMAGE)
    with pytest.raises(AttestationError, match="32 bytes"):
        compute_binding(hotkey_ss58=miner_key.ss58_address, nonce_hex="aabb",
                        weights_digest=WEIGHTS, image_digest=IMAGE)


def test_report_data_accepts_either_padding():
    b = bytes(range(32))
    assert report_data_matches(b + b"\x00" * 32, b)
    assert report_data_matches(b"\x00" * 32 + b, b)
    assert not report_data_matches(b"\xff" * 64, b)
    assert not report_data_matches(b + b"\x00" * 31, b)  # wrong length


def test_binding_absent_from_cpu_report(miner_key, nonce, policy):
    bundle, report_data = make_bundle(miner_key, nonce)
    binding = compute_binding(hotkey_ss58=bundle.hotkey, nonce_hex=nonce,
                              weights_digest=WEIGHTS, image_digest=IMAGE)
    vendor = vendor_for(report_data, binding, cpu_report_data=b"\x00" * 64)
    with pytest.raises(AttestationError, match="CPU REPORT_DATA"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_self_reported_report_data_must_match_the_signed_report(
    miner_key, nonce, policy
):
    bundle, report_data = make_bundle(miner_key, nonce)
    binding = compute_binding(hotkey_ss58=bundle.hotkey, nonce_hex=nonce,
                              weights_digest=WEIGHTS, image_digest=IMAGE)
    bundle.cpu.report_data_hex = ("00" * 64)
    vendor = vendor_for(report_data, binding)
    with pytest.raises(AttestationError, match="disagrees with the signed report"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_cpu_report_data_hex_must_be_hex(miner_key, nonce, policy):
    bundle, report_data = make_bundle(miner_key, nonce)
    binding = compute_binding(hotkey_ss58=bundle.hotkey, nonce_hex=nonce,
                              weights_digest=WEIGHTS, image_digest=IMAGE)
    bundle.cpu.report_data_hex = "zz"
    vendor = vendor_for(report_data, binding)
    with pytest.raises(AttestationError, match="valid hex"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_cvm_without_the_right_gpu_is_rejected(miner_key, nonce, policy):
    # A real CVM, a real NVIDIA token, but the token attests a different
    # binding -- i.e. a GPU that is not the one in this VM.
    bundle, report_data = make_bundle(miner_key, nonce)
    binding = compute_binding(hotkey_ss58=bundle.hotkey, nonce_hex=nonce,
                              weights_digest=WEIGHTS, image_digest=IMAGE)
    vendor = vendor_for(report_data, binding, gpu_eat_nonce="ff" * 32)
    with pytest.raises(AttestationError, match="eat_nonce"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_self_reported_eat_nonce_must_match_the_verified_claims(
    miner_key, nonce, policy
):
    bundle, vendor, _ = ok_case(miner_key, nonce)
    bundle.gpu.eat_nonce = "ab" * 32
    with pytest.raises(AttestationError, match="disagrees with the verified token"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


# --- step 5: CC state -----------------------------------------------------


@pytest.mark.parametrize("mode", ["OFF", "DEVTOOLS", "", "on"])
def test_non_confidential_gpu_modes_are_rejected(miner_key, nonce, policy, mode):
    # DEVTOOLS in particular: it looks like CC is on, and memory protection
    # is deliberately weakened. It must never count as attested.
    bundle, vendor, _ = ok_case(miner_key, nonce, cc_mode=mode)
    with pytest.raises(AttestationError, match="confidential-compute mode"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_empty_gpu_uuid_is_rejected(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce, gpu_uuid="")
    with pytest.raises(AttestationError, match="gpu_uuid"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


# --- step 6: content policy ----------------------------------------------


def test_model_not_on_the_allow_list(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce, model_id="meta-llama/Llama-3-8B")
    with pytest.raises(AttestationError, match="allow-list"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_wrong_weights_digest(miner_key, nonce, policy):
    # The miner is in a real TEE running a real GPU -- and serving weights
    # that are not the ones we pinned. Everything up to step 6 passes.
    bundle, vendor, _ = ok_case(miner_key, nonce, weights_digest="sha256:" + "99" * 32)
    with pytest.raises(AttestationError, match="weights digest"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_wrong_image_digest(miner_key, nonce, policy):
    bundle, vendor, _ = ok_case(miner_key, nonce, image_digest="sha256:" + "99" * 32)
    with pytest.raises(AttestationError, match="image digest"):
        verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                      expected_nonce=nonce, vendor=vendor, policy=policy, now_ms=NOW)


def test_unpinned_image_warns_rather_than_failing(miner_key, nonce):
    policy = AttestationPolicy(
        allowed_model_ids=frozenset({MODEL}),
        expected_weights={MODEL: WEIGHTS},
        allowed_image_digests=frozenset(),
    )
    bundle, vendor, _ = ok_case(miner_key, nonce)
    result = verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                           expected_nonce=nonce, vendor=vendor, policy=policy,
                           now_ms=NOW)
    assert result.ok
    assert any("image" in w for w in result.warnings)


def test_empty_policy_allows_anything_but_still_checks_binding(miner_key, nonce):
    bundle, vendor, _ = ok_case(miner_key, nonce, model_id="anything/at-all")
    result = verify_bundle(bundle, expected_hotkey=miner_key.ss58_address,
                           expected_nonce=nonce, vendor=vendor,
                           policy=AttestationPolicy(), now_ms=NOW)
    assert result.ok


# --- schema strictness ----------------------------------------------------


def test_bundle_payload_round_trip(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    again = AttestationBundle.from_payload(bundle.to_payload())
    assert again.to_payload() == bundle.to_payload()
    assert again.digest() == bundle.digest()


def test_unknown_bundle_field_is_rejected(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    payload = bundle.to_payload()
    payload["extra"] = 1
    with pytest.raises(AttestationError, match="unknown fields"):
        AttestationBundle.from_payload(payload)


def test_unknown_nested_field_is_rejected(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    payload = bundle.to_payload()
    payload["gpu"]["surprise"] = True
    with pytest.raises(AttestationError, match="gpu: unknown fields"):
        AttestationBundle.from_payload(payload)


def test_missing_bundle_field_is_rejected(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    payload = bundle.to_payload()
    del payload["weights_digest"]
    with pytest.raises(AttestationError, match="missing fields"):
        AttestationBundle.from_payload(payload)


def test_generated_at_must_be_an_integer(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    payload = bundle.to_payload()
    payload["generated_at_ms"] = "soon"
    with pytest.raises(AttestationError, match="must be an integer"):
        AttestationBundle.from_payload(payload)


def test_optional_fields_survive_round_trip(miner_key, nonce):
    bundle, _ = make_bundle(miner_key, nonce)
    bundle.gpu.vbios_version = "96.00.74.00.11"
    bundle.cpu.cert_chain_b64 = "Y2VydA=="
    again = AttestationBundle.from_payload(bundle.to_payload())
    assert again.gpu.vbios_version == "96.00.74.00.11"
    assert again.cpu.cert_chain_b64 == "Y2VydA=="


def test_nonces_are_unique_and_long_enough():
    ns = {new_nonce() for _ in range(200)}
    assert len(ns) == 200
    assert all(len(n) == att.NONCE_BYTES * 2 for n in ns)
