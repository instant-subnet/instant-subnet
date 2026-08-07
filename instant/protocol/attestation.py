"""Attestation bundles: what a miner proves, and how a verifier checks it.

This is the module that makes Instant more than a latency leaderboard. A
miner does not merely *claim* to run `gpt-oss-120b` in a TEE; it produces a
bundle that ties four facts together in a way it cannot forge:

1. **A confidential VM is running** — an AMD SEV-SNP attestation report (or
   Intel TDX quote) signed by the CPU vendor's root of trust.
2. **A specific GPU is in confidential-compute mode** — an NVIDIA remote
   attestation (NRAS) result, a JWT signed by NVIDIA, covering the GPU's
   measurement and CC state.
3. **A specific set of weights is loaded** — the sha256 of the resolved
   `WEIGHTS.lock` manifest, computed by the miner over the files it actually
   opened.
4. **All three belong to the same miner, right now** — a *binding* value
   that the verifier chose (the nonce) and that the miner cannot have
   pre-computed, carried inside both the SEV-SNP report and the NVIDIA
   attestation.

Point 4 is the entire game. Steps 1–3 in isolation are replayable: anyone can
copy a valid attestation off a public forum and serve it. What they cannot do
is get the AMD PSP or NVIDIA's attestation service to embed *our* fresh nonce
into a report they do not control. So:

    binding = sha256( hotkey_pubkey || nonce || weights_digest || image_digest )

goes into the SEV-SNP ``REPORT_DATA`` field (64 bytes, attacker-uncontrolled
once the report is generated) and into the NVIDIA ``eat_nonce``. A verifier
recomputes it from values it already knows and compares. If either copy
disagrees, the bundle is rejected — not downgraded, not warned about.

**What this module does and does not do.** It performs the structural and
binding checks, which are pure functions of the bundle and need no network.
It does *not* verify the AMD certificate chain or the NVIDIA JWT signature —
those need vendor roots and, for NRAS, a live call. Those live in
``instant.validator.attest_verify`` so this module stays hermetically
testable. :func:`verify_bundle` takes the results of those checks as
arguments rather than performing them, which also means a unit test can
exercise every rejection path without a GPU, a CVM, or a network.

See DESIGN.md §6.3 (bundle schema) and §6.4 (the six-step check).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json
from .ss58 import InvalidSS58Address
from .ss58 import decode as ss58_decode

#: Bundle schema version. Bump on any change to which fields are covered by
#: the binding — a verifier must never silently accept a bundle whose binding
#: was computed over a different set of inputs than it checks.
BUNDLE_VERSION = "1"

#: Nonce length in bytes. 32 is overkill for uniqueness and exactly right for
#: "no one will ever argue about the entropy".
NONCE_BYTES = 32

#: A bundle timestamped slightly ahead of the verifier is a clock, not an
#: attack. More than this and something is wrong enough to look at.
ALLOWED_CLOCK_SKEW_MS = 30_000

#: SEV-SNP REPORT_DATA is a fixed 64-byte field. We put a 32-byte binding in
#: the low half and zero the rest; some tooling left-pads and some right-pads,
#: so :func:`report_data_matches` accepts either placement rather than
#: pretending the ecosystem is consistent.
REPORT_DATA_BYTES = 64


class AttestationError(Exception):
    """A bundle failed verification. ``reason`` is safe to log and return."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------


def new_nonce() -> str:
    """A fresh verifier-chosen nonce, hex-encoded.

    The verifier generates this and sends it in the ``/attest`` request. It
    must never be derived from anything the miner can predict — not the block
    hash, not the timestamp, not a counter. Predictable means precomputable
    means the freshness guarantee is decorative.
    """
    return secrets.token_hex(NONCE_BYTES)


def compute_binding(
    *,
    hotkey_ss58: str,
    nonce_hex: str,
    weights_digest: str,
    image_digest: str,
) -> bytes:
    """The 32 bytes that must appear in both vendor attestations.

    ``sha256( hotkey_pubkey || nonce || weights_digest || image_digest )``

    The hotkey enters as its raw 32-byte public key, not its SS58 string, so
    that the binding does not depend on address formatting. The digests enter
    as their UTF-8 ``sha256:<hex>`` strings, which is what the manifest
    carries and what a human comparing values by eye will see.
    """
    try:
        pubkey = ss58_decode(hotkey_ss58)
    except InvalidSS58Address as exc:
        raise AttestationError(f"bad hotkey in binding: {exc}") from exc

    try:
        nonce = bytes.fromhex(nonce_hex)
    except ValueError as exc:
        raise AttestationError("nonce is not valid hex") from exc
    if len(nonce) != NONCE_BYTES:
        raise AttestationError(
            f"nonce must be {NONCE_BYTES} bytes, got {len(nonce)}"
        )

    h = hashlib.sha256()
    h.update(pubkey)
    h.update(nonce)
    h.update(weights_digest.encode("utf-8"))
    h.update(image_digest.encode("utf-8"))
    return h.digest()


def report_data_matches(report_data: bytes, binding: bytes) -> bool:
    """True if a 64-byte SEV-SNP ``REPORT_DATA`` carries ``binding``.

    Accepts the binding left-aligned or right-aligned within the 64-byte
    field, with the remainder zero. Different guest tooling pads differently;
    rejecting one convention would mean rejecting honest miners for a reason
    that has nothing to do with security.
    """
    if len(report_data) != REPORT_DATA_BYTES:
        return False
    pad = b"\x00" * (REPORT_DATA_BYTES - len(binding))
    return report_data in (binding + pad, pad + binding)


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CpuAttestation:
    """The CVM half of the bundle.

    ``kind`` is ``sev-snp`` or ``tdx``. ``report_b64`` is the raw vendor
    report, base64-encoded — we keep it opaque here and hand it to the vendor
    verifier rather than parsing it in two places.

    ``measurement`` is the launch measurement, hex. ``report_data_hex`` is the
    64-byte user field we bind into. ``cert_chain_b64`` carries the VCEK/VLEK
    chain when the miner ships it inline; when absent the verifier fetches
    from AMD's KDS using the values in ``report_b64``.
    """

    kind: str
    report_b64: str
    measurement: str
    report_data_hex: str
    cert_chain_b64: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "report_b64": self.report_b64,
            "measurement": self.measurement,
            "report_data_hex": self.report_data_hex,
        }
        if self.cert_chain_b64 is not None:
            payload["cert_chain_b64"] = self.cert_chain_b64
        return payload

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> CpuAttestation:
        _require(p, {"kind", "report_b64", "measurement", "report_data_hex"}, "cpu")
        _allow(p, {"cert_chain_b64"}, {"kind", "report_b64", "measurement",
                                       "report_data_hex"}, "cpu")
        return cls(
            kind=p["kind"],
            report_b64=p["report_b64"],
            measurement=p["measurement"],
            report_data_hex=p["report_data_hex"],
            cert_chain_b64=p.get("cert_chain_b64"),
        )


@dataclass(slots=True)
class GpuAttestation:
    """The GPU half of the bundle.

    ``nras_token`` is the JWT returned by
    ``https://nras.attestation.nvidia.com/v4/attest/gpu``. ``eat_nonce`` is
    the nonce NVIDIA echoes back inside it, which is where our binding lives.

    ``gpu_uuid`` is what enforces one-GPU-one-miner (DESIGN.md §8.4). Two
    registered miners presenting the same UUID is a Sybil signal, subject to
    the ``INSTANT_ALLOW_GPU_REUSE`` development escape hatch that refuses to
    engage on finney.
    """

    nras_token: str
    eat_nonce: str
    gpu_uuid: str
    cc_mode: str
    driver_version: str
    vbios_version: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "nras_token": self.nras_token,
            "eat_nonce": self.eat_nonce,
            "gpu_uuid": self.gpu_uuid,
            "cc_mode": self.cc_mode,
            "driver_version": self.driver_version,
        }
        if self.vbios_version is not None:
            payload["vbios_version"] = self.vbios_version
        return payload

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> GpuAttestation:
        req = {"nras_token", "eat_nonce", "gpu_uuid", "cc_mode", "driver_version"}
        _require(p, req, "gpu")
        _allow(p, {"vbios_version"}, req, "gpu")
        return cls(
            nras_token=p["nras_token"],
            eat_nonce=p["eat_nonce"],
            gpu_uuid=p["gpu_uuid"],
            cc_mode=p["cc_mode"],
            driver_version=p["driver_version"],
            vbios_version=p.get("vbios_version"),
        )


@dataclass(slots=True)
class AttestationBundle:
    """Everything a miner returns from ``POST /attest``.

    ``weights_digest`` is over the resolved ``WEIGHTS.lock`` — every file the
    miner loaded, with its sha256, in sorted order. ``image_digest`` is the
    container image digest from ``IMAGES.lock``. Together they answer "which
    model, built how", which is the question a user asking for gpt-oss-120b
    actually cares about.
    """

    version: str
    hotkey: str
    nonce: str
    weights_digest: str
    image_digest: str
    model_id: str
    cpu: CpuAttestation
    gpu: GpuAttestation
    generated_at_ms: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "hotkey": self.hotkey,
            "nonce": self.nonce,
            "weights_digest": self.weights_digest,
            "image_digest": self.image_digest,
            "model_id": self.model_id,
            "cpu": self.cpu.to_payload(),
            "gpu": self.gpu.to_payload(),
            "generated_at_ms": self.generated_at_ms,
        }

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> AttestationBundle:
        req = {
            "version", "hotkey", "nonce", "weights_digest", "image_digest",
            "model_id", "cpu", "gpu", "generated_at_ms",
        }
        _require(p, req, "bundle")
        _allow(p, set(), req, "bundle")
        if not isinstance(p["generated_at_ms"], int) or isinstance(
            p["generated_at_ms"], bool
        ):
            raise AttestationError("bundle.generated_at_ms must be an integer")
        return cls(
            version=p["version"],
            hotkey=p["hotkey"],
            nonce=p["nonce"],
            weights_digest=p["weights_digest"],
            image_digest=p["image_digest"],
            model_id=p["model_id"],
            cpu=CpuAttestation.from_payload(p["cpu"]),
            gpu=GpuAttestation.from_payload(p["gpu"]),
            generated_at_ms=p["generated_at_ms"],
        )

    def digest(self) -> str:
        """Stable identifier for this bundle — the ``attestation_id`` in a receipt."""
        return "sha256:" + hashlib.sha256(canonical_json(self.to_payload())).hexdigest()


def _require(p: dict[str, Any], required: set[str], where: str) -> None:
    missing = required - set(p)
    if missing:
        raise AttestationError(f"{where}: missing fields {sorted(missing)}")


def _allow(
    p: dict[str, Any], optional: set[str], required: set[str], where: str
) -> None:
    unknown = set(p) - required - optional
    if unknown:
        # Same reasoning as receipts: an unrecognised field means the sender
        # is speaking a version we do not verify. Accepting it would mean
        # attesting to content we never checked.
        raise AttestationError(f"{where}: unknown fields {sorted(unknown)}")


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AttestationPolicy:
    """What a verifier will accept.

    Loaded from ``config/attestation.toml``. Kept as data rather than
    hardcoded because the acceptable driver and image sets change on NVIDIA's
    schedule, not ours, and a driver bump should be a config PR rather than a
    release.
    """

    #: ``hard`` rejects, ``warn`` scores but flags, ``off`` skips entirely.
    #: ``off`` and ``warn`` are refused at boot when ``INSTANT_NETWORK=finney``
    #: — see ``instant.common.guards``.
    mode: str = "hard"

    #: Accepted CPU TEE kinds.
    allowed_cpu_kinds: frozenset[str] = frozenset({"sev-snp", "tdx"})

    #: NVIDIA CC states we treat as attested. ``ON`` is real confidential
    #: compute; ``DEVTOOLS`` deliberately weakens memory protection for
    #: debugging and must never count as attested on mainnet.
    allowed_cc_modes: frozenset[str] = frozenset({"ON"})

    #: Model IDs a miner may serve, from ``config/models.toml``.
    allowed_model_ids: frozenset[str] = frozenset()

    #: Expected weights digests, keyed by model id, from ``WEIGHTS.lock``.
    expected_weights: dict[str, str] = field(default_factory=dict)

    #: Accepted container image digests, from ``IMAGES.lock``. Empty means
    #: "do not check", which is right during development and wrong at launch.
    allowed_image_digests: frozenset[str] = frozenset()

    #: How old a bundle may be before it must be regenerated. A bundle is
    #: pinned to a nonce, so this is a belt-and-braces bound on how long a
    #: cached bundle stays usable for scoring.
    max_age_ms: int = 6 * 60 * 60 * 1000


# --------------------------------------------------------------------------
# The six-step check (DESIGN.md §6.4)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class VendorChecks:
    """Results of the checks this module cannot do offline.

    Supplied by ``instant.validator.attest_verify``, which owns the AMD KDS
    certificate chain and the live NRAS call. Passing them in as data is what
    lets every branch below be unit-tested without hardware.
    """

    #: AMD/Intel certificate chain verified up to the vendor root, and the
    #: report signature verified against the endorsement key.
    cpu_chain_valid: bool

    #: NRAS JWT signature verified against NVIDIA's JWKS, and
    #: ``x-nvidia-overall-att-result`` is true.
    gpu_token_valid: bool

    #: ``REPORT_DATA`` as extracted from the parsed CPU report — not the
    #: miner's self-reported hex string, which is why it arrives separately.
    cpu_report_data: bytes

    #: ``eat_nonce`` as extracted from the *verified* JWT claims.
    gpu_eat_nonce: str


@dataclass(slots=True)
class AttestationResult:
    """Outcome of verifying one bundle."""

    ok: bool
    attestation_id: str
    gpu_uuid: str
    model_id: str
    warnings: list[str] = field(default_factory=list)


def verify_bundle(
    bundle: AttestationBundle,
    *,
    expected_hotkey: str,
    expected_nonce: str,
    vendor: VendorChecks,
    policy: AttestationPolicy,
    now_ms: int,
) -> AttestationResult:
    """Run the six-step check. Raises :class:`AttestationError` on failure.

    The steps, in the order they appear in DESIGN.md §6.4:

    1. **Structure and identity** — schema version, and the bundle is from the
       hotkey we asked, carrying the nonce we chose.
    2. **Freshness** — the bundle is not from the future and not stale.
    3. **Vendor trust** — both vendor signature chains verified. This is the
       step that costs a network call; it is deliberately not first, because
       the cheap checks above eliminate most bad bundles for free.
    4. **Binding** — the recomputed binding appears in *both* the SEV-SNP
       ``REPORT_DATA`` and the NVIDIA ``eat_nonce``. One is not enough: a CVM
       with no GPU, or a GPU on a non-confidential host, each pass one side.
    5. **CC state** — the GPU is actually in confidential-compute mode, not
       in ``DEVTOOLS``, not ``OFF``.
    6. **Content policy** — model, weights and image are ones we allow.

    Order matters and is not arbitrary: cheap and definitive first, expensive
    and networked in the middle, policy last so that a policy rejection is
    reported as a policy rejection rather than masked by a structural one.
    """
    warnings: list[str] = []

    # --- 1. Structure and identity -------------------------------------
    if bundle.version != BUNDLE_VERSION:
        raise AttestationError(
            f"unsupported bundle version {bundle.version!r}, expected {BUNDLE_VERSION!r}"
        )
    if bundle.hotkey != expected_hotkey:
        raise AttestationError(
            f"bundle is for {bundle.hotkey}, expected {expected_hotkey}"
        )
    if not secrets.compare_digest(bundle.nonce, expected_nonce):
        # A mismatched nonce is the signature of a replayed bundle, so this
        # gets a constant-time compare even though the nonce is not secret —
        # it costs nothing and removes an argument.
        raise AttestationError("bundle nonce does not match the challenge")

    # --- 2. Freshness ---------------------------------------------------
    age_ms = now_ms - bundle.generated_at_ms
    if age_ms < -ALLOWED_CLOCK_SKEW_MS:
        raise AttestationError("bundle was generated in the future")
    if age_ms > policy.max_age_ms:
        raise AttestationError(f"bundle is stale by {age_ms - policy.max_age_ms}ms")

    # --- 3. Vendor trust ------------------------------------------------
    if bundle.cpu.kind not in policy.allowed_cpu_kinds:
        raise AttestationError(f"CPU TEE kind {bundle.cpu.kind!r} is not accepted")
    if not vendor.cpu_chain_valid:
        raise AttestationError("CPU attestation chain did not verify")
    if not vendor.gpu_token_valid:
        raise AttestationError("NVIDIA attestation token did not verify")

    # --- 4. Binding -----------------------------------------------------
    binding = compute_binding(
        hotkey_ss58=bundle.hotkey,
        nonce_hex=bundle.nonce,
        weights_digest=bundle.weights_digest,
        image_digest=bundle.image_digest,
    )

    if not report_data_matches(vendor.cpu_report_data, binding):
        raise AttestationError("binding is not present in CPU REPORT_DATA")

    # The miner also states REPORT_DATA in the bundle. We check it against
    # the parsed report so that a mismatch is reported honestly rather than
    # the self-reported copy being quietly ignored.
    try:
        claimed = bytes.fromhex(bundle.cpu.report_data_hex.removeprefix("0x"))
    except ValueError as exc:
        raise AttestationError("cpu.report_data_hex is not valid hex") from exc
    if claimed != vendor.cpu_report_data:
        raise AttestationError(
            "cpu.report_data_hex disagrees with the signed report"
        )

    if not _eat_nonce_matches(vendor.gpu_eat_nonce, binding):
        raise AttestationError("binding is not present in the NVIDIA eat_nonce")
    if not _eat_nonce_matches(bundle.gpu.eat_nonce, binding):
        raise AttestationError(
            "gpu.eat_nonce disagrees with the verified token claims"
        )

    # --- 5. CC state ----------------------------------------------------
    if bundle.gpu.cc_mode not in policy.allowed_cc_modes:
        raise AttestationError(
            f"GPU confidential-compute mode is {bundle.gpu.cc_mode!r}, "
            f"accepted: {sorted(policy.allowed_cc_modes)}"
        )
    if not bundle.gpu.gpu_uuid:
        raise AttestationError("gpu.gpu_uuid is empty")

    # --- 6. Content policy ----------------------------------------------
    if policy.allowed_model_ids and bundle.model_id not in policy.allowed_model_ids:
        raise AttestationError(f"model {bundle.model_id!r} is not on the allow-list")

    expected = policy.expected_weights.get(bundle.model_id)
    if expected is not None and bundle.weights_digest != expected:
        raise AttestationError(
            f"weights digest for {bundle.model_id} is {bundle.weights_digest}, "
            f"expected {expected}"
        )
    if expected is None and policy.expected_weights:
        warnings.append(
            f"no pinned weights digest for {bundle.model_id}; served unpinned"
        )

    if policy.allowed_image_digests:
        if bundle.image_digest not in policy.allowed_image_digests:
            raise AttestationError(
                f"image digest {bundle.image_digest} is not on the allow-list"
            )
    else:
        warnings.append("image digest allow-list is empty; image not pinned")

    return AttestationResult(
        ok=True,
        attestation_id=bundle.digest(),
        gpu_uuid=bundle.gpu.gpu_uuid,
        model_id=bundle.model_id,
        warnings=warnings,
    )


def _eat_nonce_matches(eat_nonce: str, binding: bytes) -> bool:
    """NVIDIA echoes the nonce as a hex string; compare bytes, not case."""
    try:
        raw = bytes.fromhex(eat_nonce.removeprefix("0x"))
    except ValueError:
        return False
    return secrets.compare_digest(raw, binding)
