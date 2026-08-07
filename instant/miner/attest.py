"""Producing attestation bundles, on the miner side.

Two implementations behind one interface:

* :class:`HardwareAttestor` — the real thing. Shells out to ``snpguest`` for
  a SEV-SNP report and to NVIDIA's attestation SDK for a GPU token, both
  bound to the verifier's nonce.
* :class:`StubAttestor` — produces a structurally valid bundle with fake
  vendor material, for Tier 0 development where there is no GPU. It is
  reachable only when ``attestation_mode == "off"``, which
  :mod:`instant.common.guards` refuses to allow on mainnet.

The stub exists because the alternative is worse. Without it, every protocol
test needs confidential hardware, so in practice the attestation path gets
tested once by hand and then never again. With it, the *shape* of the flow —
challenge, generate, bind, return — runs on every CI commit, and only the
vendor material differs. The stub marks itself unmistakably (``STUB`` in the
report and a ``cc_mode`` that no policy accepts) so that a stub bundle can
never be mistaken for a real one even if it somehow escapes development.

Bundle generation is slow — hundreds of milliseconds for the SEV-SNP report,
plus a round trip to NVIDIA's service. It must never happen on the inference
path. :class:`AttestationCache` regenerates on a fresh nonce and otherwise
serves the last bundle, and generation runs in a thread so a slow ``snpguest``
cannot stall the event loop and make every concurrent request look slow.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Protocol

from ..protocol.attestation import (
    BUNDLE_VERSION,
    AttestationBundle,
    CpuAttestation,
    GpuAttestation,
    compute_binding,
)

log = logging.getLogger("instant.miner.attest")

NRAS_URL = "https://nras.attestation.nvidia.com/v4/attest/gpu"


class AttestationUnavailable(Exception):
    """The miner cannot produce a bundle right now."""


class Attestor(Protocol):
    """Produces a bundle bound to a verifier-chosen nonce."""

    def generate(self, nonce_hex: str) -> AttestationBundle:  # pragma: no cover
        ...


# --------------------------------------------------------------------------
# Weights and image identity
# --------------------------------------------------------------------------


def weights_digest(lockfile: Path) -> str:
    """Digest of ``WEIGHTS.lock``.

    The lockfile lists every weight file with its sha256, sorted. We hash the
    lockfile rather than re-hashing 60GB of safetensors on every attestation
    — but the lockfile is only meaningful if it was generated from the files
    actually on disk, which is what ``scripts/lock_weights.py`` does at image
    build time, and which is what the image digest then covers.
    """
    if not lockfile.exists():
        raise AttestationUnavailable(f"missing {lockfile}")
    return "sha256:" + hashlib.sha256(lockfile.read_bytes()).hexdigest()


def image_digest(lockfile: Path) -> str:
    """Container image digest from ``IMAGES.lock``."""
    if not lockfile.exists():
        raise AttestationUnavailable(f"missing {lockfile}")
    data = json.loads(lockfile.read_text())
    digest = data.get("runtime_image_digest")
    if not digest:
        raise AttestationUnavailable("IMAGES.lock has no runtime_image_digest")
    return digest


# --------------------------------------------------------------------------
# Real hardware
# --------------------------------------------------------------------------


class HardwareAttestor:
    """SEV-SNP + NVIDIA CC attestation.

    Both vendor calls receive the same 32-byte binding: SEV-SNP takes it in
    ``REPORT_DATA``, NVIDIA takes it as ``eat_nonce``. Neither of them will
    embed a value chosen after the fact, which is what makes the pair of
    reports evidence about *this* miner at *this* moment rather than two
    documents that happen to be valid.
    """

    def __init__(
        self,
        *,
        hotkey_ss58: str,
        model_id: str,
        weights_lock: Path,
        images_lock: Path,
        snpguest_bin: str = "snpguest",
        gpu_index: int = 0,
    ):
        self.hotkey_ss58 = hotkey_ss58
        self.model_id = model_id
        self.weights_lock = weights_lock
        self.images_lock = images_lock
        self.snpguest_bin = snpguest_bin
        self.gpu_index = gpu_index

    def generate(self, nonce_hex: str) -> AttestationBundle:
        wd = weights_digest(self.weights_lock)
        idg = image_digest(self.images_lock)
        binding = compute_binding(
            hotkey_ss58=self.hotkey_ss58,
            nonce_hex=nonce_hex,
            weights_digest=wd,
            image_digest=idg,
        )

        cpu = self._sev_snp_report(binding)
        gpu = self._nvidia_attestation(binding)

        return AttestationBundle(
            version=BUNDLE_VERSION,
            hotkey=self.hotkey_ss58,
            nonce=nonce_hex,
            weights_digest=wd,
            image_digest=idg,
            model_id=self.model_id,
            cpu=cpu,
            gpu=gpu,
            generated_at_ms=int(time.time() * 1000),
        )

    def _sev_snp_report(self, binding: bytes) -> CpuAttestation:
        # REPORT_DATA is 64 bytes; the binding is 32. Left-aligned, zero
        # padded -- the verifier accepts either convention, but we should be
        # consistent about which one we emit.
        report_data = binding + b"\x00" * 32

        try:
            proc = subprocess.run(
                [self.snpguest_bin, "report", "-", "-", "--random"],
                input=report_data.hex().encode(),
                capture_output=True,
                timeout=30,
                check=True,
            )
        except FileNotFoundError as exc:
            raise AttestationUnavailable(
                f"{self.snpguest_bin} not found — is this a SEV-SNP guest?"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AttestationUnavailable("snpguest timed out") from exc
        except subprocess.CalledProcessError as exc:
            raise AttestationUnavailable(
                f"snpguest failed: {exc.stderr.decode(errors='replace')[:400]}"
            ) from exc

        raw = proc.stdout
        return CpuAttestation(
            kind="sev-snp",
            report_b64=base64.b64encode(raw).decode("ascii"),
            measurement=_snp_measurement(raw),
            report_data_hex=report_data.hex(),
        )

    def _nvidia_attestation(self, binding: bytes) -> GpuAttestation:
        # nvtrust's verifier CLI takes a nonce and returns the NRAS token
        # plus the local claims. We shell out rather than importing it so
        # that a broken nvtrust install cannot take the miner process down
        # at import time -- it degrades one endpoint instead of all of them.
        try:
            proc = subprocess.run(
                [
                    "python3", "-m", "nv_attestation_sdk.cli",
                    "--nonce", binding.hex(),
                    "--gpu", str(self.gpu_index),
                    "--json",
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
        except FileNotFoundError as exc:
            raise AttestationUnavailable("nv_attestation_sdk not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise AttestationUnavailable(
                f"NVIDIA attestation timed out — {NRAS_URL} unreachable?"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise AttestationUnavailable(
                f"NVIDIA attestation failed: {exc.stderr.decode(errors='replace')[:400]}"
            ) from exc

        try:
            payload = json.loads(proc.stdout)
        except ValueError as exc:
            raise AttestationUnavailable("NVIDIA attestation returned non-JSON") from exc

        try:
            return GpuAttestation(
                nras_token=payload["token"],
                eat_nonce=payload["eat_nonce"],
                gpu_uuid=payload["gpu_uuid"],
                cc_mode=payload["cc_mode"],
                driver_version=payload["driver_version"],
                vbios_version=payload.get("vbios_version"),
            )
        except KeyError as exc:
            raise AttestationUnavailable(
                f"NVIDIA attestation payload missing {exc}"
            ) from exc


def _snp_measurement(raw: bytes) -> str:
    """Launch measurement: 48 bytes at offset 0x90 of an SEV-SNP report."""
    if len(raw) < 0x90 + 48:
        raise AttestationUnavailable(
            f"SEV-SNP report is {len(raw)} bytes, too short to contain a measurement"
        )
    return raw[0x90:0x90 + 48].hex()


# --------------------------------------------------------------------------
# Development stub
# --------------------------------------------------------------------------


class StubAttestor:
    """Structurally valid, cryptographically meaningless.

    Reachable only with ``INSTANT_ATTESTATION_MODE=off``, which
    :func:`instant.common.guards.enforce` refuses on mainnet. Every field
    that a verifier checks against policy is deliberately set to something
    no production policy accepts, so a stub bundle fails closed rather than
    fails open if it ever reaches a real verifier.
    """

    def __init__(self, *, hotkey_ss58: str, model_id: str, gpu_uuid: str | None = None):
        self.hotkey_ss58 = hotkey_ss58
        self.model_id = model_id
        # Distinct per hotkey so that several dev miners on one box do not
        # collide on the GPU-uniqueness rule for the wrong reason.
        self.gpu_uuid = gpu_uuid or (
            "GPU-STUB-" + hashlib.sha256(hotkey_ss58.encode()).hexdigest()[:24]
        )

    def generate(self, nonce_hex: str) -> AttestationBundle:
        wd = "sha256:" + hashlib.sha256(f"stub-weights:{self.model_id}".encode()).hexdigest()
        idg = "sha256:" + hashlib.sha256(b"stub-image").hexdigest()
        binding = compute_binding(
            hotkey_ss58=self.hotkey_ss58,
            nonce_hex=nonce_hex,
            weights_digest=wd,
            image_digest=idg,
        )
        return AttestationBundle(
            version=BUNDLE_VERSION,
            hotkey=self.hotkey_ss58,
            nonce=nonce_hex,
            weights_digest=wd,
            image_digest=idg,
            model_id=self.model_id,
            cpu=CpuAttestation(
                kind="sev-snp",
                report_b64=base64.b64encode(b"STUB-ATTESTATION-NOT-REAL").decode(),
                measurement="00" * 48,
                report_data_hex=(binding + b"\x00" * 32).hex(),
            ),
            gpu=GpuAttestation(
                nras_token="stub.stub.stub",
                eat_nonce=binding.hex(),
                gpu_uuid=self.gpu_uuid,
                # No production policy accepts STUB. If this bundle ever
                # reaches a real verifier it is rejected at step 5.
                cc_mode="STUB",
                driver_version="0.0.0-stub",
            ),
            generated_at_ms=int(time.time() * 1000),
        )


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


class AttestationCache:
    """Serialises and caches bundle generation.

    A fresh nonce always forces regeneration — that is the freshness
    guarantee and it is not negotiable. The cache exists for the case where
    several verifiers challenge with the *same* nonce, and to keep a slow
    ``snpguest`` off the event loop.

    The lock matters: without it, ten simultaneous ``/attest`` calls launch
    ten ``snpguest`` processes, which on a busy miner is a self-inflicted
    denial of service on the endpoint that proves you are honest.
    """

    def __init__(self, attestor: Attestor):
        self._attestor = attestor
        self._lock = asyncio.Lock()
        self._nonce: str | None = None
        self._bundle: AttestationBundle | None = None
        self._generated_ms: int = 0

    @property
    def last(self) -> AttestationBundle | None:
        return self._bundle

    @property
    def last_generated_ms(self) -> int:
        return self._generated_ms

    async def get(self, nonce_hex: str, *, force: bool = False) -> AttestationBundle:
        async with self._lock:
            if not force and self._nonce == nonce_hex and self._bundle is not None:
                return self._bundle
            started = time.perf_counter()
            bundle = await asyncio.to_thread(self._attestor.generate, nonce_hex)
            took_ms = int((time.perf_counter() - started) * 1000)
            log.info("generated attestation bundle in %dms", took_ms)
            self._nonce = nonce_hex
            self._bundle = bundle
            self._generated_ms = bundle.generated_at_ms
            return bundle
