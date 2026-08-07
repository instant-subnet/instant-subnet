"""The Instant wire protocol.

This package is the contract between four independently-deployed things: the
miner, the validator, the platform gateway (which is TypeScript, not Python),
and any third party who wants to audit what the subnet claims. Because of
that last one, everything here is deliberately boring and explicit — no
implicit coercion, no version negotiation, no "be liberal in what you accept".

The dependency rule: ``instant.protocol`` may import ``sr25519``, ``base58``
and ``pydantic``. It may **not** import ``bittensor``. The SDK pulls in a
substrate client, async machinery and a wallet layer; none of that belongs in
the module that a validator runs in a tight loop or that CI runs on every
commit. Keys enter through the :class:`~instant.protocol.keys.Signer`
structural protocol, which ``bittensor.Keypair`` already satisfies.
"""

from .attestation import (
    BUNDLE_VERSION,
    AttestationBundle,
    AttestationError,
    AttestationPolicy,
    AttestationResult,
    CpuAttestation,
    GpuAttestation,
    VendorChecks,
    compute_binding,
    new_nonce,
    verify_bundle,
)
from .canonical import (
    NonCanonicalValue,
    body_sha256,
    bps,
    canonical_json,
    digest,
)
from .epistula import (
    ALLOWED_DELTA_MS,
    ALLOWED_FUTURE_MS,
    EpistulaError,
    ReplayGuard,
    VerifiedRequest,
    generate_headers,
    verify_headers,
)
from .keys import LocalKeypair, Signer, sign, verify
from .receipts import (
    Receipt,
    ReceiptError,
    SignedReceipt,
    merkle_root,
)
from .ss58 import BITTENSOR_SS58_FORMAT, InvalidSS58Address

__all__ = [
    # canonical
    "canonical_json",
    "digest",
    "body_sha256",
    "bps",
    "NonCanonicalValue",
    # keys
    "Signer",
    "LocalKeypair",
    "sign",
    "verify",
    # ss58
    "BITTENSOR_SS58_FORMAT",
    "InvalidSS58Address",
    # epistula
    "generate_headers",
    "verify_headers",
    "VerifiedRequest",
    "EpistulaError",
    "ReplayGuard",
    "ALLOWED_DELTA_MS",
    "ALLOWED_FUTURE_MS",
    # receipts
    "Receipt",
    "SignedReceipt",
    "ReceiptError",
    "merkle_root",
    # attestation
    "AttestationBundle",
    "CpuAttestation",
    "GpuAttestation",
    "AttestationPolicy",
    "AttestationResult",
    "AttestationError",
    "VendorChecks",
    "compute_binding",
    "new_nonce",
    "verify_bundle",
    "BUNDLE_VERSION",
]
