"""Instant — a Bittensor subnet for fast, attested inference.

Layout:

* ``instant.protocol`` — wire format. Signing, receipts, attestation bundles,
  request/response schemas. No Bittensor SDK dependency, no network, no GPU.
  Everything here is testable in milliseconds on a laptop.
* ``instant.miner`` — the miner: an Epistula-gated proxy in front of vLLM.
* ``instant.validator`` — the validator: probing, scoring, weight setting.
* ``instant.common`` — config loading and the boot-time guards that stop a
  development flag from reaching mainnet.
"""

__version__ = "0.1.0"
