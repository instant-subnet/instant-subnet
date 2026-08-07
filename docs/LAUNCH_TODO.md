# Instant Subnet — Launch TODO

**Reconciled:** August 6, 2026

**Immediate objective:** provision the miner/validator hosts and prove the signed
platform-to-miner path before adding validator writes

This is the public-safe technical checklist. The July 29 private master TODO in Downloads
also contains company, banking, contract, tax, and account work. Those items were not
copied into this public repository and their status has not been re-verified here.

## Current critical path

- [x] Serve the static platform from OCI behind nginx and Cloudflare.
- [x] Extract the subnet scaffold into the repository.
- [x] Resolve the Bittensor/FastAPI dependency conflict.
- [x] Pin `async-substrate-interface<2.0` so `import bittensor` works.
- [x] Confirm all 417 tests and full-project Ruff lint pass through the project venv.
- [x] Inspect the installed btcli surface sufficiently to create the subnet safely.
- [x] Create netuid 5 with Alice and verify it is active.
- [x] Enable subtoken on netuid 5.
- [x] Verify `SubtensorModule.SubtokenEnabled[5] == True`.
- [x] Apply localnet hyperparameters: tempo 10 and commit-reveal off.
- [ ] Select a vLLM-supported GPU/image for `openai/gpt-oss-20b`; do not provision
      the previously considered Ada SKU on VRAM alone.
- [ ] Provision one GPU miner host and one validator host.
- [ ] Retire the exposed plaintext-export validator hotkey; create/register a fresh
      dedicated validator hotkey instead of copying that file to a host.
- [ ] Select and record one miner hotkey from the `miner` wallet.
- [ ] Fund the `validator` coldkey from Alice.
- [ ] Register the miner and validator; record both UIDs.
- [ ] Stake the validator.
- [ ] Wait one tempo and verify `validator_permit` on its hotkey.
- [ ] Submit one live weight vector and require success, not `(False, None)`.
- [ ] Check alpha-pool liquidity/taoflow and observe first emissions.

## Next deployment sequence

1. Validate the exact GPU image against the official GPT-OSS/vLLM recipe, then create
   the miner and validator instances.
2. Install this repository into a fresh Python 3.11 venv on each host.
3. Put only the role-specific wallet on each host and fill its `.env.<role>` file.
4. Run the configuration and chain preflights in `README.md`.
5. Start vLLM, then the miner, platform, and single PM2 validator process.
6. Require `/readyz` and one full chat completion through the platform before adding
   probes or any new chain write.

Do not run another subnet-creation command: netuid 5 already exists.

## Buildout after the chain gate

- [x] Build the localnet-only, single-miner Python/FastAPI platform gateway.
- [ ] Implement gateway API keys, dynamic routing, rate limits, and quotas.
- [ ] Implement gateway receipt verification, storage, telemetry, and audit endpoints.
- [x] Implement exact-byte Epistula platform-to-miner signing and streaming relay.
- [x] Implement miner localnet connection, registration check, and axon announcement.
- [ ] Deploy and smoke-test the miner against the platform and localnet.
- [ ] Implement validator `probe.py` for direct and shadow probes.
- [ ] Implement validator attestation verification orchestration.
- [ ] Implement validator platform client and telemetry/receipt auditing.
- [ ] Implement validator `weights.py` with retry and explicit result checking.
- [x] Implement the read-only validator metagraph loop, local operations API, and
      executable entry point.
- [ ] Implement validator scoring-epoch orchestration.
- [ ] Run the four-way miner/validator/chain/platform smoke test.
- [ ] Exercise component failure modes before production deployment.

## Repository and operator documentation

- [x] Reconcile the July 29 design into `docs/DESIGN.md`.
- [x] Recover the technical launch checklist into this file.
- [x] Add a repository `README.md` with the current development quickstart.
- [ ] Add `docs/MINING.md` and `docs/VALIDATING.md`.
- [ ] Add `WEIGHTS.lock` and `IMAGES.lock` before enabling hard attestation.
- [x] Add per-role env examples with explicit localnet configuration.
- [x] Add single-process PM2 definitions for miner, validator, and gateway.
- [ ] Add a pinned/reboot-safe vLLM process definition after the GPU image is selected.
- [ ] Add DigitalOcean and OCI bootstrap/deploy scripts.
- [x] Fix active configuration documentation to use netuid 5 and port 80.
- [ ] Decide whether `INSTANT_NETUID` should remain defaulted in code or become required.
- [ ] Remove the landing page's `noindex, nofollow` before launch.
- [ ] Configure `api.instantsubnet.com` as DNS-only before streaming traffic.
- [ ] Complete and verify the `.ai` domain redirect.

## Infrastructure

- [ ] Resolve the Azure NCCadsH100v5 quota request for production confidential GPUs.
- [ ] Supply the miner deployment host/IP.
- [ ] Supply the validator deployment host/IP.
- [ ] Inspect how subtensor runs on `68.183.141.180` and record the process manager,
      binary/image, flags, and chain spec.
- [ ] Benchmark the production model under GPU confidential-computing mode.
- [ ] Add monitoring for process health, weight cadence, and emissions.
- [ ] Write restart, recovery, and incident runbooks.

## Production gates

- [ ] Rehearse registration and weight submission on the intended pre-production chain.
- [ ] Serve `openai/gpt-oss-120b`, not a development tier.
- [ ] Enforce `INSTANT_ATTESTATION_MODE=hard` on Finney.
- [ ] Refuse GPU reuse, unpinned weights, and development model tiers on Finney.
- [ ] Publish miner/validator documentation that a new operator can follow.
- [ ] Register the platform hotkey if platform identity remains metagraph-based.
- [ ] Set on-chain subnet identity, repository URL, and contact details.
- [ ] Verify the public dashboard reflects live miners, weights, and attestation state.

## Public-content constraints

- [ ] Keep the landing page and launch material focused on fast inference.
- [ ] Link the X account, GitHub, and the landing page.
- [ ] Do not mention prior subnets, private commercial arrangements, or buybacks.

## Deferred until plumbing works

- [ ] Self-serve signup and API keys.
- [ ] Billing and payments.
- [ ] Multi-model routing.
- [ ] Fine-tuning and agentic tooling.
- [ ] Advanced speed work such as speculative decoding and disaggregated prefill.
- [ ] Multi-region routing.
