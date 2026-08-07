# Instant Subnet — Reconciled Design and Build Plan

**Status:** netuid 5 active; deployable localnet miner/validator/platform skeleton implemented

**Updated:** August 6, 2026

**Current target:** provision the miner and validator hosts and prove one signed inference path
**Production product:** fast, OpenAI-compatible inference for `openai/gpt-oss-120b`

## 0. Authority and scope

This document reconciles three sources:

1. the July 29 v0.1 design draft (`DESIGN_3.md` and its rendered HTML);
2. the August 1 Claude handoff for session `session_01BadVtqgRWSJRm8QsMCGRD3`;
3. the implementation and versioned configuration in this repository.

When they disagree, use this order of authority:

1. Current code and `config/*.toml` define implemented behavior.
2. The later handoff defines the current localnet and deployment procedure.
3. The July 29 draft defines architectural intent for surfaces not implemented yet.

This is a design and status document, not a credential store. Do not add passwords,
mnemonics, private keys, tokens, or recovery material.

Public material for Instant focuses only on fast inference, the X account, GitHub, and
the landing page. Do not mention prior subnets, private commercial arrangements, or
buybacks in public-facing content.

## 1. Current objective and build order

The immediate objective is to deploy and connect four components on the existing
localnet:

1. **Chain** — netuid 5 is active on `ws://68.183.141.180:80`.
2. **Miner** — deploy a `dev1` GPU miner, register it, and announce its axon.
3. **Validator** — deploy the read-only PM2 service, then register/stake its hotkey.
4. **Platform** — deploy the single-miner gateway and prove one Epistula-signed request.

Infrastructure comes first. The current phase establishes process, chain, and HTTP
plumbing before adding probes, scoring epochs, and weight writes.

Current status:

- The static platform is served from OCI.
- The Python package, protocol layer, miner, scoring engine, and validator state store
  exist.
- Netuid 5 exists with subtoken enabled, tempo 10, and commit/reveal disabled.
- A read-only validator loop and operations API, a single-miner platform gateway, role
  env examples, and PM2 definitions now exist.
- The corrected environment passes all 417 tests and full-project lint.
- Dedicated miner and validator hosts have not been provisioned. Validator probes,
  scoring epochs, weight submission, and production platform auth are not implemented.

## 2. Product definition

Instant is a Bittensor subnet that sells fast inference for one pinned open model,
served from hardware that can prove which model and workload it loaded.

The customer-facing surface is OpenAI-compatible and streaming. Production v1 serves
`openai/gpt-oss-120b`. Multi-model routing, fine-tuning, agents, batch inference, and
advanced speed optimizations are deferred until the basic network works end to end.

### 2.1 Development model ladder

The model is selected only through `INSTANT_MODEL_TIER` and `config/models.toml`:

| Tier | Model | Purpose | Attestation |
|---|---|---|---|
| `dev0` | `Qwen/Qwen2.5-0.5B-Instruct` | CPU-only protocol plumbing and CI | `off` |
| `dev1` | `openai/gpt-oss-20b` | same-family GPU integration rig | `hard` |
| `prod` | `openai/gpt-oss-120b` | production | `hard` |

Mainnet refuses to boot with a non-production tier. The development ladder changes the
model behind the protocol; it does not create separate protocol branches.

### 2.2 Serving stack

The miner process is Python/FastAPI in front of vLLM. vLLM serves the selected model on
localhost. The miner owns authentication, capacity control, attestation, receipts,
chain registration, axon announcement, and proxying.

Target SLOs live in `config/slo.toml` and are scoring inputs, not marketing promises:

| Metric | Target |
|---|---|
| Miner TTFT p95 | 250 ms or less |
| Sustained throughput p50 | 100 tokens/s or more |
| Epoch success rate | 99% or more |
| Attestation age | 60 minutes or less |

All consensus-relevant measurements are integers: milliseconds, milli-tokens/s, and
basis points.

## 3. Architecture

```text
User
  │ API key / OpenAI-compatible request
  ▼
Instant platform gateway (Python/FastAPI, single-miner skeleton)
  │ Epistula-signed request
  ▼
Miner (Python/FastAPI) ──► vLLM
  │ signed receipt
  └────────────────────────► platform telemetry

Validator (Python, read-only chain loop implemented)
  ├─ metagraph discovery ───► subtensor
  ├─ direct/shadow probes ──► future build layer
  ├─ receipt/stat audits ───► future build layer
  └─ set_weights ───────────► future bounded writer
```

The chain holds registration, validator permits, weights, and emissions. It is not
modified by this project.

### 3.1 Routing decision

The platform proxies user requests directly to miners. Validators observe out of band;
they never relay customer inference.

This avoids an extra latency and queueing hop, keeps third-party validators out of the
customer availability path, and keeps prompts out of validator infrastructure.

The trust gap is closed with three signals:

- **Shadow probes** enter through the same platform-to-miner path as user traffic.
- **Direct probes** let validators check miners and bound platform misreporting.
- **Miner-signed receipts** let validators audit platform aggregates. The platform can
  withhold a receipt, but it cannot fabricate one.

If the platform is unavailable, validators must still be able to score from direct
probes and continue setting weights.

### 3.2 Platform implementation decision

The July 29 draft specified a Node/Fastify gateway with duplicated Zod schemas. That
decision is superseded. The gateway will be Python/FastAPI so it can reuse the tested
Epistula, SS58, canonical JSON, receipt, and Pydantic schema code in this repository.

The website/dashboard remains a separate web surface. The static landing page is
deployed. The FastAPI gateway is implemented as a localnet-only, single-miner relay but
has not yet been deployed and intentionally has no public API-key/quota layer.

## 4. Identity and authentication

Machine-to-machine requests use Epistula v2 and chain identities. End users use API
keys because they are not chain participants.

### 4.1 Epistula v2

The signed message is exactly:

```text
sha256(body).hexdigest() . uuid . timestamp . signed_for
```

`signed_for` is the empty string when absent. Implemented timing constants are:

- past window: `ALLOWED_DELTA_MS = 8000`
- future allowance: `ALLOWED_FUTURE_MS = 2000`
- secret-signature bucket: `SECRET_INTERVAL_MS = 10000`

Replay protection requires both the timestamp window and a UUID cache covering the
window. `Epistula-Signed-For` binds a request to its intended miner.

### 4.2 Actor identities

| Actor | Identity and authorization |
|---|---|
| Miner | registered hotkey on the target netuid |
| Validator | registered hotkey with `validator_permit` |
| Platform | dedicated hotkey accepted by miners |
| User | `isk_...` API key stored as a one-way hash by the platform |

On localnet, the miner may start before a populated accept-list for bootstrapping. On
non-local networks, the platform/validator accept-list is enforced.

## 5. Miner HTTP surface

The implemented miner surface is:

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | public | readiness, version, model, attestation mode |
| `GET` | `/capacity` | public | concurrency and queue state |
| `GET` | `/manifest` | public | model, digests, latest attestation identity |
| `POST` | `/attest` | Epistula | answer a verifier nonce challenge |
| `POST` | `/v1/chat/completions` | Epistula | OpenAI-compatible inference |

Streaming responses end with a miner-signed `receipt` SSE event. Non-streaming responses
carry `X-Instant-Receipt` and `X-Instant-Receipt-Sig` headers. Receipt self-timings are
diagnostic only; observer timings drive scoring.

The miner refreshes validator permits from the metagraph and announces its axon through
the Bittensor SDK. The only runtime `import bittensor` is lazy in the miner entrypoint.

## 6. Planned platform and validator surfaces

These contracts remain architectural intent and will be finalized when their code is
written.

### 6.1 User-facing platform

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

The gateway authenticates an API key, applies quota/rate limits, selects an attested
healthy miner, signs the miner request, relays SSE without buffering, verifies the final
receipt, and records usage.

### 6.2 Miner-facing platform

- announce endpoint and endpoint verification
- heartbeat/capacity updates
- attestation push/pull
- current network/model/SLO configuration

### 6.3 Validator-facing platform

- live miner/routing view
- aggregate stats backed by receipt counts and digests
- raw receipt sampling/audit path
- shadow challenge injection
- cached attestation retrieval

### 6.4 Validator operational surface

The validator will expose localhost-only health, metrics, and score breakdowns. Its
outer loop, probes, platform client, weight submission, HTTP app, and executable entry
point are still missing. Only deterministic scoring and SQLite persistence currently
exist.

## 7. Attestation

TEE attestation is a production gate, not a score bonus. A miner either proves the
approved hardware/workload/model binding or it does not earn weight.

The bundle binds:

- miner hotkey and netuid;
- verifier nonce and issue time;
- GPU evidence and GPU UUIDs;
- CPU confidential-VM evidence;
- model ID, weights digest, image digest, engine, and context length;
- an sr25519 signature by the miner hotkey.

The production verification order is freshness, GPU evidence, CPU evidence, workload
digests, chain binding/signature, and GPU uniqueness.

Network policy is configuration, not a separate code path:

- localnet: `INSTANT_ATTESTATION_MODE=off`
- Finney: `INSTANT_ATTESTATION_MODE=hard`

The code supports `warn` for non-mainnet development, but it is not the launch policy.
Mainnet refuses `warn`, `off`, GPU reuse, unpinned weights, and non-production models.

## 8. Scoring and weights

The configured quality formula is:

```text
quality = 0.45L + 0.20T + 0.25R + 0.10C
```

Where:

- `L` is TTFT p95 against the 250 ms target;
- `T` is throughput p50 against 100 tokens/s;
- `R` is weighted success reliability;
- `C` is log-scaled demonstrated capacity.

Source mixes are versioned in `config/scoring.toml`. Shadow probes dominate latency and
throughput, direct probes and shadow probes share reliability, and capacity comes from
telemetry.

Consensus arithmetic is integer-only:

- `BPS_ONE = 10000`
- `MAX_WEIGHT_U16 = 65535`
- `LOG2_FRAC_BITS = 24`
- throughput is stored as tokens/s × 1000
- percentiles use nearest-rank ceiling

The epoch calculation is:

1. derive component quality from observations;
2. apply EMA to quality (`ema_alpha_bps = 3000`);
3. apply the attestation/probe gate and penalties;
4. cube the resulting score;
5. normalize deterministically to a u16 vector.

EMA is applied before the gate. A failed gate therefore produces zero weight immediately
instead of retaining historical weight. Carry-forward reputation is keyed by hotkey,
not UID, because UIDs are recycled.

Gate details include at least 20 successful probes, two misses before gate-out, and a
one-epoch cooldown. Exact values in `config/scoring.toml` are authoritative.

## 9. Chain and network configuration

### 9.1 Existing localnet

| Item | Value |
|---|---|
| Endpoint | `ws://68.183.141.180:80` |
| Chain | Bittensor localnet |
| Runtime | `4.0.0-dev-4a2e4b1282d`, spec version 393 |
| Block cadence | standard approximately 12-second blocks |
| Existing netuids | 0 through 5 |
| Instant subnet | netuid 5, active |
| Sudo account | Alice |

The localnet is shared and contains live work on netuid 3. Do not wipe, rebuild, or
runtime-upgrade it. Netuid 5 was created separately for Instant and must not be
recreated.

Every localnet process must receive these explicitly:

```text
INSTANT_NETWORK=local
INSTANT_CHAIN_ENDPOINT=ws://68.183.141.180:80
INSTANT_NETUID=5
INSTANT_ATTESTATION_MODE=off
```

The code now defaults to netuid 5 for local development, but deployments must still set
all four values explicitly so a copied process cannot silently connect to the wrong
chain.

### 9.2 Hyperparameters

| Parameter | Localnet | Finney initial |
|---|---:|---:|
| `tempo` | 10 | 360 |
| `commit_reveal_weights_enabled` | false | false |
| `weights_rate_limit` | 100 | 100 |
| `min_allowed_weights` | 1 | 1 |
| `max_allowed_uids` | 256 | 256 |
| `immunity_period` | 5000 | 5000 |
| `activity_cutoff` | 5000 | 5000 |
| `serving_rate_limit` | 50 | 50 |
| `liquid_alpha_enabled` | false | false |

Commit-reveal is intentionally off initially. It can be revisited after the weight path
is proven and the deployed runtime behavior is understood.

### 9.3 Subtoken activation gate

On this dev chain, a newly created subnet can have subtoken disabled. The failure chain
is:

```text
Subtoken disabled
  -> add_stake returns SubtokenDisabled
  -> validator never receives validator_permit
  -> set_weights returns (False, None)
```

This is silent enough to waste hours. Netuid 5 has already passed this gate:
`SubtensorModule.SubtokenEnabled[5] == True`. For any future localnet subnet, verify the
same storage value before registering or staking.

After weights work, check alpha-pool liquidity and taoflow before diagnosing zero
emissions as an application bug. Netuid 3 required manual liquidity before emissions
flowed.

## 10. Repository and implementation status

```text
instant-subnet/
├── README.md
├── .env.{miner,validator,platform}.example
├── deploy/pm2/      one forked PM2 process per application role
├── docs/
│   ├── DESIGN.md
│   └── LAUNCH_TODO.md
├── config/
│   ├── models.toml
│   ├── scoring.toml
│   └── slo.toml
├── instant/
│   ├── common/       configuration and boot-time guards
│   ├── protocol/     Epistula, SS58, canonical JSON, receipts, attestation, schemas
│   ├── miner/        working FastAPI miner and Bittensor entrypoint
│   ├── validator/    chain observer/API plus deterministic scoring and SQLite state
│   └── platform/     localnet-only single-miner signed gateway
├── tests/            417 tests
└── pyproject.toml
```

Implemented and tested:

- canonical JSON and sr25519 signing adapters;
- Epistula v2 signing, verification, replay defense, and secret signatures;
- receipt construction, verification, streaming transport, and Merkle roots;
- attestation schemas, policy guards, and miner attestation integration;
- miner HTTP surface and vLLM proxy;
- miner registration check, metagraph refresh, and axon announcement;
- deterministic scoring, EMA, gates, penalties, u16 normalization;
- validator SQLite migrations and epoch persistence;
- validator read-only metagraph loop with liveness/readiness/miner discovery APIs;
- single-miner platform relay with exact-byte Epistula signing and streaming receipt
  preservation; and
- role env examples, PM2 process definitions, README, and test workflow.

Not implemented:

- validator probes, attestation verification orchestration, telemetry ingestion,
  scoring epoch orchestration, and bounded weight submission;
- platform API keys, database, dynamic routing, quotas, receipt verification/storage,
  telemetry, and dashboard API;
- pinned GPU/vLLM deployment, attestation lockfiles, host bootstrap automation, and the
  full four-way smoke test.

## 11. Dependencies

The supported environment intentionally pins:

```toml
fastapi ~= 0.110.1
bittensor == 9.12.2
async-substrate-interface >= 1.5.6, < 2.0
```

Bittensor 9.x requires the FastAPI 0.110 line. Async-substrate-interface 2.x introduces
a `cyscale`/`scalecodec` namespace conflict with Bittensor 9.12.2, so the `<2.0` ceiling
is required. `btcli` is a separate package; the installed CLI is 9.22.1 and may expose
commands that do not match the older spec-393 runtime. Always inspect `--help` before
constructing a chain mutation command.

## 12. Deployment topology

Current infrastructure:

- OCI Ampere A1 at `129.80.21.251`: static platform site behind nginx and Cloudflare.
- DigitalOcean at `68.183.141.180`: shared Bittensor localnet.
- Miner and validator deployment hosts: not yet provisioned.
- Confidential GPU capacity: Azure NCCadsH100v5 quota remains the production hardware
  critical path.

`api.instantsubnet.com` should be DNS-only when the streaming gateway is deployed so a
free Cloudflare proxy timeout/body limit does not sit in the inference path. The live
nginx file must be read again before changes; the handoff copy is only a transcription.

## 13. End-to-end gates

1. **Package:** the current venv installs; all 417 tests and full lint pass. Complete.
2. **Chain:** netuid 5 exists and is owned/active. Complete.
3. **Subtoken/hyperparameters:** enabled, tempo 10, commit/reveal off. Complete.
4. **Hosts:** provision compatible miner hardware and a validator VM.
5. **Registration:** register the dedicated miner and validator; record hotkeys and UIDs.
6. **Plumbing:** deploy all three services and route one signed streaming request.
7. **Permit:** stake the validator and verify a fresh `validator_permit` snapshot.
8. **Validator buildout:** run direct/shadow probes and persist one scored epoch.
9. **Weights:** submit one live vector through a bounded writer and require success.
10. **Emissions:** verify liquidity/taoflow and observe emissions.
11. **Failure modes:** kill each component in turn and confirm bounded, visible failure.

## 14. Open decisions and blockers

- Miner GPU choice and its verified GPT-OSS/vLLM image. Current official vLLM guidance
  does not yet treat Ada Lovelace as a supported GPT-OSS target.
- Miner and validator deployment IPs and firewall allow-list.
- How the localnet node is run on its droplet (systemd, Docker, or bare process).
- The alpha-pool liquidity operation required for netuid 5 emissions.
- Production confidential-GPU quota and measured CC-mode performance.
- Final platform API/database implementation details.

## 15. Superseded July 29 assumptions

Do not copy these from the original draft into scripts or runbooks:

| July 29 draft | Current decision |
|---|---|
| Node/Fastify gateway | Python/FastAPI gateway |
| localnet on `:9944`, target netuid 2 | endpoint `:80`, target netuid 5 |
| wipe/re-bootstrap confusing localnet | preserve the shared spec-393 chain |
| possible fast blocks | standard blocks, tempo 10 |
| commit-reveal enabled at launch | disabled initially |
| mainnet attestation `warn` fallback | `hard` on Finney; `off` only on localnet |
| no implementation started | deployable plumbing skeleton; 417 tests pass |

## 16. References inside the repository

- `config/models.toml` — model ladder
- `config/slo.toml` — scoring targets
- `config/scoring.toml` — consensus weights and constants
- `instant/protocol/epistula.py` — exact signing and timing behavior
- `instant/protocol/receipts.py` — signed receipt format
- `instant/protocol/attestation.py` — attestation format and verification primitives
- `instant/miner/app.py` — implemented miner API
- `instant/platform/app.py` — initial signed platform relay
- `instant/validator/runtime.py` — read-only chain loop and miner discovery
- `instant/validator/score.py` — deterministic consensus arithmetic
- `instant/validator/state.py` — persisted validator state
- `README.md` — deployment and smoke-test quickstart
- `docs/LAUNCH_TODO.md` — current execution checklist
