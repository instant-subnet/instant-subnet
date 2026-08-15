# Instant Subnet

This public repository contains one thing: the validator service for Instant on
Finney subnet 46.

The production default is a chain-only prelaunch full-burn-vote loop:

```text
read one finalized chain snapshot
  -> require Burn mode and commit/reveal disabled
  -> resolve the current subnet-owner hotkey and UID
  -> submit {owner UID: 65535}
  -> wait for finalization
  -> wait and repeat
```

This mode does not call the platform, score miners, or use report state. After an
explicit launch decision, `INSTANT_BURN_MINER_EMISSIONS=false` switches to the small
signed-report scoring loop documented below. Validators never probe or connect to
miners, and this MVP does not implement commit/reveal.

## Repository boundary

`instant-subnet` owns only:

- the validator process;
- the signed report-v1 contract;
- one deterministic scoring formula;
- one direct Bittensor weight writer;
- PM2 configuration and focused tests.

Miner installation and inference live in `instant-miner-kit`. Customer APIs, API
keys, miner routing, metrics, and routing-state decisions live in `instant-platform`.
The separate H200 service in `instant-verifier` returns pass/fail assertions only.
The platform alone decides the resulting routing action, including temporary hold,
operator review, and re-enablement.

This repository contains no platform server, Miner Kit runtime, verifier runtime,
UI, local-chain infrastructure, mock model, migration, backwards-compatibility
layer, or private deployment topology.

## Requirements

- Python 3.11
- a validator wallet registered and permitted on Finney subnet 46
- Node.js and PM2 for the long-running process
- the platform's published report-signing SS58 address after burn mode is disabled

Wallet secrets stay in the normal Bittensor wallet directory and never enter this
repository or `.env`.

## Install

```sh
git clone https://github.com/instant-subnet/instant-subnet.git
cd instant-subnet
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

Fill in the validator wallet. `INSTANT_PLATFORM_SIGNER` is needed only after burn
mode is disabled. Protect the private configuration file:

```sh
chmod 600 .env
```

`.env` is ignored by Git. The checked-in example defaults to `finney` and netuid
`46`. Leave `INSTANT_CHAIN_ENDPOINT` empty to use Bittensor's trusted Finney
selection. An operator may explicitly set network, netuid, and a WebSocket endpoint
in their private `.env` for another environment; alternate chains are never the
public default.

## Start with PM2

Install once and start the single process:

```sh
pm2 start ecosystem.config.cjs
pm2 logs instant-validator
```

Persist it across reboots only after the logs show the expected configuration:

```sh
pm2 save
pm2 startup
```

The service has no interactive CLI or operations API. PM2 starts
`python -m instant_validator`, the process loads `.env`, and the loop begins
immediately.

### Safe first run

`INSTANT_ENABLE_WEIGHT_WRITES=false` is the checked-in safety default. In this mode
the process performs the finalized chain checks and logs the exact prospective
owner UID/weight vector without submitting an extrinsic. It refuses to proceed unless
the subnet is in protocol `Burn` mode and commit/reveal is disabled.

Before deployment, the subnet owner or Root must disable commit/reveal. Keep the
existing writers running until the dry run observes that change at a finalized block.

After those checks pass, the operator reviews the dry-run vector and enables the
first write:

```text
INSTANT_ENABLE_WEIGHT_WRITES=true
```

Then restart the process:

```sh
pm2 restart instant-validator
```

## Prelaunch full burn vote

Burn is fixed at 100% in code; there is no percentage setting. The current owner UID
and runtime weight version are resolved from one finalized chain block, so UID 238 or
any other mutable UID is never hard-coded. With protocol Burn mode enabled, the
owner-directed miner emission is recorded as burned instead of paid to that miner.

## Report v1

This path is active only when `INSTANT_BURN_MINER_EMISSIONS=false`.

The platform endpoint returns one immutable completed-period JSON document. Its exact
golden example is in `tests/fixtures/report-v1.json`.

Top-level fields:

```text
schema_version, report_id, network, netuid,
period_start_block, period_end_block, created_at_ms,
signer, miners, digest, signature
```

Each miner row contains:

```text
uid, hotkey, requests, successes, failures,
prompt_tokens, completion_tokens,
ttft_p50_ms, ttft_p95_ms, tokens_per_second_p50,
toploc_verified, toploc_failed, toploc_timed_out
```

All numeric values are integers. Request counts must balance, and every request must
have one terminal TOPLOC outcome. The validator canonicalizes every field except
`digest` and `signature` as sorted compact UTF-8 JSON. It verifies the SHA-256 digest
and the platform signer's SR25519 signature over those exact bytes.

The platform owns miner registration checks before a miner enters this report. The
validator does not repeat the platform's Finney UID/hotkey lookup in the MVP.

## Scoring v1

Scoring uses integers only and is intentionally easy to audit:

| Metric | Share |
|---|---:|
| request success rate | 40% |
| verified TOPLOC rate | 40% |
| p95 time-to-first-token against a 750 ms target | 10% |
| p50 throughput against a 50 token/s target | 10% |

Each component is capped at 10,000 basis points. A miner with no successful work or
no verified TOPLOC result scores zero. Positive scores are scaled proportionally so
the highest emitted unsigned-16-bit weight is 65,535. UIDs are always emitted in
ascending order.

Changing the formula changes consensus behavior and requires a reviewed code release;
there are no hidden tuning modes or legacy policies.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `INSTANT_NETWORK` | `finney` | Bittensor network and expected report network |
| `INSTANT_NETUID` | `46` | subnet and expected report netuid |
| `INSTANT_CHAIN_ENDPOINT` | empty | optional explicit private WebSocket override |
| `INSTANT_WALLET_NAME` | `validator` | validator wallet name |
| `INSTANT_WALLET_HOTKEY` | `default` | validator hotkey name |
| `INSTANT_WALLET_PATH` | `~/.bittensor/wallets` | wallet root |
| `INSTANT_BURN_MINER_EMISSIONS` | `true` | chain-only prelaunch burn gate |
| `INSTANT_PLATFORM_REPORT_URL` | public Instant endpoint | scoring-mode report-v1 URL |
| `INSTANT_PLATFORM_SIGNER` | empty | required trusted signer in scoring mode |
| `INSTANT_POLL_INTERVAL_SECONDS` | `60` | delay between cycles; example uses `3600` |
| `INSTANT_REPORT_MAX_AGE_SECONDS` | `86400` | stale-report limit |
| `INSTANT_STATE_PATH` | `var/validator-state.json` | scoring-mode last report |
| `INSTANT_ENABLE_WEIGHT_WRITES` | `false` | explicit Finney write gate |

See `.env.example` for the complete set.

## Development

```sh
python -m pip install -e '.[dev]'
ruff check src tests scripts
pytest -q
python scripts/check_repository.py
```

The repository guard fails if the entire tracked repository reaches 12,500 lines,
if an unexpected service tree is added, or if public files reintroduce local-chain
defaults, literal WebSocket IPs, or retired entrypoints.
