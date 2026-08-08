# Instant Subnet

Instant is a Bittensor subnet for low-latency, OpenAI-compatible inference. This
repository currently contains the first deployable **localnet plumbing**:

- a miner that joins netuid 5, announces its axon, authenticates Epistula requests,
  and proxies inference to a local OpenAI-compatible worker;
- a validator whose continuous PM2 service is read-only, plus explicit one-shot
  commands for authenticated telemetry ingestion, deterministic scoring, and a
  fail-closed localnet weight submission; and
- a single-miner platform gateway with Bearer API-key authentication, exact-byte
  Epistula forwarding, receipt verification, durable SQLite telemetry, and an
  Epistula-authenticated validator stats endpoint; and
- a runnable loopback-only mock worker for exercising the complete socket path on a
  CPU droplet without claiming to serve a real model.

The platform gateway and mock worker are deliberately local/test-only. The gateway does
not yet have quotas, billing, or dynamic miner routing. The mock identifies itself as
`instant/mock-echo`; it cannot start on Finney and must never be presented as GPT-OSS.

The full test suite and Ruff lint are the repository's deployment preflight.

## Localnet state

| Item | Value |
|---|---|
| WebSocket endpoint | `ws://68.183.141.180:80` |
| Runtime spec | 393 |
| Instant netuid | 5 |
| Subtoken | enabled |
| Tempo | 10 blocks |
| Commit/reveal | disabled |

Netuid 5 already exists. The code in this repository does not create a subnet or make
chain writes during its initial validator loop.

## Topology

```text
OCI platform host
  nginx/TLS -> instant-platform :8090
                    |
                    | Epistula-signed HTTP
                    v
DigitalOcean miner (current CPU plumbing host: 165.227.197.158)
  instant-miner :8091 -> instant-mock-vllm :8000 (loopback)
          |
          | axon announcement + metagraph reads
          v
Localnet :80 <---------- DigitalOcean validator
                         instant-validator :8092 (localhost operations API)
```

Only the miner port must cross hosts during this phase. The model worker and validator
operations API should not be publicly exposed.

## Install

Use Python 3.11. Create a fresh virtual environment on every new host; do not copy a
virtual environment from another machine.

```sh
git clone git@github.com:instant-subnet/instant-subnet.git /opt/instant-subnet
cd /opt/instant-subnet
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

`btcli` is a separate package from the pinned Bittensor SDK and is only needed for
wallet and chain administration. The runtime processes use `bittensor==9.12.2` from
the project environment.

Keep wallets outside the repository. Run each PM2 process as the same Unix account
that owns its required Bittensor wallet files. The miner runtime needs the registered
hotkey plus its owner's public `coldkeypub.txt`; it never needs the private coldkey.

## Configure each role

Each host gets only its own real env file:

```sh
cp .env.miner.example .env.miner
cp .env.validator.example .env.validator
cp .env.platform.example .env.platform
```

Replace every `REPLACE_...` value. Never commit the resulting `.env.*` files. In
particular:

- `INSTANT_PLATFORM_SS58` on the miner is the public SS58 address of the platform
  signing hotkey.
- `INSTANT_PLATFORM_MINER_SS58` on the platform is the registered miner hotkey.
- `INSTANT_PLATFORM_MINER_UID` is that hotkey's current netuid-5 UID; validators reject
  telemetry whose `(uid, hotkey)` pair does not match their metagraph.
- `INSTANT_PLATFORM_VALIDATOR_SS58` is the validator allowed to call the signed stats
  endpoint.
- `INSTANT_PLATFORM_API_KEY_SHA256` is a SHA-256 digest, never the raw client token.
- The miner and validator hotkeys must be registered on netuid 5 before their chain
  checks can succeed.
- The platform signer may be a dedicated local wallet; its public address must match
  the miner allow-list value exactly.

After filling a role file, restrict it with `chmod 600 .env.<role>`. Env files contain
public configuration and wallet *names*, never mnemonics, seeds, passwords, or private
keys.

Generate one high-entropy localnet client token, store its digest in `.env.platform`,
and keep the raw value only in the client/operator secret store:

```sh
TOKEN="isk_$(openssl rand -hex 32)"
printf %s "$TOKEN" | sha256sum
```

### Create and register a fresh miner wallet securely

Create and register the wallet on a secure operator/admin machine, not on the miner.
Do not use a development URI such as `Alice` or put a mnemonic in an env file. In the
commands below, `<SECURE_WALLET_PATH>` is a protected directory on that admin machine:

```sh
umask 077
btcli wallet create \
  --wallet-path <SECURE_WALLET_PATH> \
  --wallet-name miner \
  --hotkey mock1 \
  --n-words 24 \
  --use-password
```

Keep the encrypted private coldkey, coldkey mnemonic, and seed on that secure machine
or in offline custody. Fund the new coldkey on localnet for the registration recycle or
fee, then inspect the transaction before confirming it:

```sh
btcli wallet balance \
  --network ws://68.183.141.180:80 \
  --wallet-path <SECURE_WALLET_PATH> \
  --wallet-name miner

btcli subnets register \
  --network ws://68.183.141.180:80 \
  --netuid 5 \
  --wallet-path <SECURE_WALLET_PATH> \
  --wallet-name miner \
  --hotkey mock1 \
  --safe-register
```

After registration, transfer exactly these two files to the miner over an authenticated
channel:

```text
/home/instant/.bittensor/wallets/miner/coldkeypub.txt
/home/instant/.bittensor/wallets/miner/hotkeys/mock1
```

`coldkeypub.txt` is public identity material and must come from the coldkey that owns
`mock1` in the netuid-5 metagraph. The hotkey is the fresh operational signing key. Do
**not** transfer the private coldkey, its mnemonic, or its seed. The miner must not have
this file:

```text
/home/instant/.bittensor/wallets/miner/coldkey
```

Lock down and verify the deployed runtime wallet before starting PM2:

```sh
sudo chown -R instant:instant /home/instant/.bittensor/wallets/miner
sudo chmod 700 /home/instant/.bittensor/wallets/miner
sudo chmod 700 /home/instant/.bittensor/wallets/miner/hotkeys
sudo chmod 600 /home/instant/.bittensor/wallets/miner/coldkeypub.txt
sudo chmod 600 /home/instant/.bittensor/wallets/miner/hotkeys/mock1
test ! -e /home/instant/.bittensor/wallets/miner/coldkey
```

The PM2 user must be able to read and use the hotkey without an interactive prompt; a
password prompt at service start becomes a restart loop. `python -m instant.miner
--check-chain` verifies that `coldkeypub.txt` exists, is readable, is a valid public
identity, and matches the registered hotkey owner before constructing or announcing an
axon. The platform signing hotkey does not need subnet registration, but its SS58
address must exactly match the miner's allow-list.

All hosts need synchronized clocks. Epistula accepts requests only within an 8-second
past window with a 2-second future allowance, so enable and verify system time sync.

## Preflight checks

Load the role's environment before running its commands:

```sh
set -a
. ./.env.validator
set +a
```

Substitute `miner` or `platform` for the other hosts.

```sh
# Configuration only; does not load a wallet or contact the chain.
python -m instant.miner --check
python -m instant.mock_vllm --check
python -m instant.validator --check
python -m instant.platform --check

# Live plumbing checks.
python -m instant.miner --check-chain
python -m instant.validator --once
python -m instant.platform --check-miner
```

`instant-validator --once` prints a JSON snapshot. Before registration or stake, a
successful chain connection can still report `registered: false` or
`validator_permit: false`; that is an account state, not a transport failure.

## Score and set weights manually

The PM2 validator command has no scoring or write flag. It synchronizes health and the
metagraph only; boot and restart never submit a transaction.

After the platform allow-lists this validator and `INSTANT_PLATFORM_SS58` names the
platform signing hotkey, fetch one signed stats window and run the deterministic scorer:

```sh
python -m instant.validator --score-once
curl -fsS http://127.0.0.1:8092/telemetry
curl -fsS http://127.0.0.1:8092/scoring
curl -fsS http://127.0.0.1:8092/scores
```

The validator verifies the signature over the exact response bytes, requires it to be
addressed to this validator, checks freshness and aggregate invariants, and rejects a
platform `(uid, hotkey)` that differs from the current serving metagraph roster. A
logical epoch is committed only once, so repeating the command in the same platform
window cannot advance EMA or cooldown twice.

The existing scoring gate still requires at least 20 successful direct/shadow probes;
platform telemetry does not count toward it. `--score-once` runs a finite batch of 20
direct, Epistula-signed probes per serving miner, at most four concurrently, and counts
only responses with a miner receipt binding the exact request/response bytes and both
hotkeys. If fewer than 20 succeed, the scorer honestly persists an all-zero vector and
the writer refuses it. Do not lower `config/scoring.toml` to get around this gate.

Only after inspecting a nonzero `/scores` result, run one localnet attempt by enabling
the flag for that command alone:

```sh
INSTANT_ENABLE_WEIGHT_WRITES=true \
  python -m instant.validator --set-weights-once
curl -fsS http://127.0.0.1:8092/weights
```

The writer requires local network mode, runtime spec 393, mechanism 0, commit/reveal
off, matching version key, a registered/staked validator with a current permit, an
unchanged UID-to-hotkey mapping, the chain's min/max and rate-limit rules, and a
nonzero vector summing to 65535. It composes `set_mechanism_weights` directly, signs
with the hotkey, waits for finalization, submits at most once, reads the vector and
`LastUpdate` back, and records the outcome. It never calls the pinned SDK's unbounded
high-level `set_weights` helper.

## Start the miner worker

The checked-in miner example uses the mock tier first. It binds only to loopback, emits
real delayed SSE frames, and advertises only `instant/mock-echo`:

```sh
python -m instant.mock_vllm
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Use this to prove wallet, chain, axon, Epistula, streaming, receipt, and platform
telemetry plumbing on `165.227.197.158`. It is not a model benchmark and produces no TEE
attestation.

### Replace the mock with GPT-OSS later

Install vLLM in a separate GPU environment so its CUDA dependencies cannot disturb
the subnet SDK environment. Do not provision the previously considered RTX 4000 Ada
only because it has enough nominal VRAM: the current
[official GPT-OSS vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/OpenAI/GPT-OSS.html)
says Ada Lovelace support is still being worked on. Validate the exact image/build
first, or choose hardware the recipe explicitly supports.

For the first `dev1` rig, the intended process shape is:

```sh
source /opt/instant-subnet/.venv-vllm/bin/activate
vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192
```

When switching, set `INSTANT_MODEL_TIER=dev1` in the miner, platform, and validator role
files and use the real worker rather than the mock PM2 process. Pin the vLLM/CUDA build
after the GPU image is selected. The value passed to
`--max-model-len` must match `INSTANT_MAX_MODEL_LEN`. Start and verify vLLM before
starting `instant-miner`. This initial README shows vLLM in the foreground
intentionally: its exact install and restart definition must be pinned to the selected
GPU image. Add it to PM2/systemd before calling the miner host reboot-safe.

## Run with PM2

There is one PM2 file per host under `deploy/pm2/`. Each definition uses fork mode and
exactly one process. Do not run the validator in PM2 cluster mode: two validator loops
would eventually submit duplicate weight transactions.

Install a supported Node.js LTS release and PM2 on each host, then confirm both
`node --version` and `pm2 --version` before starting a role. Prefer a dedicated
`instant` Unix user that owns `/opt/instant-subnet`, its `logs/` directory, and that
role's wallet files.

On the miner host, load `.env.miner` once and start the loopback mock before the miner:

```sh
cd /opt/instant-subnet
set -a
. ./.env.miner
set +a
mkdir -p logs
pm2 start deploy/pm2/mock-vllm.ecosystem.config.cjs --update-env
pm2 start deploy/pm2/miner.ecosystem.config.cjs --update-env
pm2 status
pm2 logs instant-mock-vllm --lines 100
pm2 logs instant-miner --lines 100
pm2 save
```

On another role host:

```sh
cd /opt/instant-subnet
set -a
. ./.env.validator
set +a
mkdir -p logs
pm2 start deploy/pm2/validator.ecosystem.config.cjs --update-env
pm2 status
pm2 logs instant-validator --lines 100
pm2 save
pm2 startup
```

On the validator host, create the parent directory named by `INSTANT_STATE_DB` and
make it writable by the PM2 user before the first start. With the example path:

```sh
sudo install -d -o instant -g instant /var/lib/instant-subnet
```

Replace `instant` if the service runs under a different Unix account.

On the platform host, also create the SQLite telemetry directory before starting PM2:

```sh
sudo install -d -o instant -g instant /opt/instant-subnet/data
```

Use `miner.ecosystem.config.cjs` and `platform.ecosystem.config.cjs` on their respective
hosts. The
last command prints the privileged command needed to enable reboot startup; run that
printed command once, then run `pm2 save` again.

## Smoke test

```sh
curl -fsS http://127.0.0.1:8092/health       # validator host
curl -fsS http://127.0.0.1:8092/livez        # process is alive
curl -fsS http://127.0.0.1:8092/readyz       # 200 only after chain authority is fresh
curl -fsS http://127.0.0.1:8091/health       # miner host
curl -fsS http://127.0.0.1:8090/health       # platform host
curl -fsS http://127.0.0.1:8090/readyz       # 200 only when the miner is ready
curl -fsS http://127.0.0.1:8090/v1/models \
  -H "Authorization: Bearer $TOKEN"          # platform -> miner
```

Once vLLM reports ready, exercise the complete signed request path from the platform
host:

```sh
curl -sS http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"instant/mock-echo","messages":[{"role":"user","content":"Reply with pong."}],"stream":true}'
```

The two `/v1` routes reject missing or invalid Bearer tokens before forwarding. Keep the
gateway bound to localhost behind nginx/TLS anyway: a single static API key is a plumbing
credential, not a quota or abuse-control system. Configure nginx with streaming buffering
disabled (`proxy_buffering off`) and do not expose port 8090 directly to the internet.

The platform persists each attempted miner request in `INSTANT_PLATFORM_STATE_DB` and
counts an HTTP 2xx as successful only when the miner receipt verifies against the exact
request and observed response bytes. `GET /validator/v1/stats` uses Epistula rather than
the customer Bearer token, is replay-protected, and signs its exact JSON response back to
the validator. The initial time-window report uses block bounds `0,0`; it must not be
interpreted as a chain-finalized epoch boundary.

## Network policy

- Chain host `:80`: reachable from the miner, validator, and platform hosts.
- Miner `:8091`: allow only the platform and validator source IPs.
- Miner vLLM `:8000`: localhost only.
- Validator `:8092`: localhost or a tightly controlled monitoring network only.
- Platform `:8090`: localhost; any nginx test route is TLS-protected and source-IP
  restricted (or requires temporary basic auth).
- SSH `:22`: restrict to operator source IPs and prefer key authentication.

Epistula authenticates platform requests but does not encrypt prompts. Until the
OCI-to-miner hop uses TLS or a private tunnel such as WireGuard, use only synthetic
localnet prompts and keep miner `:8091` source-IP restricted.

## What comes next

After the three health paths and one platform-routed inference request work:

1. register/stake the dedicated validator and confirm its permit;
2. add direct and shadow miner probes;
3. feed observations into the existing deterministic scorer and SQLite state store;
4. add bounded, explicit weight submission—never the Bittensor 9.12.2 helper path that
   can loop indefinitely after a failed extrinsic;
5. add quotas, raw receipt sampling, and dynamic routing; and
6. exercise failure modes before enabling any production traffic.

See [the reconciled design](docs/DESIGN.md) and
[the launch checklist](docs/LAUNCH_TODO.md) for the full plan.
