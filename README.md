# Instant Subnet

Instant is a Bittensor subnet for low-latency, OpenAI-compatible inference. This
repository currently contains the first deployable **localnet plumbing**:

- a miner that joins netuid 5, announces its axon, authenticates Epistula requests,
  and proxies inference to a local vLLM worker;
- a read-only validator service that connects to the local chain, discovers serving
  miners, reports registration and validator-permit state, and exposes an operations
  API; and
- a minimal platform gateway that signs a request for one configured miner and
  preserves miner-signed receipts for both regular and streaming responses.

The platform gateway is deliberately localnet-only. It does not yet have customer API
keys, quotas, billing, telemetry persistence, or dynamic miner routing. The validator
does not yet probe miners, run scoring epochs, or submit weights. Those are the next
build layer after the three processes are deployed and connected.

The current test suite has **418 passing tests** and passes full-project Ruff lint.

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
DigitalOcean GPU miner
  instant-miner :8091 -> vLLM :8000
          |
          | axon announcement + metagraph reads
          v
Localnet :80 <---------- DigitalOcean validator
                         instant-validator :8092 (localhost operations API)
```

Only the miner port must cross hosts during this phase. vLLM and the validator
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
that owns its required Bittensor wallet files.

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
- The miner and validator hotkeys must be registered on netuid 5 before their chain
  checks can succeed.
- The platform signer may be a dedicated local wallet; its public address must match
  the miner allow-list value exactly.

After filling a role file, restrict it with `chmod 600 .env.<role>`. Env files contain
public configuration and wallet *names*, never mnemonics, seeds, passwords, or private
keys.

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

## Start the miner model worker

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

Pin the vLLM/CUDA build after the GPU image is selected. The value passed to
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

On a role host:

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
curl -fsS http://127.0.0.1:8090/v1/models    # platform -> miner
```

Once vLLM reports ready, exercise the complete signed request path from the platform
host:

```sh
curl -sS http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"Reply with pong."}],"stream":false}'
```

For this skeleton, the platform endpoint has no end-user authentication. Bind it to
localhost and keep its nginx route private with a source-IP allow-list or temporary
basic authentication. TLS by itself does **not** prevent anonymous GPU use. Configure
nginx with streaming buffering disabled (`proxy_buffering off`) and do not expose port
8090 directly to the internet.

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
5. add platform API keys, quotas, telemetry, receipt auditing, and dynamic routing; and
6. exercise failure modes before enabling any production traffic.

See [the reconciled design](docs/DESIGN.md) and
[the launch checklist](docs/LAUNCH_TODO.md) for the full plan.
