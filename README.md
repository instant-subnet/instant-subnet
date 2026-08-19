# Instant Validator

The Instant Validator reads one signed, finalized Platform report, validates it against
SN46 chain state, scores every Miner deterministically, records the result, and exits.
Cron runs it every five minutes; an already-processed epoch exits without processing or
submitting it again.

## Install

Requirements: Python 3.11+, Git, cron, `flock`, Node.js, and PM2.

```bash
git clone https://github.com/instant-subnet/instant-subnet.git
cd instant-subnet
cp .env.example .env
```

Set `INSTANT_PLATFORM_SIGNER` and the Validator wallet values in `.env`, then run:

```bash
python3 scripts/start_validator.py
```

Finney and netuid 46 are the defaults. Development or local networks require explicit
`.env` overrides.

## Operate

```bash
# Run once now. Duplicate epochs are safely ignored.
scripts/run_validator.sh

# Check GitHub main and apply an available fast-forward update now.
python3 scripts/update_validator.py --once

# View updater and scoring logs.
pm2 logs instant-validator-updater
tail -f logs/validator.log
```

The PM2 updater checks `main` every five minutes. It shares a filesystem lock with the
cron scorer, so an update and a scoring run cannot overlap. Each accepted report and each
Miner's raw metrics, component scores, final score, and normalized weight are written to
the scoring log.

## Troubleshooting

- `report_already_processed` is normal: this Validator has already handled that signed
  epoch report.
- A signature, network, netuid, block-bound, or UID/hotkey mismatch fails the run without
  updating durable report state.
- If an update cannot fast-forward cleanly, the checkout is left unchanged and the updater
  logs `update_failed`.
- Confirm the configured wallet exists and that `INSTANT_PLATFORM_SIGNER` matches the
  Platform report signer before investigating chain or report errors.

## Project documentation

- [Architecture overview](docs/ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
