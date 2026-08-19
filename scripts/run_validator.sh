#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

exec .venv/bin/instant-validator run-once
