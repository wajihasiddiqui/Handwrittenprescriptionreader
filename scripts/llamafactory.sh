#!/usr/bin/env bash
# Thin wrappers → python src/llamafactory_train.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python src/llamafactory_train.py "$@"
