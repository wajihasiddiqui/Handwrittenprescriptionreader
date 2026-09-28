#!/usr/bin/env bash
# Usage: bash scripts/train_glm_ocr.sh [lora|full]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:-lora}"
exec python src/llamafactory_train.py train --mode "$MODE"
