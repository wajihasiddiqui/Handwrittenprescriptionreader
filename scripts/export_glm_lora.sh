#!/usr/bin/env bash
# Merge LoRA adapter into a standalone HF folder under models/glm-ocr-finetuned
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv_finetune}"
LLAMA_DIR="$ROOT/third_party/LLaMA-Factory"
ADAPTER="${1:-$LLAMA_DIR/saves/glm-ocr/lora/sft}"
EXPORT_DIR="${2:-$ROOT/models/glm-ocr-finetuned}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing $VENV_DIR — run setup first"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ ! -d "$ADAPTER" ]]; then
  echo "Adapter not found: $ADAPTER"
  echo "Train first: bash scripts/train_glm_ocr.sh lora"
  exit 1
fi

mkdir -p "$EXPORT_DIR"
export DISABLE_VERSION_CHECK=1

llamafactory-cli export \
  --model_name_or_path zai-org/GLM-OCR \
  --adapter_name_or_path "$ADAPTER" \
  --template glm_ocr \
  --export_dir "$EXPORT_DIR" \
  --trust_remote_code true

echo "Merged model → $EXPORT_DIR"
