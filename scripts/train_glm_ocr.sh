#!/usr/bin/env bash
# Train GLM-OCR on GPU (LoRA by default).
# Usage: bash scripts/train_glm_ocr.sh [lora|full]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-lora}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv_finetune}"
LLAMA_DIR="$ROOT/third_party/LLaMA-Factory"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing $VENV_DIR — run: bash scripts/setup_glm_finetune.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ ! -d "$LLAMA_DIR" ]]; then
  echo "Missing LLaMA-Factory — run: bash scripts/setup_glm_finetune.sh"
  exit 1
fi

python "$ROOT/src/prepare_glm_finetune.py"

case "$MODE" in
  lora)
    CFG="$ROOT/finetune_glm/configs/lora.yaml"
    ;;
  full)
    CFG="$ROOT/finetune_glm/configs/full.yaml"
    ;;
  *)
    echo "Usage: bash scripts/train_glm_ocr.sh [lora|full]"
    exit 1
    ;;
esac

# Copy YAML into LLaMA-Factory so relative output_dir paths stay inside it
DEST_CFG="$LLAMA_DIR/glm_ocr_${MODE}_prescriptions.yaml"
cp "$CFG" "$DEST_CFG"

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "==> Training mode=$MODE cfg=$DEST_CFG gpu=$CUDA_VISIBLE_DEVICES"
cd "$LLAMA_DIR"
llamafactory-cli train "$DEST_CFG"
echo "==> Done. Check: $LLAMA_DIR/saves/glm-ocr/"
