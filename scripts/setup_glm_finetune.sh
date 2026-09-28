#!/usr/bin/env bash
# One-time setup on the GPU machine for GLM-OCR fine-tuning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv_finetune}"
LLAMA_DIR="$ROOT/third_party/LLaMA-Factory"

echo "==> Project: $ROOT"
echo "==> Venv:    $VENV_DIR"

chmod +x "$ROOT/scripts/"*.sh 2>/dev/null || true

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found ($PYTHON_BIN). Install Python 3.10+."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -r "$ROOT/finetune_glm/requirements-finetune.txt"

mkdir -p "$ROOT/third_party"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "==> Cloning LLaMA-Factory..."
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_DIR"
fi

echo "==> Installing LLaMA-Factory..."
python -m pip install -e "$LLAMA_DIR"
if [[ -f "$LLAMA_DIR/requirements/metrics.txt" ]]; then
  python -m pip install -r "$LLAMA_DIR/requirements/metrics.txt" || true
fi
# GLM-OCR needs a newer transformers than LLaMA-Factory pins by default
python -m pip install -U "transformers>=5.3.0"

# Torch+CUDA: prefer existing torch if present; otherwise install CUDA wheel hint
python - <<'PY'
import importlib.util
print("torch installed:", bool(importlib.util.find_spec("torch")))
try:
    import torch
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check failed:", e)
PY

echo "==> Registering prescription dataset..."
python "$ROOT/src/prepare_glm_finetune.py" || true

echo
echo "Setup done."
echo "Next:"
echo "  1) Edit data/glm_finetune/labels.csv and add images/"
echo "  2) python src/prepare_glm_finetune.py"
echo "  3) bash scripts/train_glm_ocr.sh lora"
