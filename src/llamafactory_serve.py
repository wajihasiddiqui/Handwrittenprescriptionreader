"""Start LLaMA-Factory OpenAI-compatible API for GLM-OCR inference.

Usage (GPU or CPU machine):

  python src/llamafactory_serve.py

Then run the pipeline — OCR will call http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "third_party" / "LLaMA-Factory"
VENV_DIR = ROOT / ".venv_finetune"
INFER_CFG = ROOT / "finetune_glm" / "configs" / "infer.yaml"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_cli() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "llamafactory-cli.exe"
    return VENV_DIR / "bin" / "llamafactory-cli"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GLM-OCR via LLaMA-Factory API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES (empty = auto/CPU)")
    parser.add_argument(
        "--adapter",
        default="",
        help="Optional LoRA adapter path (after training)",
    )
    args = parser.parse_args()

    if not LLAMA_DIR.exists() or not _venv_python().exists():
        print("LLaMA-Factory not set up yet. Run first:")
        print("  python src/llamafactory_train.py setup")
        sys.exit(1)

    dest = LLAMA_DIR / "glm_ocr_infer_prescriptions.yaml"
    text = INFER_CFG.read_text(encoding="utf-8")
    if args.adapter:
        adapter = Path(args.adapter)
        if not adapter.is_absolute():
            adapter = (ROOT / adapter).resolve()
        text += f"\nadapter_name_or_path: {adapter.as_posix()}\nfinetuning_type: lora\n"
    dest.write_text(text, encoding="utf-8")

    env = os.environ.copy()
    env["DISABLE_VERSION_CHECK"] = "1"
    env["API_PORT"] = str(args.port)
    env["API_HOST"] = str(args.host)
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cli = _venv_cli()
    vpy = _venv_python()
    if cli.exists():
        cmd = [str(cli), "api", str(dest)]
    else:
        cmd = [str(vpy), "-m", "llamafactory.cli", "api", str(dest)]

    print("Starting LLaMA-Factory GLM-OCR API...")
    print(f"  config: {dest}")
    print(f"  url:    http://127.0.0.1:{args.port}/v1/chat/completions")
    print("Keep this window open, then run pipeline.py in another terminal.")
    subprocess.check_call(cmd, cwd=str(LLAMA_DIR), env=env)


if __name__ == "__main__":
    main()
