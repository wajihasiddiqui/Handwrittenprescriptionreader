"""Cross-platform LLaMA-Factory trainer for GLM-OCR prescription fine-tune.

Run on a GPU machine:

  python src/llamafactory_train.py setup
  python src/llamafactory_train.py prepare
  python src/llamafactory_train.py train --mode lora
  python src/llamafactory_train.py export
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LLAMA_DIR = ROOT / "third_party" / "LLaMA-Factory"
VENV_DIR = ROOT / ".venv_finetune"
REQ = ROOT / "finetune_glm" / "requirements-finetune.txt"
CONFIGS = {
    "lora": ROOT / "finetune_glm" / "configs" / "lora.yaml",
    "full": ROOT / "finetune_glm" / "configs" / "full.yaml",
}


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_cli() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "llamafactory-cli.exe"
    return VENV_DIR / "bin" / "llamafactory-cli"


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(">", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.check_call(cmd, cwd=str(cwd or ROOT), env=merged)


def cmd_setup(_: argparse.Namespace) -> None:
    py = sys.executable
    if not VENV_DIR.exists():
        _run([py, "-m", "venv", str(VENV_DIR)])
    vpy = _venv_python()
    _run([str(vpy), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    _run([str(vpy), "-m", "pip", "install", "-r", str(REQ)])

    LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (LLAMA_DIR / ".git").exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/hiyouga/LLaMA-Factory.git",
                str(LLAMA_DIR),
            ]
        )
    _run([str(vpy), "-m", "pip", "install", "-e", str(LLAMA_DIR)])
    metrics = LLAMA_DIR / "requirements" / "metrics.txt"
    if metrics.exists():
        try:
            _run([str(vpy), "-m", "pip", "install", "-r", str(metrics)])
        except subprocess.CalledProcessError:
            print("WARN: metrics extras failed; continuing")
    _run([str(vpy), "-m", "pip", "install", "-U", "transformers>=5.3.0"])

    _run(
        [
            str(vpy),
            "-c",
            "import torch; print('cuda', torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')",
        ]
    )
    print("\nSetup OK.")
    print("Next: python src/llamafactory_train.py prepare")
    print("Then:  python src/llamafactory_train.py train --mode lora")


def cmd_prepare(_: argparse.Namespace) -> None:
    from prepare_finetune import main as prepare_main

    prepare_main([])


def cmd_train(args: argparse.Namespace) -> None:
    mode = args.mode
    cfg = CONFIGS.get(mode)
    if cfg is None or not cfg.exists():
        raise SystemExit(f"Unknown mode or missing config: {mode}")
    if not LLAMA_DIR.exists():
        raise SystemExit("LLaMA-Factory missing. Run: python src/llamafactory_train.py setup")
    vpy = _venv_python()
    if not vpy.exists():
        raise SystemExit(f"Missing finetune venv python: {vpy}\nRun setup first.")

    # Always refresh ShareGPT dataset into LLaMA-Factory/data
    from prepare_finetune import main as prepare_main

    prepare_main([])

    dest_cfg = LLAMA_DIR / f"glm_ocr_{mode}_prescriptions.yaml"
    shutil.copy2(cfg, dest_cfg)

    cli = _venv_cli()
    env = {
        "DISABLE_VERSION_CHECK": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
    }
    if cli.exists():
        cmd = [str(cli), "train", str(dest_cfg)]
    else:
        cmd = [str(vpy), "-m", "llamafactory.cli", "train", str(dest_cfg)]
    _run(cmd, cwd=LLAMA_DIR, env=env)
    print(f"Training finished. Check: {LLAMA_DIR / 'saves' / 'glm-ocr'}")


def cmd_export(args: argparse.Namespace) -> None:
    adapter = Path(args.adapter) if args.adapter else LLAMA_DIR / "saves" / "glm-ocr" / "lora" / "sft"
    export_dir = Path(args.output) if args.output else ROOT / "models" / "glm-ocr-finetuned"
    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}\nTrain first with --mode lora")
    export_dir.mkdir(parents=True, exist_ok=True)
    vpy = _venv_python()
    cli = _venv_cli()
    env = {"DISABLE_VERSION_CHECK": "1"}
    base = [str(cli)] if cli.exists() else [str(vpy), "-m", "llamafactory.cli"]
    cmd = base + [
        "export",
        "--model_name_or_path",
        "zai-org/GLM-OCR",
        "--adapter_name_or_path",
        str(adapter),
        "--template",
        "glm_ocr",
        "--export_dir",
        str(export_dir),
        "--trust_remote_code",
        "true",
    ]
    _run(cmd, env=env)
    print(f"Merged model → {export_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-OCR fine-tune via LLaMA-Factory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Create venv + clone/install LLaMA-Factory")
    p_setup.set_defaults(func=cmd_setup)

    p_prep = sub.add_parser("prepare", help="Build ShareGPT dataset from labels.csv")
    p_prep.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="Train with LLaMA-Factory")
    p_train.add_argument("--mode", choices=["lora", "full"], default="lora")
    p_train.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    p_train.set_defaults(func=cmd_train)

    p_exp = sub.add_parser("export", help="Merge LoRA adapter to HF folder")
    p_exp.add_argument("--adapter", default="", help="Path to LoRA adapter dir")
    p_exp.add_argument("--output", default="", help="Export directory")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
