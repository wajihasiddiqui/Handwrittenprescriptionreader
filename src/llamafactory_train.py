"""Fine-tune GLM-OCR with LLaMA-Factory from data/glm_finetune/labels.csv.

Flow:
  labels.csv + images → ShareGPT JSON → LLaMA-Factory LoRA/full SFT → export

Commands:
  python src/llamafactory_train.py setup
  python src/llamafactory_train.py prepare
  python src/llamafactory_train.py train --mode lora
  python src/llamafactory_train.py export
  python src/llamafactory_train.py ollama --name glm-ocr-rx
"""

from __future__ import annotations

import argparse
import json
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
LABELS = ROOT / "data" / "glm_finetune" / "labels.csv"
LOCAL_JSON = ROOT / "data" / "glm_finetune" / "prescriptions.json"


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


def _count_sharegpt_samples() -> int:
    if not LOCAL_JSON.exists():
        return 0
    try:
        data = json.loads(LOCAL_JSON.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _cuda_available(vpy: Path) -> bool:
    try:
        probe = subprocess.check_output(
            [str(vpy), "-c", "import torch; print(int(torch.cuda.is_available()))"],
            text=True,
        ).strip()
        return probe == "1"
    except Exception:
        return False


def _write_train_yaml(
    src_cfg: Path,
    dest_cfg: Path,
    n_samples: int,
    *,
    use_cpu: bool = False,
) -> None:
    """Copy train yaml; adapt for tiny datasets and CPU-only machines."""
    lines: list[str] = []
    seen_use_cpu = False
    for line in src_cfg.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if n_samples < 10:
            if stripped.startswith("val_size:"):
                lines.append("val_size: 0.0")
                continue
            if stripped.startswith("eval_strategy:"):
                lines.append('eval_strategy: "no"')
                continue
            if stripped.startswith("eval_steps:"):
                continue
        if use_cpu:
            if stripped.startswith("bf16:"):
                lines.append("bf16: false")
                continue
            if stripped.startswith("fp16:"):
                lines.append("fp16: false")
                continue
            if stripped.startswith("use_cpu:"):
                lines.append("use_cpu: true")
                seen_use_cpu = True
                continue
            if stripped.startswith("preprocessing_num_workers:"):
                lines.append("preprocessing_num_workers: 0")
                continue
            if stripped.startswith("dataloader_num_workers:"):
                lines.append("dataloader_num_workers: 0")
                continue
            if stripped.startswith("gradient_accumulation_steps:"):
                # Smaller effective batch on CPU to reduce RAM pressure
                lines.append("gradient_accumulation_steps: 2")
                continue
            if stripped.startswith("num_train_epochs:"):
                lines.append("num_train_epochs: 1")
                continue
        lines.append(line)

    if use_cpu and not seen_use_cpu:
        # Insert near train section
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if not inserted and line.strip().startswith("bf16:"):
                out.append("use_cpu: true")
                inserted = True
        if not inserted:
            out.append("use_cpu: true")
        lines = out

    dest_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if n_samples < 10:
        print(f"Small dataset ({n_samples} samples): val_size=0, eval disabled")
    if use_cpu:
        print("CPU mode: bf16=false, use_cpu=true, epochs=1 (very slow)")


def cmd_setup(args: argparse.Namespace) -> None:
    py = sys.executable
    if not VENV_DIR.exists():
        _run([py, "-m", "venv", str(VENV_DIR)])
    vpy = _venv_python()
    _run([str(vpy), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    # Install CUDA PyTorch first (default pip torch is often CPU-only on Windows)
    cuda_index = str(getattr(args, "torch_index", None) or "https://download.pytorch.org/whl/cu124")
    print(f"Installing CUDA PyTorch from {cuda_index} ...")
    _run(
        [
            str(vpy),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "--index-url",
            cuda_index,
        ]
    )
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
            "import torch; print('torch', torch.__version__); "
            "print('cuda', torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')",
        ]
    )
    if not _cuda_available(vpy):
        print("\nWARNING: torch.cuda.is_available() is False.")
        print("  1) Confirm NVIDIA driver: nvidia-smi")
        print("  2) Reinstall CUDA torch:")
        print("     python src\\llamafactory_train.py fix-torch")
    print("\nSetup OK.")
    print("Next:")
    print("  1. Put images in data\\glm_finetune\\images\\")
    print("  2. Fill data\\glm_finetune\\labels.csv  (image,text,task)")
    print("  3. python src\\llamafactory_train.py prepare")
    print("  4. python src\\llamafactory_train.py train --mode lora")


def cmd_fix_torch(args: argparse.Namespace) -> None:
    """Reinstall CUDA-enabled PyTorch into .venv_finetune."""
    vpy = _venv_python()
    if not vpy.exists():
        raise SystemExit("Missing .venv_finetune. Run: python src\\llamafactory_train.py setup")
    cuda_index = str(args.torch_index or "https://download.pytorch.org/whl/cu124")
    print(f"Reinstalling CUDA PyTorch from {cuda_index} ...")
    _run(
        [
            str(vpy),
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    )
    _run(
        [
            str(vpy),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "--index-url",
            cuda_index,
        ]
    )
    _run(
        [
            str(vpy),
            "-c",
            "import torch; print('torch', torch.__version__); "
            "print('cuda', torch.cuda.is_available()); "
            "print(torch.version.cuda); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')",
        ]
    )
    if not _cuda_available(vpy):
        raise SystemExit(
            "Still no CUDA. Check nvidia-smi, then try cu121:\n"
            "  python src\\llamafactory_train.py fix-torch --torch-index "
            "https://download.pytorch.org/whl/cu121"
        )
    print("CUDA PyTorch OK.")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Build ShareGPT from labels.csv and sync into LLaMA-Factory/data."""
    from prepare_finetune import main as prepare_main

    prep_args: list[str] = []
    if not args.auto_ocr:
        prep_args.append("--no-auto-ocr")
    if args.force_ocr:
        prep_args.append("--force-ocr")
    if args.include_raw:
        prep_args.append("--include-raw")
    prepare_main(prep_args)
    n = _count_sharegpt_samples()
    print(f"Prepare done. ShareGPT samples ready: {n}")


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
    if not LABELS.exists():
        raise SystemExit(
            f"Missing {LABELS}\n"
            f"Create it with columns: image,text,task\n"
            f"Example: copy data\\glm_finetune\\labels.example.csv data\\glm_finetune\\labels.csv"
        )

    # Always rebuild ShareGPT from labels.csv (default: no auto-OCR)
    if not args.skip_prepare:
        prep = argparse.Namespace(
            auto_ocr=args.auto_ocr,
            force_ocr=False,
            include_raw=args.include_raw,
        )
        cmd_prepare(prep)

    n = _count_sharegpt_samples()
    if n < 1:
        raise SystemExit(
            "No ShareGPT samples. Check labels.csv has image+text rows "
            "and image files exist under data/glm_finetune/images/ or data/raw/"
        )

    llama_json = LLAMA_DIR / "data" / "prescriptions.json"
    if not llama_json.exists():
        raise SystemExit(
            f"Dataset not synced to {llama_json}\n"
            f"Run: python src\\llamafactory_train.py prepare"
        )

    dest_cfg = LLAMA_DIR / f"glm_ocr_{mode}_prescriptions.yaml"
    has_cuda = _cuda_available(vpy)
    force_cpu = bool(getattr(args, "cpu", False))
    if not has_cuda and not force_cpu:
        raise SystemExit(
            "CUDA not available in .venv_finetune (GPU machine needs CUDA PyTorch).\n"
            "Fix:\n"
            "  1) nvidia-smi\n"
            "  2) python src\\llamafactory_train.py fix-torch\n"
            "  3) .\\.venv_finetune\\Scripts\\python.exe -c "
            "\"import torch; print(torch.cuda.is_available())\"\n"
            "Or force slow CPU train:  python src\\llamafactory_train.py train --cpu ..."
        )
    if force_cpu:
        print("WARNING: --cpu set. Training on CPU will be extremely slow.")
    _write_train_yaml(cfg, dest_cfg, n, use_cpu=force_cpu or not has_cuda)

    cli = _venv_cli()
    env = {
        "DISABLE_VERSION_CHECK": "1",
    }
    if has_cuda and not force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if cli.exists():
        cmd = [str(cli), "train", str(dest_cfg)]
    else:
        cmd = [str(vpy), "-m", "llamafactory.cli", "train", str(dest_cfg)]
    print(f"Training GLM-OCR ({mode}) on {n} labeled samples from labels.csv ...")
    _run(cmd, cwd=LLAMA_DIR, env=env)
    out = LLAMA_DIR / "saves" / "glm-ocr" / mode / "sft"
    print(f"Training finished. Adapter/weights: {out}")
    print("Next: python src\\llamafactory_train.py export")


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

    # Point in-process OCR at the adapter for immediate use
    ocr_cfg_path = ROOT / "configs" / "ocr.json"
    if ocr_cfg_path.exists():
        ocr = json.loads(ocr_cfg_path.read_text(encoding="utf-8"))
        lf = ocr.setdefault("llamafactory", {})
        rel = os.path.relpath(adapter, ROOT).replace("\\", "/")
        lf["adapter"] = rel
        lf["finetuning_type"] = "lora"
        ocr_cfg_path.write_text(json.dumps(ocr, indent=2) + "\n", encoding="utf-8")
        print(f"Updated configs/ocr.json llamafactory.adapter → {rel}")


def cmd_ollama(args: argparse.Namespace) -> None:
    """Register merged HF export as an Ollama model (inference only)."""
    export_dir = Path(args.model_dir) if args.model_dir else ROOT / "models" / "glm-ocr-finetuned"
    name = args.name.strip() or "glm-ocr-rx"
    if not export_dir.exists() or not any(export_dir.iterdir()):
        raise SystemExit(
            f"Merged model not found: {export_dir}\n"
            f"Run first:\n"
            f"  python src\\llamafactory_train.py train --mode lora\n"
            f"  python src\\llamafactory_train.py export"
        )

    abs_model = export_dir.resolve().as_posix()
    modelfile = ROOT / "finetune_glm" / "Modelfile.generated"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM {abs_model}",
                "TEMPLATE {{ .Prompt }}",
                "RENDERER glm-ocr",
                "PARSER glm-ocr",
                "PARAMETER temperature 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _run(["ollama", "create", name, "-f", str(modelfile)])
    print(f"Ollama model created: {name}")
    print("Set configs/ocr.json:")
    print('  "backend": "ollama"')
    print(f'  "ollama": {{ "model": "{name}" }}')


def cmd_status(_: argparse.Namespace) -> None:
    print(f"labels.csv:     {LABELS.exists()}  ({LABELS})")
    print(f"ShareGPT JSON:  {LOCAL_JSON.exists()}  samples={_count_sharegpt_samples()}")
    print(f"LLaMA-Factory:  {LLAMA_DIR.exists()}  ({LLAMA_DIR})")
    print(f"finetune venv:  {_venv_python().exists()}  ({_venv_python()})")
    adapter = LLAMA_DIR / "saves" / "glm-ocr" / "lora" / "sft"
    print(f"LoRA adapter:   {adapter.exists()}  ({adapter})")
    export_dir = ROOT / "models" / "glm-ocr-finetuned"
    print(f"exported HF:    {export_dir.exists()}  ({export_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GLM-OCR from labels.csv via LLaMA-Factory"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Create venv + clone/install LLaMA-Factory")
    p_setup.add_argument(
        "--torch-index",
        default="https://download.pytorch.org/whl/cu124",
        help="PyTorch wheel index (default: cu124)",
    )
    p_setup.set_defaults(func=cmd_setup)

    p_fix = sub.add_parser("fix-torch", help="Reinstall CUDA PyTorch into .venv_finetune")
    p_fix.add_argument(
        "--torch-index",
        default="https://download.pytorch.org/whl/cu124",
        help="PyTorch wheel index (default: cu124)",
    )
    p_fix.set_defaults(func=cmd_fix_torch)

    p_prep = sub.add_parser("prepare", help="Read labels.csv → ShareGPT for LLaMA-Factory")
    p_prep.add_argument(
        "--auto-ocr",
        action="store_true",
        help="Fill empty label text with GLM-OCR before building ShareGPT",
    )
    p_prep.add_argument("--force-ocr", action="store_true")
    p_prep.add_argument("--include-raw", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="Train from labels.csv with LLaMA-Factory")
    p_train.add_argument("--mode", choices=["lora", "full"], default="lora")
    p_train.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    p_train.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (bf16 off, use_cpu=true). Very slow.",
    )
    p_train.add_argument(
        "--auto-ocr",
        action="store_true",
        help="Allow GLM-OCR to fill empty labels during prepare step",
    )
    p_train.add_argument(
        "--include-raw",
        action="store_true",
        help="Also resolve images from data/raw/",
    )
    p_train.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip rebuilding ShareGPT (use existing prescriptions.json)",
    )
    p_train.set_defaults(func=cmd_train)

    p_exp = sub.add_parser("export", help="Merge LoRA adapter to HF folder")
    p_exp.add_argument("--adapter", default="", help="Path to LoRA adapter dir")
    p_exp.add_argument("--output", default="", help="Export directory")
    p_exp.set_defaults(func=cmd_export)

    p_ollama = sub.add_parser(
        "ollama",
        help="Create Ollama model from exported HF merge (inference only)",
    )
    p_ollama.add_argument("--name", default="glm-ocr-rx", help="Ollama model name")
    p_ollama.add_argument(
        "--model-dir",
        default="",
        help="Merged HF folder (default: models/glm-ocr-finetuned)",
    )
    p_ollama.set_defaults(func=cmd_ollama)

    p_status = sub.add_parser("status", help="Show finetune paths / readiness")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
