"""Fine-tune PP-OCRv5 with PaddleOCR tools/train.py (CPU)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PADDLEOCR = ROOT / "PaddleOCR"
CONFIG = ROOT / "configs" / "ppocrv5_finetune_train.yml"
PRETRAIN = ROOT / "models" / "PP-OCRv5_mobile_rec_pretrained.pdparams"
SAVE_DIR = ROOT / "output" / "medical_rec"
INFER_DIR = ROOT / "models" / "medical_rec_infer"
FINETUNE = ROOT / "data" / "finetune"
TRAIN_LIST = FINETUNE / "train.txt"
VAL_LIST = FINETUNE / "val.txt"
PRETRAIN_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
)


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def download_pretrain() -> None:
    PRETRAIN.parent.mkdir(parents=True, exist_ok=True)
    if PRETRAIN.exists() and PRETRAIN.stat().st_size > 1_000_000:
        return
    import urllib.request

    print(f"Downloading {PRETRAIN_URL}")
    urllib.request.urlretrieve(PRETRAIN_URL, PRETRAIN)


def run(cmd: list[str]) -> int:
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(PADDLEOCR))


def export_best() -> None:
    best = SAVE_DIR / "best_accuracy"
    if not (SAVE_DIR / "best_accuracy.pdparams").exists():
        latest = SAVE_DIR / "latest.pdparams"
        if not latest.exists():
            print("No trained weights to export.")
            return
        best = SAVE_DIR / "latest"
    cmd = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        posix(CONFIG),
        "-o",
        f"Global.pretrained_model={best.as_posix()}",
        f"Global.save_inference_dir={posix(INFER_DIR)}",
        "Global.use_gpu=False",
    ]
    code = run(cmd)
    if code != 0:
        raise SystemExit(code)
    print(f"Exported inference model -> {INFER_DIR}")


def main() -> None:
    if not PADDLEOCR.exists():
        raise SystemExit("Clone PaddleOCR into the project first.")
    if not TRAIN_LIST.exists():
        raise SystemExit(
            f"Missing {TRAIN_LIST}\n"
            "Run this first:\n"
            "  python src\\prepare_finetune.py"
        )
    download_pretrain()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = posix(FINETUNE)
    cmd = [
        sys.executable,
        "tools/train.py",
        "-c",
        posix(CONFIG),
        "-o",
        f"Global.pretrained_model={posix(PRETRAIN)}",
        "Global.use_gpu=False",
        f"Global.save_model_dir={posix(SAVE_DIR)}",
        f"Train.dataset.data_dir={data_dir}",
        f"Train.dataset.label_file_list=[{posix(TRAIN_LIST)}]",
        f"Eval.dataset.data_dir={data_dir}",
        f"Eval.dataset.label_file_list=[{posix(VAL_LIST)}]",
    ]
    code = run(cmd)
    if code != 0:
        raise SystemExit(code)
    export_best()


if __name__ == "__main__":
    main()
