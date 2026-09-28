"""Build a ShareGPT multimodal dataset for GLM-OCR fine-tuning (LLaMA-Factory).

Input:
  data/glm_finetune/labels.csv   columns: image,text
  data/glm_finetune/images/      image files

Output:
  data/glm_finetune/prescriptions.json
  data/glm_finetune/prescription_images/  (copied/linked images for LLaMA-Factory)

Also syncs into third_party/LLaMA-Factory/data/ when that folder exists.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "glm_finetune"
IMAGES_DIR = SRC_DIR / "images"
LABELS_CSV = SRC_DIR / "labels.csv"
OUT_JSON = SRC_DIR / "prescriptions.json"
OUT_IMG_DIR = SRC_DIR / "prescription_images"
PROMPT = "Text Recognition:"

LLAMA_DATA = ROOT / "third_party" / "LLaMA-Factory" / "data"
DATASET_SNIPPET = ROOT / "finetune_glm" / "dataset_info.snippet.json"


def _read_labels(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        example = SRC_DIR / "labels.example.csv"
        raise SystemExit(
            f"Missing {path}\n"
            f"Copy {example.name} to labels.csv and fill image,text rows."
        )
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit("labels.csv must have columns: image,text")
        for i, row in enumerate(reader, start=2):
            image = str(row.get("image") or "").strip()
            text = str(row.get("text") or "").strip()
            if not image and not text:
                continue
            if not image or not text:
                raise SystemExit(f"Row {i}: both image and text are required")
            rows.append((image, text))
    if not rows:
        raise SystemExit("labels.csv has no data rows")
    return rows


def _resolve_image(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    under = IMAGES_DIR / name
    if under.is_file():
        return under
    raise FileNotFoundError(f"Image not found: {name} (looked in {IMAGES_DIR})")


def build_samples(rows: list[tuple[str, str]]) -> list[dict]:
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    missing: list[str] = []
    for idx, (image_name, text) in enumerate(rows, start=1):
        try:
            src = _resolve_image(image_name)
        except FileNotFoundError:
            missing.append(image_name)
            continue
        dest_name = f"{idx:05d}_{src.name}"
        dest = OUT_IMG_DIR / dest_name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        # Paths inside the JSON must be relative to LLaMA-Factory/data/
        rel = f"prescription_images/{dest_name}"
        samples.append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>{PROMPT}"},
                    {"role": "assistant", "content": text},
                ],
                "images": [rel],
            }
        )
    if missing:
        preview = "\n  - ".join(missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise SystemExit(
            "Missing image files under data/glm_finetune/images/:\n"
            f"  - {preview}{more}\n"
            "Add the files or fix labels.csv names."
        )
    return samples


def _merge_dataset_info(target: Path) -> None:
    snippet = json.loads(DATASET_SNIPPET.read_text(encoding="utf-8"))
    if target.exists():
        info = json.loads(target.read_text(encoding="utf-8"))
    else:
        info = {}
    info.update(snippet)
    target.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sync_to_llama_factory(samples: list[dict]) -> None:
    if not LLAMA_DATA.exists():
        print(f"LLaMA-Factory data dir not found yet: {LLAMA_DATA}")
        print("Run scripts/setup_glm_finetune.sh on the GPU machine first.")
        return
    img_dest = LLAMA_DATA / "prescription_images"
    if img_dest.exists():
        shutil.rmtree(img_dest)
    shutil.copytree(OUT_IMG_DIR, img_dest)
    (LLAMA_DATA / "prescriptions.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _merge_dataset_info(LLAMA_DATA / "dataset_info.json")
    print(f"Synced {len(samples)} samples into {LLAMA_DATA}")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_labels(LABELS_CSV)
    samples = build_samples(rows)
    OUT_JSON.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples → {OUT_JSON}")
    print(f"Images → {OUT_IMG_DIR}")
    sync_to_llama_factory(samples)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
