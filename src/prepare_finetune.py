"""Prepare ShareGPT multimodal data for GLM-OCR fine-tuning (LLaMA-Factory).

Usage:
  python src\\prepare_finetune.py
  python src\\prepare_finetune.py --labels data\\glm_finetune\\labels.csv

Input (default):
  data/glm_finetune/labels.csv     columns: image,text[,task]
  data/glm_finetune/images/        image files referenced by labels.csv

Output:
  data/glm_finetune/prescriptions.json          ShareGPT JSON
  data/glm_finetune/prescription_images/        copied images
  third_party/LLaMA-Factory/data/...           synced when present

ShareGPT sample shape (required by LLaMA-Factory + template glm_ocr):
  {
    "messages": [
      {"role": "user", "content": "<image>Text Recognition:"},
      {"role": "assistant", "content": "Tab Ascard 75mg OD"}
    ],
    "images": ["prescription_images/00001_rx.png"]
  }

Then train:
  python src\\llamafactory_train.py train --mode lora
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "glm_finetune"
DEFAULT_IMAGES = DEFAULT_DIR / "images"
DEFAULT_LABELS = DEFAULT_DIR / "labels.csv"
DEFAULT_OUT_JSON = DEFAULT_DIR / "prescriptions.json"
DEFAULT_OUT_IMAGES = DEFAULT_DIR / "prescription_images"
DATASET_SNIPPET = ROOT / "finetune_glm" / "dataset_info.snippet.json"
LLAMA_DATA = ROOT / "third_party" / "LLaMA-Factory" / "data"

TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


def _task_prompt(task: str) -> str:
    key = (task or "text").strip().lower()
    if key not in TASK_PROMPTS:
        raise SystemExit(f"Unknown task {task!r}. Use: {', '.join(TASK_PROMPTS)}")
    return TASK_PROMPTS[key]


def _read_labels(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        example = DEFAULT_DIR / "labels.example.csv"
        raise SystemExit(
            f"Missing {path}\n"
            f"Copy example and edit:\n"
            f"  copy {example.relative_to(ROOT)} data\\glm_finetune\\labels.csv"
        )

    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("labels.csv is empty or has no header")
        fields = {str(x).strip().lower(): str(x) for x in reader.fieldnames}
        if "image" not in fields or "text" not in fields:
            raise SystemExit("labels.csv must have columns: image,text[,task]")
        img_key = fields["image"]
        txt_key = fields["text"]
        task_key = fields.get("task")

        for i, row in enumerate(reader, start=2):
            image = str(row.get(img_key) or "").strip()
            text = str(row.get(txt_key) or "").strip()
            task = str(row.get(task_key) or "text").strip() if task_key else "text"
            if not image and not text:
                continue
            if image.startswith("example_"):
                continue
            if not image or not text:
                raise SystemExit(f"Row {i}: both image and text are required")
            rows.append((image, text, task or "text"))

    if not rows:
        raise SystemExit("labels.csv has no data rows")
    return rows


def _resolve_image(name: str, images_dir: Path) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    if candidate.is_absolute() and candidate.exists():
        return candidate
    under = images_dir / name
    if under.is_file():
        return under
    under2 = images_dir / Path(name).name
    if under2.is_file():
        return under2
    # also allow data/raw/
    raw = ROOT / "data" / "raw" / Path(name).name
    if raw.is_file():
        return raw
    raise FileNotFoundError(name)


def build_sharegpt_samples(
    rows: list[tuple[str, str, str]],
    images_dir: Path,
    out_images_dir: Path,
) -> list[dict]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    missing: list[str] = []

    for idx, (image_name, text, task) in enumerate(rows, start=1):
        try:
            src = _resolve_image(image_name, images_dir)
        except FileNotFoundError:
            missing.append(image_name)
            continue

        dest_name = f"{idx:05d}_{src.name}"
        dest = out_images_dir / dest_name
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dest)

        # Path must be relative to LLaMA-Factory/data/ after sync
        rel = f"prescription_images/{dest_name}"
        prompt = _task_prompt(task)
        assistant_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        samples.append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": assistant_text},
                ],
                "images": [rel],
            }
        )

    if missing:
        preview = "\n  - ".join(missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise SystemExit(
            "Missing image files (put them under data/glm_finetune/images/ or data/raw/):\n"
            f"  - {preview}{more}"
        )
    if not samples:
        raise SystemExit("No ShareGPT samples built")
    return samples


def validate_sharegpt(samples: list[dict]) -> None:
    for i, sample in enumerate(samples, start=1):
        if "messages" not in sample or "images" not in sample:
            raise SystemExit(f"Sample {i}: missing messages/images")
        msgs = sample["messages"]
        if len(msgs) < 2:
            raise SystemExit(f"Sample {i}: need user + assistant messages")
        if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            raise SystemExit(f"Sample {i}: roles must be user then assistant")
        content = str(msgs[0].get("content") or "")
        if "<image>" not in content:
            raise SystemExit(f"Sample {i}: user content must include <image>")
        images = sample["images"]
        if not isinstance(images, list) or len(images) != content.count("<image>"):
            raise SystemExit(
                f"Sample {i}: <image> count must match images list length"
            )
        if not str(msgs[1].get("content") or "").strip():
            raise SystemExit(f"Sample {i}: assistant text is empty")


def _merge_dataset_info(target: Path) -> None:
    snippet = json.loads(DATASET_SNIPPET.read_text(encoding="utf-8"))
    info = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    info.update(snippet)
    target.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sync_to_llama_factory(samples: list[dict], out_images_dir: Path) -> None:
    if not LLAMA_DATA.exists():
        print(f"LLaMA-Factory data dir not found yet: {LLAMA_DATA}")
        print("Run: python src\\llamafactory_train.py setup")
        print("Dataset JSON is still ready locally for later sync.")
        return

    img_dest = LLAMA_DATA / "prescription_images"
    if img_dest.exists():
        shutil.rmtree(img_dest)
    shutil.copytree(out_images_dir, img_dest)

    (LLAMA_DATA / "prescriptions.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _merge_dataset_info(LLAMA_DATA / "dataset_info.json")
    print(f"Synced {len(samples)} samples → {LLAMA_DATA}")
    print("Registered LLaMA-Factory dataset name: prescriptions")


def write_dataset_card(out_dir: Path, n: int) -> None:
    card = out_dir / "DATASET.md"
    card.write_text(
        "\n".join(
            [
                "# GLM-OCR ShareGPT dataset",
                "",
                f"- samples: {n}",
                "- file: prescriptions.json",
                "- images: prescription_images/",
                "- LLaMA-Factory dataset key: prescriptions",
                "- template: glm_ocr",
                "",
                "Train:",
                "  python src/llamafactory_train.py train --mode lora",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build ShareGPT dataset for GLM-OCR / LLaMA-Factory training"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-images", type=Path, default=DEFAULT_OUT_IMAGES)
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not copy into third_party/LLaMA-Factory/data",
    )
    args = parser.parse_args(argv)

    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.out_images.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_labels(args.labels)
    samples = build_sharegpt_samples(rows, args.images_dir, args.out_images)
    validate_sharegpt(samples)

    args.out_json.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(args.out_json.parent, len(samples))

    print(f"ShareGPT samples: {len(samples)}")
    print(f"JSON:   {args.out_json}")
    print(f"Images: {args.out_images}")
    print("Example sample:")
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))

    if not args.no_sync:
        sync_to_llama_factory(samples, args.out_images)

    print("\nNext:")
    print("  python src\\llamafactory_train.py setup")
    print("  python src\\llamafactory_train.py train --mode lora")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
