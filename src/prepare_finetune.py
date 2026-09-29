"""Prepare ShareGPT multimodal data for GLM-OCR fine-tuning (LLaMA-Factory).

Usage:
  # Auto-OCR new images → update labels.csv → build ShareGPT JSON
  python src\\prepare_finetune.py

  # Manual labels only (no GLM-OCR)
  python src\\prepare_finetune.py --no-auto-ocr

Input (default):
  data/glm_finetune/images/        put training images here
  data/glm_finetune/labels.csv     auto-filled by GLM-OCR (you can edit later)

Output:
  data/glm_finetune/labels.csv                  updated with OCR text
  data/glm_finetune/prescriptions.json          ShareGPT JSON
  data/glm_finetune/prescription_images/        copied images
  third_party/LLaMA-Factory/data/...           synced when present

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DIR = ROOT / "data" / "glm_finetune"
DEFAULT_IMAGES = DEFAULT_DIR / "images"
DEFAULT_LABELS = DEFAULT_DIR / "labels.csv"
DEFAULT_OUT_JSON = DEFAULT_DIR / "prescriptions.json"
DEFAULT_OUT_IMAGES = DEFAULT_DIR / "prescription_images"
DATASET_SNIPPET = ROOT / "finetune_glm" / "dataset_info.snippet.json"
LLAMA_DATA = ROOT / "third_party" / "LLaMA-Factory" / "data"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

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


def _list_images(images_dir: Path, include_raw: bool = False) -> list[Path]:
    found: dict[str, Path] = {}
    dirs = [images_dir]
    if include_raw:
        dirs.append(ROOT / "data" / "raw")
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                found.setdefault(path.name, path)
    return list(found.values())


def _load_label_rows(path: Path) -> list[dict[str, str]]:
    """Load labels.csv rows (image/text/task). Empty file → []."""
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        fields = {str(x).strip().lower(): str(x) for x in reader.fieldnames}
        if "image" not in fields:
            raise SystemExit("labels.csv must have an 'image' column")
        img_key = fields["image"]
        txt_key = fields.get("text")
        task_key = fields.get("task")
        for row in reader:
            image = str(row.get(img_key) or "").strip()
            if not image or image.startswith("example_"):
                continue
            text = str(row.get(txt_key) or "").strip() if txt_key else ""
            task = str(row.get(task_key) or "text").strip() if task_key else "text"
            rows.append({"image": image, "text": text, "task": task or "text"})
    return rows


def _write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text", "task"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "text": row.get("text", ""),
                    "task": row.get("task") or "text",
                }
            )


def _labeled_names(rows: list[dict[str, str]]) -> set[str]:
    return {r["image"] for r in rows if str(r.get("text") or "").strip()}


def auto_fill_labels_with_glm_ocr(
    labels_path: Path,
    images_dir: Path,
    *,
    include_raw: bool = False,
    force: bool = False,
) -> list[dict[str, str]]:
    """Scan images, run GLM-OCR on missing/empty labels, write labels.csv."""
    from ocr_engine import ocr_image

    images_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_label_rows(labels_path)
    by_name = {r["image"]: dict(r) for r in existing}
    images = _list_images(images_dir, include_raw=include_raw)
    if not images:
        raise SystemExit(
            f"No images found in {images_dir}\n"
            f"Put .png/.jpg files there, then re-run."
        )

    added = 0
    updated = 0
    for img_path in images:
        name = img_path.name
        row = by_name.get(name)
        has_text = bool(row and str(row.get("text") or "").strip())
        if has_text and not force:
            continue

        print(f"GLM-OCR → {name} ...")
        lines = ocr_image(img_path)
        text = "\n".join(lines).strip()
        if not text:
            print(f"  WARN: empty OCR for {name} (row kept blank)")
        if row is None:
            by_name[name] = {"image": name, "text": text, "task": "text"}
            added += 1
        else:
            row["text"] = text
            row["task"] = row.get("task") or "text"
            updated += 1

    # Keep previous order, then append newly discovered names
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in existing:
        name = r["image"]
        if name in by_name:
            ordered.append(by_name[name])
            seen.add(name)
    for img_path in images:
        name = img_path.name
        if name not in seen and name in by_name:
            ordered.append(by_name[name])
            seen.add(name)

    _write_labels(labels_path, ordered)
    print(
        f"labels.csv updated: +{added} new, {updated} OCR-filled "
        f"→ {labels_path}"
    )
    return ordered


def _rows_ready_for_train(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    ready: list[tuple[str, str, str]] = []
    for r in rows:
        image = str(r.get("image") or "").strip()
        text = str(r.get("text") or "").strip()
        task = str(r.get("task") or "text").strip() or "text"
        if image and text:
            ready.append((image, text, task))
    return ready


def _read_labels(path: Path) -> list[tuple[str, str, str]]:
    rows = _load_label_rows(path)
    ready = _rows_ready_for_train(rows)
    if not path.exists():
        example = DEFAULT_DIR / "labels.example.csv"
        raise SystemExit(
            f"Missing {path}\n"
            f"Put images in data\\glm_finetune\\images\\ and run:\n"
            f"  python src\\prepare_finetune.py\n"
            f"(or copy {example.name} and fill text manually with --no-auto-ocr)"
        )
    if not ready:
        raise SystemExit(
            f"{path} has no rows with both image and text.\n"
            f"Run with auto-OCR (default) or fill text manually."
        )
    return ready


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
                "- labels.csv is auto-filled by GLM-OCR (edit if needed)",
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
        description="Auto-OCR images with GLM-OCR and build ShareGPT training data"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-images", type=Path, default=DEFAULT_OUT_IMAGES)
    parser.add_argument(
        "--auto-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run GLM-OCR on new/empty images and update labels.csv (default: on)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Re-OCR all images even if labels.csv already has text",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also scan data/raw/ for images",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not copy into third_party/LLaMA-Factory/data",
    )
    args = parser.parse_args(argv)

    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.out_images.parent.mkdir(parents=True, exist_ok=True)

    if args.auto_ocr:
        auto_fill_labels_with_glm_ocr(
            args.labels,
            args.images_dir,
            include_raw=args.include_raw,
            force=args.force_ocr,
        )

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
    print("  Review/edit data\\glm_finetune\\labels.csv if OCR text needs fixes")
    print("  python src\\llamafactory_train.py setup")
    print("  python src\\llamafactory_train.py train --mode lora")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
