"""End-to-end prescription pipeline: OCR (PP-OCRv5) then NER (all drug DBs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ner_layer import extract_entities, format_medicines_table
from ocr_engine import ocr_image

ROOT = SRC.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


def run_pipeline(image_path: Path) -> dict:
    lines = ocr_image(image_path, OUTPUT)
    ocr_text = "\n".join(lines).strip()
    (OUTPUT / "last_ocr.txt").write_text(ocr_text or "(no text found)", encoding="utf-8")

    ner = extract_entities(" ".join(lines) if lines else "", lines=lines)
    (OUTPUT / "last_ner.json").write_text(json.dumps(ner, indent=2), encoding="utf-8")

    result = {
        "image": str(image_path.resolve()),
        "ocr_lines": lines,
        "ocr_text": ocr_text,
        "ner": ner,
    }
    (OUTPUT / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("Usage: python src\\pipeline.py data\\raw\\prescription.jpg")
        sys.exit(1)
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    result = run_pipeline(image_path)
    print("\n--- OCR ---")
    print(result["ocr_text"] or "(no text found)")
    print("\n--- Drug DBs ---")
    stats = (result["ner"] or {}).get("drug_db_stats") or {}
    print(json.dumps(stats, indent=2))
    print("\n--- MEDICINES ---")
    print(format_medicines_table((result["ner"] or {}).get("drugs") or []))
    print("\n--- NER ---")
    print(json.dumps((result["ner"] or {}).get("drugs") or [], indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUTPUT / 'last_result.json'}")


if __name__ == "__main__":
    main()
