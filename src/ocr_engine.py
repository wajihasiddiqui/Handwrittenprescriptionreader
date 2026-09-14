"""PP-OCRv5 text extraction (detection + recognition)."""

from __future__ import annotations

from pathlib import Path

from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parents[1]
FINETUNED_REC = ROOT / "models" / "medical_rec_infer"

_OCR = None


def _rec_dir() -> str | None:
    if (FINETUNED_REC / "inference.yml").exists() or (FINETUNED_REC / "inference.json").exists():
        return str(FINETUNED_REC)
    return None


def get_ocr() -> PaddleOCR:
    global _OCR
    if _OCR is None:
        kwargs = dict(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        rec_dir = _rec_dir()
        if rec_dir:
            kwargs["text_recognition_model_dir"] = rec_dir
        _OCR = PaddleOCR(**kwargs)
    return _OCR


def collect_texts(result) -> list[str]:
    texts: list[str] = []
    for res in result:
        rec = getattr(res, "rec_texts", None)
        if rec is None and hasattr(res, "get"):
            rec = res.get("rec_texts")
        if rec:
            texts.extend(rec)
    if texts:
        return texts
    for res in result:
        data = res if isinstance(res, dict) else getattr(res, "json", None) or {}
        if isinstance(data, dict):
            texts.extend(data.get("rec_texts") or [])
    return texts


def ocr_image(image_path: Path, output_dir: Path | None = None) -> list[str]:
    result = get_ocr().predict(str(image_path))
    if output_dir is not None:
        for res in result:
            # Do not call res.print() — Paddle dumps a huge config and can
            # crash Windows consoles with RecursionError in logging.
            if hasattr(res, "save_to_json"):
                res.save_to_json(str(output_dir))
    return collect_texts(result)
