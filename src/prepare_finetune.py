"""Crop labeled lines from real prescriptions and build a PaddleX rec dataset."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_engine import get_ocr

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "finetune"
IMAGES = OUT / "images"
DICT_SRC = Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_mobile_rec" / "inference.yml"

# Do not train on leftover OCR garbage from this page.
SKIP_LABELS = {
    "n",
    "1",
    "2",
    "do",
    "ft",
    "Aar.",
    "tahs.",
    "Fovw op w o E eamnèd",
    "△=Awe rnai",
    "o.d KDA Se",
}

# Correct OCR mistakes on Test1.png (ground truth for this page only).
LABEL_FIX = {
    "Patient Nae": "Patient Name",
    "Raen Soeed": "Raheen Saeed",
    "Medical Recond Number": "Medical Record Number",
    "Altrgiies": "Allergies",
    "Weighl": "Weight",
    "Dote": "Date",
    "23yo merened": "23 y/o M",
    "fonadol": "Panadol",
    "cougn": "cough",
    "syp ce225": "Syp Acefyl",
    "cBe": "CBC",
    "MpiCT": "MP/ICT",
    "sevev.": "fever",
    "raigoox1": "Vigix 0+0+1",
    "2+2+2": "2+2+2",
    "Tab": "Tab",
    "Rx": "Rx",
    "PRESCRIPTION": "PRESCRIPTION",
    "DOW UNIVERSITY HOSPITAL": "DOW UNIVERSITY HOSPITAL",
    "DOW OPD BLOCK": "DOW OPD BLOCK",
    "Diagnosis": "Diagnosis",
    "Age": "Age",
    "115596724": "115596724",
}


def load_official_dict() -> list[str]:
    import yaml

    data = yaml.safe_load(DICT_SRC.read_text(encoding="utf-8"))
    chars = data["PostProcess"]["character_dict"]
    return [str(c) for c in chars]


def fix_label(text: str) -> str:
    t = text.strip()
    if t in LABEL_FIX:
        return LABEL_FIX[t]
    key = t.rstrip(".")
    return LABEL_FIX.get(key, t)


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img
    if rng.random() < 0.7:
        out = out.rotate(rng.uniform(-6, 6), expand=True, fillcolor=(255, 255, 255))
    if rng.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.5:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    return out


def crop_boxes(image_bgr: np.ndarray, boxes) -> list[np.ndarray]:
    crops = []
    h, w = image_bgr.shape[:2]
    for box in boxes:
        arr = np.array(box).reshape(-1, 2)
        x1, y1 = np.clip(arr.min(axis=0), 0, [w - 1, h - 1]).astype(int)
        x2, y2 = np.clip(arr.max(axis=0), 0, [w, h]).astype(int)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crops.append(image_bgr[y1:y2, x1:x2])
    return crops


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    photos = list(RAW.glob("*.png")) + list(RAW.glob("*.jpg")) + list(RAW.glob("*.jpeg"))
    photos = [p for p in photos if p.name.upper() != "PUT_PHOTOS_HERE.TXT"]
    if not photos:
        raise SystemExit("No images in data/raw")

    ocr = get_ocr()
    samples: list[tuple[str, str]] = []
    rng = random.Random(7)
    idx = 0

    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None:
            continue
        result = ocr.predict(str(photo))
        for res in result:
            data = res if isinstance(res, dict) else getattr(res, "json", None) or {}
            texts = list(data.get("rec_texts") or [])
            polys = data.get("rec_polys") or data.get("dt_polys")
            if not texts or polys is None:
                continue
            crops_and_labels = []
            for box, raw in zip(polys, texts):
                h, w = image.shape[:2]
                arr = np.array(box).reshape(-1, 2)
                x1, y1 = np.clip(arr.min(axis=0), 0, [w - 1, h - 1]).astype(int)
                x2, y2 = np.clip(arr.max(axis=0), 0, [w, h]).astype(int)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crops_and_labels.append((image[y1:y2, x1:x2], raw))
            for crop, raw in crops_and_labels:
                label = fix_label(raw)
                if not label or label.startswith("(") or label in SKIP_LABELS or len(label) < 2:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                base = Image.fromarray(rgb)
                versions = [base] + [augment(base, rng) for _ in range(6)]
                for ver in versions:
                    name = f"rec_{idx:05d}.png"
                    ver.save(IMAGES / name)
                    samples.append((f"images/{name}", label))
                    idx += 1

    if len(samples) < 4:
        raise SystemExit(f"Not enough crops: {len(samples)}")

    rng.shuffle(samples)
    val_n = max(2, len(samples) // 10)
    val = samples[:val_n]
    train = samples[val_n:]

    def write_list(path: Path, rows: list[tuple[str, str]]) -> None:
        path.write_text("\n".join(f"{p}\t{t}" for p, t in rows), encoding="utf-8")

    write_list(OUT / "train.txt", train)
    write_list(OUT / "val.txt", val)

    dict_path = OUT / "dict.txt"
    if DICT_SRC.exists():
        dict_path.write_text("\n".join(load_official_dict()), encoding="utf-8")
    else:
        chars = sorted({ch for _, lab in samples for ch in lab})
        dict_path.write_text("\n".join(chars), encoding="utf-8")

    print(f"train={len(train)} val={len(val)} -> {OUT}")


if __name__ == "__main__":
    main()
