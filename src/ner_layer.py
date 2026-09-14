"""Phase 4: NER layer — RxNorm drug match plus medical codes from drug_db.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

from drug_db import DrugDatabase, load_drug_database

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

DURATION_RE = re.compile(
    r"\b(?:x\s*)?(\d+\s*(?:days?|weeks?|months?))\b",
    re.IGNORECASE,
)
HYPHEN_SCHEDULE_RE = re.compile(r"\b([01]-[01]-[01])\b")
PLUS_SCHEDULE_RE = re.compile(r"\b(\d+\s*\+\s*\d+\s*\+\s*\d+)\b")

_DB: DrugDatabase | None = None
_NLP = None
_BACKEND = None


def get_db() -> DrugDatabase:
    global _DB
    if _DB is None:
        _DB = load_drug_database()
    return _DB


def dose_pattern(db: DrugDatabase) -> re.Pattern[str]:
    units = "|".join(re.escape(u) for u in db.dose_units if u)
    return re.compile(rf"\b(\d+(?:\.\d+)?\s*(?:{units}))\b", re.IGNORECASE)


def correct_drug_name(ocr_text: str, db: DrugDatabase | None = None) -> tuple[str, int]:
    db = db or get_db()
    result = process.extractOne(
        ocr_text,
        db.search_terms,
        scorer=fuzz.ratio,
        score_cutoff=db.fuzzy_threshold,
    )
    if result:
        return db.canonical_name(result[0]), int(result[1])
    if db.live_fallback:
        live = _rxnorm_live_match(ocr_text, db)
        if live:
            return live
    return ocr_text, 0


def _rxnorm_live_match(term: str, db: DrugDatabase) -> tuple[str, int] | None:
    import requests

    url = db.api_base.rstrip("/") + "/approximateTerm.json"
    response = requests.get(
        url,
        params={"term": term, "maxEntries": 5},
        timeout=15,
        headers={"User-Agent": "AI-For-Medical-Report/1.0"},
    )
    response.raise_for_status()
    group = response.json().get("approximateGroup") or {}
    candidates = group.get("candidate") or []
    if isinstance(candidates, dict):
        candidates = [candidates]
    if not candidates:
        return None
    top = candidates[0]
    rxcui = str(top.get("rxcui") or "")
    score = int(float(top.get("score") or 0))
    name = str(top.get("name") or "").strip()
    if not name and rxcui:
        props = requests.get(
            db.api_base.rstrip("/") + f"/rxcui/{rxcui}/properties.json",
            timeout=15,
            headers={"User-Agent": "AI-For-Medical-Report/1.0"},
        )
        props.raise_for_status()
        name = str((props.json().get("properties") or {}).get("name") or "").strip()
    if not name:
        return None
    return name, min(score, 100)


def build_nlp(db: DrugDatabase):
    import spacy

    # Drug names come from RxNorm (thousands). Only code patterns go in spaCy.
    patterns = (
        [{"label": "FREQUENCY", "pattern": freq} for freq in sorted(db.frequencies)]
        + [{"label": "ROUTE", "pattern": route} for route in sorted(db.routes)]
        + [{"label": "SCHEDULE", "pattern": sched} for sched in sorted(db.schedules)]
    )

    try:
        nlp = spacy.load("en_core_sci_sm")
        backend = "scispacy:en_core_sci_sm"
        ruler = nlp.add_pipe("entity_ruler", name="med_ruler", before="ner", config={"overwrite_ents": True})
    except OSError:
        nlp = spacy.blank("en")
        backend = "spacy.blank+EntityRuler"
        ruler = nlp.add_pipe("entity_ruler", name="med_ruler")

    ruler.add_patterns(patterns)
    return nlp, backend


def get_nlp():
    global _NLP, _BACKEND
    if _NLP is None:
        _NLP, _BACKEND = build_nlp(get_db())
    return _NLP, _BACKEND


class PrescriptionValidator:
    def __init__(self, db: DrugDatabase | None = None) -> None:
        self.db = db or get_db()

    def validate(self, entities: dict) -> list[str]:
        errors: list[str] = []
        freq = entities.get("frequency")
        if freq and freq not in self.db.frequencies:
            errors.append(f"Unrecognized frequency: {freq}")
        if not entities.get("drug"):
            errors.append("Drug name not found")
        if entities.get("drug") in self.db.require_inr_for and not entities.get("inr_monitoring"):
            errors.append(f"{entities['drug']} requires INR monitoring note")
        return errors


SKIP_WORDS = {
    "TAB",
    "CAP",
    "SYP",
    "SYRUP",
    "DROP",
    "PATIENT",
    "HOSPITAL",
    "PRESCRIPTION",
    "DIAGNOSIS",
    "DOCTOR",
    "SIGNATURE",
    "DISPENSED",
    "PHARMACIST",
    "RECORD",
    "NUMBER",
    "WEIGHT",
    "AGE",
    "DATE",
    "BLOCK",
    "UNIVERSITY",
}


def find_drug(text: str, db: DrugDatabase) -> tuple[str | None, int]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    skip = db.frequencies | db.routes | SKIP_WORDS
    best: tuple[str, int] | None = None
    for width in (3, 2, 1):
        for i in range(0, len(tokens) - width + 1):
            window = tokens[i : i + width]
            if any(tok.upper() in skip for tok in window):
                continue
            phrase = " ".join(window)
            name, score = correct_drug_name(phrase, db)
            if score < db.fuzzy_threshold:
                continue
            if width == 1 and len(phrase) < 5 and score < 95:
                continue
            if best is None or score > best[1] or (score == best[1] and len(phrase) > 4):
                best = (name, score)
    if best:
        return best
    return None, 0


def extract_entities(text: str) -> dict:
    db = get_db()
    nlp, backend = get_nlp()
    doc = nlp(text)

    frequency = None
    route = None
    schedule = None
    dosage = None
    duration = None
    spacy_ents: list[dict] = []

    for ent in doc.ents:
        spacy_ents.append({"text": ent.text, "label": ent.label_})
        if ent.label_ == "FREQUENCY" and not frequency:
            frequency = ent.text.upper()
        elif ent.label_ == "ROUTE" and not route:
            route = ent.text.upper()
        elif ent.label_ == "SCHEDULE" and not schedule:
            schedule = ent.text

    drug, drug_score = find_drug(text, db)

    dose_match = dose_pattern(db).search(text)
    if dose_match:
        dosage = re.sub(r"\s+", "", dose_match.group(1))

    dur_match = DURATION_RE.search(text)
    if dur_match:
        duration = dur_match.group(1).lower()

    hyphen = HYPHEN_SCHEDULE_RE.search(text)
    plus = PLUS_SCHEDULE_RE.search(text)
    if hyphen:
        schedule = hyphen.group(1)
    elif plus:
        schedule = re.sub(r"\s+", "", plus.group(1))

    for freq in db.frequencies:
        if re.search(rf"\b{re.escape(freq)}\b", text, re.IGNORECASE):
            frequency = freq
            break

    entities = {
        "drug": drug,
        "drug_rxcui": db.rxcui_for(drug),
        "drug_confidence": drug_score,
        "dosage": dosage,
        "schedule": schedule,
        "frequency": frequency,
        "route": route,
        "duration": duration,
        "inr_monitoring": None,
        "raw_text": text,
        "ner_backend": backend,
        "drug_db": str(db.path),
        "drug_db_source": db.source,
        "spacy_entities": spacy_ents,
    }
    entities["errors"] = PrescriptionValidator(db).validate(entities)
    entities["needs_review"] = True
    return entities


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    if len(sys.argv) >= 2 and sys.argv[1] != "--ocr":
        text = " ".join(sys.argv[1:])
    else:
        ocr_file = OUTPUT / "last_ocr.txt"
        if not ocr_file.exists():
            print("No OCR text. Run ocr_baseline.py first, or pass text:")
            print('  python src\\ner_layer.py "Amoxicillin 500mg 1-0-1 TDS x 5 days"')
            sys.exit(1)
        text = " ".join(ocr_file.read_text(encoding="utf-8").split())

    entities = extract_entities(text)
    out = OUTPUT / "last_ner.json"
    out.write_text(json.dumps(entities, indent=2), encoding="utf-8")
    print(json.dumps(entities, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
