"""Phase 4: NER layer — match drugs from RxNorm + Orange Book + local brands."""

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
PLUS_SCHEDULE_RE = re.compile(r"\b(\d+\s*\+\s*\d+(?:\s*\+\s*\d+)?)\b")

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


def correct_drug_name(ocr_text: str, db: DrugDatabase | None = None) -> tuple[str, int, str | None]:
    name, score, source, _generic = match_drug_term(ocr_text, db)
    return name, score, source


def match_drug_term(
    ocr_text: str, db: DrugDatabase | None = None
) -> tuple[str, int, str | None, str | None]:
    db = db or get_db()
    phrases = [ocr_text]
    compact = ocr_text.replace(" ", "")
    if compact != ocr_text:
        phrases.append(compact)

    best: tuple[str, int, str | None, str | None] | None = None
    for phrase in phrases:
        result = process.extractOne(
            phrase,
            db.search_terms,
            scorer=fuzz.ratio,
            processor=str.lower,
            score_cutoff=db.fuzzy_threshold,
        )
        if not result:
            continue
        matched, score, _ = result
        score = int(score)
        if best is None or score > best[1]:
            best = (matched, score, db.source_for(matched), db.canonical_name(matched))
    if best:
        return best
    if db.live_fallback:
        live = _rxnorm_live_match(ocr_text, db)
        if live:
            return live[0], live[1], "rxnorm", live[0]
    return ocr_text, 0, None, None


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

    # Drug names come from RxNorm + Orange Book + local brands. Only codes go in spaCy.
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
        drugs = entities.get("drugs") or []
        names = [str(d.get("drug") or d.get("name") or "") for d in drugs]
        names = [n for n in names if n]
        for item in drugs:
            freq = item.get("frequency")
            if freq and freq not in self.db.frequencies:
                errors.append(f"Unrecognized frequency: {freq}")
        if not names:
            errors.append("Drug name not found")
        require_inr = {x.lower() for x in self.db.require_inr_for}
        if not entities.get("inr_monitoring"):
            for name in names:
                if name.lower() in require_inr:
                    errors.append(f"{name} requires INR monitoring note")
                    break
        return errors


SKIP_WORDS = {
    "TAB",
    "CAP",
    "SYP",
    "SYRUP",
    "DROP",
    "DROPS",
    "INJ",
    "INJECTION",
    "TABLET",
    "CAPSULE",
    "ONCE",
    "DAILY",
    "DAY",
    "DAYS",
    "PATIENT",
    "HOSPITAL",
    "PRESCRIPTION",
    "PRESORINTION",
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
    "SLIP",
    "LOCATION",
    "PRINTED",
    "ALLERGIES",
    "BUILDING",
    "FLOOR",
    "VISIT",
    "KARACHI",
    "AGA",
    "KHAN",
}


def _accept_match(phrase: str, width: int, score: int, db: DrugDatabase) -> bool:
    if score < db.fuzzy_threshold:
        return False
    if width == 1 and len(phrase) < 5:
        return score >= 95
    if len(phrase) < 6:
        return score >= 90
    return True


SIG_STOP_RE = re.compile(
    r"\b(?:NOREEN|NASIR|DOCTOR|SIGNATURE|PHARMACIST|PRINTED|KARACHI|DISPENSED)\b",
    re.IGNORECASE,
)


def _letter_tokens(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"[A-Za-z][A-Za-z\-]{2,}", text))


def find_drugs(text: str, db: DrugDatabase) -> list[dict]:
    matches = _letter_tokens(text)
    tokens = [m.group(0) for m in matches]
    skip = {w.upper() for w in (db.frequencies | db.routes | SKIP_WORDS)}
    candidates: list[dict] = []
    for width in (3, 2, 1):
        for i in range(0, len(tokens) - width + 1):
            window = tokens[i : i + width]
            if any(tok.upper() in skip for tok in window):
                continue
            phrase = " ".join(window)
            name, score, source, generic = match_drug_term(phrase, db)
            if not _accept_match(phrase, width, score, db):
                continue
            if source in ("rxnorm", "orange_book") and score < 90:
                continue
            candidates.append(
                {
                    "start": i,
                    "end": i + width,
                    "name": name,
                    "generic": generic or name,
                    "confidence": score,
                    "source": source,
                    "matched": phrase,
                }
            )

    candidates.sort(key=lambda c: (c["start"], -(c["end"] - c["start"]), -c["confidence"]))
    kept: list[dict] = []
    for cand in candidates:
        if kept and cand["start"] < kept[-1]["end"]:
            prev = kept[-1]
            better_score = cand["confidence"] > prev["confidence"]
            longer = cand["confidence"] == prev["confidence"] and (cand["end"] - cand["start"]) > (
                prev["end"] - prev["start"]
            )
            if better_score or longer:
                kept[-1] = cand
            continue
        kept.append(cand)

    chosen: list[dict] = []
    seen: set[str] = set()
    for cand in kept:
        key = cand["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(cand)
    return chosen


def find_drug(text: str, db: DrugDatabase) -> tuple[str | None, int, str | None]:
    drugs = find_drugs(text, db)
    if not drugs:
        return None, 0, None
    top = drugs[0]
    return top["name"], top["confidence"], top["source"]


ROUTE_HINTS = [
    (re.compile(r"\b(slc|s/?c|s\.c\.?|subcut|subcutaneous)\b", re.I), "SC"),
    (re.compile(r"\b(i\.?m\.?|intramuscular)\b", re.I), "IM"),
    (re.compile(r"\b(i\.?v\.?|intravenous)\b", re.I), "IV"),
    (re.compile(r"\b(p\.?o\.?|oral|tab(?:let)?s?|cap(?:sule)?s?|syp|syrup)\b", re.I), "PO"),
    (re.compile(r"\b(sl|sublingual)\b", re.I), "SL"),
    (re.compile(r"\b(top(?:ical)?)\b", re.I), "TOP"),
]

SCHEDULE_FROM_OCR = [
    (re.compile(r"[1il]\s*[+\-]\s*[0o]\s*[+\-]\s*[1il]", re.I), "1+0+1"),
    (re.compile(r"[1il]\s*[+\-]\s*[1il]\s*[+\-]\s*[1il]", re.I), "1+1+1"),
    (re.compile(r"[1il]\s*[+\-]\s*[0o]\s*[+\-]\s*[0o]", re.I), "1+0+0"),
    (re.compile(r"[0o]\s*[+\-]\s*[0o]\s*[+\-]\s*[1il]", re.I), "0+0+1"),
    (re.compile(r"[0o]\s*[+\-]\s*[1il]\s*[+\-]\s*[0o]", re.I), "0+1+0"),
    (re.compile(r"[1il]\+[0o]71", re.I), "1+0+1"),
    (re.compile(r"[v1il]\+[0o]t[1il]", re.I), "1+0+1"),
    (re.compile(r"1\s*/\s*2\s*\+\s*0\s*\+\s*1\s*/\s*2"), "1/2+0+1/2"),
]

FREQ_FROM_SCHEDULE = {
    "1+0+1": "BD",
    "1-0-1": "BD",
    "1+1+1": "TDS",
    "1-1-1": "TDS",
    "1+0+0": "OD",
    "1-0-0": "OD",
    "0+0+1": "HS",
    "0-0-1": "HS",
    "0+1+0": "OD",
    "0-1-0": "OD",
    "1/2+0+1/2": "BD",
}

INJECTABLE_DRUGS = {"lantus", "insulin glargine", "insugen-g", "basaglar"}


def _parse_schedule(snippet: str) -> str | None:
    hyphen = HYPHEN_SCHEDULE_RE.search(snippet)
    if hyphen:
        return hyphen.group(1)
    for pattern, value in SCHEDULE_FROM_OCR:
        if pattern.search(snippet):
            return value
    plus = PLUS_SCHEDULE_RE.search(snippet)
    if plus:
        return re.sub(r"\s+", "", plus.group(1))
    times = re.findall(r"\b(\d{1,2}\s*(?:am|pm))\b", snippet, re.I)
    if len(times) >= 2:
        return "+".join(t.replace(" ", "") for t in times)
    return None


def _parse_frequency(snippet: str, schedule: str | None) -> str | None:
    if re.search(r"once\s*(?:a\s*)?day|once\s*daily", snippet, re.I):
        return "OD"
    if re.search(r"\b(tds|tid)\b", snippet, re.I):
        return "TDS"
    if re.search(r"\b(bd|bid)\b", snippet, re.I):
        return "BD"
    if re.search(r"\b(qid|qds)\b", snippet, re.I):
        return "QDS"
    if re.search(r"\b(hs|nocte)\b", snippet, re.I):
        return "HS"
    if re.search(r"\b(stat)\b", snippet, re.I):
        return "STAT"
    if re.search(r"\b(prn|sos)\b", snippet, re.I):
        return "PRN"
    if re.search(r"(?<![A-Za-z])OD(?![A-Za-z])", snippet):
        return "OD"
    if schedule:
        return FREQ_FROM_SCHEDULE.get(schedule.replace(" ", ""))
    return None


def _parse_route(snippet: str, drug_name: str | None = None) -> str | None:
    for pattern, route in ROUTE_HINTS:
        if pattern.search(snippet):
            return route
    if drug_name and drug_name.lower() in INJECTABLE_DRUGS:
        return "SC"
    return "PO"


def _parse_dosage(snippet: str, db: DrugDatabase, schedule: str | None, route: str | None) -> str | None:
    dose_match = dose_pattern(db).search(snippet)
    if dose_match:
        return re.sub(r"\s+", "", dose_match.group(1))
    units = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(u|iu|units?)\b",
        snippet,
        re.I,
    )
    if units:
        return f"{units.group(1)} units"
    if route == "SC":
        insulin = re.search(r"\b(\d+(?:\.\d+)?)\s*[uUmM]\b", snippet)
        if insulin:
            return f"{insulin.group(1)} units"
    cleaned = snippet
    if schedule:
        cleaned = PLUS_SCHEDULE_RE.sub(" ", cleaned)
        cleaned = HYPHEN_SCHEDULE_RE.sub(" ", cleaned)
        cleaned = re.sub(r"[1il]\+[0o]71", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"[v1il]\+[0o]t[1il]", " ", cleaned, flags=re.I)
    unit_num = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)\b",
        cleaned,
        re.I,
    )
    if unit_num:
        return f"{unit_num.group(1)}{unit_num.group(2).lower()}"
    bare = re.search(r"\b(\d{2,4}(?:\.\d+)?|\d+\.\d+)\b", cleaned)
    if bare:
        value = bare.group(1)
        if route == "SC":
            return f"{value} units"
        return f"{value}mg"
    dotted = re.search(r"\b(\d+)\.(?!\d)", cleaned)
    if dotted:
        return f"{dotted.group(1)}mg"
    return None


def _sig_from_text(snippet: str, db: DrugDatabase, drug_name: str | None = None) -> dict:
    schedule = _parse_schedule(snippet)
    duration = None
    dur_match = DURATION_RE.search(snippet)
    if dur_match:
        duration = dur_match.group(1).lower()
    frequency = _parse_frequency(snippet, schedule)
    route = _parse_route(snippet, drug_name)
    dosage = _parse_dosage(snippet, db, schedule, route)
    if not frequency and schedule:
        frequency = FREQ_FROM_SCHEDULE.get(schedule.replace(" ", ""))
    if frequency == "OD" and not schedule:
        schedule = "1-0-0"
    return {
        "dosage": dosage,
        "schedule": schedule,
        "frequency": frequency,
        "route": route,
        "duration": duration,
    }


def _clip_sig_snippet(snippet: str) -> str:
    stop = SIG_STOP_RE.search(snippet)
    if stop:
        snippet = snippet[: stop.start()]
    return snippet[:120]


def _row(drug: dict, sig: dict, db: DrugDatabase) -> dict:
    name = drug["name"]
    generic = drug.get("generic") or db.canonical_name(name)
    return {
        "drug": name,
        "generic": generic,
        "dosage": sig.get("dosage"),
        "schedule": sig.get("schedule"),
        "frequency": sig.get("frequency"),
        "route": sig.get("route"),
        "duration": sig.get("duration"),
        "rxcui": db.rxcui_for(name) or db.rxcui_for(generic),
        "confidence": drug["confidence"],
        "source": drug["source"],
        "matched": drug["matched"],
    }


def _attach_sig(drugs: list[dict], text: str, db: DrugDatabase) -> list[dict]:
    token_matches = _letter_tokens(text)
    rows: list[dict] = []
    for i, drug in enumerate(drugs):
        char_from = token_matches[drug["end"] - 1].end()
        if i + 1 < len(drugs):
            char_to = token_matches[drugs[i + 1]["start"]].start()
        else:
            char_to = len(text)
        snippet = _clip_sig_snippet(text[char_from:char_to])
        rows.append(_row(drug, _sig_from_text(snippet, db, drug["name"]), db))
    return rows


def _is_stop_line(line: str) -> bool:
    return bool(SIG_STOP_RE.search(line))


def _drugs_from_lines(lines: list[str], db: DrugDatabase) -> list[dict]:
    rows: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or _is_stop_line(line):
            i += 1
            continue
        found = find_drugs(line, db)
        if not found:
            i += 1
            continue
        drug = found[0]
        prev = lines[i - 1].strip() if i else ""
        extra: list[str] = [prev, line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or _is_stop_line(nxt) or find_drugs(nxt, db):
                break
            extra.append(nxt)
            j += 1
        snippet = _clip_sig_snippet(" ".join(extra))
        rows.append(_row(drug, _sig_from_text(snippet, db, drug["name"]), db))
        i = j
    return rows


def format_medicines_table(drugs: list[dict]) -> str:
    headers = ["DRUG", "GENERIC", "DOSAGE", "SCHEDULE", "FREQUENCY", "ROUTE", "DURATION"]
    table = [headers]
    for item in drugs:
        table.append(
            [
                str(item.get("drug") or "-"),
                str(item.get("generic") or "-"),
                str(item.get("dosage") or "-"),
                str(item.get("schedule") or "-"),
                str(item.get("frequency") or "-"),
                str(item.get("route") or "-"),
                str(item.get("duration") or "-"),
            ]
        )
    widths = [max(len(row[col]) for row in table) for col in range(len(headers))]
    lines = []
    for idx, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * widths[col] for col in range(len(headers))))
    return "\n".join(lines)


def extract_entities(text: str, lines: list[str] | None = None) -> dict:
    db = get_db()
    nlp, backend = get_nlp()
    doc = nlp(text)
    spacy_ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    if lines:
        drugs = _drugs_from_lines(lines, db)
    else:
        found = find_drugs(text, db)
        drugs = _attach_sig(found, text, db)

    entities = {
        "drugs": drugs,
        "raw_text": text,
        "ner_backend": backend,
        "drug_db": str(db.path),
        "drug_db_source": db.source,
        "drug_db_stats": db.source_counts,
        "spacy_entities": spacy_ents,
        "inr_monitoring": None,
    }
    entities["errors"] = PrescriptionValidator(db).validate(entities)
    entities["needs_review"] = True
    return entities


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    lines: list[str] | None = None
    if len(sys.argv) >= 2 and sys.argv[1] != "--ocr":
        text = " ".join(sys.argv[1:])
        lines = [text]
    else:
        ocr_file = OUTPUT / "last_ocr.txt"
        if not ocr_file.exists():
            print("No OCR text. Run ocr_baseline.py first, or pass text:")
            print('  python src\\ner_layer.py "Amoxicillin 500mg 1-0-1 TDS x 5 days"')
            sys.exit(1)
        raw = ocr_file.read_text(encoding="utf-8")
        text = " ".join(raw.split())
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    entities = extract_entities(text, lines=lines)
    out = OUTPUT / "last_ner.json"
    out.write_text(json.dumps(entities, indent=2), encoding="utf-8")
    print(format_medicines_table(entities.get("drugs") or []))
    print()
    print(json.dumps(entities.get("drugs") or [], indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
