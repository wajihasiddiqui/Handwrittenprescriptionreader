"""Download RxNorm ingredient and brand names from NLM RxNav into a local cache."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from drug_db import DEFAULT_DB_PATH, ROOT, rxnorm_cache_path

TIMEOUT = 180


def fetch_concepts(api_base: str, prescribable_only: bool, tty: list[str]) -> list[dict]:
    path = "/Prescribe/allconcepts.json" if prescribable_only else "/allconcepts.json"
    url = api_base.rstrip("/") + path
    params = {"tty": " ".join(tty)}
    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    group = data.get("minConceptGroup") or {}
    concepts = group.get("minConcept") or []
    if isinstance(concepts, dict):
        concepts = [concepts]
    cleaned: list[dict] = []
    seen: set[str] = set()
    for item in concepts:
        name = str(item.get("name") or "").strip()
        rxcui = str(item.get("rxcui") or "").strip()
        term_type = str(item.get("tty") or "").strip()
        if not name or not rxcui:
            continue
        if "{" in name or len(name) > 80:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"rxcui": rxcui, "name": name, "tty": term_type})
    cleaned.sort(key=lambda row: row["name"].lower())
    return cleaned


def main() -> None:
    cfg = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
    rx = cfg.get("rxnorm") or {}
    api_base = rx.get("api_base", "https://rxnav.nlm.nih.gov/REST")
    tty = rx.get("tty") or ["IN", "PIN", "BN"]
    prescribable = bool(rx.get("prescribable_only", True))
    cache = rxnorm_cache_path(cfg)

    print(f"Downloading RxNorm ({'prescribable' if prescribable else 'full'}) TTY={tty} ...")
    try:
        concepts = fetch_concepts(api_base, prescribable, tty)
    except requests.HTTPError:
        if prescribable:
            print("Prescribable endpoint failed; trying full RxNorm ...")
            concepts = fetch_concepts(api_base, False, tty)
        else:
            raise
    payload = {
        "source": "RxNorm",
        "publisher": "U.S. National Library of Medicine (RxNav REST API)",
        "api": api_base,
        "prescribable_only": prescribable,
        "tty": tty,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(concepts),
        "concepts": concepts,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved {len(concepts)} RxNorm names to {cache}")


if __name__ == "__main__":
    main()
