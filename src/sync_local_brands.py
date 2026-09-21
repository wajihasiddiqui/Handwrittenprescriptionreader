"""Fetch public Pakistan brand lists and merge them into dict/local_brands.json."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz, process

from drug_db import DEFAULT_DB_PATH, ROOT, local_brands_path, rxnorm_cache_path
from drap_price import BASE_URL as DRAP_URL
from drap_price import extract_inn, fetch_all_products, first_brand_token, short_brand, short_generic
from pharmapedia import BASE_URL as PHARMAPEDIA_URL
from pharmapedia import fetch_all_medicines

PAREN_SALT_RE = re.compile(r"\s*\([^)]*\)")

GENERIC_SYNONYMS = {
    "paracetamol": "acetaminophen",
    "glibenclamide": "glyburide",
    "co-amoxiclav": "amoxicillin / clavulanate",
    "amoxicillin clavulanate": "amoxicillin / clavulanate",
    "amoxicillin + clavulanic acid": "amoxicillin / clavulanate",
    "salbutamol": "albuterol",
    "frusemide": "furosemide",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "lignocaine": "lidocaine",
    "pethidine": "meperidine",
    "hyoscine": "scopolamine",
    "vitamin d3": "cholecalciferol",
    "vitamin d": "cholecalciferol",
    "folic acid": "folic acid",
    "escitalopram": "escitalopram",
}


def _load_rxnorm_lookup(root: Path = ROOT) -> tuple[list[str], dict[str, str]]:
    raw = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
    cache_file = rxnorm_cache_path(raw, root)
    names: list[str] = []
    name_to_rxcui: dict[str, str] = {}
    if not cache_file.exists():
        return names, name_to_rxcui
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    for row in cache.get("concepts") or []:
        name = str(row.get("name") or "").strip()
        rxcui = str(row.get("rxcui") or "").strip()
        if not name:
            continue
        names.append(name)
        name_to_rxcui[name.lower()] = rxcui
    return names, name_to_rxcui


def _clean_generic(value: str) -> str:
    text = PAREN_SALT_RE.sub(" ", value)
    text = " ".join(text.replace("/", " / ").replace("+", " + ").split()).strip(" -/")
    return GENERIC_SYNONYMS.get(text.lower(), text)


def _map_generic(
    generic: str,
    rx_names: list[str],
    name_to_rxcui: dict[str, str],
) -> tuple[str, str]:
    cleaned = _clean_generic(generic)
    if not cleaned:
        return "", ""
    key = cleaned.lower()
    if key in name_to_rxcui:
        return cleaned, name_to_rxcui[key]
    synonym = GENERIC_SYNONYMS.get(key)
    if synonym and synonym.lower() in name_to_rxcui:
        return synonym, name_to_rxcui[synonym.lower()]
    if not rx_names:
        return cleaned, ""
    hit = process.extractOne(
        cleaned,
        rx_names,
        scorer=fuzz.token_sort_ratio,
        processor=str.lower,
        score_cutoff=90,
    )
    if not hit:
        return cleaned, ""
    matched = hit[0]
    return matched, name_to_rxcui.get(matched.lower(), "")


def _add_alias(groups: dict[str, dict], canonical: str, rxcui: str, alias: str) -> None:
    alias = " ".join(alias.split()).strip()
    canonical = " ".join(canonical.split()).strip()
    if not alias or not canonical:
        return
    if alias.lower() in {"tab", "cap", "syp", "inj", "tablet", "capsule"}:
        return
    key = canonical.lower()
    row = groups.setdefault(
        key,
        {"name": canonical, "rxcui": rxcui, "aliases": []},
    )
    if rxcui and not row.get("rxcui"):
        row["rxcui"] = rxcui
        row["name"] = canonical
    aliases = row["aliases"]
    seen = {a.lower() for a in aliases}
    if alias.lower() not in seen and alias.lower() != key:
        aliases.append(alias)


def collect_drap(groups: dict[str, dict], rx_names: list[str], name_to_rxcui: dict[str, str], *, refresh: bool) -> None:
    products = fetch_all_products(refresh=refresh)
    mapped: dict[str, tuple[str, str]] = {}
    for row in products:
        product = str(row.get("product_name") or "").strip()
        generic_raw = str(row.get("generic_name") or "").strip()
        brand = short_brand(product) or first_brand_token(product)
        generic = extract_inn(generic_raw) or short_generic(generic_raw) or generic_raw
        if not brand:
            continue
        aliases = [brand, product, first_brand_token(product)]
        canonical, rxcui = ("", "")
        if generic and generic.lower() != brand.lower():
            gkey = generic.lower()
            if gkey not in mapped:
                mapped[gkey] = _map_generic(generic, rx_names, name_to_rxcui)
            canonical, rxcui = mapped[gkey]
        target = canonical or brand
        for alias in aliases:
            _add_alias(groups, target, rxcui, alias)


def collect_pharmapedia(
    groups: dict[str, dict],
    rx_names: list[str],
    name_to_rxcui: dict[str, str],
    *,
    refresh: bool,
) -> None:
    payload = fetch_all_medicines(refresh=refresh)
    mapped: dict[str, tuple[str, str]] = {}
    for row in payload.get("brands") or []:
        brand = str(row.get("name") or "").strip()
        generic_raw = str(row.get("generic_name") or "").strip()
        if not brand:
            continue
        generic = _clean_generic(generic_raw) or generic_raw
        canonical, rxcui = ("", "")
        if generic and generic.lower() != brand.lower():
            gkey = generic.lower()
            if gkey not in mapped:
                mapped[gkey] = _map_generic(generic, rx_names, name_to_rxcui)
            canonical, rxcui = mapped[gkey]
        target = canonical or generic or brand
        _add_alias(groups, target, rxcui, brand)
        if generic:
            _add_alias(groups, target, rxcui, generic)
    for row in payload.get("generics") or []:
        generic = _clean_generic(str(row.get("name") or ""))
        if not generic:
            continue
        canonical, rxcui = _map_generic(generic, rx_names, name_to_rxcui)
        _add_alias(groups, canonical or generic, rxcui, generic)


def collect_fetched(
    rx_names: list[str],
    name_to_rxcui: dict[str, str],
    *,
    refresh_drap: bool = False,
    refresh_pharmapedia: bool = False,
) -> dict[str, dict]:
    groups: dict[str, dict] = {}

    print("Fetching Pharmapedia medicines ...")
    collect_pharmapedia(groups, rx_names, name_to_rxcui, refresh=refresh_pharmapedia)
    print("Fetching DRAP price index (all pages) ...")
    collect_drap(groups, rx_names, name_to_rxcui, refresh=refresh_drap)
    return groups


def merge_existing(groups: dict[str, dict], existing: list[dict]) -> dict[str, dict]:
    for entry in existing:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        rxcui = str(entry.get("rxcui") or "").strip()
        _add_alias(groups, name, rxcui, name)
        for alias in entry.get("aliases") or []:
            _add_alias(groups, name, rxcui, str(alias))
    return groups


def to_brand_list(groups: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for row in groups.values():
        aliases = sorted({a.strip() for a in row["aliases"] if a.strip()}, key=str.lower)
        rows.append(
            {
                "name": row["name"],
                "rxcui": row.get("rxcui") or "",
                "aliases": aliases,
            }
        )
    rows.sort(key=lambda item: item["name"].lower())
    return rows


def main() -> None:
    import sys

    refresh_drap = "--refresh-drap" in sys.argv
    refresh_pharmapedia = "--refresh-pharmapedia" in sys.argv
    raw_cfg = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
    brands_path = local_brands_path(raw_cfg, ROOT)
    if brands_path is None:
        brands_path = ROOT / "dict" / "local_brands.json"

    existing: list[dict] = []
    if brands_path.exists():
        existing = json.loads(brands_path.read_text(encoding="utf-8")).get("brands") or []

    print("Loading RxNorm lookup ...")
    rx_names, name_to_rxcui = _load_rxnorm_lookup(ROOT)
    print(f"RxNorm names: {len(rx_names)}")

    groups = collect_fetched(
        rx_names,
        name_to_rxcui,
        refresh_drap=refresh_drap,
        refresh_pharmapedia=refresh_pharmapedia,
    )
    print(f"Fetched generic groups: {len(groups)}")
    groups = merge_existing(groups, existing)
    brands = to_brand_list(groups)
    alias_count = sum(len(item["aliases"]) for item in brands)
    mapped = sum(1 for item in brands if item.get("rxcui"))

    payload = {
        "version": 2,
        "region": "PK",
        "notes": (
            "Pakistani trade names scraped from Pharmapedia Pro and the DRAP "
            "public price index, mapped to RxNorm generics where possible."
        ),
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [
            PHARMAPEDIA_URL,
            DRAP_URL,
            "dict/local_brands.json (previous aliases)",
        ],
        "count": len(brands),
        "alias_count": alias_count,
        "mapped_to_rxnorm": mapped,
        "brands": brands,
    }
    brands_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(brands)} generics / {alias_count} aliases ({mapped} with RxCUI)")
    print(f"Wrote {brands_path}")


if __name__ == "__main__":
    main()
