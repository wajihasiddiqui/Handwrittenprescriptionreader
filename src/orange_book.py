"""Normalize Orange Book ingredients and merge trade names into the drug database."""

from __future__ import annotations

import json
from pathlib import Path

SALT_SUFFIXES = (
    " hydrochloride",
    " hcl",
    " sodium",
    " calcium",
    " potassium",
    " magnesium",
    " sulfate",
    " sulphate",
    " mesylate",
    " maleate",
    " tartrate",
    " succinate",
    " fumarate",
    " dihydrate",
    " anhydrous",
    " benzoate",
    " acetate",
    " chloride",
    " bromide",
    " nitrate",
    " phosphate",
    " citrate",
    " tosylate",
    " besylate",
    " lactate",
)


def normalize_ob_ingredient(raw: str) -> str:
    part = raw.split(";")[0].strip().lower()
    for suffix in SALT_SUFFIXES:
        if part.endswith(suffix):
            part = part[: -len(suffix)].strip()
    return part


def canonical_for_ingredient(
    ingredient: str,
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
) -> str:
    """Prefer an RxNorm spelling when the Orange Book ingredient matches."""
    key = normalize_ob_ingredient(ingredient)
    if key in name_to_rxcui:
        return alias_to_canonical.get(key, key)
    if key in alias_to_canonical:
        return alias_to_canonical[key]
    return key


def merge_orange_book_cache(
    cache_path: Path,
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
    term_source: dict[str, str] | None = None,
    known: set[str] | None = None,
) -> int:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    known = known if known is not None else {t.lower() for t in search_terms}
    term_source = term_source if term_source is not None else {}
    added = 0
    for row in payload.get("brands") or []:
        trade = str(row.get("trade_name") or "").strip()
        ingredient = str(row.get("ingredient") or "").strip()
        if not trade or not ingredient:
            continue
        canonical = canonical_for_ingredient(ingredient, alias_to_canonical, name_to_rxcui)
        trade_key = trade.lower()
        ing_key = normalize_ob_ingredient(ingredient)
        if ing_key and ing_key not in known:
            search_terms.append(canonical)
            known.add(ing_key)
            alias_to_canonical.setdefault(ing_key, canonical)
            term_source.setdefault(ing_key, "orange_book")
            added += 1
        if trade_key == ing_key or trade_key == canonical.lower():
            continue
        if trade_key not in known:
            search_terms.append(trade)
            known.add(trade_key)
            added += 1
        alias_to_canonical[trade_key] = canonical
        term_source[trade_key] = "orange_book"
    return added
