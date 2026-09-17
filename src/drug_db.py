"""Load RxNorm, FDA Orange Book, and local brand aliases from dict/drug_db.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from orange_book import merge_orange_book_cache

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "dict" / "drug_db.json"


@dataclass
class DrugDatabase:
    path: Path
    source: str
    fuzzy_threshold: int
    dose_units: list[str]
    frequencies: set[str]
    routes: set[str]
    schedules: set[str]
    require_inr_for: set[str]
    canonical_names: list[str]
    search_terms: list[str]
    alias_to_canonical: dict[str, str] = field(default_factory=dict)
    name_to_rxcui: dict[str, str] = field(default_factory=dict)
    term_source: dict[str, str] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    live_fallback: bool = False
    api_base: str = "https://rxnav.nlm.nih.gov/REST"

    def canonical_name(self, matched: str) -> str:
        return self.alias_to_canonical.get(matched.lower(), matched)

    def rxcui_for(self, name: str | None) -> str | None:
        if not name:
            return None
        return self.name_to_rxcui.get(name.lower())

    def source_for(self, term: str | None) -> str | None:
        if not term:
            return None
        return self.term_source.get(term.lower())


def rxnorm_cache_path(raw: dict, root: Path = ROOT) -> Path:
    rel = (raw.get("rxnorm") or {}).get("cache_path", "dict/rxnorm_cache.json")
    path = Path(rel)
    return path if path.is_absolute() else root / path


def local_brands_path(raw: dict, root: Path = ROOT) -> Path | None:
    rel = (raw.get("local_brands") or {}).get("path")
    if not rel:
        return None
    path = Path(rel)
    return path if path.is_absolute() else root / path


def orange_book_cache_path(raw: dict, root: Path = ROOT) -> Path:
    rel = (raw.get("orange_book") or {}).get("cache_path", "dict/orange_book_cache.json")
    path = Path(rel)
    return path if path.is_absolute() else root / path


def _add_term(
    name: str,
    rxcui: str,
    canonical_names: list[str],
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
    term_source: dict[str, str],
    source: str,
    known: set[str],
) -> None:
    key = name.lower()
    canonical_names.append(name)
    if key not in known:
        search_terms.append(name)
        known.add(key)
    alias_to_canonical[key] = name
    term_source.setdefault(key, source)
    if rxcui:
        name_to_rxcui[key] = rxcui


def _merge_brand_entries(
    entries: list[dict],
    canonical_names: list[str],
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
    term_source: dict[str, str],
    source: str,
    known: set[str],
) -> int:
    added = 0
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        rxcui = str(entry.get("rxcui") or "").strip()
        key = name.lower()
        if key not in alias_to_canonical:
            _add_term(
                name,
                rxcui,
                canonical_names,
                search_terms,
                alias_to_canonical,
                name_to_rxcui,
                term_source,
                source,
                known,
            )
            added += 1
        elif rxcui and key not in name_to_rxcui:
            name_to_rxcui[key] = rxcui
        for alias in entry.get("aliases") or []:
            alias = str(alias).strip()
            if not alias:
                continue
            alias_key = alias.lower()
            if alias_key not in known:
                search_terms.append(alias)
                known.add(alias_key)
                added += 1
            alias_to_canonical[alias_key] = name
            term_source[alias_key] = source
            if rxcui and alias_key not in name_to_rxcui:
                name_to_rxcui[alias_key] = rxcui
    return added


def _load_rxnorm(
    raw: dict,
    canonical_names: list[str],
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
    term_source: dict[str, str],
    known: set[str],
    root: Path,
) -> int | None:
    rx_cfg = raw.get("rxnorm") or {}
    if rx_cfg.get("enabled", True) is False:
        return None
    cache_file = rxnorm_cache_path(raw, root)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"RxNorm cache missing: {cache_file}\n"
            "Run: python src\\sync_rxnorm.py"
        )
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    count = 0
    for row in cache.get("concepts") or []:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        _add_term(
            name,
            str(row.get("rxcui") or ""),
            canonical_names,
            search_terms,
            alias_to_canonical,
            name_to_rxcui,
            term_source,
            "rxnorm",
            known,
        )
        count += 1
    return count


def _load_orange_book(
    raw: dict,
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
    term_source: dict[str, str],
    known: set[str],
    root: Path,
) -> int | None:
    ob = raw.get("orange_book") or {}
    if ob.get("enabled", True) is False:
        return None
    cache_file = orange_book_cache_path(raw, root)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Orange Book cache missing: {cache_file}\n"
            "Run: python src\\sync_orange_book.py"
        )
    return merge_orange_book_cache(
        cache_file,
        search_terms,
        alias_to_canonical,
        name_to_rxcui,
        term_source=term_source,
        known=known,
    )


def _load_local_brands(
    raw: dict,
    canonical_names: list[str],
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
    term_source: dict[str, str],
    known: set[str],
    root: Path,
) -> int | None:
    local_cfg = raw.get("local_brands") or {}
    if local_cfg.get("enabled", True) is False and not (raw.get("drugs") or []):
        return None
    added = 0
    if local_cfg.get("enabled", True) is not False:
        brands_file = local_brands_path(raw, root)
        if brands_file and brands_file.exists():
            payload = json.loads(brands_file.read_text(encoding="utf-8"))
            added += _merge_brand_entries(
                payload.get("brands") or [],
                canonical_names,
                search_terms,
                alias_to_canonical,
                name_to_rxcui,
                term_source,
                "local",
                known,
            )
    added += _merge_brand_entries(
        raw.get("drugs") or [],
        canonical_names,
        search_terms,
        alias_to_canonical,
        name_to_rxcui,
        term_source,
        "local",
        known,
    )
    return added


def load_drug_database(path: Path | None = None) -> DrugDatabase:
    db_path = path or DEFAULT_DB_PATH
    raw = json.loads(db_path.read_text(encoding="utf-8"))
    rx_cfg = raw.get("rxnorm") or {}
    project_root = db_path.parent.parent

    canonical_names: list[str] = []
    search_terms: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    name_to_rxcui: dict[str, str] = {}
    term_source: dict[str, str] = {}
    known: set[str] = set()
    source_counts: dict[str, int] = {}
    loaded: list[str] = []

    rx_count = _load_rxnorm(
        raw,
        canonical_names,
        search_terms,
        alias_to_canonical,
        name_to_rxcui,
        term_source,
        known,
        project_root,
    )
    if rx_count is not None:
        source_counts["rxnorm"] = rx_count
        loaded.append("rxnorm")

    ob_count = _load_orange_book(
        raw,
        search_terms,
        alias_to_canonical,
        name_to_rxcui,
        term_source,
        known,
        project_root,
    )
    if ob_count is not None:
        source_counts["orange_book"] = ob_count
        loaded.append("orange_book")

    local_count = _load_local_brands(
        raw,
        canonical_names,
        search_terms,
        alias_to_canonical,
        name_to_rxcui,
        term_source,
        known,
        project_root,
    )
    if local_count is not None:
        source_counts["local"] = local_count
        loaded.append("local")

    source_counts["search_terms"] = len(search_terms)
    source = "+".join(loaded) if loaded else "none"

    return DrugDatabase(
        path=db_path,
        source=source,
        fuzzy_threshold=int(raw.get("fuzzy_threshold", 80)),
        dose_units=[str(u) for u in raw.get("dose_units", [])],
        frequencies={str(x).upper() for x in raw.get("frequencies", [])},
        routes={str(x).upper() for x in raw.get("routes", [])},
        schedules={str(x) for x in raw.get("schedules", [])},
        require_inr_for={str(x) for x in raw.get("validation", {}).get("require_inr_for", [])},
        canonical_names=canonical_names,
        search_terms=search_terms,
        alias_to_canonical=alias_to_canonical,
        name_to_rxcui=name_to_rxcui,
        term_source=term_source,
        source_counts=source_counts,
        live_fallback=bool(rx_cfg.get("live_fallback", False)),
        api_base=str(rx_cfg.get("api_base") or "https://rxnav.nlm.nih.gov/REST"),
    )
