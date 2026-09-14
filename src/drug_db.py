"""Load RxNorm cache plus local medical codes from dict/drug_db.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    live_fallback: bool = False
    api_base: str = "https://rxnav.nlm.nih.gov/REST"

    def canonical_name(self, matched: str) -> str:
        return self.alias_to_canonical.get(matched.lower(), matched)

    def rxcui_for(self, name: str | None) -> str | None:
        if not name:
            return None
        return self.name_to_rxcui.get(name.lower())


def rxnorm_cache_path(raw: dict, root: Path = ROOT) -> Path:
    rel = (raw.get("rxnorm") or {}).get("cache_path", "dict/rxnorm_cache.json")
    path = Path(rel)
    return path if path.is_absolute() else root / path


def _add_term(
    name: str,
    rxcui: str,
    canonical_names: list[str],
    search_terms: list[str],
    alias_to_canonical: dict[str, str],
    name_to_rxcui: dict[str, str],
) -> None:
    canonical_names.append(name)
    search_terms.append(name)
    alias_to_canonical[name.lower()] = name
    if rxcui:
        name_to_rxcui[name.lower()] = rxcui


def load_drug_database(path: Path | None = None) -> DrugDatabase:
    db_path = path or DEFAULT_DB_PATH
    raw = json.loads(db_path.read_text(encoding="utf-8"))
    rx_cfg = raw.get("rxnorm") or {}

    canonical_names: list[str] = []
    search_terms: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    name_to_rxcui: dict[str, str] = {}
    source = str(raw.get("source") or "local")

    if source == "rxnorm":
        cache_file = rxnorm_cache_path(raw, db_path.parent.parent)
        if not cache_file.exists():
            raise FileNotFoundError(
                f"RxNorm cache missing: {cache_file}\n"
                "Run: python src\\sync_rxnorm.py"
            )
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
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
            )
    else:
        for entry in raw.get("drugs") or []:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            _add_term(name, "", canonical_names, search_terms, alias_to_canonical, name_to_rxcui)
            for alias in entry.get("aliases") or []:
                alias = str(alias).strip()
                if not alias:
                    continue
                if alias not in search_terms:
                    search_terms.append(alias)
                alias_to_canonical[alias.lower()] = name

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
        live_fallback=bool(rx_cfg.get("live_fallback", False)),
        api_base=str(rx_cfg.get("api_base") or "https://rxnav.nlm.nih.gov/REST"),
    )
