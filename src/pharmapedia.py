"""Download brand/generic rows from https://pharmapedia.pro/medicines."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "dict" / "pharmapedia_cache.json"
BASE_URL = "https://pharmapedia.pro/medicines"
API_URL = "https://pharmapedia.pro/api/medicines"
TIMEOUT = 60
PAGE_SIZE = 50
REQUEST_PAUSE = 0.35
PREFIXES = (
    [chr(code) for code in range(ord("a"), ord("z") + 1)]
    + [str(digit) for digit in range(10)]
    + list("*+-.(['")
)
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": BASE_URL,
}


def _slim(row: dict) -> dict:
    kind = str(row.get("type") or "")
    slim = {
        "type": kind,
        "id": row.get("id"),
        "name": str(row.get("name") or "").strip(),
        "slug": str(row.get("slug") or "").strip(),
    }
    if kind == "brand":
        slim["company"] = str(row.get("company") or "").strip()
        slim["generic_name"] = str(row.get("generic_name") or "").strip()
    return slim


def _get_page(session: requests.Session, kind: str, query: str, offset: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = session.get(
                API_URL,
                params={"q": query, "type": kind, "limit": PAGE_SIZE, "offset": offset},
                timeout=(20, TIMEOUT),
            )
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After") or (8 + attempt * 5))
                print(f"  rate limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RuntimeError("Unexpected Pharmapedia response")
            time.sleep(REQUEST_PAUSE)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Pharmapedia {kind} q={query!r} offset={offset} failed: {last_error}")


def _fetch_prefix(session: requests.Session, kind: str, prefix: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    prefix_key = prefix.lower()
    while True:
        payload = _get_page(session, kind, prefix, offset)
        results = payload.get("results") or []
        kept: list[dict] = []
        for row in results:
            item = _slim(row)
            if not item["name"]:
                continue
            if prefix_key and not item["name"].lower().startswith(prefix_key):
                continue
            kept.append(item)
        rows.extend(kept)
        if offset == 0 or offset % 250 == 0:
            print(f"  {kind}:{prefix} offset {offset} +{len(kept)} total {len(rows)}", flush=True)
        if not payload.get("hasMore") or not results:
            break
        if results and not kept:
            break
        offset += len(results)
        if offset > 20000:
            break
    return rows


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (str(row.get("type") or ""), str(row.get("id") or row.get("slug") or row.get("name") or ""))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _split_payload(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    brands = [
        {
            "id": row.get("id"),
            "name": row["name"],
            "slug": row.get("slug") or "",
            "company": row.get("company") or "",
            "generic_name": row.get("generic_name") or "",
        }
        for row in rows
        if row.get("type") == "brand" and row.get("name")
    ]
    generics = [
        {
            "id": row.get("id"),
            "name": row["name"],
            "slug": row.get("slug") or "",
        }
        for row in rows
        if row.get("type") == "generic" and row.get("name")
    ]
    brands.sort(key=lambda item: item["name"].lower())
    generics.sort(key=lambda item: item["name"].lower())
    return brands, generics


def _save(cache_path: Path, rows: list[dict], completed: list[str]) -> dict:
    brands, generics = _split_payload(_dedupe(rows))
    payload = {
        "source": "Pharmapedia Pro",
        "url": BASE_URL,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(brands) + len(generics),
        "brand_count": len(brands),
        "generic_count": len(generics),
        "completed_prefixes": completed,
        "brands": brands,
        "generics": generics,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"  checkpoint {len(brands)} brands / {len(generics)} generics", flush=True)
    return payload


def fetch_all_medicines(cache_path: Path | None = None, *, refresh: bool = False) -> dict:
    cache_path = cache_path or DEFAULT_CACHE
    collected: list[dict] = []
    completed: list[str] = []
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        completed = list(payload.get("completed_prefixes") or [])
        for row in payload.get("brands") or []:
            collected.append({"type": "brand", **row})
        for row in payload.get("generics") or []:
            collected.append({"type": "generic", **row})
        jobs = [(kind, prefix) for kind in ("brands", "generics") for prefix in PREFIXES]
        if collected and len(completed) >= len(jobs):
            print(
                f"Using cached Pharmapedia: {payload.get('brand_count')} brands / "
                f"{payload.get('generic_count')} generics",
                flush=True,
            )
            return payload
        if collected:
            print(
                f"Resuming Pharmapedia cache: {len(completed)} prefixes, "
                f"{payload.get('brand_count')} brands",
                flush=True,
            )

    print(f"Fetching {BASE_URL} ...", flush=True)
    session = requests.Session()
    session.headers.update(UA)
    jobs = [(kind, prefix) for kind in ("brands", "generics") for prefix in PREFIXES]
    done_set = set(completed)
    for kind, prefix in jobs:
        key = f"{kind}:{prefix}"
        if key in done_set:
            continue
        rows = _fetch_prefix(session, kind, prefix)
        collected.extend(rows)
        completed.append(key)
        print(f"  finished {key} +{len(rows)}", flush=True)
        _save(cache_path, collected, completed)

    payload = _save(cache_path, collected, completed)
    print(
        f"Saved {payload['brand_count']} brands / {payload['generic_count']} generics to {cache_path}",
        flush=True,
    )
    return payload


def main() -> None:
    import sys

    refresh = "--refresh" in sys.argv
    fetch_all_medicines(refresh=refresh)


if __name__ == "__main__":
    main()
