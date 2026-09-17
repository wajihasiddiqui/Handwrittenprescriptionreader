"""Download FDA Orange Book products.txt and build a local brand-name cache."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from drug_db import DEFAULT_DB_PATH, ROOT, orange_book_cache_path

TIMEOUT = 300
DEFAULT_DOWNLOAD_URL = "https://www.fda.gov/media/76860/download"


def parse_products(text: str, *, include_otc: bool, exclude_discontinued: bool) -> list[dict]:
    """Parse tilde-delimited products.txt from the Orange Book ZIP."""
    rows: list[dict] = []
    trade_seen: set[str] = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("~")
        if len(parts) < 13:
            continue

        ingredient = parts[0].strip()
        trade_name = parts[2].strip()
        product_type = parts[12].strip().upper()
        if not ingredient or not trade_name:
            continue
        if exclude_discontinued and product_type == "DISCN":
            continue
        if not include_otc and product_type == "OTC":
            continue

        key = trade_name.lower()
        if key in trade_seen:
            continue
        trade_seen.add(key)
        rows.append(
            {
                "trade_name": trade_name,
                "ingredient": ingredient,
                "dosage_form_route": parts[1].strip(),
                "strength": parts[4].strip(),
                "appl_type": parts[5].strip(),
                "appl_no": parts[6].strip(),
                "product_type": product_type,
            }
        )

    rows.sort(key=lambda row: row["trade_name"].lower())
    return rows


def download_products_zip(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": "HandwrittenPrescriptionReader/1.0"},
    )
    response.raise_for_status()
    return response.content


def extract_products_txt(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        name = next((n for n in archive.namelist() if n.lower().endswith("products.txt")), None)
        if not name:
            raise FileNotFoundError("products.txt not found inside Orange Book ZIP")
        return archive.read(name).decode("latin-1")


def main() -> None:
    cfg = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
    ob = cfg.get("orange_book") or {}
    if not ob.get("enabled", True):
        print("Orange Book sync disabled in dict/drug_db.json")
        return

    url = str(ob.get("download_url") or DEFAULT_DOWNLOAD_URL)
    cache = orange_book_cache_path(cfg, ROOT)
    include_otc = bool(ob.get("include_otc", True))
    exclude_discontinued = bool(ob.get("exclude_discontinued", True))

    print(f"Downloading FDA Orange Book from {url} ...")
    zip_bytes = download_products_zip(url)
    products_text = extract_products_txt(zip_bytes)
    brands = parse_products(products_text, include_otc=include_otc, exclude_discontinued=exclude_discontinued)

    payload = {
        "source": "FDA Orange Book",
        "publisher": "U.S. Food and Drug Administration",
        "download_url": url,
        "include_otc": include_otc,
        "exclude_discontinued": exclude_discontinued,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(brands),
        "brands": brands,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved {len(brands)} Orange Book trade names to {cache}")


if __name__ == "__main__":
    main()
