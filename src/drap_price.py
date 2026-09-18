"""Download all product rows from https://e.dra.gov.pk/public/price."""

from __future__ import annotations

import html as html_lib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "dict" / "drap_price_cache.json"
BASE_URL = "https://e.dra.gov.pk/public/price"
TIMEOUT = 60
MAX_WORKERS = 6
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
}

ROW_RE = re.compile(
    r'<div class="text-sm font-medium text-gray-900">(.*?)</div>\s*'
    r'<div class="max-w-\[200px\] text-sm text-gray-500[^"]*"[^>]*>(.*?)</div>',
    re.S,
)
LAST_PAGE_RE = re.compile(
    r"Showing page.*?of.*?<span class=\"font-medium\">(\d+)</span>",
    re.S,
)
FORM_SPLIT = re.compile(
    r"\s+(?:injection|tablet|capsule|syrup|suspension|cream|ointment|drops|"
    r"infusion|solution|sachet|gel|lotion|spray|inhaler|vial|ampoule|pfs|tab|cap|syp|inj)\b",
    re.I,
)


def _plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return " ".join(text.split()).strip()


def parse_last_page(html: str) -> int:
    match = LAST_PAGE_RE.search(html)
    return int(match.group(1)) if match else 1


def parse_products(html: str) -> list[dict]:
    rows: list[dict] = []
    for brand_html, generic_html in ROW_RE.findall(html):
        brand = _plain(brand_html)
        generic = _plain(generic_html)
        if not brand:
            continue
        rows.append({"product_name": brand, "generic_name": generic})
    return rows


def short_brand(product_name: str) -> str:
    core = product_name.split("(")[0].strip()
    core = FORM_SPLIT.split(core, maxsplit=1)[0].strip()
    core = re.sub(r"\s+\d[\d./%]*\s*(?:mg|ml|mcg|g|iu|units?|%|s)?\b.*$", "", core, flags=re.I)
    core = re.sub(r"\s+\d.*$", "", core).strip(" -/")
    return core or product_name


def short_generic(generic_name: str) -> str:
    text = re.split(r"\bEach\b", generic_name, maxsplit=1, flags=re.I)[0]
    text = FORM_SPLIT.split(text, maxsplit=1)[0]
    text = re.sub(r"\s+\d.*$", "", text).strip(" -,:;()")
    return text[:100]


_THREAD = threading.local()


def _session() -> requests.Session:
    sess = getattr(_THREAD, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(UA)
        _THREAD.session = sess
    return sess


def _get_page(page: int) -> str:
    last_error: Exception | None = None
    for _attempt in range(4):
        try:
            url = BASE_URL if page == 1 else f"{BASE_URL}?page={page}"
            response = _session().get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"DRAP page {page} failed: {last_error}")


def fetch_all_products(cache_path: Path | None = None, *, refresh: bool = False) -> list[dict]:
    cache_path = cache_path or DEFAULT_CACHE
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        products = payload.get("products") or []
        if products:
            print(f"Using cached DRAP prices: {len(products)} products")
            return products

    print(f"Fetching {BASE_URL} ...")
    first_html = _get_page(1)
    last_page = parse_last_page(first_html)
    products = parse_products(first_html)
    print(f"DRAP pages: {last_page} (20 products/page)")

    if last_page > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_get_page, page): page for page in range(2, last_page + 1)}
            done = 1
            for future in as_completed(futures):
                page = futures[future]
                html = future.result()
                products.extend(parse_products(html))
                done += 1
                if done % 50 == 0 or done == last_page:
                    print(f"  scraped {done}/{last_page} pages, {len(products)} rows")

    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in products:
        key = (row["product_name"].lower(), row["generic_name"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    payload = {
        "source": "DRAP Pharmaceutical Product Price Index",
        "url": BASE_URL,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pages": last_page,
        "count": len(unique),
        "products": unique,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(unique)} DRAP products to {cache_path}")
    return unique
