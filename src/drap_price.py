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


SALT_RE = re.compile(
    r"\b(?:dihydrochloride|hydrochloride|hcl|2hcl|di-?hcl|sodium|potassium|"
    r"calcium|magnesium|sulphate|sulfate|acetate|maleate|tartrate|"
    r"bp|usp|specs?|specification)\b",
    re.I,
)
CONTAINS_RE = re.compile(
    r"each\s+(?:film[-\s]?coated\s+)?(?:tablet|capsule|5\s*ml|ml|vial|sachet)[^\w]{0,40}contains[:\s-]*",
    re.I,
)


def extract_inn(text: str) -> str:
    """Pull a likely generic/INN out of a DRAP composition string."""
    t = html_lib.unescape(text or "")
    t = CONTAINS_RE.sub(" ", t)
    t = SALT_RE.sub(" ", t)
    t = re.split(r"\d", t, maxsplit=1)[0]
    t = re.sub(r"[^A-Za-z +/]+", " ", t)
    t = " ".join(t.split()).strip(" -/")
    if len(t) < 4:
        return ""
    return t[:100]


def first_brand_token(product_name: str) -> str:
    token = re.split(r"[\s(,/]+", product_name.strip())[0]
    token = re.sub(r"[^A-Za-z0-9\-]+", "", token)
    if len(token) < 4:
        return ""
    if token.lower() in {"each", "tablet", "capsule", "syrup", "oral", "film"}:
        return ""
    return token


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


def parse_manufacturer_ids(html: str) -> list[str]:
    block = re.search(
        r'<select name="manufacturer".*?</select>',
        html,
        re.S | re.I,
    )
    if not block:
        return []
    return re.findall(r'<option value="(\d+)">', block.group(0))


def fetch_manufacturer_products() -> list[dict]:
    session = _session()
    home = session.get(BASE_URL, timeout=TIMEOUT)
    home.raise_for_status()
    token_match = re.search(r'name="_token" value="([^"]+)"', home.text)
    if not token_match:
        raise RuntimeError("DRAP CSRF token not found")
    token = token_match.group(1)
    mfr_ids = parse_manufacturer_ids(home.text)
    print(f"DRAP manufacturers: {len(mfr_ids)}")
    products: list[dict] = []
    for i, mfr_id in enumerate(mfr_ids, start=1):
        last_error: Exception | None = None
        html = ""
        for _attempt in range(3):
            try:
                response = session.post(
                    BASE_URL,
                    data={"_token": token, "manufacturer": mfr_id},
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                html = response.text
                token_refresh = re.search(r'name="_token" value="([^"]+)"', html)
                if token_refresh:
                    token = token_refresh.group(1)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error:
            print(f"  skip manufacturer {mfr_id}: {last_error}")
            continue
        products.extend(parse_products(html))
        last_page = parse_last_page(html)
        if last_page > 40:
            continue
        for page in range(2, last_page + 1):
            page_html = session.get(
                f"{BASE_URL}?page={page}&manufacturer={mfr_id}",
                timeout=TIMEOUT,
            ).text
            products.extend(parse_products(page_html))
        if i % 25 == 0 or i == len(mfr_ids):
            print(f"  manufacturers {i}/{len(mfr_ids)}, extra rows {len(products)}")
    return products


def _dedupe(products: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in products:
        key = (row["product_name"].lower(), row["generic_name"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def fetch_all_products(
    cache_path: Path | None = None,
    *,
    refresh: bool = False,
    include_manufacturers: bool = False,
) -> list[dict]:
    cache_path = cache_path or DEFAULT_CACHE
    products: list[dict] = []
    last_page = 0
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        products = payload.get("products") or []
        last_page = int(payload.get("pages") or 0)
        if products and not include_manufacturers:
            print(f"Using cached DRAP prices: {len(products)} products")
            return products
        if products:
            print(f"Using cached DRAP prices: {len(products)} products")

    if not products:
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
                    html = future.result()
                    products.extend(parse_products(html))
                    done += 1
                    if done % 50 == 0 or done == last_page:
                        print(f"  scraped {done}/{last_page} pages, {len(products)} rows")

    if include_manufacturers:
        extra = fetch_manufacturer_products()
        print(f"Manufacturer extra rows: {len(extra)}")
        products.extend(extra)

    unique = _dedupe(products)
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
