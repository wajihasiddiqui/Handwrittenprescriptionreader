"""GLM-OCR via LLaMA-Factory OpenAI-compatible API only."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ocr.json"

_OCR_CFG: dict | None = None


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_ocr_config() -> dict:
    global _OCR_CFG
    if _OCR_CFG is None:
        if CONFIG_PATH.exists():
            _OCR_CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            _OCR_CFG = {}
    return _OCR_CFG


def get_backend() -> str:
    return "llamafactory"


def last_backend() -> str:
    return "llamafactory"


def _cfg() -> dict:
    return load_ocr_config().get("llamafactory") or load_ocr_config().get("glm_ocr") or {}


def _prepare_image_bytes(image_path: Path, max_side: int = 1600) -> bytes:
    import io

    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _dedupe_ocr_text(text: str) -> str:
    lines: list[str] = []
    for part in text.splitlines():
        part = part.strip()
        if not part:
            continue
        if set(part) <= {"`", " ", "*"}:
            if lines:
                break
            continue
        lines.append(part)
    if not lines:
        return ""
    anchor = lines[0]
    for i in range(1, len(lines)):
        if lines[i] == anchor and i >= 3:
            lines = lines[:i]
            break
    if len(lines) >= 4:
        half = len(lines) // 2
        first = lines[:half]
        second = lines[half : half + len(first)]
        if first and first == second:
            lines = first
    cleaned: list[str] = []
    for line in lines:
        if cleaned and cleaned[-1] == line:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _post_json(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ensure_api(host: str, port: int) -> None:
    try:
        urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5).read()
    except Exception as exc:
        raise RuntimeError(
            "LLaMA-Factory API is not running.\n"
            "Terminal 1:  python src\\llamafactory_serve.py\n"
            "Terminal 2:  python src\\pipeline.py data\\raw\\YourImage.png\n"
            f"Expected API: http://{host}:{port}/v1"
        ) from exc


def _ocr_llamafactory(image_path: Path) -> str:
    cfg = _cfg()
    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or 8000)
    model = str(cfg.get("model") or "zai-org/GLM-OCR")
    prompt = str(cfg.get("prompt") or "Text Recognition:")
    timeout = int(cfg.get("timeout") or 900)
    max_side = int(cfg.get("max_image_side") or 1600)
    max_tokens = int(cfg.get("max_tokens") or cfg.get("num_predict") or 768)

    _ensure_api(host, port)
    raw = _prepare_image_bytes(image_path, max_side=max_side)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    try:
        payload = _post_json(
            f"http://{host}:{port}/v1/chat/completions", body, timeout=timeout
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"LLaMA-Factory HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"LLaMA-Factory timed out after {timeout}s on {image_path.name}"
        ) from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty LLaMA-Factory response: {str(payload)[:400]}")
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


def ocr_image(image_path: Path, output_dir: Path | None = None) -> list[str]:
    """Run GLM-OCR through LLaMA-Factory API and return text lines."""
    _load_dotenv()
    text = _dedupe_ocr_text(_ocr_llamafactory(Path(image_path)))
    lines = [part for part in text.splitlines() if part.strip()]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(image_path).stem}_ocr.json").write_text(
            json.dumps(
                {
                    "backend": "llamafactory",
                    "text": text,
                    "lines": lines,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return lines
