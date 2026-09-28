"""GLM-OCR text extraction — local on this laptop via Ollama (no cloud API key)."""

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
    return "glm-ocr-local"


def last_backend() -> str:
    return "glm-ocr-local"


def _glm_cfg() -> dict:
    return load_ocr_config().get("glm_ocr") or {}


def _check_ollama(host: str = "127.0.0.1", port: int = 11434) -> None:
    url = f"http://{host}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not running on this laptop.\n"
            "1) Install: https://ollama.com/download\n"
            "2) Pull model:  ollama pull glm-ocr:latest\n"
            "3) Start the Ollama Windows app, then retry."
        ) from exc

    names = [
        str(row.get("name") or row.get("model") or "")
        for row in (payload.get("models") or [])
    ]
    names = [n for n in names if n]
    if not any(n.startswith("glm-ocr") for n in names):
        raise RuntimeError(
            "Ollama is running, but glm-ocr is not installed yet.\n"
            "Run:  ollama pull glm-ocr:latest\n"
            f"Installed models: {names or '(none)'}"
        )


def _prepare_image_bytes(image_path: Path, max_side: int = 1600) -> tuple[bytes, str]:
    """Downscale large photos so CPU OCR finishes in reasonable time."""
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
        return buf.getvalue(), "image/jpeg"


def _ollama_generate_vision(image_path: Path) -> str:
    cfg = _glm_cfg()
    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or 11434)
    model = str(cfg.get("model") or "glm-ocr:latest")
    timeout = int(cfg.get("timeout") or 900)
    prompt = str(cfg.get("prompt") or "Text Recognition:")
    max_side = int(cfg.get("max_image_side") or 1600)

    _check_ollama(host, port)
    raw, _mime = _prepare_image_bytes(image_path, max_side=max_side)
    body = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(raw).decode("ascii")],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": int(cfg.get("num_predict") or 2048),
        },
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s while reading {image_path.name}.\n"
            "CPU OCR can be slow. Keep Ollama running and retry."
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc

    if payload.get("error"):
        raise RuntimeError(f"Ollama error: {payload['error']}")
    return str(payload.get("response") or "").strip()


def ocr_image(image_path: Path, output_dir: Path | None = None) -> list[str]:
    """Run local GLM-OCR through Ollama on this laptop and return text lines."""
    _load_dotenv()
    text = _ollama_generate_vision(Path(image_path))
    lines: list[str] = []
    for part in text.splitlines():
        part = part.strip()
        if not part:
            continue
        if set(part) <= {"`", " "}:
            continue
        lines.append(part)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(image_path).stem}_glmocr.json").write_text(
            json.dumps(
                {
                    "backend": "glm-ocr-local",
                    "model": (_glm_cfg().get("model") or "glm-ocr:latest"),
                    "text": text,
                    "lines": lines,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return lines
