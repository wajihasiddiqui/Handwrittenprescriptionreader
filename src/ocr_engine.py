"""GLM-OCR via Ollama (inference) and/or LLaMA-Factory ChatModel (in-process).

Training is always LLaMA-Factory (not Ollama).
Ollama is only for running glm-ocr:latest (or a merged fine-tune).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ocr.json"
LLAMA_SRC = ROOT / "third_party" / "LLaMA-Factory" / "src"

_OCR_CFG: dict | None = None
_CHAT_MODEL = None
_LAST_BACKEND = "ollama"


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
    cfg = load_ocr_config()
    backend = str(cfg.get("backend") or "ollama").strip().lower()
    if backend in {"llamafactory", "llamafactory-inprocess", "inprocess"}:
        return "llamafactory-inprocess"
    return "ollama"


def last_backend() -> str:
    return _LAST_BACKEND


def _ollama_cfg() -> dict:
    return load_ocr_config().get("ollama") or {}


def _lf_cfg() -> dict:
    return load_ocr_config().get("llamafactory") or {}


def _ensure_llamafactory_importable() -> None:
    try:
        import llamafactory  # noqa: F401

        return
    except ImportError:
        pass
    if LLAMA_SRC.exists() and str(LLAMA_SRC) not in sys.path:
        sys.path.insert(0, str(LLAMA_SRC))
    try:
        import llamafactory  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LLaMA-Factory is not installed in this Python environment.\n"
            "Run once:\n"
            "  python src\\llamafactory_train.py setup\n"
            "Or use Ollama for inference:\n"
            "  set configs/ocr.json backend to \"ollama\"\n"
            "  ollama pull glm-ocr:latest"
        ) from exc


def _prepare_image(image_path: Path, max_side: int = 1600):
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    return img


def _image_to_b64(image_path: Path, max_side: int = 1600) -> str:
    import io

    img = _prepare_image(image_path, max_side=max_side)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


def _ocr_ollama(image_path: Path) -> str:
    """Call Ollama native /api/generate (recommended for GLM-OCR vision)."""
    global _LAST_BACKEND
    _LAST_BACKEND = "ollama"
    _load_dotenv()
    cfg = _ollama_cfg()
    host = str(cfg.get("host") or "http://127.0.0.1:11434").rstrip("/")
    model = str(cfg.get("model") or "glm-ocr:latest")
    prompt = str(cfg.get("prompt") or "Text Recognition:")
    max_side = int(cfg.get("max_image_side") or 1600)
    max_tokens = int(cfg.get("max_tokens") or 768)
    timeout = int(cfg.get("timeout_sec") or 300)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_image_to_b64(Path(image_path), max_side=max_side)],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    url = f"{host}/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama not reachable at {url}\n"
            f"Install/start Ollama, then:\n"
            f"  ollama pull glm-ocr:latest\n"
            f"Detail: {exc}"
        ) from exc

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")
    return str((data or {}).get("response") or "").strip()


def get_chat_model():
    """Load GLM-OCR once via LLaMA-Factory and reuse it."""
    global _CHAT_MODEL
    if _CHAT_MODEL is not None:
        return _CHAT_MODEL

    _load_dotenv()
    _ensure_llamafactory_importable()
    from llamafactory.chat import ChatModel

    cfg = _lf_cfg()
    model_path = str(cfg.get("model") or "zai-org/GLM-OCR")
    local_ft = ROOT / "models" / "glm-ocr-finetuned"
    if local_ft.exists() and any(local_ft.iterdir()):
        model_path = str(local_ft)

    args: dict = {
        "model_name_or_path": model_path,
        "template": str(cfg.get("template") or "glm_ocr"),
        "infer_backend": str(cfg.get("infer_backend") or "huggingface"),
        "trust_remote_code": True,
    }
    adapter = str(cfg.get("adapter") or "").strip()
    if adapter:
        adapter_path = Path(adapter)
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        args["adapter_name_or_path"] = str(adapter_path)
        args["finetuning_type"] = str(cfg.get("finetuning_type") or "lora")

    print(f"Loading GLM-OCR in-process: {args['model_name_or_path']} ...")
    _CHAT_MODEL = ChatModel(args)
    print("GLM-OCR ready.")
    return _CHAT_MODEL


def _ocr_inprocess(image_path: Path) -> str:
    global _LAST_BACKEND
    _LAST_BACKEND = "llamafactory-inprocess"
    cfg = _lf_cfg()
    prompt = str(cfg.get("prompt") or "Text Recognition:")
    max_side = int(cfg.get("max_image_side") or 1600)
    max_tokens = int(cfg.get("max_tokens") or 768)

    chat_model = get_chat_model()
    image = _prepare_image(Path(image_path), max_side=max_side)
    messages = [{"role": "user", "content": f"<image>{prompt}"}]
    responses = chat_model.chat(
        messages,
        images=[image],
        temperature=0.0,
        do_sample=False,
        max_new_tokens=max_tokens,
    )
    if not responses:
        return ""
    first = responses[0]
    text = getattr(first, "response_text", None)
    if text is None and isinstance(first, dict):
        text = first.get("response_text") or first.get("text")
    return str(text or "").strip()


def ocr_image(image_path: Path, output_dir: Path | None = None) -> list[str]:
    """OCR one image using configs/ocr.json backend (ollama or in-process)."""
    backend = get_backend()
    if backend == "llamafactory-inprocess":
        raw = _ocr_inprocess(Path(image_path))
    else:
        raw = _ocr_ollama(Path(image_path))

    text = _dedupe_ocr_text(raw)
    lines = [part for part in text.splitlines() if part.strip()]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(image_path).stem}_ocr.json").write_text(
            json.dumps(
                {
                    "backend": last_backend(),
                    "text": text,
                    "lines": lines,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return lines
