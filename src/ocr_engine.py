"""GLM-OCR in-process via LLaMA-Factory ChatModel (no API server)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ocr.json"
LLAMA_SRC = ROOT / "third_party" / "LLaMA-Factory" / "src"

_OCR_CFG: dict | None = None
_CHAT_MODEL = None


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
    return "llamafactory-inprocess"


def last_backend() -> str:
    return "llamafactory-inprocess"


def _cfg() -> dict:
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
            "Then either:\n"
            "  .\\.venv_finetune\\Scripts\\python.exe src\\pipeline.py data\\raw\\Test1.png\n"
            "or install into your active venv:\n"
            "  pip install -e third_party\\LLaMA-Factory\n"
            "  pip install -U \"transformers>=5.3.0\" torch torchvision pillow"
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


def get_chat_model():
    """Load GLM-OCR once and reuse it for later images."""
    global _CHAT_MODEL
    if _CHAT_MODEL is not None:
        return _CHAT_MODEL

    _load_dotenv()
    _ensure_llamafactory_importable()
    from llamafactory.chat import ChatModel

    cfg = _cfg()
    model_path = str(cfg.get("model") or "zai-org/GLM-OCR")
    # Prefer local merged fine-tune if present.
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
    cfg = _cfg()
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
    """Load GLM-OCR in this process and OCR one image (no server)."""
    text = _dedupe_ocr_text(_ocr_inprocess(Path(image_path)))
    lines = [part for part in text.splitlines() if part.strip()]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(image_path).stem}_ocr.json").write_text(
            json.dumps(
                {
                    "backend": "llamafactory-inprocess",
                    "text": text,
                    "lines": lines,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return lines
