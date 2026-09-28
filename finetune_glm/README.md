# GLM-OCR with LLaMA-Factory only (no Ollama)

This project uses **LLaMA-Factory** for:
1. Serving GLM-OCR for prescription OCR
2. Optional LoRA / full fine-tuning on GPU

Official train guide: https://github.com/zai-org/GLM-OCR/tree/main/examples/finetune

## Hardware

| Task | Need |
|------|------|
| OCR infer | CPU possible (slow) / GPU better |
| LoRA train | ≥ 8 GB VRAM |
| Full train | ≥ 24 GB VRAM |

## One-time setup

```powershell
python src\llamafactory_train.py setup
```

## Run OCR (2 terminals)

**Terminal 1 — API (keep open):**
```powershell
python src\llamafactory_serve.py
```

**Terminal 2 — pipeline:**
```powershell
.\.venv\Scripts\python.exe src\pipeline.py data\raw\Test1.png
```

Expect: `--- OCR (llamafactory) ---`

## Train (optional)

```powershell
python src\llamafactory_train.py prepare
python src\llamafactory_train.py train --mode lora
python src\llamafactory_train.py export
python src\llamafactory_serve.py --adapter third_party\LLaMA-Factory\saves\glm-ocr\lora\sft
```

## Main files

| Path | Purpose |
|------|---------|
| `src/ocr_engine.py` | Calls LLaMA-Factory `/v1/chat/completions` only |
| `src/llamafactory_serve.py` | Start OCR API |
| `src/llamafactory_train.py` | setup / prepare / train / export |
| `configs/ocr.json` | API host/port/model |
| `finetune_glm/configs/infer.yaml` | LLaMA-Factory infer config |
| `finetune_glm/configs/lora.yaml` | LoRA train config |
