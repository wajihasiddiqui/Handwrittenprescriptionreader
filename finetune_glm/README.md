# GLM-OCR fine-tune (GPU machine)

Train a domain-adapted GLM-OCR on your prescription handwriting, then use it with Ollama or a local server.

Official guide: https://github.com/zai-org/GLM-OCR/tree/main/examples/finetune

## Hardware

| Method | GPU VRAM |
|--------|----------|
| LoRA (recommended) | ≥ 8 GB |
| Full SFT | ≥ 24 GB |

## On this laptop (prepare only)

1. Put prescription crops/photos in `data/glm_finetune/images/`
2. Copy `data/glm_finetune/labels.example.csv` → `data/glm_finetune/labels.csv`
3. Fill `image,text` rows (image = filename under `images/`, text = correct OCR)
4. Build ShareGPT JSON:

```powershell
.\.venv\Scripts\python.exe src\prepare_glm_finetune.py
```

5. Commit/push code + `data/glm_finetune/prescriptions.json` (and images if allowed)

Do **not** train on CPU laptop — push and train on the GPU machine.

## On the GPU machine

```bash
# 1) Clone / pull this repo
cd Handwrittenprescriptionreader

# 2) One-time setup (creates venv, clones LLaMA-Factory, installs deps)
bash scripts/setup_glm_finetune.sh

# 3) Build dataset if not already built
python src/prepare_glm_finetune.py

# 4) Train LoRA (default)
bash scripts/train_glm_ocr.sh lora

# Full fine-tune (needs ~24GB):
# bash scripts/train_glm_ocr.sh full

# 5) Merge LoRA into a standalone folder
bash scripts/export_glm_lora.sh
```

Outputs:
- LoRA adapter: `third_party/LLaMA-Factory/saves/glm-ocr/lora/sft`
- Merged model: `models/glm-ocr-finetuned/`

## After training — use the model

### Option A: vLLM / transformers (easiest with merged weights)

Point a local OpenAI-compatible server at `models/glm-ocr-finetuned`, then set that host/port in your OCR config.

### Option B: Ollama

Create a Modelfile from the merged GGUF/conversion path (requires converting HF → GGUF with a community converter). Prefer vLLM on the GPU box if you stay there.

## Dataset format (ShareGPT)

Each sample:

```json
{
  "messages": [
    {"role": "user", "content": "<image>Text Recognition:"},
    {"role": "assistant", "content": "Ascard 75mg OD"}
  ],
  "images": ["prescription_images/ascard_001.png"]
}
```

Prompt must stay `Text Recognition:` for text OCR (official GLM-OCR convention).

## Files in this folder

| File | Purpose |
|------|---------|
| `configs/lora.yaml` | LoRA training config |
| `configs/full.yaml` | Full SFT config |
| `dataset_info.snippet.json` | Register dataset inside LLaMA-Factory |
