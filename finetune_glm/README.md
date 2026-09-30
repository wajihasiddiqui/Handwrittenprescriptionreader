# GLM-OCR: Ollama (run) + LLaMA-Factory (train)

**Ollama does not train.** Use it only to run `glm-ocr:latest` (or your fine-tuned model).  
**Fine-tune with LLaMA-Factory** on HuggingFace `zai-org/GLM-OCR`.

## 1) Inference with Ollama (latest)

```powershell
ollama pull glm-ocr:latest
# or: scripts\pull_ollama_glmocr.bat
python src\pipeline.py data\raw\Test1.png
```

`configs/ocr.json` → `"backend": "ollama"`, model `glm-ocr:latest`.

## 2) Prepare ShareGPT + fine-tune

1. Put images in `data/glm_finetune/images/`
2. Auto-OCR → `labels.csv` → ShareGPT:

```powershell
python src\prepare_finetune.py
```

3. Fix bad OCR text in `labels.csv`, then train:

```powershell
python src\llamafactory_train.py setup
python src\llamafactory_train.py train --mode lora
python src\llamafactory_train.py export
```

## 3) Run the fine-tuned model in Ollama

```powershell
python src\llamafactory_train.py ollama --name glm-ocr-rx
```

Then set in `configs/ocr.json`:

```json
"backend": "ollama",
"ollama": { "model": "glm-ocr-rx" }
```

## Alternate inference (no Ollama)

Set `"backend": "llamafactory-inprocess"` to load the HF/merged model inside Python.
