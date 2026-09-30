# Fine-tune GLM-OCR from labels.csv (LLaMA-Factory)

**Train tool:** LLaMA-Factory  
**Data source:** `data/glm_finetune/labels.csv` + images  
**Base model:** `zai-org/GLM-OCR`

Ollama is only for running a model after training — it does not train.

## 1) Labels

Edit `data/glm_finetune/labels.csv`:

```csv
image,text,task
rx01.png,"Tab Ascard 75mg OD",text
rx02.png,"Syp Acefyl 2+2+2",text
```

Put image files in `data/glm_finetune/images/` (or `data/raw/`).

## 2) One-shot (Windows)

```powershell
scripts\run_finetune.bat
```

## 3) Step by step

```powershell
python src\llamafactory_train.py setup
python src\llamafactory_train.py prepare --include-raw
python src\llamafactory_train.py train --mode lora --include-raw
python src\llamafactory_train.py export
```

Optional: pack for Ollama

```powershell
python src\llamafactory_train.py ollama --name glm-ocr-rx
```

Check readiness:

```powershell
python src\llamafactory_train.py status
```

## Notes

- `prepare` reads **labels.csv** and builds ShareGPT (`prescriptions.json`).
- Add `--auto-ocr` only if you want empty `text` cells filled by GLM-OCR first.
- Needs a **GPU** for practical training (LoRA ~8GB+ VRAM).
- After export, `configs/ocr.json` gets `llamafactory.adapter` set to the LoRA path.
