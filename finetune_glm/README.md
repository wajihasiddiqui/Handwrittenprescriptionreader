# GLM-OCR fine-tune with LLaMA-Factory

Train GLM-OCR on your prescription handwriting using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

Official upstream guide: https://github.com/zai-org/GLM-OCR/tree/main/examples/finetune

## Hardware

| Mode | GPU VRAM |
|------|----------|
| `lora` (recommended) | ≥ 8 GB |
| `full` | ≥ 24 GB |

## 1) Prepare labels (laptop or GPU PC)

1. Put images in `data/glm_finetune/images/`
2. Create labels:

```powershell
copy data\glm_finetune\labels.example.csv data\glm_finetune\labels.csv
notepad data\glm_finetune\labels.csv
```

Format:

```csv
image,text
rx001.png,"Tab Ascard 75mg OD"
rx002.png,"Inj Lantus 6 units SC OD"
```

3. Build ShareGPT JSON:

```powershell
python src\llamafactory_train.py prepare
```

## 2) Setup LLaMA-Factory (GPU machine, once)

```bash
# Linux / macOS / Git Bash
python src/llamafactory_train.py setup
```

Windows PowerShell:

```powershell
python src\llamafactory_train.py setup
```

Or:

```powershell
.\scripts\llamafactory.bat setup
```

This will:
- create `.venv_finetune`
- clone `third_party/LLaMA-Factory`
- install deps + `llamafactory-cli`
- register dataset `prescriptions`

Install CUDA PyTorch yourself if CUDA is not detected:

```powershell
.\.venv_finetune\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 3) Train

```powershell
python src\llamafactory_train.py train --mode lora
```

Full fine-tune:

```powershell
python src\llamafactory_train.py train --mode full
```

Shortcut:

```powershell
.\scripts\llamafactory.bat train
```

Checkpoints go to:

`third_party/LLaMA-Factory/saves/glm-ocr/lora/sft`

## 4) Export / merge LoRA

```powershell
python src\llamafactory_train.py export
```

Merged model:

`models/glm-ocr-finetuned/`

## Files

| Path | Purpose |
|------|---------|
| `src/llamafactory_train.py` | Main entry: setup / prepare / train / export |
| `src/prepare_glm_finetune.py` | CSV → ShareGPT JSON for LLaMA-Factory |
| `finetune_glm/configs/lora.yaml` | LoRA training config (`template: glm_ocr`) |
| `finetune_glm/configs/full.yaml` | Full SFT config |
| `finetune_glm/dataset_info.snippet.json` | Registers `prescriptions` dataset |
| `scripts/llamafactory.bat` | Windows shortcuts |

## After training

Point an inference server (vLLM / transformers) at `models/glm-ocr-finetuned`, or convert for Ollama if needed.

Daily OCR on laptop can keep using Ollama `glm-ocr:latest` until you replace it with the fine-tuned weights.
