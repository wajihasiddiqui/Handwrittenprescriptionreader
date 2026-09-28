# GLM-OCR guide (LLaMA-Factory only)

This project reads prescription photos with **GLM-OCR** served by **LLaMA-Factory**.  
**Ollama is not used.**

## What you need

1. Python 3.12 + project `.venv`
2. LLaMA-Factory setup (`python src\llamafactory_train.py setup`)
3. Two terminals when reading a photo (API + pipeline)

## First-time setup

```powershell
cd C:\Users\HP\source\repos\Handwrittenprescriptionreader
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install pillow requests rapidfuzz spacy pyyaml

python src\llamafactory_train.py setup
```

If CUDA is available on a GPU machine:

```powershell
.\.venv_finetune\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Run OCR every time

**Terminal 1 (keep open):**
```powershell
python src\llamafactory_serve.py
```

**Terminal 2:**
```powershell
.\.venv\Scripts\python.exe src\pipeline.py data\raw\Test1.png
```

Or:
```powershell
.\scripts\start_llamafactory_api.bat
.\scripts\run_local_ocr.bat data\raw\Test1.png
```

Check:
```powershell
.\scripts\check_local_ocr.bat
```

## Fine-tune (GPU)

1. Put images in `data\glm_finetune\images\`
2. Edit `data\glm_finetune\labels.csv` (`image,text`)
3. Train:

```powershell
python src\llamafactory_train.py prepare
python src\llamafactory_train.py train --mode lora
python src\llamafactory_train.py export
```

Serve fine-tuned weights:

```powershell
python src\llamafactory_serve.py --adapter third_party\LLaMA-Factory\saves\glm-ocr\lora\sft
```

## Common errors

| Message | Fix |
|---------|-----|
| LLaMA-Factory API is not running | Start `python src\llamafactory_serve.py` |
| Missing `.venv_finetune` | Run `python src\llamafactory_train.py setup` |
| CUDA / torch issues | Install CUDA torch in `.venv_finetune` |

## Output files

- `output\last_ocr.txt`
- `output\last_ner.json`
- `output\last_result.json`
