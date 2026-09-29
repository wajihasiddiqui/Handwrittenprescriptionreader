# GLM-OCR guide (in-process LLaMA-Factory)

## Idea

The pipeline loads **GLM-OCR inside the same process**.  
No Ollama. No API server for normal use.

## Setup once

```powershell
cd C:\Users\HP\source\repos\Handwrittenprescriptionreader
python src\llamafactory_train.py setup
```

## Run OCR

```powershell
.\.venv_finetune\Scripts\python.exe src\pipeline.py data\raw\Test1.png
```

Or:

```powershell
.\scripts\run_local_ocr.bat data\raw\Test1.png
```

Expect: `--- OCR (llamafactory-inprocess) ---`

## Fine-tune on GPU

```powershell
python src\llamafactory_train.py prepare
python src\llamafactory_train.py train --mode lora
python src\llamafactory_train.py export
```

## If import fails

Run setup, then use `.venv_finetune`:

```powershell
.\.venv_finetune\Scripts\python.exe src\pipeline.py data\raw\Test1.png
```
