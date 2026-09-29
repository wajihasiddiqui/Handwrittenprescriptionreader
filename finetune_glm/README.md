# GLM-OCR with LLaMA-Factory (in-process, no server)

The pipeline loads **GLM-OCR inside the same Python process** using LLaMA-Factory `ChatModel`.  
You do **not** need to start an API server for normal OCR.

## Prepare ShareGPT dataset

1. Put images in `data/glm_finetune/images/`
2. Run prepare (GLM-OCR fills `labels.csv` automatically, then builds ShareGPT):

```powershell
python src\prepare_finetune.py
```

Or: `scripts\prepare_finetune.bat`

Options:
- `--force-ocr` — re-OCR every image (overwrite existing text)
- `--include-raw` — also scan `data/raw/`
- `--no-auto-ocr` — use only hand-edited `labels.csv`

3. Optional: fix bad OCR text in `data/glm_finetune/labels.csv`
4. Train:

```powershell
python src\llamafactory_train.py train --mode lora
```

