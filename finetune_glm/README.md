# GLM-OCR with LLaMA-Factory (in-process, no server)

The pipeline loads **GLM-OCR inside the same Python process** using LLaMA-Factory `ChatModel`.  
You do **not** need to start an API server for normal OCR.

## Prepare ShareGPT dataset

1. Put images in `data/glm_finetune/images/` (or use files already in `data/raw/`)
2. Edit labels:

```powershell
copy data\glm_finetune\labels.example.csv data\glm_finetune\labels.csv
notepad data\glm_finetune\labels.csv
```

Columns: `image,text` (optional `task`: text|table|formula)

3. Build ShareGPT JSON for LLaMA-Factory:

```powershell
python src\prepare_finetune.py
```

Outputs:
- `data/glm_finetune/prescriptions.json`
- `data/glm_finetune/prescription_images/`
- syncs into `third_party/LLaMA-Factory/data/` when setup was run

4. Train:

```powershell
python src\llamafactory_train.py train --mode lora
```

