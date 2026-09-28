# GLM-OCR step by step

This project reads a photo of a handwritten prescription with **GLM-OCR**.
The model runs on your computer through **Ollama**. There is no cloud API key.

A human must still check every result.

Do the parts in order. Part A is enough to read photos on this laptop.
Part B is only for later, on a computer that has an NVIDIA GPU.

---

## Words

| Word | Meaning |
|------|---------|
| OCR | Turn a photo into text. |
| GLM-OCR | The model that reads the writing. |
| Ollama | The app that runs GLM-OCR on this PC. |
| `.venv` | A private Python folder for this project only. |
| NER | After the text is read, pick out the medicine name, dose, and how often to take it. |
| Fine-tune | Extra practice on your own handwriting. Not needed to start. |
| LLaMA-Factory | The trainer used only on the GPU computer. It teaches GLM-OCR from your labeled photos. |

The flow is:

1. Photo goes in.
2. GLM-OCR (through Ollama) writes the text.
3. The project guesses the medicine details.
4. You open the saved files and check them.

---

# Part A — Read prescriptions on this laptop

## What you install

Install these, in this order:

1. **Python 3.12**
2. **Ollama**
3. The model **glm-ocr:latest** (Ollama downloads it)
4. Three Python packages: **pillow**, **rapidfuzz**, **requests**

You also need:

- This folder: `C:\Users\HP\source\repos\Handwrittenprescriptionreader`
- Internet for the first download
- About 4 GB of free disk for the model
- 8 GB of RAM (16 GB is more comfortable)

A graphics card is not required for Part A. On a normal laptop CPU, one photo can take several minutes. Leave the Ollama app open while a photo is being read.

You do not install PaddleOCR for this path.

---

## Step 1 — Install Python

1. Download Python 3.12 from https://www.python.org/downloads/
2. On the first installer screen, tick **Add python.exe to PATH**.
3. Open PowerShell and check:

```powershell
python --version
```

You want a line like `Python 3.12.x`.

---

## Step 2 — Create the project Python folder

```powershell
cd C:\Users\HP\source\repos\Handwrittenprescriptionreader
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pillow rapidfuzz requests
```

Use `.\.venv\Scripts\python.exe` for every later Python command in this project.

---

## Step 3 — Install Ollama

1. Download the Windows installer from https://ollama.com/download
2. Install it.
3. Open the Ollama app. It stays in the system tray (near the clock).

---

## Step 4 — Download the GLM-OCR model

This is the large download. Keep the internet connection up until it finishes.

```powershell
ollama pull glm-ocr:latest
ollama list
```

`ollama list` must show a name that starts with `glm-ocr`.

---

## Step 5 — Check that everything works

```powershell
cd C:\Users\HP\source\repos\Handwrittenprescriptionreader
scripts\check_local_ocr.bat
```

When it prints `Setup OK`, continue.

If it says Ollama is not running, open the Ollama app and run the check again.
If it says `glm-ocr` is not installed, repeat Step 4.

---

## Step 6 — Read one prescription

1. Put the photo in `data\raw\`. Example: `data\raw\Test1.png`
2. Run:

```powershell
scripts\run_local_ocr.bat data\raw\Test1.png
```

The same job, written out:

```powershell
.\.venv\Scripts\python.exe src\pipeline.py data\raw\Test1.png
```

Use the real file name. If the photo is `.jpg`, pass the `.jpg` path.

---

## Step 7 — Open the results

| File | What it is |
|------|------------|
| `output\last_ocr.txt` | The text GLM-OCR read |
| `output\last_ner.json` | Guessed medicine, dose, and frequency |
| `output\last_result.json` | Both together |

Compare the text with the photo. If a medicine name is wrong, that is a reading error. A person still has to approve the answer before anyone uses it.

The prompt sent to the model is `Text Recognition:`. It is already set in `configs\ocr.json`. Leave it as it is.

---

## If something breaks

| What you see | What to do |
|--------------|------------|
| `Ollama is not running` | Open the Ollama app, then retry |
| `glm-ocr is not installed` | Run `ollama pull glm-ocr:latest` |
| `python` is not recognized | Reinstall Python and tick **Add python.exe to PATH** |
| `File not found` | Pass the real path, including `.png` or `.jpg` |
| Timed out after 900 seconds | CPU reading is slow. Keep Ollama open and run the same photo again |
| Medicine name looks wrong | The photo was misread. Check `output\last_ocr.txt` against the paper |

---

# Part B — Fine-tune GLM-OCR (GPU computer only)

Skip this until Part A works and you have many photos with the correct text written by a person.

Do not train on this laptop. Training needs an NVIDIA GPU:

| Method | GPU memory | Command |
|--------|------------|---------|
| LoRA (use this) | 8 GB or more | `bash scripts/train_glm_ocr.sh lora` |
| Full training | 24 GB or more | `bash scripts/train_glm_ocr.sh full` |

## Why this project uses LLaMA-Factory

LLaMA-Factory is not used when you read a photo on this laptop. Ollama does that.

Fine-tuning means showing GLM-OCR your prescription photos and the correct text. Someone would otherwise have to write the whole training program: load the model, pair each photo with its label, run LoRA, save checkpoints, and merge the result. LLaMA-Factory already does those steps. The official GLM-OCR fine-tune guide uses it, so this project uses it too.

The work is split like this:

1. Your scripts turn `labels.csv` and the photos into `data\glm_finetune\prescriptions.json`.
2. LLaMA-Factory reads that file and trains the model `zai-org/GLM-OCR`.
3. The training command is `llamafactory-cli train`.
4. The settings are in `finetune_glm\configs\lora.yaml`: LoRA, the `glm_ocr` template, and the prompt `Text Recognition:`.

The name comes from an older tool built for LLaMA models. In this project it trains GLM-OCR. It does not download or use a LLaMA model.

Install it only on the GPU computer. `scripts/setup_glm_finetune.sh` clones it into `third_party/LLaMA-Factory` and installs it inside `.venv_finetune`. This laptop does not need it.

## Step 8 — On this laptop, prepare labels only

1. Put photos in `data\glm_finetune\images\`
2. Copy `data\glm_finetune\labels.example.csv` to `data\glm_finetune\labels.csv` if that file is not already filled in.
3. Each row is the file name and the correct text:

```csv
image,text
example_ascard.png,"Tab Ascard 75mg OD"
```

`image` is the file name inside `data\glm_finetune\images\`. `text` is exactly what the photo should say.

4. Build the training file:

```powershell
.\.venv\Scripts\python.exe src\prepare_glm_finetune.py
```

Or:

```powershell
scripts\prepare_glm_finetune.bat
```

That writes `data\glm_finetune\prescriptions.json`.

---

## Step 9 — On the GPU computer, install and train

Copy or git-pull this project onto the GPU machine, then:

```bash
cd Handwrittenprescriptionreader
bash scripts/setup_glm_finetune.sh
python src/prepare_glm_finetune.py
bash scripts/train_glm_ocr.sh lora
bash scripts/export_glm_lora.sh
```

`setup_glm_finetune.sh` creates a separate environment (`.venv_finetune`), installs the training libraries, and clones LLaMA-Factory. It does not replace the laptop `.venv`.

When training finishes:

- LoRA adapter: `third_party/LLaMA-Factory/saves/glm-ocr/lora/sft`
- Merged model: `models/glm-ocr-finetuned/`

More detail is in `finetune_glm/README.md`.

---

## Everyday checklist

1. Open the Ollama app.
2. Open PowerShell in this project folder.
3. Run `scripts\run_local_ocr.bat` with your photo.
4. Read `output\last_ocr.txt` and `output\last_result.json`.
5. A pharmacist or doctor checks the answer.

---

# All commands, one by one

Open PowerShell for the laptop commands. Open a terminal on the GPU computer for the Linux commands. Run each command, wait until it finishes, then run the next one.

## This laptop — first time only

1. Check Python:

```powershell
python --version
```

2. Go to the project folder:

```powershell
cd C:\Users\HP\source\repos\Handwrittenprescriptionreader
```

3. Create the private Python folder:

```powershell
python -m venv .venv
```

4. Update pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

5. Install the three packages:

```powershell
.\.venv\Scripts\python.exe -m pip install pillow rapidfuzz requests
```

6. Download GLM-OCR (Ollama app must be open):

```powershell
ollama pull glm-ocr:latest
```

7. Confirm the model is installed:

```powershell
ollama list
```

8. Check Ollama and the project together:

```powershell
scripts\check_local_ocr.bat
```

You want the line `Setup OK`.

## This laptop — every photo

9. Read one prescription. Change `Test1.png` to your file name:

```powershell
scripts\run_local_ocr.bat data\raw\Test1.png
```

The same job without the batch file:

```powershell
.\.venv\Scripts\python.exe src\pipeline.py data\raw\Test1.png
```

## This laptop — prepare fine-tune labels

Do this only when you are ready for Part B. Put the photos in `data\glm_finetune\images\` and fill `data\glm_finetune\labels.csv` first.

10. If `labels.csv` does not exist yet, copy the example:

```powershell
copy data\glm_finetune\labels.example.csv data\glm_finetune\labels.csv
```

11. Build the training JSON:

```powershell
.\.venv\Scripts\python.exe src\prepare_glm_finetune.py
```

The same job as a batch file:

```powershell
scripts\prepare_glm_finetune.bat
```

## GPU computer — install LLaMA-Factory and train

Do not run these on the laptop. From the project folder on the GPU machine:

12. One-time setup. This creates `.venv_finetune`, installs the training libraries, and clones LLaMA-Factory:

```bash
bash scripts/setup_glm_finetune.sh
```

That script runs these commands for you:

```bash
python3 -m venv .venv_finetune
source .venv_finetune/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r finetune_glm/requirements-finetune.txt
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
python -m pip install -e third_party/LLaMA-Factory
python -m pip install -U "transformers>=5.3.0"
python src/prepare_glm_finetune.py
```

13. Build the dataset again after the photos and labels are on this machine:

```bash
python src/prepare_glm_finetune.py
```

14. Train with LoRA (the method to use):

```bash
bash scripts/train_glm_ocr.sh lora
```

That script runs LLaMA-Factory:

```bash
llamafactory-cli train glm_ocr_lora_prescriptions.yaml
```

Full training, only if the GPU has about 24 GB of memory:

```bash
bash scripts/train_glm_ocr.sh full
```

15. Merge the LoRA result into one model folder:

```bash
bash scripts/export_glm_lora.sh
```

That script runs:

```bash
llamafactory-cli export --model_name_or_path zai-org/GLM-OCR --adapter_name_or_path third_party/LLaMA-Factory/saves/glm-ocr/lora/sft --template glm_ocr --export_dir models/glm-ocr-finetuned --trust_remote_code true
```

The merged model is `models/glm-ocr-finetuned/`. The LoRA adapter is `third_party/LLaMA-Factory/saves/glm-ocr/lora/sft`.
