@echo off
REM Fine-tune GLM-OCR from labels.csv using LLaMA-Factory
setlocal
cd /d "%~dp0.."

echo === 1) Setup (venv + LLaMA-Factory) ===
python src\llamafactory_train.py setup
if errorlevel 1 exit /b 1

echo.
echo === 2) Prepare ShareGPT from labels.csv ===
echo Put images in data\glm_finetune\images\
echo labels file: data\glm_finetune\labels.csv  (columns: image,text,task)
python src\llamafactory_train.py prepare --include-raw
if errorlevel 1 exit /b 1

echo.
echo === 3) Train LoRA ===
python src\llamafactory_train.py train --mode lora --include-raw --skip-prepare
if errorlevel 1 exit /b 1

echo.
echo === 4) Export merged model ===
python src\llamafactory_train.py export
if errorlevel 1 exit /b 1

echo.
echo Done. Optional Ollama pack:
echo   python src\llamafactory_train.py ollama --name glm-ocr-rx
exit /b 0
