@echo off
REM Auto-OCR images with GLM-OCR → update labels.csv → ShareGPT for LLaMA-Factory
setlocal
cd /d "%~dp0.."

if not exist "data\glm_finetune\images" mkdir "data\glm_finetune\images"

echo Put training images in: data\glm_finetune\images\
echo Then this script runs GLM-OCR and fills labels.csv automatically.
echo.

REM Prefer finetune venv (has LLaMA-Factory) when present
set "PY=python"
if exist ".venv_finetune\Scripts\python.exe" set "PY=.venv_finetune\Scripts\python.exe"

"%PY%" src\prepare_finetune.py %*
echo.
echo Review labels if needed:
echo   notepad data\glm_finetune\labels.csv
echo Train on GPU:
echo   python src\llamafactory_train.py setup
echo   python src\llamafactory_train.py train --mode lora
exit /b %ERRORLEVEL%
