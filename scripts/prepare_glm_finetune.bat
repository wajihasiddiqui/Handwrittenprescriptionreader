@echo off
REM Laptop helper: prepare labels only. Training must run on a GPU Linux/Windows machine.
setlocal
cd /d "%~dp0.."

if not exist "data\glm_finetune\labels.csv" (
  copy /Y "data\glm_finetune\labels.example.csv" "data\glm_finetune\labels.csv" >nul
  echo Created data\glm_finetune\labels.csv — edit it, then re-run.
)

".\.venv\Scripts\python.exe" src\prepare_glm_finetune.py
echo.
echo On the GPU machine after git push/pull:
echo   bash scripts/setup_glm_finetune.sh
echo   bash scripts/train_glm_ocr.sh lora
echo   bash scripts/export_glm_lora.sh
exit /b %ERRORLEVEL%
