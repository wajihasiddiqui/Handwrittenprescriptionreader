@echo off
REM Build ShareGPT dataset for GLM-OCR / LLaMA-Factory
setlocal
cd /d "%~dp0.."

if not exist "data\glm_finetune\labels.csv" (
  copy /Y "data\glm_finetune\labels.example.csv" "data\glm_finetune\labels.csv" >nul
  echo Created data\glm_finetune\labels.csv
  echo Edit image,text rows, put files in data\glm_finetune\images\ or data\raw\
  echo Then re-run this script.
)

python src\prepare_finetune.py
echo.
echo Train on GPU:
echo   python src\llamafactory_train.py setup
echo   python src\llamafactory_train.py train --mode lora
exit /b %ERRORLEVEL%
