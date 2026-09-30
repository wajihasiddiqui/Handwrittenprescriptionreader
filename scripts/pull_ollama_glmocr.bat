@echo off
REM Pull official GLM-OCR into Ollama (inference only — training uses LLaMA-Factory)
setlocal
cd /d "%~dp0.."

where ollama >nul 2>&1
if errorlevel 1 (
  echo Install Ollama first: https://ollama.com/download
  exit /b 1
)

echo Pulling glm-ocr:latest ...
ollama pull glm-ocr:latest
if errorlevel 1 exit /b 1

echo.
echo Done. OCR backend is already set to ollama in configs\ocr.json
echo Run: python src\pipeline.py data\raw\YourImage.png
echo.
echo Fine-tune still uses LLaMA-Factory (not Ollama):
echo   python src\llamafactory_train.py setup
echo   python src\llamafactory_train.py train --mode lora
echo   python src\llamafactory_train.py export
echo   python src\llamafactory_train.py ollama --name glm-ocr-rx
exit /b 0
