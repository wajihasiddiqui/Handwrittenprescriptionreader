@echo off
REM Check Ollama + glm-ocr:latest for local OCR
setlocal
cd /d "%~dp0.."

where ollama >nul 2>&1
if errorlevel 1 (
  echo Ollama not found in PATH.
  echo Install from https://ollama.com/download
  exit /b 1
)

echo === ollama version ===
ollama --version

echo.
echo === installed models ===
ollama list

echo.
echo Pull latest GLM-OCR if missing:
echo   ollama pull glm-ocr:latest
echo.
echo configs\ocr.json backend should be: ollama
echo model: glm-ocr:latest
echo.
echo Test:
echo   python src\pipeline.py data\raw\YourImage.png
exit /b 0
