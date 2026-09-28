@echo off
REM Verify local GLM-OCR + Ollama setup for this project
setlocal
cd /d "%~dp0.."

set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%OLLAMA%" set "OLLAMA=ollama"

echo === Ollama ===
"%OLLAMA%" --version
if errorlevel 1 (
  echo Ollama not found. Install from https://ollama.com/download
  exit /b 1
)

echo.
echo === Models ===
"%OLLAMA%" list

echo.
echo === API ===
powershell -NoProfile -Command "try { (Invoke-RestMethod http://127.0.0.1:11434/api/tags).models | ForEach-Object { $_.name }; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 (
  echo Starting Ollama serve...
  start "" "%OLLAMA%" serve
  timeout /t 4 /nobreak >nul
)

echo.
echo === Project Python ===
".\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'src'); from ocr_engine import get_backend, _check_ollama; print('backend', get_backend()); _check_ollama(); print('Ollama + glm-ocr OK')"
if errorlevel 1 exit /b 1

echo.
echo Setup OK. Run:
echo   .\.venv\Scripts\python.exe src\pipeline.py data\raw\YourImage.png
exit /b 0
