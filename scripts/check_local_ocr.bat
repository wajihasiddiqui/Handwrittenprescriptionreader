@echo off
REM Check LLaMA-Factory OCR setup (no Ollama)
setlocal
cd /d "%~dp0.."

echo === Finetune venv / LLaMA-Factory ===
if not exist ".venv_finetune\Scripts\python.exe" (
  echo Missing .venv_finetune
  echo Run: python src\llamafactory_train.py setup
  exit /b 1
)
if not exist "third_party\LLaMA-Factory" (
  echo Missing third_party\LLaMA-Factory
  echo Run: python src\llamafactory_train.py setup
  exit /b 1
)
echo OK: LLaMA-Factory present

echo.
echo === API http://127.0.0.1:8000/v1/models ===
powershell -NoProfile -Command "try { (Invoke-RestMethod http://127.0.0.1:8000/v1/models) | Out-String; exit 0 } catch { Write-Host 'API not running'; Write-Host 'Start: python src\llamafactory_serve.py'; exit 1 }"
if errorlevel 1 exit /b 1

echo.
echo === Project OCR backend ===
".\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'src'); from ocr_engine import get_backend; print('backend', get_backend())"

echo.
echo Setup OK. Run pipeline in another terminal while API stays open:
echo   .\.venv\Scripts\python.exe src\pipeline.py data\raw\YourImage.png
exit /b 0
