@echo off
REM Check in-process LLaMA-Factory OCR setup (no server)
setlocal
cd /d "%~dp0.."

echo === LLaMA-Factory install ===
if exist ".venv_finetune\Scripts\python.exe" (
  set "PY=.venv_finetune\Scripts\python.exe"
) else (
  set "PY=.\.venv\Scripts\python.exe"
)

"%PY%" -c "import sys; sys.path.insert(0,'src'); from ocr_engine import _ensure_llamafactory_importable, get_backend; _ensure_llamafactory_importable(); print('backend', get_backend()); print('LLaMA-Factory import OK')"
if errorlevel 1 (
  echo.
  echo Run setup first:
  echo   python src\llamafactory_train.py setup
  exit /b 1
)

echo.
echo Setup OK. Single command run:
echo   %PY% src\pipeline.py data\raw\YourImage.png
exit /b 0
