@echo off
REM Run OCR in-process with LLaMA-Factory (no API server)
setlocal
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: scripts\run_local_ocr.bat data\raw\Test1.png
  exit /b 1
)

if exist ".venv_finetune\Scripts\python.exe" (
  set "PY=.venv_finetune\Scripts\python.exe"
) else (
  set "PY=.\.venv\Scripts\python.exe"
)

echo Running in-process GLM-OCR: %~1
"%PY%" src\pipeline.py "%~1"
exit /b %ERRORLEVEL%
