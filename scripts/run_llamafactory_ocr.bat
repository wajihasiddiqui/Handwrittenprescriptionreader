@echo off
REM Run pipeline using LLaMA-Factory OCR (API must already be running)
setlocal
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: scripts\run_llamafactory_ocr.bat data\raw\Test1.png
  echo First start API: scripts\start_llamafactory_api.bat
  exit /b 1
)
set OCR_BACKEND=llamafactory
".\.venv\Scripts\python.exe" src\pipeline.py "%~1"
exit /b %ERRORLEVEL%
