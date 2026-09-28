@echo off
REM Run OCR pipeline (LLaMA-Factory API must already be running)
setlocal
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: scripts\run_local_ocr.bat data\raw\Test1.png
  echo First start API: scripts\start_llamafactory_api.bat
  exit /b 1
)
powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:8000/v1/models | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  echo LLaMA-Factory API is not running.
  echo Start it first: scripts\start_llamafactory_api.bat
  exit /b 1
)
echo Running: %~1
".\.venv\Scripts\python.exe" src\pipeline.py "%~1"
exit /b %ERRORLEVEL%
