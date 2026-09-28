@echo off
REM One-click local OCR pipeline (Ollama + GLM-OCR on this laptop)
setlocal
cd /d "%~dp0.."
set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"

if "%~1"=="" (
  echo Usage: scripts\run_local_ocr.bat data\raw\Test1.png
  exit /b 1
)

where ollama >nul 2>&1
if errorlevel 1 (
  echo Ollama not found. Install from https://ollama.com/download
  exit /b 1
)

ollama list | findstr /I "glm-ocr" >nul
if errorlevel 1 (
  echo Pulling glm-ocr:latest ...
  ollama pull glm-ocr:latest
)

echo Running: %~1
".\.venv\Scripts\python.exe" src\pipeline.py "%~1"
exit /b %ERRORLEVEL%
