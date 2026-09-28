@echo off
REM LLaMA-Factory GLM-OCR fine-tune helpers (GPU machine recommended)
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
  echo Usage:
  echo   scripts\llamafactory.bat setup
  echo   scripts\llamafactory.bat prepare
  echo   scripts\llamafactory.bat train
  echo   scripts\llamafactory.bat train full
  echo   scripts\llamafactory.bat export
  exit /b 1
)

set "CMD=%~1"
if /I "%CMD%"=="setup" (
  python src\llamafactory_train.py setup
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="prepare" (
  python src\llamafactory_train.py prepare
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="train" (
  if /I "%~2"=="full" (
    python src\llamafactory_train.py train --mode full
  ) else (
    python src\llamafactory_train.py train --mode lora
  )
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="export" (
  python src\llamafactory_train.py export
  exit /b %ERRORLEVEL%
)

echo Unknown command: %CMD%
exit /b 1
