@echo off
REM Start LLaMA-Factory API for GLM-OCR (keep this window open)
setlocal
cd /d "%~dp0.."
echo Starting LLaMA-Factory GLM-OCR API on http://127.0.0.1:8000
echo Keep this window open, then run pipeline in another terminal.
python src\llamafactory_serve.py %*
exit /b %ERRORLEVEL%
