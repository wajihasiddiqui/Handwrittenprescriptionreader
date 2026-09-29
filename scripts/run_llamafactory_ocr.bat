@echo off
REM Same as run_local_ocr.bat — in-process LLaMA-Factory OCR
setlocal
cd /d "%~dp0.."
call "%~dp0run_local_ocr.bat" %*
exit /b %ERRORLEVEL%
