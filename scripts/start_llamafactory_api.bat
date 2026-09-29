@echo off
REM Deprecated: server mode is optional now. Model loads inside pipeline.py
setlocal
cd /d "%~dp0.."
echo NOTE: Server is no longer required.
echo The model loads in-process when you run:
echo   scripts\run_local_ocr.bat data\raw\Test1.png
echo.
echo If you still want the API server:
python src\llamafactory_serve.py %*
exit /b %ERRORLEVEL%
