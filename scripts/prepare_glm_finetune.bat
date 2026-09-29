@echo off
REM Alias → scripts\prepare_finetune.bat
setlocal
cd /d "%~dp0.."
call "%~dp0prepare_finetune.bat"
exit /b %ERRORLEVEL%
