@echo off
setlocal
cd /d "%~dp0.."
if not exist "data\glm_finetune\labels.csv" (
  copy /Y "data\glm_finetune\labels.example.csv" "data\glm_finetune\labels.csv" >nul
  echo Created data\glm_finetune\labels.csv — edit it with real image names + text.
)
python src\llamafactory_train.py prepare
echo.
echo Next on GPU machine:
echo   python src\llamafactory_train.py setup
echo   python src\llamafactory_train.py train --mode lora
echo   python src\llamafactory_train.py export
exit /b %ERRORLEVEL%
