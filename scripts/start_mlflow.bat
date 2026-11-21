
@echo off
cd /d "%~dp0.."

".venv\Scripts\python.exe" -m scripts.start_mlflow

pause