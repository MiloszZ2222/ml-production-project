@echo off
cd /d "%~dp0.."

set AWS_PROFILE=ml-project
set AWS_DEFAULT_REGION=eu-north-1

if not exist "..\mlflow-storage" mkdir "..\mlflow-storage"

".venv\Scripts\mlflow.exe" server --backend-store-uri "sqlite:///../mlflow-storage/mlflow.db" --artifacts-destination "s3://ml-production-private-858525799461-eu-north-1-an/mlflow-artifacts/" --serve-artifacts --host 127.0.0.1 --port 5000

pause
