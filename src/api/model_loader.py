import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import mlflow.sklearn
from sklearn.pipeline import Pipeline

from src.config import settings


def load_model() -> Pipeline:
    """Download and load the approved model from S3."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        archive_path = temp_path / "model.zip"
        model_path = temp_path / "model"

        # Download model from S3
        with urlopen(
            settings.deployment.model_url,
            timeout=settings.deployment.download_timeout,
        ) as response, archive_path.open("wb") as file:
            shutil.copyfileobj(response, file)

        # Verify model integrity
        actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()

        if actual_hash != settings.deployment.model_sha256:
            raise ValueError("Model SHA-256 verification failed.")

        # Extract model archive
        model_path.mkdir()

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(model_path)

        # Load MLflow model
        pipeline = mlflow.sklearn.load_model(str(model_path))

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded model is not a sklearn Pipeline.")

    return pipeline
