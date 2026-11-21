import os
import subprocess
import sys
from urllib.parse import urlparse

from src.config import PROJECT_ROOT, settings


def main() -> None:
    """Start the MLflow server using centralized configuration."""

    tracking_uri = urlparse(settings.mlflow.tracking_uri)

    if (
        tracking_uri.scheme != "http"
        or tracking_uri.hostname not in {"127.0.0.1", "localhost"}
        or tracking_uri.port is None
        or tracking_uri.path not in {"", "/"}
    ):
        raise ValueError(
            "The local MLflow server requires a tracking URI "
            "such as http://127.0.0.1:5000."
        )

    host = tracking_uri.hostname
    port = tracking_uri.port

    # Resolve the SQLite database path relative to the project root.
    db_path = (PROJECT_ROOT / settings.mlflow.backend_store_path).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    backend_store_uri = f"sqlite:///{db_path.as_posix()}"

    artifact_destination = (
        f"s3://{settings.aws.private_bucket}/"
        f"{settings.mlflow.artifact_prefix.strip('/')}/"
    )

    # Configure AWS for the MLflow server process.
    env = os.environ.copy()

    if settings.aws.profile:
        env["AWS_PROFILE"] = settings.aws.profile
    else:
        env.pop("AWS_PROFILE", None)

    env["AWS_DEFAULT_REGION"] = settings.aws.region

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--artifacts-destination",
        artifact_destination,
        "--serve-artifacts",
        "--host",
        host,
        "--port",
        str(port),
    ]

    print("Starting MLflow server...", flush=True)
    print(f"Tracking URI: {settings.mlflow.tracking_uri}", flush=True)
    print(f"Backend store: {db_path}", flush=True)
    print(f"Artifact destination: {artifact_destination}", flush=True)
    print(f"AWS region: {settings.aws.region}", flush=True)

    subprocess.run(
        command,
        env=env,
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
