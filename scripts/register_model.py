import argparse

import mlflow
from mlflow import MlflowClient

from src.config import settings


def main() -> None:
    """Register an MLflow model and optionally assign the champion alias."""

    parser = argparse.ArgumentParser(description="Register an MLflow model.")

    parser.add_argument(
        "--model-uri",
        required=True,
        help="URI of the logged model to register.",
    )

    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="Assign the configured model alias to the new version.",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    client = MlflowClient()

    model_version = mlflow.register_model(
        model_uri=args.model_uri,
        name=settings.mlflow.model_name,
    )

    print(f"Registered model: {settings.mlflow.model_name}")
    print(f"Version: {model_version.version}")

    if args.set_champion:
        client.set_registered_model_alias(
            name=settings.mlflow.model_name,
            alias=settings.mlflow.model_alias,
            version=model_version.version,
        )

        print(f"Alias assigned: {settings.mlflow.model_alias}")


if __name__ == "__main__":
    main()
