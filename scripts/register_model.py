import argparse
import os

import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "house-price-predictor"


def main():
    parser = argparse.ArgumentParser(description="Register an MLflow model.")
    parser.add_argument(
        "--model-uri",
        required=True,
        help="URI of the logged model to register.",
    )
    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="Assign the champion alias to the new version.",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    model_version = mlflow.register_model(
        model_uri=args.model_uri,
        name=MODEL_NAME,
    )

    print(f"Registered model: {MODEL_NAME}")
    print(f"Version: {model_version.version}")

    if args.set_champion:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=model_version.version,
        )
        print("Alias assigned: champion")


if __name__ == "__main__":
    main()
