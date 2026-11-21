from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data import load_house_prices
from src.train import train_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Train, evaluate, and log the model using MLflow."""

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    df = load_house_prices(source="private")

    X = df.drop(
        columns=[
            settings.data.id_column,
            settings.data.target_column,
        ]
    )
    y = df[settings.data.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.training.test_size,
        random_state=settings.training.random_state,
    )

    with mlflow.start_run(run_name="xgboost-tuned-full-pipeline"):
        mlflow.log_params(settings.training.model_params)
        mlflow.log_param(
            "random_state",
            settings.training.random_state,
        )
        mlflow.log_param(
            "test_size",
            settings.training.test_size,
        )

        pipeline = train_model(X_train, y_train)

        predictions = pipeline.predict(X_test)
        predictions_positive = np.maximum(predictions, 0)

        rmsle = np.sqrt(mean_squared_log_error(y_test, predictions_positive))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        mlflow.log_metrics(
            {
                "rmsle": rmsle,
                "rmse": rmse,
                "mae": mae,
            }
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="house-price-pipeline",
            serialization_format="cloudpickle",
            code_paths=[str(PROJECT_ROOT / "src")],
        )

        print(f"RMSLE: {rmsle:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()
