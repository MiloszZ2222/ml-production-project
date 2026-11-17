import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.model_selection import train_test_split

from src.data import load_house_prices
from src.train import BEST_PARAMS, RANDOM_STATE, train_model


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("house-prices-xgboost")

    df = load_house_prices(source="private")

    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    with mlflow.start_run(run_name="xgboost-tuned-full-pipeline"):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)

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
            code_paths=["src"],
        )

        print(f"RMSLE: {rmsle:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()
