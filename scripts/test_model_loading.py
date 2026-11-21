import mlflow
import mlflow.sklearn
import numpy as np

from src.config import settings
from src.data import load_house_prices


def main() -> None:
    """Load the champion model from MLflow and generate predictions."""

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    model_uri = (
        f"models:/{settings.mlflow.model_name}" f"@{settings.mlflow.model_alias}"
    )

    # Load the registered champion model
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Load input data from S3
    df = load_house_prices(source="private")

    X = df.drop(
        columns=[
            settings.data.id_column,
            settings.data.target_column,
        ]
    )

    # Generate predictions without retraining
    predictions = pipeline.predict(X.head(5))

    print(f"Model URI: {model_uri}")
    print(f"Predictions: {np.round(predictions, 2)}")


if __name__ == "__main__":
    main()
