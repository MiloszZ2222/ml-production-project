import mlflow
import mlflow.sklearn
import numpy as np

from src.data import load_house_prices

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "house-prices-xgboost"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    # Find the experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    # Find successfully completed runs, newest first
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        raise RuntimeError("No finished runs found.")

    # Find a model associated with one of the runs
    model_uri = None

    for run_id in runs["run_id"]:
        models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
            output_format="list",
        )

        if models:
            model_uri = models[0].model_uri
            break

    if model_uri is None:
        raise RuntimeError("No logged model found.")

    # Load the trained pipeline from MLflow
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Load input data from S3
    df = load_house_prices(source="private")
    X = df.drop(columns=["Id", "SalePrice"])

    # Generate predictions without retraining
    predictions = pipeline.predict(X.head(5))

    print("Model URI:", model_uri)
    print("Predictions:", np.round(predictions, 2))


if __name__ == "__main__":
    main()
