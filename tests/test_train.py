from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train import save_model, train_model

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"


def test_train_model_produces_valid_predictions():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    X_small = X.iloc[:100]
    y_small = y.iloc[:100]

    pipeline = train_model(X_small, y_small)

    predictions = pipeline.predict(X.iloc[100:110])

    assert isinstance(pipeline, Pipeline)
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()


def test_model_serialization(tmp_path):
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    pipeline = train_model(X.iloc[:100], y.iloc[:100])

    X_new = X.iloc[100:110]
    predictions_before = pipeline.predict(X_new)

    model_path = tmp_path / "house_price_pipeline.joblib"

    save_model(pipeline, model_path)

    loaded_pipeline = joblib.load(model_path)
    predictions_after = loaded_pipeline.predict(X_new)

    assert model_path.exists()
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )
