import joblib
import numpy as np

from sklearn.pipeline import Pipeline

from src.train import save_model, train_model


def make_sample_target(X):
    return 80_000 + 120 * X["GrLivArea"] + 2 * X["LotArea"]


def test_train_model_produces_valid_predictions(house_data_factory):
    X = house_data_factory(n_samples=110)
    y = make_sample_target(X)

    X_train = X.iloc[:100]
    y_train = y.iloc[:100]

    X_test = X.iloc[100:]

    pipeline = train_model(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert isinstance(pipeline, Pipeline)
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()


def test_model_serialization(tmp_path, house_data_factory):
    X = house_data_factory(n_samples=110)
    y = make_sample_target(X)

    X_train = X.iloc[:100]
    y_train = y.iloc[:100]

    X_new = X.iloc[100:]

    pipeline = train_model(X_train, y_train)

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
