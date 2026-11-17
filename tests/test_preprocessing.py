import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor


def test_complete_preprocessor(house_data_factory):
    X = house_data_factory()
    X_original = X.copy(deep=True)

    preprocessor = build_preprocessor()

    # Verify that every input feature is accounted for
    selected_columns = set()

    for _, _, columns in preprocessor.transformers:
        selected_columns.update(columns)

    assert len(X.columns) == 79
    assert selected_columns == set(X.columns)

    result = preprocessor.fit_transform(X)
    names = preprocessor.get_feature_names_out()

    assert result.shape[0] == len(X)
    assert result.shape[1] == len(names)
    assert len(names) == len(set(names))
    assert np.isfinite(result).all()

    # Verify that preprocessing did not modify input data
    pd.testing.assert_frame_equal(X, X_original)

    result_df = pd.DataFrame(result, columns=names)

    # Verify key engineered features
    np.testing.assert_allclose(
        result_df["HouseAge"],
        [10, 3, 0],
    )

    np.testing.assert_allclose(
        result_df["YearsSinceRemodel"],
        [5, 1, 0],
    )

    np.testing.assert_allclose(
        result_df["TotalSF"],
        [2300, 1200, 2800],
    )

    np.testing.assert_allclose(
        result_df["TotalBathrooms"],
        [2.5, 2.0, 4.0],
    )

    np.testing.assert_allclose(
        result_df["MasVnrArea"],
        [0.0, 100.0, 100.0],
    )


def test_preprocessor_handles_new_data(house_data_factory):
    X_train = house_data_factory()

    preprocessor = build_preprocessor()
    train_result = preprocessor.fit_transform(X_train)

    # Save feature names learned during training
    feature_names_before = preprocessor.get_feature_names_out()

    # Simulate a new observation
    X_new = X_train.iloc[[0]].copy()

    # Category not observed during training
    X_new["Neighborhood"] = "NewNeighborhood"

    # Missing numerical value
    X_new["LotFrontage"] = np.nan

    # Transform new data without fitting again
    new_result = preprocessor.transform(X_new)

    feature_names_after = preprocessor.get_feature_names_out()

    # Verify consistent number of output features
    assert new_result.shape == (1, train_result.shape[1])

    # Verify that output contains only finite values
    assert np.isfinite(new_result).all()

    # Verify that feature names and their order remain unchanged
    np.testing.assert_array_equal(
        feature_names_before,
        feature_names_after,
    )


def test_preprocessor_on_larger_synthetic_data(house_data_factory):
    X = house_data_factory(n_samples=150)

    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    assert result.shape[0] == 150
    assert result.shape[1] == len(feature_names)

    assert len(feature_names) == len(set(feature_names))
    assert np.isfinite(result).all()
