import numpy as np
import pandas as pd

from src.preprocessing import (
    NOMINAL_ABSENCE_COLUMNS,
    NOMINAL_REGULAR_COLUMNS,
    NUMERIC_PASSTHROUGH_COLUMNS,
    ORDINAL_CATEGORIES,
    build_preprocessor,
)


def make_sample_house_data():
    # Start with all remaining numeric features
    data = {column: [1.0, 2.0, 3.0] for column in NUMERIC_PASSTHROUGH_COLUMNS}

    # Provide valid ordinal categories
    for column, categories in ORDINAL_CATEGORIES.items():
        data[column] = [categories[-1]] * 3

    # Provide sample nominal categories
    for column in NOMINAL_REGULAR_COLUMNS:
        data[column] = ["Sample"] * 3

    for column in NOMINAL_ABSENCE_COLUMNS:
        data[column] = [np.nan, "Present", np.nan]

    # Custom transformer inputs
    data.update(
        {
            "MasVnrArea": [0.0, 100.0, np.nan],
            "MasVnrType": [np.nan, "Stone", np.nan],
            "YrSold": [2010, 2008, 2007],
            "YearBuilt": [2000, 2005, 2007],
            "YearRemodAdd": [2005, 2007, 2008],
            "GrLivArea": [1500, 1200, 1800],
            "TotalBsmtSF": [800, 0, 1000],
            "FullBath": [2, 1, 2],
            "HalfBath": [1, 0, 1],
            "BsmtFullBath": [0, 1, 1],
            "BsmtHalfBath": [0, 0, 1],
            "LotFrontage": [50.0, np.nan, 90.0],
            "LotArea": [8000, 9000, 10000],
            "1stFlrSF": [1000, 1100, 1200],
            "GarageYrBlt": [2000.0, np.nan, 1995.0],
            "MSSubClass": [20, 60, 30],
            "Electrical": ["SBrkr", np.nan, "SBrkr"],
        }
    )

    return pd.DataFrame(data)


def test_complete_preprocessor():
    X = make_sample_house_data()
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


def test_preprocessor_handles_new_data():
    X_train = make_sample_house_data()

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


def test_preprocessor_on_real_data():
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"

    df = pd.read_csv(data_path)

    X = df.drop(columns=["Id", "SalePrice"])

    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    assert result.shape[0] == len(X)
    assert result.shape[1] == len(feature_names)

    assert len(feature_names) == len(set(feature_names))
    assert np.isfinite(result).all()
