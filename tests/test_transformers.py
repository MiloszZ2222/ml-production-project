import numpy as np
import pandas as pd
import pytest

from src.transformers import MasonryTypeImputer, PositiveMedianImputer


def test_positive_median_imputer():
    X = np.array([[0.0], [100.0], [200.0], [300.0], [np.nan]])

    transformer = PositiveMedianImputer()

    result = transformer.fit_transform(X)

    expected = np.array(
        [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0], [200.0, 1.0]]
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        [[0.0], [0.0], [0.0]],
        [[np.nan], [np.nan], [np.nan]],
        [[-10.0], [-20.0], [0.0]],
    ],
)
def test_positive_median_imputer_without_positive_values(values):
    X = np.array(values)

    transformer = PositiveMedianImputer()

    with pytest.raises(ValueError, match="No positive finite values available"):
        transformer.fit(X)


@pytest.mark.parametrize(
    "X", [np.array([100.0, 200.0, np.nan]), np.array([[100.0, 10.0], [200.0, 20.0]])]
)
def test_positive_median_imputer_invalid_shape(X):
    transformer = PositiveMedianImputer()

    with pytest.raises(
        ValueError, match="Expected a two-dimensional input with one column"
    ):
        transformer.fit(X)


@pytest.mark.parametrize(
    "masonry_type, masonry_area, expected",
    [
        (np.nan, 0.0, "None"),
        (np.nan, 150.0, "Unknown"),
        (np.nan, np.nan, "Unknown"),
        ("Stone", 200.0, "Stone"),
    ],
)
def test_masonry_type_imputer(masonry_type, masonry_area, expected):
    X = pd.DataFrame({"MasVnrType": [masonry_type], "MasVnrArea": [masonry_area]})

    transformer = MasonryTypeImputer()
    result = transformer.fit_transform(X)

    assert result.shape == (1, 1)
    assert result[0, 0] == expected


def test_masonry_type_imputer_does_not_modify_input():
    X = pd.DataFrame(
        {
            "MasVnrType": [np.nan, "Stone", np.nan],
            "MasVnrArea": [0.0, 200.0, 150.0],
        }
    )
    # Save the original input data
    X_original = X.copy(deep=True)

    transformer = MasonryTypeImputer()
    transformer.fit_transform(X)

    # Verify that the original DataFrame was not modified
    pd.testing.assert_frame_equal(X, X_original)


def test_masonry_type_imputer_invalid_input_type():
    X = np.array(
        [
            ["Stone", 200.0],
            ["None", 0.0],
        ],
        dtype=object,
    )

    transformer = MasonryTypeImputer()

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        transformer.fit(X)


def test_masonry_type_imputer_missing_columns():
    X = pd.DataFrame({"MasVnrType": ["Stone", "None"]})

    transformer = MasonryTypeImputer()

    with pytest.raises(ValueError, match="Required masonry columns are missing"):
        transformer.fit(X)
