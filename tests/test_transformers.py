import numpy as np
import pytest

from src.transformers import PositiveMedianImputer


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
