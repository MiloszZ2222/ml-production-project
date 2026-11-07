import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PositiveMedianImputer(TransformerMixin, BaseEstimator):
    """Impute missing values using the median of positive observations."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)

        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("Expected a two-dimensional input with one column.")

        positive_values = values[(values > 0) & np.isfinite(values)]

        if positive_values.size == 0:
            raise ValueError("No positive finite values available.")

        self.median_ = np.median(positive_values)
        self.n_features_in_ = 1

        return self

    def transform(self, X):
        check_is_fitted(self, "median_")

        values = np.asarray(X, dtype=float).copy()

        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("Expected a two-dimensional input with one column.")

        missing_indicator = np.isnan(values).astype(int)

        values[np.isnan(values)] = self.median_

        return np.hstack([values, missing_indicator])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "median_")

        feature_name = "MasVnrArea" if input_features is None else input_features[0]

        return np.array([feature_name, f"{feature_name}_missing"], dtype=object)
