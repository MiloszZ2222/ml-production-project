import numpy as np
import pandas as pd
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


class MasonryTypeImputer(TransformerMixin, BaseEstimator):
    """Impute masonry type based on the original masonry area."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        required = ["MasVnrType", "MasVnrArea"]

        if not all(col in X.columns for col in required):
            raise ValueError("Required masonry columns are missing.")

        self.n_features_in_ = 2
        self.feature_names_in_ = np.array(required, dtype=object)

        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")

        data = X[self.feature_names_in_].copy()
        # Ensure the categorical column can store string values
        data["MasVnrType"] = data["MasVnrType"].astype(object)

        missing_type = data["MasVnrType"].isna()

        no_veneer = missing_type & (data["MasVnrArea"] == 0)

        data.loc[no_veneer, "MasVnrType"] = "None"

        data.loc[missing_type & ~no_veneer, "MasVnrType"] = "Unknown"

        return data[["MasVnrType"]].to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")

        return np.asarray(["MasVnrType"], dtype=object)


class AgeFeaturesTransformer(TransformerMixin, BaseEstimator):
    """Create property age and years-since-remodeling features."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        required = ["YrSold", "YearBuilt", "YearRemodAdd"]

        if not all(col in X.columns for col in required):
            raise ValueError("Required age columns are missing.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")

        X = X.copy()

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

        X["YearsSinceRemodel"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)

        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")

        if input_features is None:
            input_features = self.feature_names_in_

        return np.asarray(
            [*input_features, "HouseAge", "YearsSinceRemodel"], dtype=object
        )


class AreaFeaturesTransformer(TransformerMixin, BaseEstimator):
    """Create total above-ground and basement area feature."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        required = ["GrLivArea", "TotalBsmtSF"]

        if not all(col in X.columns for col in required):
            raise ValueError("Required area columns are missing.")

        self.feature_names_in_ = np.asarray(required, dtype=object)
        self.n_feature_names_in_ = len(required)

        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")

        data = X[self.feature_names_in_].copy()

        total_sf = data["GrLivArea"] + data["TotalBsmtSF"]

        return total_sf.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")

        return np.asarray(["TotalSF"], dtype=object)
