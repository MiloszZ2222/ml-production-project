import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    NOMINAL_ABSENCE_COLUMNS,
    NOMINAL_REGULAR_COLUMNS,
    NUMERIC_PASSTHROUGH_COLUMNS,
    ORDINAL_CATEGORIES,
)


def make_sample_house_data(n_samples=3):
    data = {column: [1.0, 2.0, 3.0] for column in NUMERIC_PASSTHROUGH_COLUMNS}

    for column, categories in ORDINAL_CATEGORIES.items():
        data[column] = [categories[-1]] * 3

    for column in NOMINAL_REGULAR_COLUMNS:
        data[column] = ["Sample"] * 3

    for column in NOMINAL_ABSENCE_COLUMNS:
        data[column] = [np.nan, "Present", np.nan]

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

    base = pd.DataFrame(data)

    if n_samples == 3:
        return base

    indices = np.arange(n_samples)
    X = base.iloc[indices % 3].reset_index(drop=True).copy()

    # Introduce deterministic variation in numerical features
    X["YrSold"] = 2006 + indices % 10
    X["YearBuilt"] = 1970 + indices % 30
    X["YearRemodAdd"] = X["YearBuilt"] + indices % 7

    X["GrLivArea"] = 1200 + (indices % 50) * 20
    X["TotalBsmtSF"] = 500 + (indices % 50) * 15
    X["LotArea"] = 8000 + (indices % 50) * 100
    X["1stFlrSF"] = 900 + (indices % 50) * 10

    return X


@pytest.fixture
def house_data_factory():
    return make_sample_house_data
