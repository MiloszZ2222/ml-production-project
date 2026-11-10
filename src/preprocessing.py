import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from src.transformers import (
    AgeFeaturesTransformer,
    AreaFeaturesTransformer,
    BathroomFeaturesTransformer,
    MasonryTypeImputer,
    PositiveMedianImputer,
)

# --------------------------------------------------
# 1. Ordinal feature configuration
# --------------------------------------------------

QUALITY_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_WITH_ABSENCE = ["None", *QUALITY_ORDER]

ORDINAL_CATEGORIES = {
    "ExterQual": QUALITY_ORDER,
    "ExterCond": QUALITY_ORDER,
    "HeatingQC": QUALITY_ORDER,
    "KitchenQual": QUALITY_ORDER,
    "BsmtQual": QUALITY_WITH_ABSENCE,
    "BsmtCond": QUALITY_WITH_ABSENCE,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "FireplaceQu": QUALITY_WITH_ABSENCE,
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "GarageQual": QUALITY_WITH_ABSENCE,
    "GarageCond": QUALITY_WITH_ABSENCE,
    "PoolQC": QUALITY_WITH_ABSENCE,
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "LandSlope": ["Gtl", "Mod", "Sev"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "Street": ["Grvl", "Pave"],
    "CentralAir": ["N", "Y"],
}

ORDINAL_ABSENCE_COLUMNS = [
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]

ORDINAL_REGULAR_COLUMNS = [
    column for column in ORDINAL_CATEGORIES if column not in ORDINAL_ABSENCE_COLUMNS
]


# --------------------------------------------------
# 2. Nominal feature configuration
# --------------------------------------------------

NOMINAL_ABSENCE_COLUMNS = ["Alley", "GarageType", "Fence", "MiscFeature"]

NOMINAL_REGULAR_COLUMNS = [
    "MSZoning",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "Foundation",
    "Heating",
    "Electrical",
    "SaleType",
    "SaleCondition",
    "MSSubClass",
]


# --------------------------------------------------
# 3. Numeric features passed without modification
# --------------------------------------------------

NUMERIC_PASSTHROUGH_COLUMNS = [
    "OverallQual",
    "OverallCond",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
]


# --------------------------------------------------
# 4. Reusable pipeline factories
# --------------------------------------------------


def make_one_hot_pipeline(strategy):
    """Create a nominal categorical preprocessing pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=strategy,
                    fill_value=("None" if strategy == "constant" else None),
                    keep_empty_features=True,
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def make_ordinal_pipeline(columns, strategy):
    """Create an ordinal preprocessing pipeline."""

    categories = [ORDINAL_CATEGORIES[column] for column in columns]

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=strategy,
                    fill_value=("None" if strategy == "constant" else None),
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


# --------------------------------------------------
# 5. Complete preprocessing pipeline
# --------------------------------------------------


def build_preprocessor():
    """Build an unfitted preprocessing pipeline for House Prices."""

    masonry_type_pipeline = Pipeline(
        steps=[
            ("imputer", MasonryTypeImputer()),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    lot_frontage_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log_transform",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            # Custom feature engineering
            ("masonry_area", PositiveMedianImputer(), ["MasVnrArea"]),
            ("masonry_type", masonry_type_pipeline, ["MasVnrType", "MasVnrArea"]),
            (
                "age_features",
                AgeFeaturesTransformer(),
                ["YrSold", "YearBuilt", "YearRemodAdd"],
            ),
            ("area_features", AreaFeaturesTransformer(), ["GrLivArea", "TotalBsmtSF"]),
            (
                "bathroom_features",
                BathroomFeaturesTransformer(),
                ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
            ),
            # Numeric preprocessing
            ("lot_frontage", lot_frontage_pipeline, ["LotFrontage"]),
            (
                "log_features",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                ["LotArea", "1stFlrSF", "GrLivArea"],
            ),
            (
                "garage_year",
                SimpleImputer(strategy="constant", fill_value=0),
                ["GarageYrBlt"],
            ),
            # Ordinal categorical features
            (
                "ordinal_regular",
                make_ordinal_pipeline(
                    ORDINAL_REGULAR_COLUMNS, strategy="most_frequent"
                ),
                ORDINAL_REGULAR_COLUMNS,
            ),
            (
                "ordinal_absence",
                make_ordinal_pipeline(ORDINAL_ABSENCE_COLUMNS, strategy="constant"),
                ORDINAL_ABSENCE_COLUMNS,
            ),
            # Nominal categorical features
            (
                "nominal_regular",
                make_one_hot_pipeline(strategy="most_frequent"),
                NOMINAL_REGULAR_COLUMNS,
            ),
            (
                "nominal_absence",
                make_one_hot_pipeline(strategy="constant"),
                NOMINAL_ABSENCE_COLUMNS,
            ),
            # Remaining numeric features
            ("numeric_remaining", "passthrough", NUMERIC_PASSTHROUGH_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0,
    )

    return preprocessor
