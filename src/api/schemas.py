from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """Input schema for a single house price prediction."""

    model_config = ConfigDict(extra="forbid")

    # --------------------------------------------------
    # 1. Building classification and location
    # --------------------------------------------------

    MSSubClass: int
    MSZoning: str | None
    LotFrontage: float | None = Field(default=..., ge=0)
    LotArea: int = Field(gt=0, le=100_000)
    Street: str | None
    Alley: str | None
    LotShape: str | None
    LandContour: str | None
    Utilities: str | None
    LotConfig: str | None
    LandSlope: str | None
    Neighborhood: str | None
    Condition1: str | None
    Condition2: str | None
    BldgType: str | None
    HouseStyle: str | None

    # --------------------------------------------------
    # 2. Building quality and construction
    # --------------------------------------------------

    OverallQual: int = Field(ge=1, le=10)
    OverallCond: int = Field(ge=1, le=10)

    YearBuilt: int = Field(ge=1800, le=2030)
    YearRemodAdd: int = Field(ge=1800, le=2030)

    RoofStyle: str | None
    RoofMatl: str | None
    Exterior1st: str | None
    Exterior2nd: str | None

    MasVnrType: str | None
    MasVnrArea: float | None = Field(
        default=...,
        ge=0,
        le=2_000,
    )

    ExterQual: str | None
    ExterCond: str | None
    Foundation: str | None

    # --------------------------------------------------
    # 3. Basement
    # --------------------------------------------------

    BsmtQual: str | None
    BsmtCond: str | None
    BsmtExposure: str | None
    BsmtFinType1: str | None

    BsmtFinSF1: int = Field(ge=0)
    BsmtFinType2: str | None
    BsmtFinSF2: int = Field(ge=0)
    BsmtUnfSF: int = Field(ge=0)

    TotalBsmtSF: int = Field(ge=0, le=3_500)

    # --------------------------------------------------
    # 4. Heating and utilities
    # --------------------------------------------------

    Heating: str | None
    HeatingQC: str | None
    CentralAir: str | None
    Electrical: str | None

    # --------------------------------------------------
    # 5. Living area and rooms
    # --------------------------------------------------

    FirstFlrSF: int = Field(
        ge=0,
        le=3_500,
        alias="1stFlrSF",
    )

    SecondFlrSF: int = Field(
        ge=0,
        le=3_500,
        alias="2ndFlrSF",
    )

    LowQualFinSF: int = Field(ge=0)
    GrLivArea: int = Field(gt=0, le=5_000)

    BsmtFullBath: int = Field(ge=0)
    BsmtHalfBath: int = Field(ge=0)
    FullBath: int = Field(ge=0)
    HalfBath: int = Field(ge=0)

    BedroomAbvGr: int = Field(ge=0)
    KitchenAbvGr: int = Field(ge=0)
    KitchenQual: str | None
    TotRmsAbvGrd: int = Field(ge=0)
    Functional: str | None

    Fireplaces: int = Field(ge=0)
    FireplaceQu: str | None

    # --------------------------------------------------
    # 6. Garage
    # --------------------------------------------------

    GarageType: str | None
    GarageYrBlt: float | None = Field(
        default=...,
        ge=1800,
        le=2030,
    )

    GarageFinish: str | None
    GarageCars: int = Field(ge=0, le=10)
    GarageArea: int = Field(ge=0, le=1_200)

    GarageQual: str | None
    GarageCond: str | None
    PavedDrive: str | None

    # --------------------------------------------------
    # 7. Outdoor features
    # --------------------------------------------------

    WoodDeckSF: int = Field(ge=0)
    OpenPorchSF: int = Field(ge=0)
    EnclosedPorch: int = Field(ge=0)

    ThreeSsnPorch: int = Field(
        ge=0,
        alias="3SsnPorch",
    )

    ScreenPorch: int = Field(ge=0)

    PoolArea: int = Field(ge=0, le=1_000)
    PoolQC: str | None

    Fence: str | None
    MiscFeature: str | None
    MiscVal: int = Field(ge=0)

    # --------------------------------------------------
    # 8. Sale information
    # --------------------------------------------------

    MoSold: int = Field(ge=1, le=12)
    YrSold: int = Field(ge=2006, le=2010)
    SaleType: str | None
    SaleCondition: str | None


class PredictionResponse(BaseModel):
    """Response schema for a single house price prediction."""

    predicted_price: float
