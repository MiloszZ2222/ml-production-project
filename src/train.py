from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42985

BEST_PARAMS = {
    "n_estimators": 606,
    "max_depth": 3,
    "learning_rate": 0.04898338493290398,
    "min_child_weight": 4,
    "subsample": 0.6047050917409649,
    "colsample_bytree": 0.7164888104969732,
    "reg_alpha": 1.4687221482811246,
    "reg_lambda": 0.0036028306853463006,
}


def build_model_pipeline() -> Pipeline:
    """Create the final preprocessing and XGBoost pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **BEST_PARAMS),
            ),
        ]
    )


def train_model(X, y) -> Pipeline:
    """Build and fit the final model pipeline."""
    pipeline = build_model_pipeline()
    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """Save a fitted model pipeline to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, path)
