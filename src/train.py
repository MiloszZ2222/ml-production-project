from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import settings
from src.preprocessing import build_preprocessor


def build_model_pipeline() -> Pipeline:
    """Create the final preprocessing and XGBoost pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    random_state=settings.training.random_state,
                    n_jobs=-1,
                    **settings.training.model_params,
                ),
            ),
        ]
    )


def train_model(X, y) -> Pipeline:
    """Build and fit the final model pipeline."""
    pipeline = build_model_pipeline()
    pipeline.fit(X, y)
    return pipeline
