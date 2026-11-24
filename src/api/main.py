from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from sklearn.pipeline import Pipeline

from src.api.model_loader import load_model
from src.api.schemas import HouseFeatures, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    print("Loading ML model...")

    app.state.model = load_model()

    print("ML model loaded successfully!")

    yield

    app.state.model = None

    print("Application shutting down.")


app = FastAPI(
    title="House Prices API",
    description="REST API for house price prediction using XGBoost.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model(request: Request) -> Pipeline:
    """Provide the loaded model to API endpoints."""

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return model


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "House Prices API is running!"}


@app.get("/model-info")
def model_info(model: Annotated[Pipeline, Depends(get_model)]) -> dict:
    """Return information about the loaded ML pipeline."""

    return {
        "model_loaded": True,
        "pipeline_steps": list(model.named_steps.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    house: HouseFeatures, model: Annotated[Pipeline, Depends(get_model)]
) -> PredictionResponse:
    """Predict the sale price of a single house."""

    features = pd.DataFrame([house.model_dump(by_alias=True)])

    prediction = model.predict(features)[0]

    return PredictionResponse(predicted_price=float(prediction))
