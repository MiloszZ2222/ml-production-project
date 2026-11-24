import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_model


@pytest.fixture
def house_data() -> dict:
    """Load a valid example house from a local JSON file."""

    fixture_path = Path(__file__).parent / "fixtures" / "house.json"

    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def mock_model() -> Mock:
    """Create a fake model returning a fixed prediction."""

    model = Mock()
    model.predict.return_value = [206048.0]

    return model


@pytest.fixture
def client(mock_model: Mock):
    """Create an API test client with a mocked model."""

    app.dependency_overrides[get_model] = lambda: mock_model

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_root(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "House Prices API is running!"}


def test_predict_success(
    client: TestClient,
    house_data: dict,
    mock_model: Mock,
):
    response = client.post("/predict", json=house_data)

    assert response.status_code == 200
    assert response.json() == {"predicted_price": 206048.0}

    mock_model.predict.assert_called_once()

    features = mock_model.predict.call_args.args[0]

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (1, 79)
    assert "1stFlrSF" in features.columns
    assert "FirstFlrSF" not in features.columns


def test_predict_invalid_quality(
    client: TestClient,
    house_data: dict,
):
    house_data["OverallQual"] = 15

    response = client.post("/predict", json=house_data)

    assert response.status_code == 422


def test_predict_negative_area(
    client: TestClient,
    house_data: dict,
):
    house_data["LotArea"] = -500

    response = client.post("/predict", json=house_data)

    assert response.status_code == 422


def test_predict_missing_feature(
    client: TestClient,
    house_data: dict,
):
    del house_data["GrLivArea"]

    response = client.post("/predict", json=house_data)

    assert response.status_code == 422


def test_predict_unknown_feature(
    client: TestClient,
    house_data: dict,
):
    house_data["UnknownFeature"] = 123

    response = client.post("/predict", json=house_data)

    assert response.status_code == 422
