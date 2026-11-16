import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

API_DIRECTORY = Path(__file__).resolve().parents[1]
if str(API_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(API_DIRECTORY))

import main


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    return {
        "historical_spend": 500.0,
        "order_count": 4,
        "units_purchased": 20,
        "distinct_products": 8,
        "active_days": 3,
        "days_since_last_purchase": 2,
        "mean_order_spend": 125.0,
        "historical_return_lines": 0,
        "historical_cancellation_lines": 0,
        "historical_adjustment_lines": 0,
        "country": "United Kingdom",
    }


def test_root_health_and_swagger_docs(client):
    root_response = client.get("/")
    health_response = client.get("/health")
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert root_response.status_code == 200
    assert root_response.json()["prediction"] == "POST /predict"
    assert health_response.status_code == 200
    assert health_response.json()["model_available"] is True
    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200
    paths = openapi_response.json()["paths"]
    assert "/predict" in paths
    assert "503" in paths["/predict"]["post"]["responses"]
    assert "503" in paths["/health"]["get"]["responses"]


def test_predict_returns_nonnegative_spend(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["target"] == "future_spend"
    assert body["model"] == "Ridge"
    assert body["prediction"] >= 0
    assert np.isfinite(body["prediction"])


def test_unseen_country_is_supported(client, valid_payload):
    payload = {**valid_payload, "country": "New Country"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] >= 0


@pytest.mark.parametrize(
    "changes",
    [
        {"historical_spend": -1},
        {"order_count": 0},
        {"days_since_last_purchase": -1},
        {"country": "   "},
        {"unexpected": "field"},
    ],
)
def test_invalid_inputs_return_safe_validation_errors(client, valid_payload, changes):
    payload = {**valid_payload, **changes}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]
    assert all("input" not in error for error in response.json()["detail"])


def test_missing_required_feature_returns_validation_error(client, valid_payload):
    payload = {key: value for key, value in valid_payload.items() if key != "country"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "country"


def test_model_unavailable_returns_service_unavailable(monkeypatch):
    def unavailable_model():
        raise FileNotFoundError("private path should not be returned")

    monkeypatch.setattr(main, "load_model_bundle", unavailable_model)
    with TestClient(main.app) as client:
        health_response = client.get("/health")
        prediction_response = client.post(
            "/predict",
            json={
                "historical_spend": 500.0,
                "order_count": 4,
                "units_purchased": 20,
                "distinct_products": 8,
                "active_days": 3,
                "days_since_last_purchase": 2,
                "mean_order_spend": 125.0,
                "historical_return_lines": 0,
                "historical_cancellation_lines": 0,
                "historical_adjustment_lines": 0,
                "country": "United Kingdom",
            },
        )

    assert health_response.status_code == 503
    assert health_response.json() == {
        "status": "degraded",
        "model_available": False,
    }
    assert prediction_response.status_code == 503
    assert "private path" not in prediction_response.text


def test_prediction_failure_does_not_expose_internal_error(
    monkeypatch,
    valid_payload,
):
    with TestClient(main.app) as client:
        artifact = client.app.state.model_bundle

        def fail_prediction(_features):
            raise RuntimeError("sensitive internal detail")

        monkeypatch.setattr(artifact["estimator"], "predict", fail_prediction)
        response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction could not be generated."
    assert "sensitive internal detail" not in response.text


def test_request_features_match_saved_pipeline_schema(client, valid_payload):
    artifact = client.app.state.model_bundle
    features = pd.DataFrame([valid_payload], columns=artifact["input_features"])
    expected = max(float(artifact["estimator"].predict(features)[0]), 0.0)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(expected)
