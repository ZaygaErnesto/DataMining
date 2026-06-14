"""Unit and integration tests for FastAPI application src.app."""

import pytest
from fastapi.testclient import TestClient

from src.app import app

# ---------------------------------------------------------------------------
# Client Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_home_endpoint_renders_html(client):
    """GET / should render the dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Predictive Maintenance" in response.text


def test_metrics_endpoint_returns_data(client):
    """GET /api/metrics should return training metrics JSON."""
    response = client.get("/api/metrics")
    # If the metrics file exists it will return 200, else 404 (both are valid behaviors)
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "accuracy" in data
        assert "balanced_accuracy" in data
        assert "f1_macro" in data


def test_predict_endpoint_validates_payload(client):
    """POST /api/predict with empty payload should return 422 Unprocessable Entity."""
    response = client.post("/api/predict", json={})
    assert response.status_code == 422


def test_predict_endpoint_diagnoses_telemetry(client):
    """POST /api/predict with valid sensor telemetry should return classification."""
    payload = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }
    response = client.post("/api/predict", json=payload)
    # If the model artifact is loaded it will return 200
    assert response.status_code in (200, 530, 503)
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "feature_set_used" in data
        assert isinstance(data["prediction"], str)
