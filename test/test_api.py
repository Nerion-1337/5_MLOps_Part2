# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

from api.main import app, MODEL_STATE


@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI avec exécution du cycle de vie lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_healthcheck(client):
    """Vérifie que la route /health renvoie un statut opérationnel et les métriques de MLflow."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ready", "not_ready"]
    assert "optimal_threshold" in data
    assert "run_id" in data


def test_predict_valid_payload(client):
    """Vérifie l'inférence avec des données conformes."""
    payload = {
        "client_id": 100002,
        "features": {
            "AMT_CREDIT": 406597.5,
            "AMT_INCOME_TOTAL": 202500.0,
            "EXT_SOURCE_2": 0.26,
            "EXT_SOURCE_3": 0.13,
            "DAYS_BIRTH": -9461
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == 100002
    assert 0.0 <= data["probability"] <= 1.0
    assert data["decision"] in ["ACCORDE", "REFUSE"]
    assert data["status"] == "SUCCESS"
    assert "threshold" in data


def test_predict_invalid_credit_amount(client):
    """Vérifie que l'API rejette un montant de crédit négatif (Erreur 422)."""
    payload = {
        "client_id": 100003,
        "features": {
            "AMT_CREDIT": -50000.0,
            "AMT_INCOME_TOTAL": 100000.0
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_income_amount(client):
    """Vérifie que l'API rejette un revenu total négatif ou nul (Erreur 422)."""
    payload = {
        "client_id": 100004,
        "features": {
            "AMT_CREDIT": 100000.0,
            "AMT_INCOME_TOTAL": 0.0
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_ext_source_range(client):
    """Vérifie que l'API rejette une valeur hors bornes [0.0, 1.0] sur EXT_SOURCE (Erreur 422)."""
    payload = {
        "client_id": 100005,
        "features": {
            "AMT_CREDIT": 100000.0,
            "EXT_SOURCE_2": 1.45
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_features(client):
    """Vérifie le rejet d'un payload sans variables (Erreur 422)."""
    payload = {
        "client_id": 100006,
        "features": {}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_data_types(client):
    """Vérifie le rejet quand un type string non convertible est injecté (Erreur 422)."""
    payload = {
        "client_id": 100007,
        "features": {
            "AMT_CREDIT": "cent-mille-euros"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422