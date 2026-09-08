import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.predictor import predictor

@pytest.fixture(scope="module", autouse=True)
def load_models():
    predictor.load_artifacts()

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert len(data["available_models"]) == 4

def test_get_symptoms_catalog():
    response = client.get("/api/symptoms")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 132
    assert len(data["symptoms"]) == 132
    first_sym = data["symptoms"][0]
    assert "id" in first_sym
    assert "name" in first_sym

def test_predict_happy_path_fungal_infection():
    payload = {
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
        "model_name": "random_forest",
        "top_n": 5
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["model_used"] == "random_forest"
    assert len(data["predictions"]) == 5
    top_pred = data["predictions"][0]
    assert top_pred["rank"] == 1
    assert top_pred["disease"] == "Fungal infection"
    assert top_pred["confidence_percentage"] > 70.0
    assert len(top_pred["contributing_symptoms"]) > 0
    assert "disclaimer" in data

def test_predict_all_four_models():
    models = ["naive_bayes", "decision_tree", "random_forest", "gradient_boosting"]
    payload = {"symptoms": ["itching", "skin_rash"], "top_n": 3}
    
    for model_name in models:
        payload["model_name"] = model_name
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == model_name
        assert len(data["predictions"]) == 3

def test_predict_empty_symptoms_error():
    payload = {"symptoms": [], "model_name": "random_forest"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error (min_length=1)

def test_predict_unknown_symptom_error():
    payload = {"symptoms": ["invalid_fake_symptom"], "model_name": "random_forest"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "Unknown symptom ID" in response.json()["detail"]

def test_predict_invalid_model_error():
    payload = {"symptoms": ["fever"], "model_name": "super_ai_model"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid model" in response.json()["detail"]

def test_get_models_compare_benchmarks():
    response = client.get("/api/models/compare")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    assert "random_forest" in models
    assert "naive_bayes" in models
    assert "decision_tree" in models
    assert "gradient_boosting" in models
    assert models["random_forest"]["accuracy"] > 0.90
