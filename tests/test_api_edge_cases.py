"""Additional API edge-case tests; these complement tests/test_api.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import app as api_module


def test_health_reports_model_file_availability(monkeypatch):
    monkeypatch.setattr(api_module, "MODEL_DIR", Path("/path/that/does/not/exist"))
    response = api_module.app.test_client().get("/health")
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["models"] == {"random_forest": False, "gbm_classifier": False}


def test_grade_rejects_malformed_json(monkeypatch):
    monkeypatch.setattr(api_module, "_bundle", lambda _name: {"model": object(), "feature_columns": []})
    response = api_module.app.test_client().post(
        "/predict/grade", data="{not-json", content_type="application/json"
    )
    assert response.status_code == 400
    assert "JSON object" in response.get_json()["error"]


def test_grade_returns_503_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(api_module, "_bundle", lambda _name: None)
    response = api_module.app.test_client().post(
        "/predict/grade", json={"features": {"prev_score": 60}}
    )
    assert response.status_code == 503
    assert "not trained" in response.get_json()["error"]


def test_outcome_rejects_unknown_feature(monkeypatch):
    defaults = {"total_clicks": 100, "code_module": "AAA"}
    bundle = {"model": object(), "label_encoder": object(),
              "feature_columns": ["total_clicks", "code_module"]}
    monkeypatch.setattr(api_module, "_bundle", lambda _name: bundle)
    monkeypatch.setattr(api_module, "_report", lambda _name: {"defaults": defaults})
    response = api_module.app.test_client().post(
        "/predict/outcome", json={"features": {"total_clickz": 200}}
    )
    assert response.status_code == 400
    assert "Unknown feature" in response.get_json()["error"]


def test_unknown_endpoint_returns_json_404():
    response = api_module.app.test_client().get("/not-a-real-endpoint")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Endpoint not found."


def test_prediction_endpoint_rejects_get_method():
    response = api_module.app.test_client().get("/predict/grade")
    assert response.status_code == 405
    assert "not allowed" in response.get_json()["error"]


def test_response_has_observability_headers():
    response = api_module.app.test_client().get("/health")
    assert response.headers.get("X-Request-ID")
    assert float(response.headers["X-Response-Time-ms"]) >= 0
