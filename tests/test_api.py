import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import app as api_module


class FakeRegressor:
    def predict(self, row):
        return [67.4]


class FakeClassifier:
    def predict_proba(self, row):
        return [[0.10, 0.20, 0.60, 0.10]]


class FakeEncoder:
    classes_ = ["Distinction", "Fail", "Pass", "Withdrawn"]


def install_grade_fakes(monkeypatch):
    defaults = {"prev_score": 60, "code_module": "AAA"}
    bundle = {"model": FakeRegressor(), "feature_columns": ["prev_score", "code_module"]}
    monkeypatch.setattr(api_module, "_bundle", lambda name: bundle if name == "random_forest" else None)
    monkeypatch.setattr(api_module, "_report", lambda name: {"defaults": defaults})


def test_grade_prediction_returns_score(monkeypatch):
    install_grade_fakes(monkeypatch)
    response = api_module.app.test_client().post(
        "/predict/grade", json={"features": {"prev_score": 75}}
    )
    assert response.status_code == 200
    assert response.get_json()["predicted_score"] == 67.4


def test_grade_prediction_rejects_unknown_feature(monkeypatch):
    install_grade_fakes(monkeypatch)
    response = api_module.app.test_client().post(
        "/predict/grade", json={"features": {"prev_socre": 75}}
    )
    assert response.status_code == 400
    assert "Unknown feature" in response.get_json()["error"]


def test_grade_prediction_rejects_non_numeric_value(monkeypatch):
    install_grade_fakes(monkeypatch)
    response = api_module.app.test_client().post(
        "/predict/grade", json={"features": {"prev_score": "high"}}
    )
    assert response.status_code == 400
    assert "must be numeric" in response.get_json()["error"]


def test_outcome_prediction_returns_probabilities_and_risk(monkeypatch):
    defaults = {"total_clicks": 100, "code_module": "AAA"}
    bundle = {
        "model": FakeClassifier(),
        "label_encoder": FakeEncoder(),
        "feature_columns": ["total_clicks", "code_module"],
    }
    monkeypatch.setattr(api_module, "_bundle", lambda name: bundle if name == "gbm_classifier" else None)
    monkeypatch.setattr(api_module, "_report", lambda name: {"defaults": defaults})
    response = api_module.app.test_client().post(
        "/predict/outcome", json={"features": {"total_clicks": 200}}
    )
    data = response.get_json()
    assert response.status_code == 200
    assert data["predicted_outcome"] == "Pass"
    assert data["risk_level"] == "Low"
    assert set(data["probabilities"]) == set(FakeEncoder.classes_)
