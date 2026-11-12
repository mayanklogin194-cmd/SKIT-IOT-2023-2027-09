import sys
from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from api.batch_prediction import register_batch_prediction


class DummyGradeModel:
    def predict(self, frame):
        return [125.5]


class DummyOutcomeModel:
    def predict_proba(self, frame):
        return [[0.1, 0.7, 0.1, 0.1]]


class DummyLabelEncoder:
    classes_ = ["Distinction", "Pass", "Fail", "Withdrawn"]


def make_client(model_available=True):
    app = Flask(__name__)

    def loader(name):
        if not model_available:
            return None
        if name == "random_forest":
            return {"model": DummyGradeModel()}
        return {"model": DummyOutcomeModel(), "label_encoder": DummyLabelEncoder()}

    def row_builder(bundle, defaults, features):
        if not isinstance(features, dict):
            raise ValueError("'features' must be a JSON object.")
        if "invalid" in features:
            raise ValueError("invalid feature supplied")
        return [features]

    def report_loader(name):
        return {"defaults": {"example": 0}}

    register_batch_prediction(app, loader, row_builder, report_loader, max_batch_size=2)
    return app.test_client()


def test_outcome_batch_returns_predictions_and_risk():
    response = make_client().post("/predict/batch", json={
        "prediction_type": "outcome",
        "students": [{"example": 1}, {"example": 2}]
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["successful"] == 2
    assert data["results"][0]["predicted_outcome"] == "Pass"
    assert data["results"][0]["risk_level"] == "Low"


def test_grade_batch_clamps_score_to_100():
    response = make_client().post("/predict/batch", json={
        "prediction_type": "grade", "students": [{"example": 1}]
    })
    assert response.status_code == 200
    assert response.get_json()["results"][0]["predicted_score"] == 100.0


def test_invalid_student_is_reported_without_stopping_batch():
    response = make_client().post("/predict/batch", json={
        "students": [{"invalid": True}, {"example": 1}]
    })
    data = response.get_json()
    assert data["successful"] == 1
    assert data["failed"] == 1
    assert data["errors"][0]["index"] == 0


def test_empty_batch_is_rejected():
    assert make_client().post("/predict/batch", json={"students": []}).status_code == 400


def test_oversized_batch_is_rejected():
    assert make_client().post("/predict/batch", json={
        "students": [{}, {}, {}]
    }).status_code == 413


def test_missing_model_returns_service_unavailable():
    assert make_client(False).post("/predict/batch", json={
        "students": [{}]
    }).status_code == 503


def test_unknown_prediction_type_is_rejected():
    assert make_client().post("/predict/batch", json={
        "prediction_type": "unknown", "students": [{}]
    }).status_code == 400
