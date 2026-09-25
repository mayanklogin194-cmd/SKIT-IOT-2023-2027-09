"""
FR-005/NFR-001: Flask REST API serving model predictions to the teacher
and student dashboards. Target: respond within 2 seconds of a request (NFR-001).

Owner: Mayank Rathore & Priyanshu Joshi (Sprint 7)
"""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_rf_bundle = None
_gbm_bundle = None


def _load_models():
    global _rf_bundle, _gbm_bundle
    if _rf_bundle is None:
        _rf_bundle = joblib.load(MODEL_DIR / "random_forest.joblib")
    if _gbm_bundle is None:
        _gbm_bundle = joblib.load(MODEL_DIR / "gbm_classifier.joblib")
    return _rf_bundle, _gbm_bundle


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/grade", methods=["POST"])
def predict_grade():
    """Body: {"features": {"prior_score": 62, "total_vle_clicks": 340, ...}}"""
    rf_bundle, _ = _load_models()
    payload = request.get_json(force=True)
    row = pd.DataFrame([payload["features"]]).reindex(columns=rf_bundle["feature_columns"], fill_value=0)
    pred = rf_bundle["model"].predict(row)[0]
    return jsonify({"predicted_score": round(float(pred), 2)})


@app.route("/predict/outcome", methods=["POST"])
def predict_outcome():
    """Body: {"features": {...}}"""
    _, gbm_bundle = _load_models()
    payload = request.get_json(force=True)
    row = pd.DataFrame([payload["features"]]).reindex(columns=gbm_bundle["feature_columns"], fill_value=0)
    pred_idx = gbm_bundle["model"].predict(row)[0]
    label = gbm_bundle["label_encoder"].inverse_transform([pred_idx])[0]
    proba = gbm_bundle["model"].predict_proba(row)[0].max()
    return jsonify({"predicted_outcome": label, "confidence": round(float(proba), 3)})


@app.route("/dashboard/alerts", methods=["GET"])
def early_warning_alerts():
    """Placeholder: teacher dashboard early-warning list.
    Replace with a real query once predictions are stored per-student."""
    return jsonify({"alerts": [], "note": "wire this up to a predictions table"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
