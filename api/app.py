"""
FR-005/NFR-001: Flask REST API serving model predictions to the teacher and student
dashboards. Target: respond within 2 seconds of a request (NFR-001).

Run:   python api/app.py            (add USE_SYNTHETIC=1 to serve the synthetic demo models)
Serves the built React dashboard (dashboard/dist) at "/" when it exists.

Owner: Mayank Rathore & Priyanshu Joshi (Sprint 7 preview - API)
"""
import json
import os
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
SUB = "synthetic" if os.environ.get("USE_SYNTHETIC") else ""
MODEL_DIR, REPORT_DIR = ROOT / "models" / SUB, ROOT / "reports" / SUB
DIST = ROOT / "dashboard" / "dist"
FEEDBACK_PATH = REPORT_DIR / "prediction_feedback.jsonl"

app = Flask(__name__, static_folder=str(DIST), static_url_path="")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("student_prediction_api")


@app.before_request
def start_request_metrics():
    request.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.started_at = time.perf_counter()


@app.after_request
def finish_request_metrics(response):
    elapsed_ms = (time.perf_counter() - getattr(request, "started_at", time.perf_counter())) * 1000
    response.headers["X-Request-ID"] = getattr(request, "request_id", "unknown")
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                getattr(request, "request_id", "unknown"), request.method,
                request.path, response.status_code, elapsed_ms)
    return response


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found.", "request_id": getattr(request, "request_id", "unknown")}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"error": "HTTP method not allowed for this endpoint.",
                    "request_id": getattr(request, "request_id", "unknown")}), 405

_cache = {}


def _bundle(name):
    if name not in _cache:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            return None
        _cache[name] = joblib.load(path)
    return _cache[name]


def _report(name):
    return json.loads((REPORT_DIR / name).read_text())


def _row(bundle, defaults, features):
    if not isinstance(features, dict):
        raise ValueError("'features' must be a JSON object.")
    unknown = sorted(set(features) - set(defaults))
    if unknown:
        raise ValueError("Unknown feature(s): " + ", ".join(unknown))

    row = dict(defaults)
    for key, value in features.items():
        default = defaults[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"'{key}' must be a boolean.")
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"'{key}' must be numeric.")
            if not pd.notna(value) or value in (float('inf'), float('-inf')):
                raise ValueError(f"'{key}' must be a finite number.")
        elif not isinstance(value, str):
            raise ValueError(f"'{key}' must be a string.")
        row[key] = value
    return pd.DataFrame([row])[bundle["feature_columns"]]


def _request_features():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    features = payload.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("'features' must be a JSON object.")
    return features


@app.get("/health")
def health():
    return jsonify({"status": "ok", "models": {"random_forest": (MODEL_DIR / "random_forest.joblib").exists(),
                                                "gbm_classifier": (MODEL_DIR / "gbm_classifier.joblib").exists()}})


@app.post("/predict/grade")
def predict_grade():
    """Body: {"features": {"prev_score": 62, "cum_clicks": 340, ...}}  (missing keys use typical values)"""
    bundle = _bundle("random_forest")
    if bundle is None:
        return jsonify({"error": "random_forest model not trained - run python src/main.py"}), 503
    try:
        feats = _request_features()
        row = _row(bundle, _report("form_defaults_grade.json")["defaults"], feats)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    pred = float(bundle["model"].predict(row)[0])
    return jsonify({"predicted_score": round(min(max(pred, 0), 100), 2)})


@app.post("/predict/outcome")
def predict_outcome():
    """Body: {"features": {...}} -> predicted outcome, class probabilities and an early-warning risk level."""
    bundle = _bundle("gbm_classifier")
    if bundle is None:
        return jsonify({"error": "gbm_classifier not trained - run python src/main.py"}), 503
    try:
        feats = _request_features()
        row = _row(bundle, _report("form_defaults_outcome.json")["defaults"], feats)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    proba = bundle["model"].predict_proba(row)[0]
    classes = list(bundle["label_encoder"].classes_)
    probs = dict(zip(classes, map(float, proba)))
    risk = probs.get("Fail", 0) + probs.get("Withdrawn", 0)
    level = "High" if risk >= 0.6 else "Medium" if risk >= 0.4 else "Low"
    best = max(probs, key=probs.get)
    return jsonify({"predicted_outcome": best, "confidence": round(probs[best], 3),
                    "probabilities": probs, "risk": risk, "risk_level": level})




@app.post("/feedback")
def save_prediction_feedback():
    """Store lightweight feedback about a prediction without storing input features."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    prediction_type = payload.get("prediction_type")
    feedback = payload.get("feedback")
    prediction = payload.get("prediction")
    comment = payload.get("comment", "")

    if prediction_type not in {"grade", "outcome"}:
        return jsonify({"error": "'prediction_type' must be 'grade' or 'outcome'."}), 400
    if feedback not in {"useful", "not_useful"}:
        return jsonify({"error": "'feedback' must be 'useful' or 'not_useful'."}), 400
    if prediction is None:
        return jsonify({"error": "'prediction' is required."}), 400
    if not isinstance(comment, str) or len(comment) > 500:
        return jsonify({"error": "'comment' must be a string of at most 500 characters."}), 400

    if prediction_type == "grade":
        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
            return jsonify({"error": "Grade prediction must be numeric."}), 400
        if not 0 <= prediction <= 100:
            return jsonify({"error": "Grade prediction must be between 0 and 100."}), 400
        stored_prediction = round(float(prediction), 2)
    else:
        if not isinstance(prediction, str) or prediction not in {
            "Distinction", "Pass", "Fail", "Withdrawn"
        }:
            return jsonify({"error": "Outcome prediction is not a recognized class."}), 400
        stored_prediction = prediction

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": getattr(request, "request_id", "unknown"),
        "prediction_type": prediction_type,
        "prediction": stored_prediction,
        "feedback": feedback,
        "comment": comment.strip(),
    }
    try:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Could not persist prediction feedback")
        return jsonify({"error": "Feedback could not be saved right now."}), 500

    return jsonify({"saved": True, "message": "Feedback received."}), 201


@app.get("/dashboard/summary")
def summary():
    return jsonify({"baselines": _report("baseline_metrics.json"), "rf": _report("rf_metrics.json"),
                    "gbm": _report("gbm_metrics.json"),
                    "forms": {"grade": _report("form_defaults_grade.json"),
                              "outcome": _report("form_defaults_outcome.json")}})


@app.get("/dashboard/students")
def students():
    return jsonify(_report("sample_students.json"))


@app.get("/dashboard/alerts")
def early_warning_alerts():
    """Held-out students whose predicted chance of failing or withdrawing is >= 40%."""
    alerts = [s for s in _report("sample_students.json") if s["risk"] >= 0.4]
    return jsonify({"alerts": alerts})


@app.get("/")
def index():
    if (DIST / "index.html").exists():
        return send_from_directory(DIST, "index.html")
    return jsonify({"message": "API is running. Build the dashboard: cd dashboard && npm install && npm run build"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
