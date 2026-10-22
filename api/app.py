"""
FR-005/NFR-001: Flask REST API serving model predictions to the teacher and student
dashboards. Target: respond within 2 seconds of a request (NFR-001).

Run:   python api/app.py            (add USE_SYNTHETIC=1 to serve the synthetic demo models)
Serves the built React dashboard (dashboard/dist) at "/" when it exists.

Owner: Mayank Rathore & Priyanshu Joshi (Sprint 7 preview - API)
"""
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
SUB = "synthetic" if os.environ.get("USE_SYNTHETIC") else ""
MODEL_DIR, REPORT_DIR = ROOT / "models" / SUB, ROOT / "reports" / SUB
DIST = ROOT / "dashboard" / "dist"

app = Flask(__name__, static_folder=str(DIST), static_url_path="")
CORS(app)


@app.before_request
def _start_request_timer():
    """Start a timer without logging student feature values."""
    g.request_started_at = time.perf_counter()


@app.after_request
def _log_request_duration(response):
    """Log request metadata and expose server processing time."""
    started = getattr(g, "request_started_at", None)
    elapsed_ms = (time.perf_counter() - started) * 1000 if started else 0.0
    app.logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f",
        request.method, request.path, response.status_code, elapsed_ms,
    )
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.errorhandler(404)
def _not_found(_error):
    """Return a consistent JSON response for unknown routes."""
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def _method_not_allowed(_error):
    """Return JSON when a known route receives the wrong HTTP method."""
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


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
