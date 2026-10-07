"""
FR-005/NFR-001: Flask REST API serving model predictions to the teacher and student
dashboards. Target: respond within 2 seconds of a request (NFR-001).

Run:   python api/app.py            (add USE_SYNTHETIC=1 to serve the synthetic demo models)
Serves the built React dashboard (dashboard/dist) at "/" when it exists.

Owner: Mayank Rathore & Priyanshu Joshi 
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
SUB = "synthetic" if os.environ.get("USE_SYNTHETIC") else ""
MODEL_DIR, REPORT_DIR = ROOT / "models" / SUB, ROOT / "reports" / SUB
DIST = ROOT / "dashboard" / "dist"

app = Flask(__name__, static_folder=str(DIST), static_url_path="")
CORS(app)

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
    row = dict(defaults)
    row.update({k: v for k, v in (features or {}).items() if k in row})
    return pd.DataFrame([row])[bundle["feature_columns"]]


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
    feats = (request.get_json(force=True) or {}).get("features", {})
    row = _row(bundle, _report("form_defaults_grade.json")["defaults"], feats)
    pred = float(bundle["model"].predict(row)[0])
    return jsonify({"predicted_score": round(min(max(pred, 0), 100), 2)})


@app.post("/predict/outcome")
def predict_outcome():
    """Body: {"features": {...}} -> predicted outcome, class probabilities and an early-warning risk level."""
    bundle = _bundle("gbm_classifier")
    if bundle is None:
        return jsonify({"error": "gbm_classifier not trained - run python src/main.py"}), 503
    feats = (request.get_json(force=True) or {}).get("features", {})
    row = _row(bundle, _report("form_defaults_outcome.json")["defaults"], feats)
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
