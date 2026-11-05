"""Tests for this week's model-insight, risk-threshold, and feedback work."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model_insights import build_insights
from risk_threshold_analysis import analyze_thresholds
from api import app as api_module


def test_model_insights_selects_top_features():
    rf = {
        "rmse": 12.0, "mae": 8.0, "r2": 0.5,
        "feature_importance": [
            {"feature": "prev_score", "importance": 0.7},
            {"feature": "cum_clicks", "importance": 0.3},
        ],
    }
    gbm = {
        "selected": "XGBoost",
        "models": {"XGBoost": {
            "accuracy": 0.8, "f1_macro": 0.75,
            "feature_importance": [{"feature": "avg_score", "importance": 0.9}],
        }},
    }
    result = build_insights(rf, gbm, top_n=1)
    assert result["grade_model"]["top_features"][0]["feature"] == "prev_score"
    assert result["outcome_model"]["model"] == "XGBoost"
    assert "not causation" in result["caution"]


def test_risk_thresholds_are_monotonic():
    students = [
        {"risk": 0.25, "final_result": "Pass"},
        {"risk": 0.45, "final_result": "Fail"},
        {"risk": 0.80, "final_result": "Withdrawn"},
    ]
    result = analyze_thresholds(students, thresholds=(0.3, 0.5, 0.7))
    counts = [row["alert_count"] for row in result["thresholds"]]
    assert counts == [2, 1, 1]
    assert result["thresholds"][0]["actual_fail_or_withdrawn_in_alerts"] == 2


def test_risk_threshold_rejects_invalid_threshold():
    try:
        analyze_thresholds([], thresholds=(1.2,))
    except ValueError as exc:
        assert "from 0 to 1" in str(exc)
    else:
        raise AssertionError("Invalid threshold should raise ValueError")


def test_feedback_endpoint_persists_minimal_record(monkeypatch, tmp_path):
    feedback_path = tmp_path / "prediction_feedback.jsonl"
    monkeypatch.setattr(api_module, "FEEDBACK_PATH", feedback_path)
    response = api_module.app.test_client().post(
        "/feedback",
        json={
            "prediction_type": "grade",
            "prediction": 82.5,
            "feedback": "useful",
            "comment": "Looks reasonable",
        },
    )
    assert response.status_code == 201
    record = json.loads(feedback_path.read_text(encoding="utf-8").strip())
    assert record["prediction"] == 82.5
    assert record["feedback"] == "useful"
    assert "features" not in record


def test_feedback_endpoint_rejects_invalid_grade(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    response = api_module.app.test_client().post(
        "/feedback",
        json={
            "prediction_type": "grade",
            "prediction": 140,
            "feedback": "useful",
        },
    )
    assert response.status_code == 400
