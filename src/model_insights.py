"""Create readable model insights from the project's saved evaluation reports.

Run from the repository root:
    python src/model_insights.py

This reports global feature importance and model metrics. It does not explain an
individual prediction or imply that feature importance is causal.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_insights(rf_metrics, gbm_metrics, top_n=5):
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    rf_features = rf_metrics.get("feature_importance", [])
    selected_name = gbm_metrics.get("selected")
    models = gbm_metrics.get("models", {})
    if selected_name not in models:
        raise ValueError("Selected GBM model is missing from its metrics report")
    gbm_features = models[selected_name].get("feature_importance", [])

    def top_features(items):
        return [
            {"feature": item["feature"], "importance": float(item["importance"])}
            for item in sorted(items, key=lambda x: x.get("importance", 0), reverse=True)[:top_n]
        ]

    return {
        "scope": "Global model-level feature importance; not a per-student explanation.",
        "grade_model": {
            "model": "Random Forest Regressor",
            "rmse": float(rf_metrics["rmse"]),
            "mae": float(rf_metrics["mae"]),
            "r2": float(rf_metrics["r2"]),
            "top_features": top_features(rf_features),
        },
        "outcome_model": {
            "model": selected_name,
            "accuracy": float(models[selected_name]["accuracy"]),
            "macro_f1": float(models[selected_name]["f1_macro"]),
            "top_features": top_features(gbm_features),
        },
        "caution": "Feature importance describes model usage, not causation or guaranteed effects.",
    }


def main():
    insights = build_insights(
        read_json(REPORTS / "rf_metrics.json"),
        read_json(REPORTS / "gbm_metrics.json"),
    )
    output = REPORTS / "model_insights.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print(f"Model insights written to {output}")


if __name__ == "__main__":
    main()
