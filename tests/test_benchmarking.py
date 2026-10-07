import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmarking import build_benchmark_summary


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_benchmark_summary_selects_best_models(tmp_path):
    write_json(tmp_path / "baseline_metrics.json", {
        "Linear Regression": {"rmse": 15.91, "mae": 11.2, "r2": 0.30},
        "Decision Tree": {"rmse": 14.63, "mae": 10.5, "r2": 0.40},
    })
    write_json(tmp_path / "rf_metrics.json", {
        "rmse": 14.24, "mae": 10.1, "r2": 0.44,
    })
    write_json(tmp_path / "gbm_metrics.json", {
        "selected": "LightGBM",
        "models": {
            "XGBoost": {"accuracy": 0.542, "f1_macro": 0.50, "f1_weighted": 0.52},
            "LightGBM": {"accuracy": 0.543, "f1_macro": 0.51, "f1_weighted": 0.52},
        },
    })

    summary = build_benchmark_summary(tmp_path)

    assert summary["best_regression_by_rmse"] == "Random Forest"
    assert summary["best_classification_by_accuracy"] == "LightGBM"
    assert (tmp_path / "benchmark_summary.json").exists()
