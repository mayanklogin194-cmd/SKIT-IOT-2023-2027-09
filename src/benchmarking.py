"""
Sprint 6 benchmarking helpers.

This module compares the models already produced by the project pipeline.
It deliberately does NOT invent published-paper numbers. Literature values can
be added later once the exact papers cited in Form-1 are verified.

Inputs:
    reports/baseline_metrics.json
    reports/rf_metrics.json
    reports/gbm_metrics.json

Outputs:
    reports/benchmark_summary.json
    reports/benchmark_comparison.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_benchmark_summary(reports_dir="reports"):
    reports_dir = Path(reports_dir)

    baseline = load_json(reports_dir / "baseline_metrics.json")
    rf = load_json(reports_dir / "rf_metrics.json")
    gbm = load_json(reports_dir / "gbm_metrics.json")

    regression = []
    for name, metrics in baseline.items():
        regression.append({
            "model": name,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        })
    regression.append({
        "model": "Random Forest",
        "rmse": rf["rmse"],
        "mae": rf["mae"],
        "r2": rf["r2"],
    })
    regression.sort(key=lambda x: x["rmse"])

    classification = []
    for name, metrics in gbm["models"].items():
        classification.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        })
    classification.sort(key=lambda x: x["accuracy"], reverse=True)

    summary = {
        "scope": "Internal benchmark on held-out students from the current OULAD run",
        "regression": regression,
        "classification": classification,
        "best_regression_by_rmse": regression[0]["model"],
        "best_classification_by_accuracy": classification[0]["model"],
        "literature_comparison": {
            "status": "pending_verification",
            "note": "Do not add published benchmark numbers until the exact papers cited in the project form are verified."
        }
    }

    (reports_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def plot_benchmark(summary, out_path="reports/benchmark_comparison.png"):
    regression = summary["regression"]
    labels = [x["model"] for x in regression]
    values = [x["rmse"] for x in regression]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Internal assessment-grade model benchmark")
    ax.bar_label(bars, fmt="%.2f")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    reports = Path("reports")
    summary = build_benchmark_summary(reports)
    plot_benchmark(summary)
    print("Benchmark summary and chart written to reports/.")
