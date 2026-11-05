"""Compare early-warning alert volumes at several risk thresholds.

Run from repository root:
    python src/risk_threshold_analysis.py

Input: reports/sample_students.json
Output: reports/risk_threshold_analysis.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze_thresholds(students, thresholds=(0.30, 0.40, 0.50, 0.60)):
    if not thresholds or any(not 0 <= float(t) <= 1 for t in thresholds):
        raise ValueError("Thresholds must be a non-empty list of values from 0 to 1.")
    rows = []
    total = len(students)
    for threshold in thresholds:
        alerts = [s for s in students if float(s.get("risk", 0)) >= float(threshold)]
        actual_positive = sum(
            str(s.get("final_result", "")).strip().lower() in {"fail", "withdrawn"}
            for s in alerts
        )
        rows.append({
            "threshold": float(threshold),
            "alert_count": len(alerts),
            "alert_percentage": round(100 * len(alerts) / total, 2) if total else 0.0,
            "actual_fail_or_withdrawn_in_alerts": int(actual_positive),
            "observed_fail_or_withdrawn_rate_in_alerts": round(
                actual_positive / len(alerts), 4
            ) if alerts else 0.0,
        })
    return {
        "total_students": total,
        "risk_definition": "Predicted probability of Fail plus Withdrawn.",
        "interpretation": (
            "Threshold comparisons use saved sample predictions. They help inspect "
            "alert volume and observed outcomes; they do not establish a universally optimal threshold."
        ),
        "thresholds": rows,
    }


def main():
    reports = ROOT / "reports"
    students = json.loads((reports / "sample_students.json").read_text(encoding="utf-8"))
    result = analyze_thresholds(students)
    output = reports / "risk_threshold_analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Risk threshold analysis written to {output}")


if __name__ == "__main__":
    main()
