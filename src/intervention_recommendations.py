"""Create actionable early-warning recommendations for saved student records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "reports" / "sample_students.json"
OUTPUT_FILE = ROOT / "reports" / "intervention_recommendations.json"

def load_records(path=INPUT_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("Student report must contain a JSON list")
    return [row for row in rows if isinstance(row, dict)]

def numeric_value(record, field, default=0.0):
    try:
        value = float(record.get(field, default))
    except (TypeError, ValueError):
        return float(default)
    return value if value >= 0 else float(default)

def risk_band(risk):
    if risk >= 0.70:
        return "High"
    if risk >= 0.40:
        return "Medium"
    return "Low"

def engagement_band(clicks, active_days):
    if clicks < 100 or active_days < 8:
        return "Low"
    if clicks < 300 or active_days < 20:
        return "Medium"
    return "High"

def recommend_action(risk_level, engagement_level, score):
    if risk_level == "High" and engagement_level == "Low":
        return "Contact student; offer a tutor check-in and a short study plan."
    if risk_level == "High":
        return "Review recent assessments and arrange targeted academic support."
    if risk_level == "Medium" and engagement_level == "Low":
        return "Send a learning reminder and recommend the next course activity."
    if risk_level == "Medium":
        return "Monitor the next assessment and share relevant practice material."
    if score < 50:
        return "Offer optional revision resources before the next assessment."
    return "Continue normal progress monitoring."

def build_recommendations(records):
    results = []
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for row in records:
        risk = numeric_value(row, "risk")
        clicks = numeric_value(row, "total_clicks")
        days = numeric_value(row, "active_days")
        score = numeric_value(row, "avg_score")
        risk_level = risk_band(risk)
        engagement_level = engagement_band(clicks, days)
        counts[risk_level] += 1
        results.append({
            "student_id": row.get("id_student"),
            "module": row.get("code_module", "Unknown"),
            "risk_level": risk_level,
            "engagement_level": engagement_level,
            "recommended_action": recommend_action(risk_level, engagement_level, score),
        })
    results.sort(key=lambda item: {"High": 0, "Medium": 1, "Low": 2}[item["risk_level"]])
    return {
        "records_processed": len(results),
        "risk_counts": counts,
        "recommendations": results,
        "note": "Rule-based suggestions for review; not automated decisions about students."
    }

def main():
    report = build_recommendations(load_records())
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Created {len(report['recommendations'])} recommendations at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
