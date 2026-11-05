"""Analyze engagement segments from saved student prediction records."""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "reports" / "sample_students.json"
OUTPUT_PATH = ROOT / "reports" / "engagement_segment_analysis.json"

def load_students(path=INPUT_PATH):
    """Load and validate the sample student records."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("sample_students.json must contain a list")
    valid = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            clicks = float(row.get("total_clicks", 0))
            days = float(row.get("active_days", 0))
            score = float(row.get("avg_score", 0))
        except (TypeError, ValueError):
            continue
        if clicks < 0 or days < 0:
            continue
        item = dict(row)
        item["_clicks"] = clicks
        item["_days"] = days
        item["_score"] = score
        valid.append(item)
    return valid

def engagement_score(student, click_mid, day_mid):
    """Calculate a transparent relative engagement score."""
    click_part = student["_clicks"] / click_mid if click_mid else 0
    day_part = student["_days"] / day_mid if day_mid else 0
    return round((click_part + day_part) / 2, 3)

def assign_segment(score):
    """Map relative score to low, medium, or high engagement."""
    if score < 0.75:
        return "Low"
    if score < 1.25:
        return "Medium"
    return "High"

def risk_value(student):
    """Use saved failure probability when available; otherwise use risk label."""
    try:
        return float(student.get("p_Fail", 0))
    except (TypeError, ValueError):
        return 1.0 if str(student.get("risk", "")).lower() == "high" else 0.0

def summarize_segment(name, members):
    """Create aggregate metrics without exposing individual student records."""
    count = len(members)
    outcomes = {}
    for student in members:
        outcome = str(student.get("final_result", "Unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    avg = lambda key: round(sum(s[key] for s in members) / count, 2) if count else 0
    return {
        "segment": name,
        "student_count": count,
        "average_clicks": avg("_clicks"),
        "average_active_days": avg("_days"),
        "average_assessment_score": avg("_score"),
        "average_failure_probability": round(sum(risk_value(s) for s in members) / count, 3) if count else 0,
        "outcome_counts": outcomes,
    }

def build_report(students):
    """Build all three segment summaries using sample-relative medians."""
    if not students:
        raise ValueError("No valid student records were found")
    click_mid = median(s["_clicks"] for s in students)
    day_mid = median(s["_days"] for s in students)
    groups = {"Low": [], "Medium": [], "High": []}
    for student in students:
        score = engagement_score(student, click_mid, day_mid)
        groups[assign_segment(score)].append(student)
    return {
        "source": str(INPUT_PATH.name),
        "method": "relative median-normalized clicks and active days",
        "student_records_analyzed": len(students),
        "median_clicks": click_mid,
        "median_active_days": day_mid,
        "segments": [summarize_segment(name, groups[name]) for name in groups],
        "limitations": [
            "Describes saved sample predictions only, not the full OULAD population.",
            "Engagement groups are relative to this sample.",
            "Associations do not establish that engagement causes outcomes."
        ],
    }

def main():
    students = load_students()
    report = build_report(students)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Analyzed {report['student_records_analyzed']} students")
    print(f"Report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
