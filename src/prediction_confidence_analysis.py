"""Evaluate confidence values stored with saved student predictions."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "reports" / "sample_students.json"
OUTPUT_PATH = ROOT / "reports" / "prediction_confidence_analysis.json"
PROBABILITY_FIELDS = ("p_Distinction", "p_Fail", "p_Pass", "p_Withdrawn")
LOW_LIMIT = 0.50
HIGH_LIMIT = 0.75

def load_records(path=INPUT_PATH):
    """Read the prediction sample file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Expected a list of student records")
    return records

def extract_probabilities(record):
    """Return a validated, normalized probability distribution."""
    values = []
    for field in PROBABILITY_FIELDS:
        try:
            value = float(record[field])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing or invalid probability: {field}")
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"Probability {field} must be between 0 and 1")
        values.append(value)
    total = sum(values)
    if total <= 0:
        raise ValueError("Probability total must be greater than zero")
    return [value / total for value in values]

def entropy(probabilities):
    """Calculate Shannon entropy in natural-log units."""
    return -sum(p * math.log(p) for p in probabilities if p > 0)

def analyze_record(record):
    """Add predicted class, confidence, entropy, and correctness."""
    probabilities = extract_probabilities(record)
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    predicted = PROBABILITY_FIELDS[index].replace("p_", "")
    actual = str(record.get("final_result", "Unknown"))
    confidence = probabilities[index]
    return {
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "entropy": round(entropy(probabilities), 4),
        "actual_class": actual,
        "correct": predicted.lower() == actual.lower(),
    }

def confidence_band(confidence):
    """Classify the maximum probability into a confidence band."""
    if confidence < LOW_LIMIT:
        return "Low"
    if confidence < HIGH_LIMIT:
        return "Medium"
    return "High"

def summarize_band(name, rows):
    """Aggregate confidence and observed accuracy for one band."""
    count = len(rows)
    if not count:
        return {"band": name, "count": 0, "average_confidence": 0,
                "observed_accuracy": None, "average_entropy": 0}
    return {
        "band": name,
        "count": count,
        "average_confidence": round(sum(r["confidence"] for r in rows) / count, 4),
        "observed_accuracy": round(sum(r["correct"] for r in rows) / count, 4),
        "average_entropy": round(sum(r["entropy"] for r in rows) / count, 4),
    }

def build_report(records):
    """Analyze all valid records and retain only aggregate report data."""
    analyzed = []
    skipped = 0
    for record in records:
        try:
            analyzed.append(analyze_record(record))
        except (ValueError, TypeError):
            skipped += 1
    if not analyzed:
        raise ValueError("No records contained valid class probabilities")
    bands = {"Low": [], "Medium": [], "High": []}
    for row in analyzed:
        bands[confidence_band(row["confidence"])].append(row)
    correct = sum(row["correct"] for row in analyzed)
    return {
        "records_analyzed": len(analyzed),
        "records_skipped": skipped,
        "overall_observed_accuracy": round(correct / len(analyzed), 4),
        "confidence_bands": [summarize_band(name, rows) for name, rows in bands.items()],
        "limitations": [
            "Confidence is the largest model probability, not a guarantee of correctness.",
            "Observed accuracy is measured on saved sample records only.",
            "This report does not establish probability calibration."
        ],
    }

def main():
    report = build_report(load_records())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Analyzed {report['records_analyzed']} prediction records")
    print(f"Skipped {report['records_skipped']} invalid records")
    print(f"Report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
