"""Compare model error patterns across final-result classes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "reports" / "gbm_metrics.json"
OUTPUT_FILE = ROOT / "reports" / "class_error_analysis.json"

def load_metrics(path=INPUT_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("models"), dict):
        raise ValueError("Metrics file has no model dictionary")
    return payload

def safe_divide(top, bottom):
    return top / bottom if bottom else 0.0

def analyze_matrix(labels, matrix):
    size = len(labels)
    if size == 0 or len(matrix) != size:
        raise ValueError("Confusion matrix size does not match class labels")
    if any(len(row) != size for row in matrix):
        raise ValueError("Confusion matrix must be square")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("Confusion matrix cannot contain negative counts")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("Confusion matrix contains no observations")
    class_rows = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        actual_count = sum(matrix[index])
        predicted_count = sum(matrix[r][index] for r in range(size))
        false_negative = actual_count - true_positive
        false_positive = predicted_count - true_positive
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        class_rows.append({
            "class": label,
            "actual_count": actual_count,
            "predicted_count": predicted_count,
            "correct": true_positive,
            "misclassified": false_negative,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    hardest = min(class_rows, key=lambda row: row["recall"])
    return {
        "sample_count": total,
        "accuracy": round(safe_divide(sum(matrix[i][i] for i in range(size)), total), 4),
        "class_results": class_rows,
        "lowest_recall_class": hardest["class"],
        "lowest_recall": hardest["recall"],
    }

def build_report(payload):
    output = {"selected_model": payload.get("selected"), "models": {}}
    labels_default = payload.get("classes", [])
    for model_name, model_data in payload["models"].items():
        matrix = model_data.get("confusion_matrix")
        labels = model_data.get("labels", labels_default)
        if matrix is None:
            continue
        output["models"][model_name] = analyze_matrix(labels, matrix)
    if not output["models"]:
        raise ValueError("No confusion matrices were available")
    return output

def main():
    report = build_report(load_metrics())
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved class error analysis for {len(report['models'])} models")

if __name__ == "__main__":
    main()
