"""Build per-class precision, recall and F1 analysis from saved GBM confusion matrices.

Run from project root: python src/classification_analysis.py
Input: reports/gbm_metrics.json
Output: reports/classification_analysis.json
"""
import json
from pathlib import Path


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def analyze_confusion_matrix(labels, matrix):
    if not labels or len(matrix) != len(labels):
        raise ValueError("Labels and confusion matrix dimensions must match.")
    if any(len(row) != len(labels) for row in matrix):
        raise ValueError("Confusion matrix must be square and match labels.")
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("Confusion matrix must contain at least one sample.")

    rows = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(len(labels)))
        fp = predicted - tp
        fn = support - tp
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        rows.append({"class": label, "precision": precision, "recall": recall,
                     "f1": f1, "support": support})

    macro = {key: sum(row[key] for row in rows) / len(rows)
             for key in ("precision", "recall", "f1")}
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return {"accuracy_from_matrix": correct / total, "per_class": rows,
            "macro_average": macro, "sample_count": total}


def build_classification_analysis(metrics_path="reports/gbm_metrics.json",
                                  output_path="reports/classification_analysis.json"):
    metrics_path, output_path = Path(metrics_path), Path(output_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = {"selected_model": payload["selected"], "models": {}}
    for model_name, model in payload["models"].items():
        matrix = model.get("confusion_matrix")
        labels = model.get("labels", payload.get("classes", []))
        if matrix is None:
            raise ValueError(f"No confusion_matrix found for {model_name}.")
        results["models"][model_name] = analyze_confusion_matrix(labels, matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    result = build_classification_analysis()
    print(f"Wrote per-class analysis for {len(result['models'])} models.")
