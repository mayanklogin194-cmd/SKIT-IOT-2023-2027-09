"""
FR-003/FR-004 evaluation helpers: RMSE, Accuracy, F1, grouped cross-validation,
leak-free splitting, feature-importance roll-up and comparison charts.

Owner: Kripendra Singh (Sprint 4-5 metrics) & Atharva Khandal (Sprint 6 benchmarking, to extend)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from config import RANDOM_STATE


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def regression_report(y, p) -> dict:
    return {"rmse": rmse(y, p), "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))}


def classification_report_dict(y, p, labels) -> dict:
    return {"accuracy": float(accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro")),
            "f1_weighted": float(f1_score(y, p, average="weighted")),
            "labels": list(labels),
            "confusion_matrix": confusion_matrix(y, p).tolist()}


def group_split(df, groups, test_size=0.2):
    """Train/test split where one student never appears on both sides (no leakage)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(gss.split(df, groups=groups))
    return df.iloc[tr], df.iloc[te]


def grouped_cv(model, X, y, groups, scoring, folds=3):
    scores = cross_val_score(model, X, y, groups=groups, scoring=scoring, cv=GroupKFold(folds))
    print(f"CV {scoring}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    return scores


def rolled_importance(pipe, categorical, top=15) -> list:
    """Sum one-hot columns back to their source feature so charts stay readable."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
        n = n.split("__", 1)[1]
        base = next((c for c in categorical if n.startswith(c + "_")), n)
        agg[base] = agg.get(base, 0.0) + float(v)
    total = sum(agg.values()) or 1.0
    items = sorted(((k, v / total) for k, v in agg.items()), key=lambda x: -x[1])[:top]
    return [{"feature": k, "importance": round(v, 4)} for k, v in items]


def save_json(reports_dir, name, obj) -> None:
    (Path(reports_dir) / name).write_text(json.dumps(obj, indent=2))


def plot_rmse_comparison(rmse_by_model: dict, out_path):
    """Bar chart of our models' RMSE. (Sprint 6: add the cited published benchmarks here.)"""
    labels, values = list(rmse_by_model), list(rmse_by_model.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#9aa5b1"] * (len(values) - 1) + ["#2f6feb"])
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Assessment grade prediction - RMSE by model")
    ax.bar_label(bars, fmt="%.2f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(items: list, title: str, out_path, top_n=12):
    items = items[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([i["feature"] for i in items], [i["importance"] for i in items], color="#2f6feb")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
