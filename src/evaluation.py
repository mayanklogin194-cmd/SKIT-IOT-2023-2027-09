"""
FR-003/FR-004 evaluation + NFR benchmarking: RMSE, Accuracy, F1, cross-validation,
and comparison charts (published benchmarks vs our models).


"""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def cross_validate_regressor(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    rmse_scores = -scores
    print(f"CV RMSE: mean={rmse_scores.mean():.3f}, std={rmse_scores.std():.3f}")
    return rmse_scores


def cross_validate_classifier(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"CV macro-F1: mean={scores.mean():.3f}, std={scores.std():.3f}")
    return scores


def plot_rmse_comparison(our_rmse: float, published_benchmarks: dict, out_path="reports/rmse_comparison.png"):
    """published_benchmarks e.g. {'Atthibyani (2024)': 8.9, 'Jha et al. (2019)': 9.4}"""
    labels = list(published_benchmarks.keys()) + ["Our Random Forest"]
    values = list(published_benchmarks.values()) + [our_rmse]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#9aa5b1"] * len(published_benchmarks) + ["#2f6feb"])
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Assessment grade prediction — RMSE vs published benchmarks")
    ax.bar_label(bars, fmt="%.2f")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to {out_path}")


def plot_feature_importance(model, feature_names, top_n=15, out_path="reports/feature_importance.png"):
    importances = model.feature_importances_
    order = importances.argsort()[::-1][:top_n]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#2f6feb")
    ax.set_title("Feature importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved feature importance chart to {out_path}")
