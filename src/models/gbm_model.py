"""
FR-004: Gradient Boosting classifier (XGBoost, with LightGBM as a fallback) —
predicts final student outcome (Pass / Fail / Distinction / Withdrawn).

Owner: Priyanshu Joshi & Atharva Khandal (Sprint 5)
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    BACKEND = "xgboost"
except ImportError:
    from lightgbm import LGBMClassifier
    BACKEND = "lightgbm"

DROP_COLS = ["score", "id_student", "id_assessment", "final_result",
             "code_module", "code_presentation"]


def get_feature_target(df, target="final_result"):
    y = df[target]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool"])
    return X, y


def _make_model(**kwargs):
    if BACKEND == "xgboost":
        return XGBClassifier(eval_metric="mlogloss", random_state=42, **kwargs)
    return LGBMClassifier(random_state=42, **kwargs)


def train_gbm(df, tune: bool = False, model_path: str = "models/gbm_classifier.joblib"):
    # One row per student for outcome classification (dedupe from per-assessment rows)
    student_level = df.drop_duplicates(subset="id_student")
    X, y_raw = get_feature_target(student_level)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if tune:
        param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
        search = GridSearchCV(_make_model(), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best {BACKEND} params:", search.best_params_)
    else:
        model = _make_model(n_estimators=300, max_depth=5, learning_rate=0.1)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"{BACKEND} outcome classifier — accuracy: {acc:.3f}, macro-F1: {f1:.3f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "backend": BACKEND,
    }, model_path)

    return model, acc, f1, list(X.columns)
