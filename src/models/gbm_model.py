"""
FR-004: gradient-boosting classifier (XGBoost / LightGBM) that predicts the final outcome
(Distinction / Pass / Fail / Withdrawn) from activity up to the early-warning cut-off day.
Evaluated with Accuracy and macro-F1; the better of the two boosters is kept.

Owner: Atharva Khandal (Sprint 5 - train & tune GBM classifier)
"""
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from config import RANDOM_STATE
from encoding import CLF_CAT, CLF_NUM, make_preprocessor
from evaluation import classification_report_dict, group_split, rolled_importance, save_json

DEFAULT_PARAMS = {"model__n_estimators": 150, "model__max_depth": 6,
                  "model__learning_rate": 0.05, "model__subsample": 0.8}


def train_gbm(student_df, models_dir, reports_dir, tune: bool = False):
    train, test = group_split(student_df, student_df["id_student"])
    X_cols = CLF_NUM + CLF_CAT
    le = LabelEncoder().fit(student_df["final_result"])
    y_tr, y_te = le.transform(train["final_result"]), le.transform(test["final_result"])
    weights = compute_sample_weight("balanced", y_tr)      # Distinction is rare

    boosters = {
        "XGBoost": XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                 random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
        "LightGBM": LGBMClassifier(objective="multiclass", random_state=RANDOM_STATE,
                                   n_jobs=1, verbose=-1),
    }
    space = {"model__n_estimators": [150, 250], "model__max_depth": [3, 4, 6],
             "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8, 1.0]}

    results = {}
    for name, est in boosters.items():
        pipe = Pipeline([("prep", make_preprocessor(CLF_NUM, CLF_CAT)), ("model", est)])
        if tune:
            cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
            search = RandomizedSearchCV(pipe, space, n_iter=6, cv=cv, scoring="f1_macro",
                                        random_state=RANDOM_STATE, n_jobs=1)
            search.fit(train[X_cols], y_tr, model__sample_weight=weights)
            pipe, params = search.best_estimator_, search.best_params_
        else:
            params = DEFAULT_PARAMS
            pipe.set_params(**params).fit(train[X_cols], y_tr, model__sample_weight=weights)
        rep = classification_report_dict(y_te, pipe.predict(test[X_cols]), le.classes_)
        rep.update(params={k.split("__")[1]: v for k, v in params.items()},
                   feature_importance=rolled_importance(pipe, CLF_CAT))
        results[name] = (pipe, rep)
        print(f"  {name:9s} accuracy={rep['accuracy']:.3f}  macro-F1={rep['f1_macro']:.3f}")

    best = max(results, key=lambda n: results[n][1]["f1_macro"])
    pipe, rep = results[best]
    save_json(reports_dir, "gbm_metrics.json", {
        "selected": best, "classes": list(le.classes_), "n_train": len(train), "n_test": len(test),
        "cutoff_note": "features use activity up to the early-warning cut-off day",
        "models": {n: r for n, (_, r) in results.items()}})

    # held-out students with predicted risk -> teacher early-warning list
    proba = pipe.predict_proba(test[X_cols])
    sample = test.assign(**{f"p_{c}": proba[:, i] for i, c in enumerate(le.classes_)})
    sample = sample.sample(min(40, len(sample)), random_state=RANDOM_STATE)
    sample["risk"] = sample.get("p_Fail", 0) + sample.get("p_Withdrawn", 0)
    cols = ["code_module", "id_student", "total_clicks", "active_days", "avg_score",
            "final_result", "risk"] + [f"p_{c}" for c in le.classes_]
    save_json(reports_dir, "sample_students.json",
              sample[cols].sort_values("risk", ascending=False).round(3).to_dict("records"))

    defaults = {c: (student_df[c].median() if c in CLF_NUM else student_df[c].mode()[0]) for c in X_cols}
    save_json(reports_dir, "form_defaults_outcome.json", {
        "defaults": {k: (v.item() if hasattr(v, "item") else v) for k, v in defaults.items()},
        "choices": {c: sorted(student_df[c].astype(str).unique().tolist()) for c in CLF_CAT}})

    joblib.dump({"model": pipe, "feature_columns": X_cols, "label_encoder": le},
                models_dir / "gbm_classifier.joblib", compress=3)
    return pipe, rep, X_cols
