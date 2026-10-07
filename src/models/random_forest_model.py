"""
FR-003: Random Forest regressor that forecasts an individual assessment grade (0-100)
from the learner's prior scores, VLE engagement so far and demographics. Evaluated with RMSE.

Owner: Mayank Rathore (Sprint 4 - train & tune Random Forest regressor)
"""
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from config import RANDOM_STATE
from encoding import REG_CAT, REG_NUM, make_preprocessor
from evaluation import group_split, regression_report, rolled_importance, save_json

DEFAULT_PARAMS = {"model__n_estimators": 80, "model__max_depth": 14,
                  "model__min_samples_leaf": 5, "model__max_features": 0.3}


def get_feature_target(df):
    return df[REG_NUM + REG_CAT], df["score"]


def train_random_forest(assess_df, models_dir, reports_dir, tune: bool = False):
    train, test = group_split(assess_df, assess_df["id_student"])
    X_cols = REG_NUM + REG_CAT
    pipe = Pipeline([("prep", make_preprocessor(REG_NUM, REG_CAT)),
                     ("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE))])

    params = dict(DEFAULT_PARAMS)
    if tune:  # small randomized search on a sample keeps this feasible on a free Colab CPU
        sample = train.sample(min(30000, len(train)), random_state=RANDOM_STATE)
        space = {"model__n_estimators": [40, 60], "model__max_depth": [10, 14, 18],
                 "model__min_samples_leaf": [5, 10, 20], "model__max_features": [0.3, 0.5, 0.8]}
        search = RandomizedSearchCV(pipe, space, n_iter=6, cv=3, random_state=RANDOM_STATE,
                                    scoring="neg_root_mean_squared_error", n_jobs=1)
        search.fit(sample[X_cols], sample["score"])
        params = {**search.best_params_, "model__n_estimators": 80}
        print("  best params:", params)

    pipe.set_params(**params).fit(train[X_cols], train["score"])
    report = regression_report(test["score"], pipe.predict(test[X_cols]))
    report.update(params={k.split("__")[1]: v for k, v in params.items()},
                  feature_importance=rolled_importance(pipe, REG_CAT),
                  n_train=len(train), n_test=len(test))
    print(f"  Random Forest      RMSE={report['rmse']:.2f}  MAE={report['mae']:.2f}  R2={report['r2']:.3f}")
    save_json(reports_dir, "rf_metrics.json", report)

    # defaults + dropdown choices for the dashboard's "predict next grade" form
    defaults = {c: (assess_df[c].median() if c in REG_NUM else assess_df[c].mode()[0]) for c in X_cols}
    save_json(reports_dir, "form_defaults_grade.json", {
        "defaults": {k: (v.item() if hasattr(v, "item") else v) for k, v in defaults.items()},
        "choices": {c: sorted(assess_df[c].astype(str).unique().tolist()) for c in REG_CAT}})

    joblib.dump({"model": pipe, "feature_columns": X_cols}, models_dir / "random_forest.joblib", compress=3)
    return pipe, report, X_cols
