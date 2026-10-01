"""
Sprint 4 baselines: Linear Regression + Decision Tree (compare against the Random Forest).

Owner: Priyanshu Joshi (Sprint 4 - baseline linear regression & decision tree)
"""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from config import RANDOM_STATE
from encoding import REG_CAT, REG_NUM, make_preprocessor
from evaluation import group_split, regression_report, save_json


def train_baselines(assess_df, reports_dir):
    train, test = group_split(assess_df, assess_df["id_student"])
    X = REG_NUM + REG_CAT
    models = {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=10, min_samples_leaf=30,
                                                random_state=RANDOM_STATE), False),
    }
    out = {}
    for name, (model, scale) in models.items():
        pipe = Pipeline([("prep", make_preprocessor(REG_NUM, REG_CAT, scale=scale)), ("model", model)])
        pipe.fit(train[X], train["score"])
        out[name] = regression_report(test["score"], pipe.predict(test[X]))
        print(f"  {name:18s} RMSE={out[name]['rmse']:.2f}  R2={out[name]['r2']:.3f}")
    save_json(reports_dir, "baseline_metrics.json", out)
    return out
