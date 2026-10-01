"""
Feature table builder: VLE engagement features (total clicks, active days) and the
orchestration of prior-score + encoding lists into the two model-ready tables.

Owner: Mayank Rathore (Sprint 3 - aggregate VLE clicks -> total_clicks, active_days)
"""
import pandas as pd
from config import STUDENT_KEYS, CUTOFF_DAY
from prior_score import add_prior_scores, student_prior_summary
from encoding import REG_CAT, REG_NUM, CLF_CAT, CLF_NUM


def daily_clicks(vle: pd.DataFrame) -> pd.DataFrame:
    """Collapse the ~10M click rows to one row per student per course day (+ running totals)."""
    d = (vle.groupby(STUDENT_KEYS + ["date"], observed=True, sort=False)["sum_click"]
            .sum().reset_index().sort_values(STUDENT_KEYS + ["date"]))
    d["date"] = d["date"].astype("int32")
    d["cum_clicks"] = d.groupby(STUDENT_KEYS, observed=True)["sum_click"].cumsum()
    d["cum_active_days"] = d.groupby(STUDENT_KEYS, observed=True).cumcount() + 1
    return d


def add_engagement_before_assessment(assess: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach clicks / active days accumulated up to each assessment's submission day."""
    left, right = assess.copy(), daily[STUDENT_KEYS + ["date", "cum_clicks", "cum_active_days"]].copy()
    left["_d"] = left["date_submitted"].astype("int32")
    right = right.rename(columns={"date": "_d"})
    for k in ("code_module", "code_presentation"):
        left[k], right[k] = left[k].astype(str), right[k].astype(str)
    left["id_student"], right["id_student"] = left["id_student"].astype("int64"), right["id_student"].astype("int64")
    out = pd.merge_asof(left.sort_values("_d"), right.sort_values("_d"),
                        on="_d", by=STUDENT_KEYS, direction="backward")
    out[["cum_clicks", "cum_active_days"]] = out[["cum_clicks", "cum_active_days"]].fillna(0)
    return out.drop(columns="_d").sort_index()


def student_engagement(daily: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    d = daily[daily["date"] <= cutoff]
    g = d.groupby(STUDENT_KEYS, observed=True)
    return pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "active_days": g.size(),
        "clicks_last_30d": d[d["date"] > cutoff - 30].groupby(STUDENT_KEYS, observed=True)["sum_click"].sum(),
    }).fillna(0).reset_index()


def _validate(df, numeric, categorical, name):
    missing = [c for c in numeric + categorical if c not in df.columns]
    assert not missing, f"{name}: missing feature columns {missing}"
    nans = df[numeric].isna().sum()
    assert nans.sum() == 0, f"{name}: NaNs left in {nans[nans > 0].to_dict()}"
    print(f"  {name}: {len(df):,} rows x {len(numeric) + len(categorical)} features - OK")


def build_feature_table(cleaned: dict, vle: pd.DataFrame, cutoff: int = CUTOFF_DAY):
    """Returns (assessment_features, student_features)."""
    daily = daily_clicks(vle)

    assess = add_prior_scores(cleaned["assessments"])
    assess = add_engagement_before_assessment(assess, daily)
    _validate(assess, REG_NUM, REG_CAT, "assessment table")

    students = cleaned["students"]
    # students who unregistered before the cut-off are not "early-warning" cases
    students = students[~(students["date_unregistration"] <= cutoff)]
    students = students.merge(student_engagement(daily, cutoff), on=STUDENT_KEYS, how="left")
    students = students.merge(student_prior_summary(assess, cutoff), on=STUDENT_KEYS, how="left")
    students[["total_clicks", "active_days", "clicks_last_30d", "n_submitted",
              "avg_score", "min_score"]] = students[["total_clicks", "active_days", "clicks_last_30d",
                                                     "n_submitted", "avg_score", "min_score"]].fillna(0)
    _validate(students, CLF_NUM, CLF_CAT, "student table")
    return assess, students
