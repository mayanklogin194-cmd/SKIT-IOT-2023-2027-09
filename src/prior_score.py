"""
Previous-score features: what did this student score before the current assessment?

Owner: Atharva Khandal (Sprint 3 - prev-score feature, shifted per student)
"""
import pandas as pd
from config import STUDENT_KEYS


def add_prior_scores(assess: pd.DataFrame) -> pd.DataFrame:
    """prev_score / prev_avg_score use only assessments submitted *before* the current one."""
    df = assess.sort_values(STUDENT_KEYS + ["date_submitted", "id_assessment"]).copy()
    g = df.groupby(STUDENT_KEYS, observed=True)["score"]
    df["prev_score"] = g.shift(1)
    df["prev_avg_score"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["n_prev_assessments"] = df.groupby(STUDENT_KEYS, observed=True).cumcount()
    df["is_first_assessment"] = df["prev_score"].isna().astype(int)
    fill = df["score"].mean()                       # first assessment has no history
    df["prev_score"] = df["prev_score"].fillna(fill)
    df["prev_avg_score"] = df["prev_avg_score"].fillna(fill)
    return df.sort_index()


def student_prior_summary(assess: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Per-student score summary using only assessments submitted by `cutoff`."""
    a = assess[assess["date_submitted"] <= cutoff]
    g = a.groupby(STUDENT_KEYS, observed=True)["score"]
    return pd.DataFrame({"avg_score": g.mean(), "min_score": g.min(),
                         "n_submitted": g.size()}).reset_index()
