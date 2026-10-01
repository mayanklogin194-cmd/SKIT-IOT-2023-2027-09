"""
FR-002: Engineers predictive features — prior assessment score, cumulative
VLE clicks, active days, one-hot encoded categoricals.

Owner: Mayank Rathore & Atharva Khandal (Sprint 3 — feature engineering)
"""
import pandas as pd


def add_prior_score_feature(df: pd.DataFrame) -> pd.DataFrame:
    """For each student, prior_score = mean score of their earlier submitted
    assessments (shifted so we never leak the current assessment's own score)."""
    df = df.sort_values(["id_student", "date_submitted"]).copy()
    df["prior_score"] = (
        df.groupby("id_student")["score"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    # First assessment for a student has no prior score — use cohort median
    df["prior_score"] = df["prior_score"].fillna(df["score"].median())
    return df


def add_vle_engagement_features(df: pd.DataFrame, student_vle: pd.DataFrame) -> pd.DataFrame:
    """total_vle_clicks and active_days per student, joined onto the main table."""
    engagement = (
        student_vle.groupby("id_student")
        .agg(total_vle_clicks=("sum_click", "sum"), active_days=("date", "nunique"))
        .reset_index()
    )
    df = df.merge(engagement, on="id_student", how="left")
    df["total_vle_clicks"] = df["total_vle_clicks"].fillna(0)
    df["active_days"] = df["active_days"].fillna(0)
    return df


def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost rejects feature names containing [, ] or <, which can appear
    in one-hot columns generated from raw category text (e.g. age_band '55<=')."""
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\[\]<]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def one_hot_encode(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = [c for c in
                             ["gender", "region", "highest_education", "age_band",
                              "disability", "assessment_type"]
                             if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return _sanitize_column_names(df)


def build_feature_table(merged_df: pd.DataFrame, student_vle: pd.DataFrame) -> pd.DataFrame:
    df = add_prior_score_feature(merged_df)
    df = add_vle_engagement_features(df, student_vle)
    df = one_hot_encode(df)
    return df
