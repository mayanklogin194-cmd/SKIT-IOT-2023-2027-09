"""
FR-001, FR-002 (partial): Merges studentInfo + studentAssessment + assessments
into one unified table, cleans missing values, standardizes dtypes.

Owner: Kripmatra Singh & Atharva Khandal (Sprint 2 — core data pipeline)
"""
import pandas as pd


def merge_tables(tables: dict) -> pd.DataFrame:
    assessment_scores = tables["studentAssessment"].merge(
        tables["assessments"][["id_assessment", "weight", "assessment_type"]],
        on="id_assessment",
        how="left",
    )

    merged = assessment_scores.merge(
        tables["studentInfo"],
        on="id_student",
        how="left",
    )
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize dtypes
    numeric_cols = ["score", "weight", "date_submitted", "num_of_prev_attempts", "studied_credits"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no score (can't train/evaluate on those) or no student id
    df = df.dropna(subset=["score", "id_student"])

    # Median-impute remaining numeric gaps rather than dropping more rows
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Standardize categorical text
    for col in ["gender", "disability", "final_result", "assessment_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.drop_duplicates(subset=["id_student", "id_assessment"])
    return df.reset_index(drop=True)
