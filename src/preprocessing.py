"""
FR-001: merge the linked OULAD tables, clean missing values, standardize dtypes.

Owner: Priyanshu Joshi (Sprint 2 - core data pipeline & preprocessing)
"""
import pandas as pd
from config import STUDENT_KEYS

CAT_COLS = ["code_module", "code_presentation", "gender", "region", "highest_education",
            "imd_band", "age_band", "disability", "assessment_type"]


def merge_tables(tables: dict) -> dict:
    """Returns two merged views:
      assessments - one row per (student, assessment) with demographics attached
      students    - one row per (student, course presentation) with registration info
    """
    sa = tables["studentAssessment"].merge(tables["assessments"], on="id_assessment", how="left")
    assessments = sa.merge(tables["studentInfo"], on=STUDENT_KEYS, how="inner")

    reg = tables["studentRegistration"][STUDENT_KEYS + ["date_registration", "date_unregistration"]]
    students = tables["studentInfo"].merge(reg, on=STUDENT_KEYS, how="left")
    return {"assessments": assessments, "students": students}


def clean_data(merged: dict) -> dict:
    a = merged["assessments"].copy()
    a = a.dropna(subset=["score"])                      # nothing to learn from a missing grade
    # some final exams have no planned date -> use the day they were submitted
    a["date"] = a["date"].fillna(a["date_submitted"])
    a["weight"] = a["weight"].astype(float)
    a["is_banked"] = a["is_banked"].astype(int)

    s = merged["students"].copy()
    for df in (a, s):
        df["imd_band"] = df["imd_band"].fillna("Unknown")
        for c in CAT_COLS:
            if c in df.columns:
                df[c] = df[c].astype("category")
        for c in ("id_student", "num_of_prev_attempts", "studied_credits"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int32")
    return {"assessments": a, "students": s}
