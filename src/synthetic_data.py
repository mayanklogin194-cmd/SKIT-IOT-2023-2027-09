"""
Generates fake data matching the OULAD schema, so the rest of the pipeline
can be built/tested before the real dataset is downloaded.

Owner: Mayank Rathore (Sprint 1 — dataset setup)
"""
import numpy as np
import pandas as pd


def generate_synthetic_oulad(n_students: int = 500, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    student_ids = np.arange(100000, 100000 + n_students)

    student_info = pd.DataFrame({
        "id_student": student_ids,
        "code_module": rng.choice(["AAA", "BBB", "CCC"], n_students),
        "code_presentation": rng.choice(["2013J", "2014B"], n_students),
        "gender": rng.choice(["M", "F"], n_students),
        "region": rng.choice(["East Anglian Region", "Scotland", "London Region"], n_students),
        "highest_education": rng.choice(
            ["A Level or Equivalent", "HE Qualification", "Lower Than A Level"], n_students
        ),
        "age_band": rng.choice(["0-35", "35-55", "55<="], n_students),
        "num_of_prev_attempts": rng.integers(0, 3, n_students),
        "studied_credits": rng.choice([60, 90, 120], n_students),
        "disability": rng.choice(["Y", "N"], n_students, p=[0.1, 0.9]),
        "final_result": rng.choice(
            ["Pass", "Fail", "Withdrawn", "Distinction"], n_students, p=[0.4, 0.25, 0.2, 0.15]
        ),
    })

    assessments = pd.DataFrame({
        "id_assessment": np.arange(1, 21),
        "code_module": rng.choice(["AAA", "BBB", "CCC"], 20),
        "code_presentation": rng.choice(["2013J", "2014B"], 20),
        "assessment_type": rng.choice(["TMA", "CMA", "Exam"], 20),
        "date": rng.integers(20, 240, 20),
        "weight": rng.choice([10, 20, 25, 50], 20),
    })

    rows = []
    for sid in student_ids:
        n_assess = rng.integers(3, 7)
        for aid in rng.choice(assessments["id_assessment"], n_assess, replace=False):
            rows.append({
                "id_student": sid,
                "id_assessment": aid,
                "date_submitted": rng.integers(1, 250),
                "is_banked": 0,
                "score": np.clip(rng.normal(65, 20), 0, 100).round(1),
            })
    student_assessment = pd.DataFrame(rows)

    vle_rows = []
    for sid in student_ids:
        n_clicks_days = rng.integers(10, 120)
        for _ in range(n_clicks_days):
            vle_rows.append({
                "id_student": sid,
                "date": rng.integers(-25, 250),
                "sum_click": rng.integers(1, 30),
            })
    student_vle = pd.DataFrame(vle_rows)

    return {
        "studentInfo": student_info,
        "assessments": assessments,
        "studentAssessment": student_assessment,
        "studentVle": student_vle,
    }


if __name__ == "__main__":
    tables = generate_synthetic_oulad()
    for name, df in tables.items():
        print(name, df.shape)
