"""
FR-001: load the OULAD CSV tables and check their schema.

Owner: Kripendra Singh (Sprint 1 - dataset loading scripts)
"""
from pathlib import Path
import pandas as pd

EXPECTED_SCHEMA = {
    "studentInfo": ["code_module", "code_presentation", "id_student", "gender", "region",
                    "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
                    "studied_credits", "disability", "final_result"],
    "studentAssessment": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    "assessments": ["code_module", "code_presentation", "id_assessment", "assessment_type",
                    "date", "weight"],
    "studentVle": ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
    "studentRegistration": ["code_module", "code_presentation", "id_student",
                            "date_registration", "date_unregistration"],
    "courses": ["code_module", "code_presentation", "module_presentation_length"],
    "vle": ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"],
}

# smaller dtypes so the 10M-row studentVle table fits in a free Colab session
_VLE_DTYPES = {"id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int16"}


def load_raw_tables(data_dir: str = "data") -> dict:
    tables = {}
    for name in EXPECTED_SCHEMA:
        path = Path(data_dir) / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - download OULAD into data/ (see data/README.md) "
                "or run with --synthetic")
        dtypes = _VLE_DTYPES if name == "studentVle" else None
        tables[name] = pd.read_csv(path, dtype=dtypes, na_values=["?", ""])
    return tables


def validate_schema(tables: dict) -> None:
    for name, cols in EXPECTED_SCHEMA.items():
        if name not in tables:
            raise ValueError(f"missing table: {name}")
        missing = set(cols) - set(tables[name].columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        if tables[name].empty:
            raise ValueError(f"{name}.csv is empty")
    print("Schema validation passed for all 7 OULAD tables.")


if __name__ == "__main__":
    t = load_raw_tables()
    validate_schema(t)
    for n, d in t.items():
        print(f"{n:20s} rows={len(d):>9,}  cols={d.shape[1]}")
