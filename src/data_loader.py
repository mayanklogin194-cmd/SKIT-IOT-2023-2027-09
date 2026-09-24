"""
FR-001: Loads and merges the raw OULAD tables into a single unified dataset.

Owner: Kripmatra Singh (Sprint 1-2 — data loading & pipeline)
"""
from pathlib import Path
import pandas as pd

REQUIRED_TABLES = ["studentInfo", "studentAssessment", "assessments", "studentVle"]


def load_raw_tables(data_dir: str = "data") -> dict:
    """Load the raw OULAD CSVs from `data_dir` into a dict of DataFrames."""
    data_path = Path(data_dir)
    tables = {}
    missing = []
    for name in REQUIRED_TABLES:
        csv_path = data_path / f"{name}.csv"
        if csv_path.exists():
            tables[name] = pd.read_csv(csv_path)
        else:
            missing.append(str(csv_path))

    if missing:
        raise FileNotFoundError(
            "Missing OULAD files: " + ", ".join(missing) +
            "\nEither download them (see data/README.md) or run with --synthetic."
        )
    return tables


def validate_schema(tables: dict) -> None:
    """Basic sanity checks — required columns present, no fully-empty tables."""
    expected_cols = {
        "studentInfo": {"id_student", "final_result"},
        "studentAssessment": {"id_student", "id_assessment", "score"},
        "assessments": {"id_assessment", "weight"},
        "studentVle": {"id_student", "date", "sum_click"},
    }
    for name, cols in expected_cols.items():
        df = tables[name]
        missing_cols = cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} is missing expected columns: {missing_cols}")
        if df.empty:
            raise ValueError(f"{name} loaded but is empty")
    print("Schema validation passed for:", ", ".join(tables.keys()))
