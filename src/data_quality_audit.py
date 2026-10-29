"""Create a data-quality audit for the OULAD tables.

Owner: Mayank Rathore
Purpose: Report missing values, duplicate rows, column types, and basic
numeric summaries without changing the source CSV files.
Run from the project root: python src/data_quality_audit.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import DATA_DIR  # noqa: E402
from data_loader import EXPECTED_SCHEMA, load_raw_tables, validate_schema  # noqa: E402

REPORT_PATH = ROOT / "reports" / "data_quality_audit.json"


def _number(value: Any) -> int | float | None:
    """Convert pandas/numpy values to JSON-safe Python numbers."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (int, float)) else None


def audit_table(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize quality signals for one loaded OULAD table."""
    missing_by_column = frame.isna().sum()
    missing_total = int(missing_by_column.sum())
    cell_count = int(frame.shape[0] * frame.shape[1])
    numeric = frame.select_dtypes(include="number")
    numeric_summary: dict[str, dict[str, int | float | None]] = {}

    for column in numeric.columns:
        values = numeric[column]
        numeric_summary[column] = {
            "min": _number(values.min()),
            "mean": _number(values.mean()),
            "max": _number(values.max()),
        }

    missing_columns = {
        column: int(count)
        for column, count in missing_by_column.items()
        if count > 0
    }
    expected = EXPECTED_SCHEMA.get(name, [])
    missing_schema_columns = sorted(set(expected) - set(frame.columns))

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": missing_total,
        "missing_cell_percentage": round(
            100 * missing_total / cell_count, 4
        ) if cell_count else 0.0,
        "columns_with_missing_values": missing_columns,
        "missing_expected_columns": missing_schema_columns,
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "numeric_summary": numeric_summary,
    }

def build_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build the report and a concise project-wide summary."""
    table_reports = {
        name: audit_table(name, frame) for name, frame in tables.items()
    }
    total_rows = sum(item["rows"] for item in table_reports.values())
    total_duplicates = sum(item["duplicate_rows"] for item in table_reports.values())
    total_missing = sum(item["missing_cells"] for item in table_reports.values())
    schema_issues = {
        name: details["missing_expected_columns"]
        for name, details in table_reports.items()
        if details["missing_expected_columns"]
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(DATA_DIR),
        "summary": {
            "tables_audited": len(table_reports),
            "total_rows_across_tables": total_rows,
            "total_duplicate_rows_across_tables": total_duplicates,
            "total_missing_cells": total_missing,
            "tables_with_missing_values": sum(
                bool(item["columns_with_missing_values"])
                for item in table_reports.values()
            ),
            "schema_issues_found": bool(schema_issues),
            "schema_issues": schema_issues,
        },
        "tables": table_reports,
        "interpretation": [
            "Missing values are reported, not automatically filled.",
            "Duplicate rows are counted, not automatically deleted.",
            "Review table-specific meaning before cleaning any records.",
        ],
    }

def main() -> int:
    """Load the configured dataset, audit it, and write a JSON report."""
    try:
        tables = load_raw_tables(str(DATA_DIR))
        validate_schema(tables)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Data quality audit could not run: {exc}", file=sys.stderr)
        return 1

    report = build_report(tables)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited {report['summary']['tables_audited']} tables.")
    print(f"Rows checked: {report['summary']['total_rows_across_tables']:,}")
    print(f"Duplicate rows: {report['summary']['total_duplicate_rows_across_tables']:,}")
    print(f"Missing cells: {report['summary']['total_missing_cells']:,}")
    print(f"Report saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
