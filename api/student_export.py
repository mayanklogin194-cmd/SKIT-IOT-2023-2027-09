
import csv
import json
import logging
from pathlib import Path
from flask import Blueprint, send_file, jsonify
from io import StringIO

export_bp = Blueprint("student_export", __name__)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
STUDENT_FILE = ROOT / "reports" / "sample_students.json"

EXPORT_FIELDS = [
    "id_student",
    "code_module",
    "total_clicks",
    "active_days",
    "avg_score",
    "final_result",
    "risk",
    "p_Distinction",
    "p_Fail",
    "p_Pass",
    "p_Withdrawn",
]

def load_student_records():
    """Load student records from the saved JSON report."""
    if not STUDENT_FILE.exists():
        raise FileNotFoundError("Student data file was not found")

    with STUDENT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Student data must be a list")

    return [row for row in records if isinstance(row, dict)]

def prepare_export_rows(records):
    """Select only approved export columns."""
    rows = []

    for student in records:
        row = {}

        for field in EXPORT_FIELDS:
            value = student.get(field, "")
            row[field] = value

        rows.append(row)

    return rows

def create_csv_file(records):
    """Create an in-memory CSV file."""
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=EXPORT_FIELDS,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in records:
        writer.writerow(row)

    output.seek(0)
    return output

@export_bp.route("/dashboard/students/export", methods=["GET"])
def export_students():
    """Download student prediction data as CSV."""
    try:
        students = load_student_records()

        if not students:
            return jsonify({
                "error": "No student records available"
            }), 404

        export_rows = prepare_export_rows(students)
        csv_data = create_csv_file(export_rows)

        from io import BytesIO
        file_data = BytesIO(csv_data.getvalue().encode("utf-8"))
        file_data.seek(0)

        return send_file(
            file_data,
            mimetype="text/csv",
            as_attachment=True,
            download_name="student_predictions.csv"
        )

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except (ValueError, json.JSONDecodeError) as error:
        logger.warning("Invalid student export data: %s", error)
        return jsonify({"error": "Invalid student data"}), 400

    except OSError:
        logger.exception("Student export failed")
        return jsonify({"error": "Unable to export student data"}), 500
