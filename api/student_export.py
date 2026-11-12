"""Register a CSV download endpoint for the project's saved student predictions."""
import csv
import io
import json
import logging
from pathlib import Path
from flask import jsonify, send_file

logger = logging.getLogger(__name__)

EXPORT_COLUMNS = [
    "id_student", "code_module", "total_clicks", "active_days",
    "avg_score", "final_result", "risk", "p_Distinction",
    "p_Fail", "p_Pass", "p_Withdrawn",
]

def read_student_records(report_dir):
    """Load the saved sample prediction list."""
    source = Path(report_dir) / "sample_students.json"
    if not source.exists():
        raise FileNotFoundError("Student prediction report is not available")
    with source.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Student prediction report must be a list")
    return [row for row in records if isinstance(row, dict)]

def csv_buffer(records):
    """Build CSV bytes using a fixed allow-list of columns."""
    text_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        text_stream,
        fieldnames=EXPORT_COLUMNS,
        extrasaction="ignore",
    )
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key, "") for key in EXPORT_COLUMNS})
    byte_stream = io.BytesIO(text_stream.getvalue().encode("utf-8-sig"))
    byte_stream.seek(0)
    return byte_stream

def register_student_export(app, report_dir):
    """Attach GET /dashboard/students/export to the existing Flask app."""
    @app.get("/dashboard/students/export")
    def export_student_predictions():
        try:
            records = read_student_records(report_dir)
            if not records:
                return jsonify({"error": "No student records to export"}), 404
            stream = csv_buffer(records)
            return send_file(
                stream,
                mimetype="text/csv",
                as_attachment=True,
                download_name="student_predictions.csv",
                max_age=0,
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except (ValueError, json.JSONDecodeError) as error:
            logger.warning("Student export data is invalid: %s", error)
            return jsonify({"error": "Student prediction data is invalid"}), 400
        except OSError:
            logger.exception("Unable to export student prediction data")
            return jsonify({"error": "Student export is temporarily unavailable"}), 500
    return app
