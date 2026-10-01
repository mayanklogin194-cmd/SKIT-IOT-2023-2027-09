"""
Shared paths and constants.

Owner: Mayank Rathore (Sprint 1 - repo / environment setup)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42
STUDENT_KEYS = ["code_module", "code_presentation", "id_student"]

# Early-warning cut-off: the outcome classifier only sees what happened up to this
# course day, so predictions are available while there is still time to intervene.
CUTOFF_DAY = 100


def artifact_dirs(synthetic: bool = False):
    """(models_dir, reports_dir). Synthetic runs get their own folders."""
    sub = "synthetic" if synthetic else ""
    models, reports = ROOT / "models" / sub, ROOT / "reports" / sub
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return models, reports
