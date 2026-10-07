"""
End-to-end pipeline entrypoint (Sprints 1-5; sprint 6-8 hooks marked below).
load -> validate -> profile -> merge -> clean -> features -> baselines -> RF (grades) -> GBM (outcomes)

Usage:
    python src/main.py --synthetic     # no OULAD download needed yet (writes to reports/synthetic/)
    python src/main.py                 # uses the real CSVs in data/
    python src/main.py --tune          # randomized hyper-parameter search (slower)

Owner: All members (integration owned by Priyanshu Joshi)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, artifact_dirs                                   # noqa: E402
from data_loader import load_raw_tables, validate_schema                     # noqa: E402
from profiling import run_profile                                            # noqa: E402
from preprocessing import merge_tables, clean_data                           # noqa: E402
from feature_engineering import build_feature_table                          # noqa: E402
from models.baselines import train_baselines                                 # noqa: E402
from models.random_forest_model import train_random_forest                   # noqa: E402
from models.gbm_model import train_gbm                                       # noqa: E402
from evaluation import plot_rmse_comparison, plot_feature_importance         # noqa: E402


def run_pipeline(use_synthetic: bool, tune: bool):
    models_dir, reports_dir = artifact_dirs(use_synthetic)

    print("== Sprint 1: load + validate + profile ==")
    if use_synthetic:
        from synthetic_data import generate_synthetic_oulad
        tables = generate_synthetic_oulad()
        print("Using synthetic OULAD-shaped data (results are NOT real).")
    else:
        tables = load_raw_tables(str(DATA_DIR))
    validate_schema(tables)
    run_profile({k: v for k, v in tables.items() if k != "studentVle"}, reports_dir)

    print("== Sprint 2: merge + clean ==")
    cleaned = clean_data(merge_tables(tables))

    print("== Sprint 3: feature engineering ==")
    assess_df, student_df = build_feature_table(cleaned, tables["studentVle"])

    print("== Sprint 4: baselines + Random Forest (assessment grade) ==")
    base = train_baselines(assess_df, reports_dir)
    _, rf_rep, _ = train_random_forest(assess_df, models_dir, reports_dir, tune=tune)

    print("== Sprint 5: gradient boosting (final outcome) ==")
    _, gbm_rep, _ = train_gbm(student_df, models_dir, reports_dir, tune=tune)

    # charts for the report
    plot_rmse_comparison({**{n: m["rmse"] for n, m in base.items()}, "Random Forest": rf_rep["rmse"]},
                         reports_dir / "rmse_comparison.png")
    plot_feature_importance(rf_rep["feature_importance"], "Random Forest - grade prediction",
                            reports_dir / "feature_importance_rf.png")
    plot_feature_importance(gbm_rep["feature_importance"], "GBM - outcome prediction",
                            reports_dir / "feature_importance_gbm.png")
    # TODO Sprint 6: benchmark against the published papers (Atthibyani 2024, Jha et al. 2019)
    # TODO Sprint 8: unit tests + end-to-end integration tests

    print("\nPipeline complete.")
    print(f"  Random Forest RMSE : {rf_rep['rmse']:.3f}")
    print(f"  GBM Accuracy       : {gbm_rep['accuracy']:.3f}")
    print(f"  GBM Macro-F1       : {gbm_rep['f1_macro']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use generated fake data")
    parser.add_argument("--tune", action="store_true", help="randomized hyper-parameter search")
    args = parser.parse_args()
    run_pipeline(use_synthetic=args.synthetic, tune=args.tune)
