"""
Data profiling + exploratory analysis of the raw OULAD tables.

Owner: Atharva Khandal (Sprint 1 - explore schema, data-profiling scripts)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def profile_tables(tables: dict) -> dict:
    return {
        name: {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        }
        for name, df in tables.items()
    }


def run_profile(tables: dict, reports_dir) -> None:
    reports_dir = Path(reports_dir)
    (reports_dir / "data_profile.json").write_text(json.dumps(profile_tables(tables), indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tables["studentInfo"]["final_result"].value_counts().plot.bar(ax=ax[0], color="#2f6feb")
    ax[0].set_title("Final result distribution")
    tables["studentAssessment"]["score"].dropna().plot.hist(bins=20, ax=ax[1], color="#2f6feb")
    ax[1].set_title("Assessment score distribution")
    plt.tight_layout()
    fig.savefig(reports_dir / "eda_overview.png", dpi=130)
    plt.close(fig)
    print(f"Profile + EDA chart written to {reports_dir}/")
