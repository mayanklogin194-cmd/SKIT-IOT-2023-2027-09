"""
Feature lists + one-hot encoding of categorical variables (shared by every model).

Owner: Kripendra Singh (Sprint 3 - one-hot encode categorical variables)
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEMOGRAPHIC_CAT = ["code_module", "gender", "region", "highest_education", "imd_band",
                   "age_band", "disability"]
DEMOGRAPHIC_NUM = ["num_of_prev_attempts", "studied_credits"]

# Sprint 4 - assessment-grade regression (one row per student x assessment)
REG_CAT = DEMOGRAPHIC_CAT + ["assessment_type"]
REG_NUM = DEMOGRAPHIC_NUM + ["weight", "date", "prev_score", "prev_avg_score",
                             "n_prev_assessments", "is_first_assessment",
                             "cum_clicks", "cum_active_days"]

# Sprint 5 - final-outcome classification (one row per student, data up to the cut-off day)
CLF_CAT = DEMOGRAPHIC_CAT
CLF_NUM = DEMOGRAPHIC_NUM + ["total_clicks", "active_days", "clicks_last_30d",
                             "avg_score", "min_score", "n_submitted"]


def make_preprocessor(numeric, categorical, scale: bool = False) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
