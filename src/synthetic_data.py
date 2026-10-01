"""
Generates fake OULAD-shaped tables so the pipeline can be developed and tested
before (or without) the real dataset. Numbers from synthetic runs are NOT results.

Owner: Mayank Rathore (Sprint 1 - environment setup)
"""
import numpy as np
import pandas as pd

REGIONS = ["Scotland", "London Region", "North Region", "South Region", "Wales"]
EDU = ["A Level or Equivalent", "HE Qualification", "Lower Than A Level", "Post Graduate Qualification"]
IMD = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]


def generate_synthetic_oulad(n_students: int = 1500, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    modules, pres = ["AAA", "BBB"], "2013J"

    courses = pd.DataFrame({"code_module": modules, "code_presentation": pres,
                            "module_presentation_length": 240})
    rows, aid = [], 1000
    for m in modules:
        for d, t, w in [(25, "TMA", 10), (50, "TMA", 20), (75, "TMA", 20), (100, "TMA", 20),
                        (150, "TMA", 30), (240, "Exam", 100)]:
            aid += 1
            rows.append((m, pres, aid, t, d if t != "Exam" else np.nan, w))
    assessments = pd.DataFrame(rows, columns=["code_module", "code_presentation", "id_assessment",
                                              "assessment_type", "date", "weight"])

    ability = rng.normal(0, 1, n_students)
    z = 0.9 * ability + rng.normal(0, 0.6, n_students)
    result = np.select([z < -1.0, z < -0.25, z < 1.2], ["Withdrawn", "Fail", "Pass"], "Distinction")
    info = pd.DataFrame({
        "code_module": rng.choice(modules, n_students), "code_presentation": pres,
        "id_student": np.arange(100000, 100000 + n_students),
        "gender": rng.choice(["M", "F"], n_students), "region": rng.choice(REGIONS, n_students),
        "highest_education": rng.choice(EDU, n_students, p=[.4, .25, .3, .05]),
        "imd_band": rng.choice(IMD + [np.nan], n_students),
        "age_band": rng.choice(["0-35", "35-55", "55<="], n_students, p=[.7, .27, .03]),
        "num_of_prev_attempts": rng.choice([0, 1, 2], n_students, p=[.85, .12, .03]),
        "studied_credits": rng.choice([30, 60, 90, 120], n_students, p=[.1, .6, .2, .1]),
        "disability": rng.choice(["N", "Y"], n_students, p=[.9, .1]), "final_result": result})

    unreg = np.where(result == "Withdrawn", rng.integers(20, 200, n_students), np.nan)
    registration = info[["code_module", "code_presentation", "id_student"]].assign(
        date_registration=rng.integers(-90, 0, n_students), date_unregistration=unreg)

    vle = pd.DataFrame({"id_site": np.arange(1, 41), "code_module": np.repeat(modules, 20),
                        "code_presentation": pres,
                        "activity_type": rng.choice(["resource", "forumng", "quiz", "oucontent"], 40),
                        "week_from": np.nan, "week_to": np.nan})

    vle_rows, sa_rows = [], []
    for i in range(n_students):
        sid, mod = info.id_student[i], info.code_module[i]
        last_day = int(unreg[i]) if result[i] == "Withdrawn" else 240
        n_days = int(np.clip(rng.poisson(max(35 + 18 * ability[i], 3)), 2, 200))
        days = np.unique(rng.integers(-10, max(last_day, 5), n_days))
        clicks = rng.poisson(2.5 + 1.2 * max(ability[i], -1.5), len(days)) + 1
        site_pool = vle.id_site[vle.code_module == mod].to_numpy()
        vle_rows.append(pd.DataFrame({"code_module": mod, "code_presentation": pres, "id_student": sid,
                                      "id_site": rng.choice(site_pool, len(days)),
                                      "date": days, "sum_click": clicks}))
        for _, a in assessments[assessments.code_module == mod].iterrows():
            due = a.date if not np.isnan(a.date) else 240
            if due > last_day or rng.random() < 0.06:
                continue
            score = float(np.clip(round(62 + 13 * ability[i] + rng.normal(0, 9)), 0, 100))
            sa_rows.append((a.id_assessment, sid, int(due + rng.integers(-3, 6)), 0, score))

    return {"studentInfo": info,
            "studentAssessment": pd.DataFrame(sa_rows, columns=["id_assessment", "id_student",
                                                                "date_submitted", "is_banked", "score"]),
            "assessments": assessments, "studentVle": pd.concat(vle_rows, ignore_index=True),
            "studentRegistration": registration, "courses": courses, "vle": vle}
