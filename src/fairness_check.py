"""
=============================================================
 fairness_check.py
 Employee Performance Predictor – Bias & Fairness Audit
=============================================================
 Analyzes model predictions across demographic/org slices:
   • Department-level selection rates
   • Experience-level performance gaps
   • Statistical parity difference
   • Adverse impact ratio (4/5ths rule)

 Why this matters:
   In a real HR system, biased predictions can lead to
   discriminatory decisions at scale. Auditing for fairness
   is a legal and ethical requirement in many jurisdictions.
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────
BAND_MAP    = {0: "Low", 1: "Medium", 2: "High"}
CLASS_NAMES = ["Low", "Medium", "High"]


# ── Selection Rate helper ─────────────────────────────────
def selection_rate(predictions: np.ndarray, positive_class: int = 2) -> float:
    """Fraction of employees predicted as High performer."""
    return (predictions == positive_class).mean()


# ── Department-level audit ────────────────────────────────
def audit_by_department(
    test_df:     pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    For each department, compute:
      - n_employees
      - selection_rate (% predicted High)
      - actual_high_rate (ground truth %)
      - gap (prediction - actual)
      - adverse_impact_ratio vs best department
    """
    df = test_df.copy().reset_index(drop=True)
    df["prediction"]      = [BAND_MAP[int(p)] for p in predictions]
    df["actual_band"]     = df["performance_band"].values if "performance_band" in df else "Unknown"
    df["pred_high"]       = (predictions == 2).astype(int)
    df["actual_high"]     = (df["actual_band"] == "High").astype(int)

    summary = []
    for dept, grp in df.groupby("department"):
        n              = len(grp)
        pred_high_rate = grp["pred_high"].mean()
        actual_high_rt = grp["actual_high"].mean()
        summary.append({
            "Department":           dept,
            "N_Employees":          n,
            "Predicted_High_%":     round(pred_high_rate * 100, 1),
            "Actual_High_%":        round(actual_high_rt * 100, 1),
            "Gap_%pts":             round((pred_high_rate - actual_high_rt) * 100, 1),
        })

    summary_df = pd.DataFrame(summary).sort_values("Predicted_High_%", ascending=False)

    # Adverse impact ratio: each dept / best dept
    best_rate = summary_df["Predicted_High_%"].max()
    summary_df["Adverse_Impact_Ratio"] = (summary_df["Predicted_High_%"] / best_rate).round(3)
    summary_df["Bias_Flag"] = summary_df["Adverse_Impact_Ratio"].apply(
        lambda x: "⚠️ INVESTIGATE" if x < 0.80 else "✅ OK"
    )

    return summary_df.reset_index(drop=True)


# ── Experience-level audit ────────────────────────────────
def audit_by_experience(
    test_df:     pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Bucket employees by experience tier and audit predictions.
    Buckets: Junior (0-3 yrs), Mid (4-8 yrs), Senior (9+ yrs)
    """
    df = test_df.copy().reset_index(drop=True)
    df["prediction"]  = [BAND_MAP[int(p)] for p in predictions]
    df["pred_high"]   = (predictions == 2).astype(int)
    df["actual_band"] = df["performance_band"].values if "performance_band" in df else "Unknown"
    df["actual_high"] = (df["actual_band"] == "High").astype(int)

    bins   = [-1, 3, 8, 100]
    labels = ["Junior (0–3 yrs)", "Mid (4–8 yrs)", "Senior (9+ yrs)"]
    df["exp_tier"] = pd.cut(df["experience_years"], bins=bins, labels=labels)

    summary = []
    for tier, grp in df.groupby("exp_tier", observed=True):
        if len(grp) == 0:
            continue
        summary.append({
            "Experience_Tier":  str(tier),
            "N_Employees":      len(grp),
            "Predicted_High_%": round(grp["pred_high"].mean() * 100, 1),
            "Actual_High_%":    round(grp["actual_high"].mean() * 100, 1),
        })

    return pd.DataFrame(summary)


# ── Statistical parity ────────────────────────────────────
def statistical_parity_difference(
    test_df:       pd.DataFrame,
    predictions:   np.ndarray,
    group_col:     str,
    reference_grp: str,
) -> pd.DataFrame:
    """
    Compute statistical parity difference between groups.
    SPD = P(ŷ=High | group=A) − P(ŷ=High | group=Ref)
    Ideal SPD = 0; |SPD| > 0.10 is worth investigating.
    """
    df = test_df.copy().reset_index(drop=True)
    df["pred_high"] = (predictions == 2).astype(int)

    if group_col not in df.columns:
        return pd.DataFrame()

    ref_rate = df[df[group_col] == reference_grp]["pred_high"].mean()
    rows = []
    for grp, sub in df.groupby(group_col):
        rate = sub["pred_high"].mean()
        spd  = rate - ref_rate
        rows.append({
            "Group":          grp,
            "High_Rate_%":    round(rate * 100, 1),
            "SPD":            round(spd, 3),
            "Fairness_Flag":  "⚠️ REVIEW" if abs(spd) > 0.10 else "✅ OK",
        })

    return pd.DataFrame(rows).sort_values("SPD")


# ── Full fairness report ──────────────────────────────────
def run_fairness_audit(
    test_df:     pd.DataFrame,
    predictions: np.ndarray,
    output_dir:  str = "outputs",
) -> dict:
    """
    Run the complete fairness audit and return a dict of DataFrames.
    Also saves results as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("  FAIRNESS & BIAS AUDIT")
    print("=" * 60)

    # Department
    dept_df = audit_by_department(test_df, predictions)
    dept_path = os.path.join(output_dir, "fairness_department.csv")
    dept_df.to_csv(dept_path, index=False)
    print(f"\n  📋 Department-level audit ({dept_path}):")
    print(dept_df.to_string(index=False))

    # Experience
    exp_df = audit_by_experience(test_df, predictions)
    exp_path = os.path.join(output_dir, "fairness_experience.csv")
    exp_df.to_csv(exp_path, index=False)
    print(f"\n  📋 Experience-level audit ({exp_path}):")
    print(exp_df.to_string(index=False))

    # Statistical parity by department
    spd_df = statistical_parity_difference(
        test_df, predictions,
        group_col     = "department",
        reference_grp = "Engineering",
    )
    spd_path = os.path.join(output_dir, "fairness_spd.csv")
    spd_df.to_csv(spd_path, index=False)
    print(f"\n  📋 Statistical Parity Difference (ref=Engineering):")
    print(spd_df.to_string(index=False))

    # ── Mitigation suggestions ────────────────────────────
    bias_detected = dept_df["Bias_Flag"].str.contains("INVESTIGATE").any()
    print("\n  💡 Mitigation Recommendations:")
    if bias_detected:
        print("  ⚠️  Bias detected in one or more departments. Suggested actions:")
        print("    1. Review feature distribution by department — check for systemic gaps in")
        print("       training hours or goal-setting processes")
        print("    2. Consider fairness constraints during model training (e.g., equalized odds)")
        print("    3. Calibrate model separately per department if gap persists")
        print("    4. Engage HR leadership to investigate structural inequities")
    else:
        print("  ✅ No significant bias detected across departments.")
        print("     Continue monitoring quarterly as new data arrives.")

    return {
        "department_audit":  dept_df,
        "experience_audit":  exp_df,
        "spd_audit":         spd_df,
        "bias_detected":     bias_detected,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_generator import generate_dataset
    from src.preprocessing  import run_preprocessing
    from src.model_training import train_and_compare

    df  = generate_dataset(n_employees=200, save=False)
    out = run_preprocessing(df, models_dir="../models")
    results, best = train_and_compare(
        out["X_train"], out["y_train"],
        out["X_test"],  out["y_test"],
        out["feature_names"], models_dir="../models"
    )
    preds = results[best]["y_pred"]
    run_fairness_audit(out["test_raw"], preds, output_dir="../outputs")
