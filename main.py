"""
=============================================================
 main.py
 Employee Performance Predictor – Master Orchestration Script
=============================================================
 Runs the full ML pipeline in sequence:
   1.  Generate synthetic HR dataset
   2.  Preprocess + time-based split
   3.  EDA visualizations
   4.  Train & compare 3 models
   5.  Model evaluation plots
   6.  Explainability (feature importance + sample SHAP)
   7.  HR Decision Engine (batch recommendations)
   8.  Fairness & bias audit
   9.  Save all outputs
  10.  Print case study

 Usage:
   python main.py
   python main.py --employees 1000
=============================================================
"""

import os
import sys
import argparse
import warnings
import json
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Make src importable ───────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_generator    import generate_dataset
from src.preprocessing     import run_preprocessing
from src.model_training    import train_and_compare, get_feature_importance
from src.explainability    import (
    compute_permutation_importance,
    explain_single_prediction,
    save_shap_summary_plot,
)
try:
    from src.explainability import compute_shap_values
    SHAP_SUPPORT = True
except Exception:
    SHAP_SUPPORT = False

from src.hr_decision_engine import decide, batch_decisions
from src.fairness_check     import run_fairness_audit
from src.visualizations     import (
    generate_all_eda_plots,
    generate_all_model_plots,
    plot_fairness_dept,
    plot_top_bottom_performers,
)

# ── Directories ───────────────────────────────────────────
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
GRAPHS_DIR  = os.path.join(ROOT, "outputs", "graphs")

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, GRAPHS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Banner ────────────────────────────────────────────────
def banner():
    print("\n" + "█" * 64)
    print("  EMPLOYEE PERFORMANCE PREDICTOR  |  INDUSTRY-GRADE PIPELINE")
    print("  Version 2.0  |  Built for Placement Portfolio")
    print("█" * 64 + "\n")


# ── Case Study ────────────────────────────────────────────
def print_case_study(model, scaler, feature_names: list, encoders: dict):
    """
    Simulate a real HR use-case:
    Input → Prediction → Explanation → HR Action
    """
    print("\n" + "=" * 64)
    print("  📋 REAL-WORLD CASE STUDY")
    print("=" * 64)

    employees = [
        {
            "id": "EMP0042",
            "label": "At-Risk Employee",
            "raw": {
                "age": 28, "experience_years": 3, "salary": 38000,
                "training_hours": 6, "attendance_rate": 0.74,
                "overtime_hours": 40, "projects_completed": 1,
                "manager_rating": 2.0, "peer_score": 2.3,
                "goal_achievement_pct": 38,
                "department_enc": 1, "education_enc": 1, "role_enc": 0,
            },
        },
        {
            "id": "EMP0173",
            "label": "High Performer",
            "raw": {
                "age": 35, "experience_years": 12, "salary": 95000,
                "training_hours": 28, "attendance_rate": 0.97,
                "overtime_hours": 8, "projects_completed": 6,
                "manager_rating": 4.6, "peer_score": 4.5,
                "goal_achievement_pct": 92,
                "department_enc": 0, "education_enc": 2, "role_enc": 2,
            },
        },
    ]

    for emp in employees:
        raw  = emp["raw"]
        feat = np.array([[raw[c] for c in feature_names]])
        if scaler:
            feat_scaled = scaler.transform(feat)
        else:
            feat_scaled = feat

        proba      = model.predict_proba(feat_scaled)[0]
        pred_class = int(np.argmax(proba))
        pred_band  = {0: "Low", 1: "Medium", 2: "High"}[pred_class]

        explanation = explain_single_prediction(
            model              = model,
            employee_features  = feat_scaled[0],
            feature_names      = feature_names,
            feature_values_raw = raw,
            sample_idx         = 0,
        )

        hr_dec = decide(emp["id"], pred_band, float(proba[pred_class]), raw)

        print(f"\n  👤 Employee: {emp['id']}  ({emp['label']})")
        print(f"  {'─' * 55}")
        print(f"  Predicted Band : {pred_band}")
        print(f"  Confidence     : {proba[pred_class]*100:.1f}%")
        low  = proba[0] if len(proba) > 0 else 0
        med  = proba[1] if len(proba) > 1 else 0
        high = proba[2] if len(proba) > 2 else 0

        print(f"  Probabilities  →  Low={low:.2f}  Med={med:.2f}  High={high:.2f}")
        print(f"\n  ✅ Key Strengths:")
        for m in explanation["top3_positive_drivers"]:
            print(f"    → {m}")
        print(f"\n  ⚠️  Areas of Concern:")
        for m in explanation["top3_negative_drivers"]:
            print(f"    → {m}")
        print(f"\n  📌 HR Recommended Actions:")
        for a in hr_dec.recommended_actions:
            print(f"    • {a}")
        if hr_dec.pip_required:
            print(f"\n  📄 PIP Goals:")
            for g in hr_dec.pip_goals:
                print(f"    • {g}")
        if hr_dec.promotion_ready:
            print(f"  🚀 PROMOTION CANDIDATE")
        print(f"  Retention Risk: {hr_dec.retention_risk}  |  Priority: {hr_dec.priority}")


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
def main(n_employees: int = 500):
    t0 = time.time()
    banner()

    # ── STEP 1: Generate data ─────────────────────────────
    print("STEP 1/9 — Generating Synthetic HR Dataset")
    df = generate_dataset(n_employees=n_employees, save=True, output_dir=DATA_DIR)
    print(f"  Dataset shape: {df.shape}")

    # ── STEP 2: Preprocessing ─────────────────────────────
    print("\nSTEP 2/9 — Preprocessing & Time-Based Split")
    prep = run_preprocessing(df, models_dir=MODELS_DIR)

    X_train       = prep["X_train"]
    X_test        = prep["X_test"]
    y_train       = prep["y_train"]
    y_test        = prep["y_test"]
    feature_names = prep["feature_names"]
    train_raw     = prep["train_raw"]
    test_raw      = prep["test_raw"]
    scaler        = prep["scaler"]
    encoders      = prep["label_encoders"]

    # Save feature names
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # ── STEP 3: EDA Visualizations ────────────────────────
    print("\nSTEP 3/9 — EDA Visualizations")
    generate_all_eda_plots(df, output_dir=GRAPHS_DIR)

    # ── STEP 4: Train Models ──────────────────────────────
    print("\nSTEP 4/9 — Training & Comparing Models")
    results, best_name = train_and_compare(
        X_train, y_train, X_test, y_test,
        feature_names = feature_names,
        models_dir    = MODELS_DIR,
    )
    best_model = results[best_name]["model"]

    # ── STEP 5: Feature Importance ────────────────────────
    print("\nSTEP 5/9 — Feature Importance & Explainability")
    fi_df = get_feature_importance(best_model, feature_names)
    if len(fi_df):
        fi_path = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importance saved → {fi_path}")
        print(f"\n  Top 5 features:\n{fi_df.head(5).to_string(index=False)}")

    # Permutation importance
    pi_df = compute_permutation_importance(best_model, X_test, y_test, feature_names)
    pi_path = os.path.join(OUTPUTS_DIR, "permutation_importance.csv")
    pi_df.to_csv(pi_path, index=False)
    print(f"  Permutation importance saved → {pi_path}")

    # SHAP (optional)
    shap_values = None
    if SHAP_SUPPORT:
        try:
            from src.explainability import compute_shap_values
            shap_values, _ = compute_shap_values(best_model, X_train, X_test, feature_names)
            if shap_values is not None:
                save_shap_summary_plot(
                    shap_values, X_test, feature_names,
                    output_path=os.path.join(GRAPHS_DIR, "shap_summary.png")
                )
        except Exception as e:
            print(f"  [SHAP] Skipped: {e}")

    # ── STEP 6: Model Evaluation Plots ───────────────────
    print("\nSTEP 6/9 — Model Evaluation Plots")
    generate_all_model_plots(results, fi_df, best_name, output_dir=GRAPHS_DIR)

    # ── STEP 7: HR Decision Engine ───────────────────────
    print("\nSTEP 7/9 — HR Decision Engine (Batch)")
    predictions = results[best_name]["y_pred"]
    probas      = best_model.predict_proba(X_test)
    hr_report   = batch_decisions(test_raw, predictions, probas)
    hr_path     = os.path.join(OUTPUTS_DIR, "hr_decisions.csv")
    hr_report.to_csv(hr_path, index=False)
    print(f"  HR decision report saved → {hr_path}")
    print(f"  {hr_report['Priority'].value_counts().to_string()}")

    # ── STEP 8: Fairness Audit ───────────────────────────
    print("\nSTEP 8/9 — Fairness & Bias Audit")
    fairness = run_fairness_audit(test_raw, predictions, output_dir=OUTPUTS_DIR)
    if fairness["department_audit"] is not None:
        plot_fairness_dept(fairness["department_audit"], output_dir=GRAPHS_DIR)

    # Top/bottom performers
    plot_top_bottom_performers(test_raw, predictions, output_dir=GRAPHS_DIR)

    # ── STEP 9: Case Study ────────────────────────────────
    print("\nSTEP 9/9 — Case Study Simulation")
    print_case_study(best_model, scaler, feature_names, encoders)

    # ── Summary ───────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "█" * 64)
    print(f"  ✅  PIPELINE COMPLETE  in {elapsed:.1f}s")
    print(f"  Best Model   : {best_name}")
    print(f"  Accuracy     : {results[best_name]['accuracy']:.4f}")
    print(f"  F1 (Macro)   : {results[best_name]['f1_macro']:.4f}")
    print(f"  Outputs      : {OUTPUTS_DIR}/")
    print(f"  Graphs       : {GRAPHS_DIR}/")
    print(f"\n  Next step → run the dashboard:")
    print(f"    streamlit run dashboard/app.py")
    print("█" * 64 + "\n")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Performance Predictor Pipeline")
    parser.add_argument("--employees", type=int, default=500,
                        help="Number of synthetic employees to generate (default: 500)")
    args = parser.parse_args()
    main(n_employees=args.employees)
