"""
=============================================================
 explainability.py
 Employee Performance Predictor – Explainable AI (XAI)
=============================================================
 Uses permutation importance + SHAP (if available) to explain:
   • Which features drove EACH individual prediction
   • Top 3 positive drivers
   • Top 3 negative drivers
   • HR-friendly plain-English translation
=============================================================
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance

# ── HR-friendly feature translations ─────────────────────
# Each key maps a feature name to a tuple:
#   (positive_message, negative_message)
HR_MESSAGES = {
    "goal_achievement_pct": (
        "Employee consistently met or exceeded their quarterly goals",
        "Employee struggled to meet quarterly targets — goal-setting support recommended",
    ),
    "manager_rating": (
        "High manager satisfaction signals strong performance visibility",
        "Low manager rating suggests misalignment — 1-on-1 coaching advised",
    ),
    "peer_score": (
        "Strong peer collaboration enhances team output",
        "Low peer collaboration score — team integration program may help",
    ),
    "attendance_rate": (
        "Excellent attendance reflects strong commitment and reliability",
        "Below-average attendance is a risk signal — wellbeing check recommended",
    ),
    "training_hours": (
        "Substantial training investment is improving skill readiness",
        "Low training hours are limiting growth — enroll in L&D programs",
    ),
    "projects_completed": (
        "High project completion rate demonstrates productivity",
        "Few projects completed this quarter — workload or skill gap review needed",
    ),
    "overtime_hours": (
        "Managed overtime suggests healthy work-life balance",
        "Excessive overtime may indicate workload stress or burnout risk",
    ),
    "experience_years": (
        "Deep experience base provides consistent performance foundation",
        "Limited experience — structured mentorship program recommended",
    ),
    "salary": (
        "Competitive compensation aligns with performance incentives",
        "Potential compensation mismatch may affect motivation",
    ),
    "age": (
        "Mature professional bringing stability",
        "Early-career employee — structured onboarding beneficial",
    ),
    "department_enc": (
        "Department environment supports performance",
        "Department-level challenges may be affecting performance",
    ),
    "education_enc": (
        "Educational background supports role requirements",
        "Skills gap identified — targeted upskilling recommended",
    ),
    "role_enc": (
        "Role clarity is supporting task execution",
        "Role ambiguity may be reducing effectiveness",
    ),
}

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


# ── Permutation importance (model-agnostic fallback) ──────
def compute_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute feature importance by randomly shuffling each feature
    and measuring drop in model accuracy. Works for ANY model.
    """
    print("[XAI] Computing permutation importance …")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats    = n_repeats,
        random_state = 42,
        scoring      = "f1_macro",
        n_jobs       = -1,
    )
    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": perm.importances_mean,
        "Std":        perm.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return fi_df


# ── SHAP global explanation ───────────────────────────────
def compute_shap_values(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: list):
    """
    Compute SHAP values using TreeExplainer (fast for RF/XGB)
    or KernelExplainer as fallback.
    Returns shap_values array + explainer object.
    """
    if not SHAP_AVAILABLE:
        print("[XAI] SHAP not installed. Skipping SHAP computation.")
        return None, None

    print("[XAI] Computing SHAP values …")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        print("[XAI] TreeExplainer failed. Using KernelExplainer (slower) …")
        background  = shap.sample(X_train, 100)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:200])   # limit for speed

    print(f"  SHAP values computed. Shape: {np.array(shap_values).shape}")
    return shap_values, explainer


# ── Per-employee local explanation ────────────────────────
def explain_single_prediction(
    model,
    employee_features: np.ndarray,  # 1-D array (already scaled)
    feature_names: list,
    feature_values_raw: dict,        # original (unscaled) values for display
    shap_values=None,
    sample_idx: int = 0,
) -> dict:
    """
    Explain a single employee's prediction.
    Returns a dict with:
      - predicted_class
      - predicted_band
      - probabilities
      - top3_positive_drivers
      - top3_negative_drivers
      - hr_summary (plain English)
    """
    proba = model.predict_proba(employee_features.reshape(1, -1))[0]
    pred_class = int(np.argmax(proba))
    pred_band  = CLASS_NAMES[pred_class]

    # ── Get feature contributions ─────────────────────────
    if shap_values is not None and SHAP_AVAILABLE:
        # Use SHAP values for the predicted class
        if isinstance(shap_values, list):
            contributions = shap_values[pred_class][sample_idx]
        else:
            contributions = shap_values[sample_idx, :, pred_class] \
                            if shap_values.ndim == 3 else shap_values[sample_idx]
    else:
        # Fallback: use model coefficients / feature importances
        if hasattr(model, "coef_"):
            contributions = model.coef_[pred_class] * employee_features
        elif hasattr(model, "feature_importances_"):
            contributions = model.feature_importances_ * employee_features
        else:
            contributions = np.zeros(len(feature_names))

    # ── Sort by contribution ──────────────────────────────
    contrib_df = pd.DataFrame({
        "Feature":      feature_names,
        "Contribution": contributions,
    }).sort_values("Contribution", ascending=False)

    positive_drivers = contrib_df[contrib_df["Contribution"] > 0].head(3)
    negative_drivers = contrib_df[contrib_df["Contribution"] < 0].tail(3)

    # ── Translate to HR language ──────────────────────────
    def translate(row, positive: bool) -> str:
        feat = row["Feature"]
        msg_pair = HR_MESSAGES.get(feat, (
            f"High {feat} positively contributed",
            f"Low {feat} negatively contributed",
        ))
        raw_val = feature_values_raw.get(feat, "N/A")
        msg = msg_pair[0] if positive else msg_pair[1]
        return f"{msg}  [value: {raw_val}]"

    pos_msgs = [translate(r, True)  for _, r in positive_drivers.iterrows()]
    neg_msgs = [translate(r, False) for _, r in negative_drivers.iterrows()]

    hr_summary = (
        f"Predicted Performance: {pred_band.upper()}  "
        f"(confidence {proba[pred_class]*100:.1f}%)\n\n"
        f"✅ Key Strengths:\n"
        + "\n".join(f"  • {m}" for m in pos_msgs) + "\n\n"
        f"⚠️  Areas of Concern:\n"
        + "\n".join(f"  • {m}" for m in neg_msgs)
    )

    return {
        "predicted_class":          pred_class,
        "predicted_band":           pred_band,
        "probabilities":            {CLASS_NAMES[i]: round(float(p), 3) for i, p in enumerate(proba)},
        "top3_positive_drivers":    pos_msgs,
        "top3_negative_drivers":    neg_msgs,
        "hr_summary":               hr_summary,
    }


# ── Save SHAP summary plot ────────────────────────────────
def save_shap_summary_plot(shap_values, X_test, feature_names, output_path: str = "outputs/graphs/shap_summary.png"):
    """Save SHAP beeswarm summary plot."""
    if not SHAP_AVAILABLE or shap_values is None:
        print("[XAI] SHAP not available. Skipping summary plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # For multi-class take the 'High' class SHAP values (index 2)
    sv = shap_values[2] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(
        sv, X_test,
        feature_names  = feature_names,
        show           = False,
        plot_type      = "bar",
        max_display    = 12,
    )
    plt.title("SHAP Feature Importance (High Performance Class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary plot saved → {output_path}")


if __name__ == "__main__":
    print("Explainability module loaded successfully.")
    print(f"SHAP available: {SHAP_AVAILABLE}")
