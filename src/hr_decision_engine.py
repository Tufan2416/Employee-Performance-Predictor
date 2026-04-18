"""
=============================================================
 hr_decision_engine.py
 Employee Performance Predictor – HR Decision Engine
=============================================================
 Converts model predictions + feature values into:
   • Actionable HR recommendations
   • Training suggestions
   • Promotion readiness signals
   • Performance Improvement Plan (PIP) triggers
   • Retention risk flags

 This is what turns an ML model into a BUSINESS TOOL.
=============================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ── Thresholds (tunable business rules) ──────────────────
THRESHOLDS = {
    "low_training_hours":       15,      # hours/quarter
    "critical_attendance":      0.80,    # below → flag
    "low_attendance":           0.88,
    "high_overtime":            30,      # hours/quarter → burnout risk
    "low_goal_achievement":     50,      # %
    "medium_goal_achievement":  70,
    "high_goal_achievement":    85,
    "low_manager_rating":       2.5,     # /5
    "high_manager_rating":      4.0,
    "low_peer_score":           2.5,
    "few_projects":             2,       # per quarter
    "senior_experience":        10,      # years
    "min_experience_promo":     5,       # years before promotion eligible
}


@dataclass
class HRDecision:
    """Structured output of the HR Decision Engine."""
    employee_id:             str
    predicted_band:          str
    confidence_pct:          float
    # Recommendations
    training_needed:         bool             = False
    training_topics:         List[str]        = field(default_factory=list)
    promotion_ready:         bool             = False
    pip_required:            bool             = False
    pip_goals:               List[str]        = field(default_factory=list)
    retention_risk:          str              = "Low"    # Low / Medium / High
    wellbeing_flag:          bool             = False
    recommended_actions:     List[str]        = field(default_factory=list)
    priority:                str              = "Normal" # Normal / Medium / Urgent
    summary:                 str              = ""

    def to_dict(self) -> dict:
        return {
            "Employee ID":           self.employee_id,
            "Predicted Band":        self.predicted_band,
            "Confidence (%)":        self.confidence_pct,
            "Training Needed":       self.training_needed,
            "Training Topics":       "; ".join(self.training_topics),
            "Promotion Ready":       self.promotion_ready,
            "PIP Required":          self.pip_required,
            "PIP Goals":             "; ".join(self.pip_goals),
            "Retention Risk":        self.retention_risk,
            "Wellbeing Flag":        self.wellbeing_flag,
            "Recommended Actions":   "; ".join(self.recommended_actions),
            "Priority":              self.priority,
            "HR Summary":            self.summary,
        }


def decide(
    employee_id: str,
    predicted_band: str,
    confidence: float,
    features: dict,
) -> HRDecision:
    """
    Core decision engine.

    Parameters
    ----------
    employee_id    : employee identifier
    predicted_band : "Low" | "Medium" | "High"
    confidence     : probability score (0–1) for predicted class
    features       : dict of raw (unscaled) feature values

    Returns
    -------
    HRDecision dataclass with all HR recommendations.
    """
    d = HRDecision(
        employee_id    = employee_id,
        predicted_band = predicted_band,
        confidence_pct = round(confidence * 100, 1),
    )

    # ── Helper shorthand ──────────────────────────────────
    t  = THRESHOLDS
    th = features.get("training_hours",       20)
    att= features.get("attendance_rate",      0.90)
    ot = features.get("overtime_hours",       10)
    ga = features.get("goal_achievement_pct", 60)
    mr = features.get("manager_rating",       3.0)
    ps = features.get("peer_score",           3.0)
    pc = features.get("projects_completed",   3)
    ex = features.get("experience_years",     5)
    sal= features.get("salary",               50000)

    actions = []

    # ====================================================
    # RULE SET A: Training Recommendations
    # ====================================================
    if th < t["low_training_hours"]:
        d.training_needed = True
        d.training_topics.append("Role-specific technical skills bootcamp")

    if ga < t["low_goal_achievement"]:
        d.training_needed = True
        d.training_topics.append("Goal-setting and OKR workshop")

    if ps < t["low_peer_score"]:
        d.training_needed = True
        d.training_topics.append("Collaboration and communication skills training")

    if mr < t["low_manager_rating"] and ex < t["senior_experience"]:
        d.training_needed = True
        d.training_topics.append("Leadership readiness program")

    # ====================================================
    # RULE SET B: Promotion Readiness
    # ====================================================
    if (
        predicted_band == "High"
        and ga >= t["high_goal_achievement"]
        and mr >= t["high_manager_rating"]
        and ex >= t["min_experience_promo"]
        and pc >= 4
    ):
        d.promotion_ready = True
        actions.append("🚀 Initiate promotion discussion with department head")
        actions.append("📋 Prepare succession plan documentation")

    # ====================================================
    # RULE SET C: Performance Improvement Plan (PIP)
    # ====================================================
    if predicted_band == "Low":
        d.pip_required = True
        if ga < t["low_goal_achievement"]:
            d.pip_goals.append(f"Increase goal achievement from {ga:.0f}% to ≥{t['medium_goal_achievement']}% within 90 days")
        if mr < t["low_manager_rating"]:
            d.pip_goals.append(f"Improve manager rating from {mr:.1f} to ≥{t['low_manager_rating']+0.5:.1f} through bi-weekly check-ins")
        if pc < t["few_projects"]:
            d.pip_goals.append(f"Complete ≥{t['few_projects']+1} projects per quarter with quality gate sign-off")
        if not d.pip_goals:
            d.pip_goals.append("Develop 90-day performance recovery plan with measurable KPIs")

        actions.append("⚠️  Schedule PIP review meeting within 2 weeks")
        actions.append("📅 Assign dedicated performance coach / mentor")

    # ====================================================
    # RULE SET D: Retention Risk
    # ====================================================
    risk_score = 0
    if predicted_band == "Low":        risk_score += 3
    if sal < 40000:                    risk_score += 2
    if ga < t["low_goal_achievement"]: risk_score += 1
    if mr < t["low_manager_rating"]:   risk_score += 2
    if ot > t["high_overtime"]:        risk_score += 1

    if risk_score >= 5:
        d.retention_risk = "High"
        actions.append("🔴 High attrition risk — escalate to HRBP immediately")
        actions.append("💬 Schedule stay interview within 1 week")
    elif risk_score >= 3:
        d.retention_risk = "Medium"
        actions.append("🟡 Moderate retention risk — quarterly engagement check-in advised")
    else:
        d.retention_risk = "Low"

    # ====================================================
    # RULE SET E: Wellbeing / Burnout Flag
    # ====================================================
    if ot > t["high_overtime"] or att < t["critical_attendance"]:
        d.wellbeing_flag = True
        if ot > t["high_overtime"]:
            actions.append(f"🏥 Overtime ({ot:.0f} hrs/qtr) exceeds healthy limit — review workload distribution")
        if att < t["critical_attendance"]:
            actions.append(f"🏥 Attendance ({att*100:.0f}%) is critically low — confidential wellbeing session recommended")

    # ====================================================
    # RULE SET F: Quick wins for Medium performers
    # ====================================================
    if predicted_band == "Medium":
        if th >= t["low_training_hours"] and ga >= t["medium_goal_achievement"]:
            actions.append("📈 Near High-performer threshold — consider stretch project assignment")
        else:
            actions.append("📚 Targeted skills development can unlock High performance in next quarter")

    # ====================================================
    # Priority Classification
    # ====================================================
    if d.pip_required or d.retention_risk == "High" or d.wellbeing_flag:
        d.priority = "Urgent"
    elif predicted_band == "Medium" and d.training_needed:
        d.priority = "Medium"
    else:
        d.priority = "Normal"

    # ====================================================
    # Build Summary
    # ====================================================
    training_str = ""
    if d.training_needed:
        topics = "; ".join(d.training_topics[:2])
        training_str = f"Training: {topics}. "

    promotion_str = "Promotion candidate." if d.promotion_ready else ""
    pip_str       = "PIP required." if d.pip_required else ""
    risk_str      = f"Retention risk: {d.retention_risk}."

    d.summary = (
        f"[{d.priority}] {predicted_band} performer ({d.confidence_pct}% confidence). "
        f"{training_str}{promotion_str}{pip_str}{risk_str}"
    )

    d.recommended_actions = actions[:6]  # cap at 6 actions
    return d


# ── Batch decisions ───────────────────────────────────────
def batch_decisions(
    test_df:        pd.DataFrame,
    predictions:    np.ndarray,
    probabilities:  np.ndarray,
) -> pd.DataFrame:
    """
    Apply the decision engine to all employees in the test set.
    Returns a DataFrame with HR recommendations.
    """
    band_map = {0: "Low", 1: "Medium", 2: "High"}
    records  = []

    for i, row in test_df.iterrows():
        idx         = test_df.index.get_loc(i)
        pred_band   = band_map[int(predictions[idx])]
        confidence  = float(probabilities[idx].max())

        features = {
            "training_hours":       row.get("training_hours",       20),
            "attendance_rate":      row.get("attendance_rate",       0.90),
            "overtime_hours":       row.get("overtime_hours",        10),
            "goal_achievement_pct": row.get("goal_achievement_pct",  60),
            "manager_rating":       row.get("manager_rating",        3.0),
            "peer_score":           row.get("peer_score",            3.0),
            "projects_completed":   row.get("projects_completed",    3),
            "experience_years":     row.get("experience_years",      5),
            "salary":               row.get("salary",                50000),
        }

        decision = decide(row.get("employee_id", f"EMP{i}"), pred_band, confidence, features)
        records.append(decision.to_dict())

    return pd.DataFrame(records)


# ── Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample: predict a Low performer
    result = decide(
        employee_id    = "EMP0042",
        predicted_band = "Low",
        confidence     = 0.72,
        features = {
            "training_hours":       8,
            "attendance_rate":      0.75,
            "overtime_hours":       35,
            "goal_achievement_pct": 42,
            "manager_rating":       2.1,
            "peer_score":           2.8,
            "projects_completed":   1,
            "experience_years":     3,
            "salary":               35000,
        }
    )
    print("\n" + "=" * 60)
    print("  HR DECISION ENGINE — SAMPLE OUTPUT")
    print("=" * 60)
    for k, v in result.to_dict().items():
        print(f"  {k:<25}: {v}")
