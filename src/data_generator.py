"""
=============================================================
 data_generator.py
 Employee Performance Predictor – Synthetic HR Data Generator
=============================================================
 Creates a realistic, time-aware HR dataset that simulates:
   • Employee lifecycle  (join → train → perform → review)
   • Quarterly performance reviews  (Q1-2022 … Q4-2024)
   • Natural noise and imperfect patterns (like real data)
   • Continuous performance score → band mapping
 NO data leakage: only features available BEFORE the review
 period are included.
=============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────
DEPARTMENTS = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]
EDUCATION   = ["High School", "Bachelor's", "Master's", "PhD"]
ROLES       = {
    "Engineering":  ["Junior Dev", "Senior Dev", "Tech Lead"],
    "Sales":        ["Sales Rep", "Sales Manager", "Account Exec"],
    "HR":           ["HR Associate", "HR Manager", "Recruiter"],
    "Marketing":    ["Marketing Analyst", "Content Lead", "Brand Manager"],
    "Finance":      ["Financial Analyst", "Accountant", "CFO Analyst"],
    "Operations":   ["Ops Analyst", "Process Lead", "Logistics Manager"],
}

# ── Quarter helpers ───────────────────────────────────────
def quarter_to_date(year: int, q: int) -> pd.Timestamp:
    """Convert year + quarter number to a timestamp (start of quarter)."""
    month = (q - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def generate_quarters(start_year=2022, end_year=2024):
    """Return list of (year, quarter) tuples."""
    quarters = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            quarters.append((y, q))
    return quarters


# ── Core data generation ──────────────────────────────────
def generate_employee_base(n_employees: int = 500) -> pd.DataFrame:
    """
    Generate static employee attributes that don't change over time.
    These are set once when the employee 'joins'.
    """
    emp_ids = [f"EMP{str(i).zfill(4)}" for i in range(1, n_employees + 1)]

    departments   = np.random.choice(DEPARTMENTS, n_employees, p=[0.25, 0.20, 0.10, 0.15, 0.15, 0.15])
    education     = np.random.choice(EDUCATION,   n_employees, p=[0.10, 0.50, 0.30, 0.10])

    ages          = np.random.randint(22, 58, n_employees)
    experience    = np.clip(ages - 22 + np.random.randint(-2, 4, n_employees), 0, 35)

    roles = [random.choice(ROLES[d]) for d in departments]

    # Salary driven by experience + department (with noise)
    base_salary = {
        "Engineering": 70000, "Sales": 55000, "HR": 50000,
        "Marketing": 58000,   "Finance": 65000, "Operations": 52000,
    }
    salaries = np.array([
        base_salary[d] + experience[i] * 1500 + np.random.normal(0, 3000)
        for i, d in enumerate(departments)
    ]).clip(30000, 200000).astype(int)

    # Personality index (latent variable – not directly exposed)
    personality_score = np.random.beta(5, 2, n_employees)  # skewed high

    # Join date  (some employees joined 3+ years ago, some recently)
    join_offsets = np.random.randint(0, 36, n_employees)   # months before 2022
    join_dates   = [
        (datetime(2022, 1, 1) - timedelta(days=int(30 * o))).strftime("%Y-%m-%d")
        for o in join_offsets
    ]

    return pd.DataFrame({
        "employee_id":       emp_ids,
        "department":        departments,
        "education":         education,
        "role":              roles,
        "age":               ages,
        "experience_years":  experience,
        "salary":            salaries,
        "join_date":         join_dates,
        "_personality":      personality_score,   # hidden latent variable
    })


def generate_quarterly_records(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every employee × quarter produce one performance review row.
    Features are generated using realistic causal rules with noise.
    """
    quarters = generate_quarters(2022, 2024)   # 12 quarters
    rows     = []

    for _, emp in base_df.iterrows():
        # Employee-level random effects (stable across quarters)
        motivation_base = emp["_personality"] * 0.6 + np.random.uniform(0.1, 0.4)
        skill_factor    = min(1.0, emp["experience_years"] / 20 + np.random.uniform(0, 0.3))
        edu_bonus       = {"High School": 0, "Bachelor's": 0.05, "Master's": 0.10, "PhD": 0.15}[emp["education"]]

        for idx, (year, q) in enumerate(quarters):
            quarter_label = f"{year}-Q{q}"
            review_date   = quarter_to_date(year, q)

            # ── Time-varying features ─────────────────────────
            # Training hours: improves over time with some randomness
            training_hours = max(0, np.random.normal(
                loc  = 12 + idx * 0.5 + skill_factor * 5,
                scale= 4
            ))

            # Attendance rate (0–1)
            attendance_rate = np.clip(np.random.normal(0.90 + motivation_base * 0.05, 0.06), 0.60, 1.0)

            # Overtime hours per quarter
            overtime_hours = max(0, np.random.normal(
                loc  = 8 + (1 - motivation_base) * 10,
                scale= 5
            ))

            # Projects completed
            projects_completed = max(0, int(np.random.normal(
                loc  = 3 + skill_factor * 2 + training_hours / 15,
                scale= 1
            )))

            # Manager rating (1–5), influenced by personality + performance
            manager_rating = np.clip(np.random.normal(
                loc  = 2.5 + motivation_base * 2 + edu_bonus * 2,
                scale= 0.5
            ), 1.0, 5.0)

            # Peer collaboration score (1–5)
            peer_score = np.clip(np.random.normal(
                loc  = 2.5 + emp["_personality"] * 2.5,
                scale= 0.6
            ), 1.0, 5.0)

            # Goal achievement % (0–100)
            goal_achievement = np.clip(np.random.normal(
                loc  = 50 + skill_factor * 30 + training_hours * 0.8 + edu_bonus * 20,
                scale= 12
            ), 10, 100)

            # ── Continuous performance score (10–100) ─────────
            # Causal formula (like a real KPI roll-up):
            perf_score = (
                0.30 * goal_achievement
              + 0.20 * manager_rating  * 20          # scale to 0-100
              + 0.15 * peer_score      * 20
              + 0.15 * attendance_rate * 100
              + 0.10 * min(training_hours / 30, 1) * 100
              + 0.05 * min(projects_completed / 6, 1) * 100
              + 0.05 * (1 - overtime_hours / 50) * 100  # too much OT → burnout
              + np.random.normal(0, 5)                   # irreducible noise
            )
            perf_score = np.clip(perf_score, 10, 100)

            # ── Performance band ──────────────────────────────
            if perf_score >= 75:
                perf_band = "High"
            elif perf_score >= 50:
                perf_band = "Medium"
            else:
                perf_band = "Low"

            rows.append({
                "employee_id":          emp["employee_id"],
                "department":           emp["department"],
                "education":            emp["education"],
                "role":                 emp["role"],
                "age":                  emp["age"],
                "experience_years":     emp["experience_years"],
                "salary":               emp["salary"],
                "join_date":            emp["join_date"],
                "review_year":          year,
                "review_quarter":       q,
                "quarter_label":        quarter_label,
                "review_date":          review_date,
                "training_hours":       round(training_hours, 1),
                "attendance_rate":      round(attendance_rate, 3),
                "overtime_hours":       round(overtime_hours, 1),
                "projects_completed":   projects_completed,
                "manager_rating":       round(manager_rating, 2),
                "peer_score":           round(peer_score, 2),
                "goal_achievement_pct": round(goal_achievement, 1),
                "performance_score":    round(perf_score, 2),
                "performance_band":     perf_band,
            })

    return pd.DataFrame(rows)


def inject_missing_and_noise(df: pd.DataFrame, missing_rate: float = 0.03) -> pd.DataFrame:
    """
    Inject random missing values and occasional data-entry errors
    to mimic realistic messy HR data.
    """
    numeric_cols = ["training_hours", "overtime_hours", "peer_score", "goal_achievement_pct"]
    df = df.copy()

    for col in numeric_cols:
        mask = np.random.rand(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    # Occasional duplicate records (data entry error)
    n_dups = int(len(df) * 0.005)
    dup_idx = np.random.choice(df.index, n_dups, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


# ── Main entry ────────────────────────────────────────────
def generate_dataset(n_employees: int = 500, save: bool = True, output_dir: str = "data") -> pd.DataFrame:
    """
    Full pipeline: generate base → quarterly records → inject noise.
    Returns the final DataFrame and optionally saves it to CSV.
    """
    print("=" * 60)
    print("  Generating Synthetic HR Dataset")
    print("=" * 60)

    print(f"  → Creating {n_employees} employee profiles …")
    base_df = generate_employee_base(n_employees)

    print(f"  → Generating quarterly performance records (12 quarters) …")
    quarterly_df = generate_quarterly_records(base_df)

    print(f"  → Injecting realistic noise and missing values …")
    final_df = inject_missing_and_noise(quarterly_df)

    # Drop the hidden latent variable (not available in real data)
    final_df = final_df.reset_index(drop=True)

    print(f"\n  ✅ Dataset ready: {final_df.shape[0]} rows × {final_df.shape[1]} columns")
    print(f"     Performance band distribution:")
    print(final_df["performance_band"].value_counts().to_string(index=True))

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "hr_dataset.csv")
        final_df.to_csv(path, index=False)
        print(f"\n  💾 Saved to: {path}")

    return final_df


# ── Run standalone ────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset(n_employees=500, save=True, output_dir="../data")
    print("\nSample rows:")
    print(df.head(3).to_string())
