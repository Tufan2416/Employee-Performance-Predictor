"""
=============================================================
 preprocessing.py
 Employee Performance Predictor – Data Cleaning & Splitting
=============================================================
 Responsibilities:
   1. Drop duplicates
   2. Impute missing values (median / mode strategy)
   3. Encode categoricals
   4. Feature scaling
   5. TIME-BASED train / test split  ← critical for real-world ML
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# ── Columns used as features ──────────────────────────────
FEATURE_COLS = [
    "age", "experience_years", "salary",
    "training_hours", "attendance_rate", "overtime_hours",
    "projects_completed", "manager_rating", "peer_score",
    "goal_achievement_pct",
    # encoded categoricals added during preprocessing:
    "department_enc", "education_enc", "role_enc",
]

TARGET_COL        = "performance_band"
TIME_COL          = "quarter_label"
TRAIN_CUTOFF_QLAB = "2024-Q1"   # everything BEFORE this → train; from here → test


# ── Step 1: Basic cleaning ────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates, fix dtypes, drop rows with missing targets.
    """
    print("[Preprocessing] Cleaning data …")
    original_len = len(df)

    # Drop duplicate records
    df = df.drop_duplicates(
        subset=["employee_id", "review_year", "review_quarter"],
        keep="first"
    ).copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL])

    # Ensure numeric dtypes
    numeric_cols = [
        "age", "experience_years", "salary", "training_hours",
        "attendance_rate", "overtime_hours", "projects_completed",
        "manager_rating", "peer_score", "goal_achievement_pct",
        "performance_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Rows before clean: {original_len:,}  →  after: {len(df):,}  (removed {original_len - len(df):,})")
    return df.reset_index(drop=True)


# ── Step 2: Impute missing values ─────────────────────────
def impute_missing(df: pd.DataFrame, fit: bool = True, imputer_path: str = None) -> pd.DataFrame:
    """
    Impute numeric columns with median strategy.
    fit=True  → fit imputer on training data and save it.
    fit=False → load saved imputer and transform test data.
    """
    print("[Preprocessing] Imputing missing values …")
    numeric_cols = [
        "training_hours", "overtime_hours", "peer_score",
        "goal_achievement_pct",
    ]
    existing = [c for c in numeric_cols if c in df.columns]

    if fit:
        imputer = SimpleImputer(strategy="median")
        df[existing] = imputer.fit_transform(df[existing])
        if imputer_path:
            os.makedirs(os.path.dirname(imputer_path), exist_ok=True)
            joblib.dump(imputer, imputer_path)
            print(f"  Imputer saved → {imputer_path}")
    else:
        imputer = joblib.load(imputer_path)
        df[existing] = imputer.transform(df[existing])

    return df


# ── Step 3: Encode categoricals ───────────────────────────
LABEL_ENCODERS: dict = {}   # module-level cache

def encode_categoricals(df: pd.DataFrame, fit: bool = True, encoder_path: str = None) -> pd.DataFrame:
    """
    Label-encode department, education, role.
    fit=True  → fit on train set.
    fit=False → apply saved encoders on test set.
    """
    print("[Preprocessing] Encoding categorical columns …")
    cat_cols = ["department", "education", "role"]
    global LABEL_ENCODERS

    df = df.copy()

    for col in cat_cols:
        enc_col = f"{col}_enc"
        if fit:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))
            LABEL_ENCODERS[col] = le
        else:
            le = LABEL_ENCODERS.get(col)
            if le is None:
                raise ValueError(f"Encoder for '{col}' not found. Run fit=True first.")
            df[enc_col] = le.transform(df[col].astype(str))

    if fit and encoder_path:
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(LABEL_ENCODERS, encoder_path)
        print(f"  Encoders saved → {encoder_path}")

    # Encode target label
    band_map = {"Low": 0, "Medium": 1, "High": 2}
    df["target"] = df[TARGET_COL].map(band_map)

    return df


# ── Step 4: Time-based split ──────────────────────────────
def time_based_split(df: pd.DataFrame):
    """
    Split data chronologically:
        Train → quarters BEFORE TRAIN_CUTOFF_QLAB
        Test  → quarters FROM  TRAIN_CUTOFF_QLAB onward

    Why this matters:
    ─────────────────
    • Random split leaks future information into training.
    • In real HR systems the model is trained on past quarters
      and must generalise to FUTURE employees' data.
    • Time-based split correctly simulates deployment conditions.
    """
    # Build an orderable quarter integer  e.g. "2022-Q3" → 20223
    def q_sort_key(label: str) -> int:
        y, q = label.split("-Q")
        return int(y) * 10 + int(q)

    cutoff_key = q_sort_key(TRAIN_CUTOFF_QLAB)
    df = df.copy()
    df["_q_key"] = df["quarter_label"].apply(q_sort_key)

    train_df = df[df["_q_key"] < cutoff_key].drop(columns=["_q_key"])
    test_df  = df[df["_q_key"] >= cutoff_key].drop(columns=["_q_key"])

    print(f"\n[Preprocessing] ⏱  Time-based split (cutoff: {TRAIN_CUTOFF_QLAB})")
    print(f"  Train → {len(train_df):,} rows  | quarters: {df[df['_q_key'] < cutoff_key]['quarter_label'].unique()[:3]} … ")
    print(f"  Test  → {len(test_df):,} rows   | quarters: {df[df['_q_key'] >= cutoff_key]['quarter_label'].unique()}")

    return train_df, test_df


# ── Step 5: Feature scaling ───────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = None
) -> tuple:
    """
    Fit StandardScaler on train, transform both train and test.
    Prevents data leakage from test statistics entering training.
    """
    print("[Preprocessing] Scaling features …")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved → {scaler_path}")

    return X_train_sc, X_test_sc, scaler


# ── Full pipeline ─────────────────────────────────────────
def run_preprocessing(
    df: pd.DataFrame,
    models_dir: str = "models"
) -> dict:
    """
    Orchestrate all preprocessing steps.
    Returns a dict with train/test arrays + metadata.
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = clean_data(df)

    # Time split BEFORE imputation/encoding to avoid leakage
    train_raw, test_raw = time_based_split(df)

    # Impute – fit on train only
    train_raw = impute_missing(train_raw, fit=True,
                                imputer_path=f"{models_dir}/imputer.pkl")
    test_raw  = impute_missing(test_raw, fit=False,
                                imputer_path=f"{models_dir}/imputer.pkl")

    # Encode – fit on train only
    train_raw = encode_categoricals(train_raw, fit=True,
                                     encoder_path=f"{models_dir}/encoders.pkl")
    test_raw  = encode_categoricals(test_raw, fit=False,
                                     encoder_path=f"{models_dir}/encoders.pkl")

    # Extract feature matrix + target
    available_features = [c for c in FEATURE_COLS if c in train_raw.columns]
    X_train = train_raw[available_features]
    X_test  = test_raw[available_features]
    y_train = train_raw["target"]
    y_test  = test_raw["target"]

    # Scale
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test,
        scaler_path=f"{models_dir}/scaler.pkl"
    )

    print(f"\n  ✅ Preprocessing complete!")
    print(f"     X_train: {X_train_sc.shape}  |  X_test: {X_test_sc.shape}")
    print(f"     Class distribution (train): {dict(y_train.value_counts().sort_index())}")

    return {
        "X_train":          X_train_sc,
        "X_test":           X_test_sc,
        "y_train":          y_train.values,
        "y_test":           y_test.values,
        "feature_names":    available_features,
        "train_raw":        train_raw,
        "test_raw":         test_raw,
        "scaler":           scaler,
        "label_encoders":   LABEL_ENCODERS,
    }


if __name__ == "__main__":
    # Quick sanity check
    import sys
    sys.path.insert(0, "..")
    from src.data_generator import generate_dataset
    df = generate_dataset(n_employees=100, save=False)
    result = run_preprocessing(df, models_dir="../models")
    print(result["X_train"].shape, result["X_test"].shape)
