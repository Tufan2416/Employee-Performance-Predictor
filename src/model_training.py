"""
=============================================================
 model_training.py
 Employee Performance Predictor – Model Training & Comparison
=============================================================
 Models trained:
   1. Logistic Regression  (baseline – interpretable)
   2. Random Forest        (ensemble – robust)
   3. XGBoost              (gradient boosting – high performance)

 All models use time-based split; no random leakage.
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[Warning] XGBoost not installed. Skipping XGB model.")


# ── Class labels ──────────────────────────────────────────
CLASSES     = [0, 1, 2]
CLASS_NAMES = ["Low", "Medium", "High"]


# ── Model definitions ─────────────────────────────────────
def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter      = 1000,
            C             = 1.0,
            class_weight  = "balanced",
            random_state  = 42,
            solver        = "lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators  = 300,
            max_depth     = 8,
            min_samples_leaf = 10,
            class_weight  = "balanced",
            random_state  = 42,
            n_jobs        = -1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators    = 300,
            max_depth       = 6,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            use_label_encoder = False,
            eval_metric     = "mlogloss",
            random_state    = 42,
            n_jobs          = -1,
        )
    return models


# ── Train + evaluate all models ───────────────────────────
def train_and_compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list,
    models_dir: str = "models",
) -> dict:
    """
    Train every model, evaluate on the held-out test set,
    print a comparison table, and save every trained model.

    Returns:
        results  – dict { model_name → {model, metrics, predictions} }
        best_name – name of the best model (highest macro F1)
    """
    os.makedirs(models_dir, exist_ok=True)
    models  = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n  ▶ Training: {name} ...")
        y_train = y_train - y_train.min()
        y_test  = y_test  - y_test.min()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm     = confusion_matrix(y_test, y_pred, labels=CLASSES)

        labels = sorted(list(set(y_test)))

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt,
            "confusion_matrix": cm,
            "report": classification_report(
                y_test,
                y_pred,
                labels=labels,
                target_names=[CLASS_NAMES[i] for i in labels],
                zero_division=0
            ),
        }

        # Save model
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"    Accuracy: {acc:.4f}  |  F1 (macro): {f1_mac:.4f}  |  Saved → {model_path}")

    # ── Comparison table ──────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}")
    print("─" * 60)
    for name, res in results.items():
        print(f"  {name:<25} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f} {res['f1_weighted']:>12.4f}")
    print("─" * 60)

    # ── Best model ────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    print(f"\n  🏆 Best Model: {best_name}  (F1-Macro: {results[best_name]['f1_macro']:.4f})")

    # Save best model alias
    best_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(results[best_name]["model"], best_path)
    print(f"  💾 Best model saved → {best_path}")

    # Save comparison summary as CSV
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "Model":        name,
            "Accuracy":     round(res["accuracy"],    4),
            "F1_Macro":     round(res["f1_macro"],    4),
            "F1_Weighted":  round(res["f1_weighted"], 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)

    return results, best_name


# ── Feature importance helper ─────────────────────────────
def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances for tree-based models.
    Returns a sorted DataFrame.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For multi-class LR take mean absolute coefficient
        imp = np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": imp,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return fi_df


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, "..")
    from src.data_generator  import generate_dataset
    from src.preprocessing   import run_preprocessing

    df  = generate_dataset(n_employees=300, save=False)
    out = run_preprocessing(df, models_dir="../models")

    results, best = train_and_compare(
        out["X_train"], out["y_train"],
        out["X_test"],  out["y_test"],
        out["feature_names"],
        models_dir="../models",
    )
    print("\nDone. Best model:", best)
