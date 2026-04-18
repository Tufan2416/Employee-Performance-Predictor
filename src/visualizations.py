"""
=============================================================
 visualizations.py
 Employee Performance Predictor – All Charts & Graphs
=============================================================
 Generates publication-quality plots for:
   1. EDA (distribution, correlations, trends)
   2. Model evaluation (confusion matrix, ROC)
   3. Feature importance
   4. Business insights (dept trends, training impact)
   5. Fairness audit charts
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import warnings
warnings.filterwarnings("ignore")

# ── Style settings ────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE   = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
FIG_DPI   = 150
OUTPUT_DIR = "outputs/graphs"


def _save(fig, name: str, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════
# 1. EDA PLOTS
# ══════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Bar chart of performance band distribution."""
    counts = df["performance_band"].value_counts().reindex(["High", "Medium", "Low"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[b] for b in counts.index], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Performance Band Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Employees")
    ax.set_xlabel("Performance Band")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, "01_class_distribution.png", output_dir)


def plot_feature_distributions(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Histogram grid of key numeric features by band."""
    features = ["training_hours", "attendance_rate", "goal_achievement_pct",
                "manager_rating", "peer_score", "overtime_hours"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        for band, color in PALETTE.items():
            subset = df[df["performance_band"] == band][feat].dropna()
            ax.hist(subset, bins=20, alpha=0.55, color=color, label=band, density=True)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Feature Distributions by Performance Band", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "02_feature_distributions.png", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Correlation heatmap of numeric features."""
    num_cols = ["training_hours", "attendance_rate", "goal_achievement_pct",
                "manager_rating", "peer_score", "overtime_hours",
                "experience_years", "salary", "projects_completed", "performance_score"]
    existing = [c for c in num_cols if c in df.columns]
    corr     = df[existing].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask    = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "03_correlation_heatmap.png", output_dir)


def plot_quarterly_trend(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Line chart: average performance score per quarter."""
    trend = (
        df.groupby("quarter_label")["performance_score"]
          .mean()
          .reset_index()
    )
    # Sort chronologically
    def q_key(s):
        y, q = s.split("-Q")
        return int(y) * 10 + int(q)
    trend["_key"] = trend["quarter_label"].apply(q_key)
    trend = trend.sort_values("_key")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend["quarter_label"], trend["performance_score"],
            marker="o", linewidth=2, color="#3498db", markersize=7)
    ax.fill_between(trend["quarter_label"], trend["performance_score"],
                    alpha=0.15, color="#3498db")
    ax.set_title("Average Performance Score by Quarter", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Avg. Performance Score")
    ax.set_ylim(40, 80)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return _save(fig, "04_quarterly_trend.png", output_dir)


def plot_dept_performance(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Grouped bar: performance band % by department."""
    dept_band = (
        df.groupby(["department", "performance_band"])
          .size()
          .reset_index(name="count")
    )
    pivot = dept_band.pivot(index="department", columns="performance_band", values="count").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    # Reorder columns
    for col in ["High", "Medium", "Low"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["High", "Medium", "Low"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax,
               color=[PALETTE["High"], PALETTE["Medium"], PALETTE["Low"]],
               edgecolor="white", width=0.65)
    ax.set_title("Performance Band Distribution by Department (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Department")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Band")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return _save(fig, "05_dept_performance.png", output_dir)


def plot_training_vs_performance(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Scatter: training hours vs performance score coloured by band."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for band, color in PALETTE.items():
        sub = df[df["performance_band"] == band]
        ax.scatter(sub["training_hours"], sub["performance_score"],
                   alpha=0.35, s=18, color=color, label=band)

    # Trend line
    x = df["training_hours"].dropna()
    y = df.loc[x.index, "performance_score"].dropna()
    idx = x.index.intersection(y.index)
    if len(idx) > 10:
        z = np.polyfit(x[idx], y[idx], 1)
        p = np.poly1d(z)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, p(xline), "k--", linewidth=1.5, label="Trend")

    ax.set_title("Training Hours vs Performance Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Hours per Quarter")
    ax.set_ylabel("Performance Score")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "06_training_vs_performance.png", output_dir)


# ══════════════════════════════════════════════════════════
# 2. MODEL EVALUATION PLOTS
# ══════════════════════════════════════════════════════════

def plot_confusion_matrices(results: dict, output_dir=OUTPUT_DIR):
    """Plot confusion matrices for all models side-by-side."""
    n     = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(
            confusion_matrix = res["confusion_matrix"],
            display_labels   = ["Low", "Med", "High"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f} | F1: {res['f1_macro']:.3f}",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Confusion Matrices – All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "07_confusion_matrices.png", output_dir)


def plot_model_comparison_bar(results: dict, output_dir=OUTPUT_DIR):
    """Grouped bar comparing accuracy and F1 across models."""
    names    = list(results.keys())
    accs     = [results[n]["accuracy"]    for n in names]
    f1_macs  = [results[n]["f1_macro"]    for n in names]
    f1_wts   = [results[n]["f1_weighted"] for n in names]

    x   = np.arange(len(names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - w,  accs,    w, label="Accuracy",     color="#3498db", alpha=0.85)
    ax.bar(x,      f1_macs, w, label="F1 (Macro)",   color="#e67e22", alpha=0.85)
    ax.bar(x + w,  f1_wts,  w, label="F1 (Weighted)",color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Accuracy & F1 Scores", fontsize=14, fontweight="bold")
    ax.legend()

    for i, (a, f, fw) in enumerate(zip(accs, f1_macs, f1_wts)):
        ax.text(i - w,   a  + 0.01, f"{a:.3f}",  ha="center", fontsize=8)
        ax.text(i,       f  + 0.01, f"{f:.3f}",  ha="center", fontsize=8)
        ax.text(i + w,   fw + 0.01, f"{fw:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    return _save(fig, "08_model_comparison.png", output_dir)


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str, output_dir=OUTPUT_DIR):
    """Horizontal bar chart of feature importances."""
    top = fi_df.head(12)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                   color="#3498db", alpha=0.80, edgecolor="white")
    ax.set_title(f"Top Feature Importances – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=8)
    plt.tight_layout()
    return _save(fig, "09_feature_importance.png", output_dir)


# ══════════════════════════════════════════════════════════
# 3. FAIRNESS PLOTS
# ══════════════════════════════════════════════════════════

def plot_fairness_dept(dept_df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Grouped bar: predicted vs actual High performer % by department."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(dept_df))
    w = 0.35

    ax.bar(x - w/2, dept_df["Predicted_High_%"], w, label="Predicted High %",
           color="#3498db", alpha=0.85)
    ax.bar(x + w/2, dept_df["Actual_High_%"],    w, label="Actual High %",
           color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(dept_df["Department"], rotation=20, ha="right")
    ax.set_title("Predicted vs Actual High Performers by Department", fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.tight_layout()
    return _save(fig, "10_fairness_dept.png", output_dir)


# ══════════════════════════════════════════════════════════
# 4. BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════

def plot_top_bottom_performers(test_df: pd.DataFrame, predictions, output_dir=OUTPUT_DIR):
    """Identify and visualise top/bottom 10 employees by predicted probability."""
    df = test_df.copy().reset_index(drop=True)
    df["pred_band"] = predictions

    high_risk = df[df["pred_band"] == 0][["employee_id", "department", "experience_years",
                                           "goal_achievement_pct", "manager_rating"]].head(10)
    top_perf  = df[df["pred_band"] == 2][["employee_id", "department", "experience_years",
                                           "goal_achievement_pct", "manager_rating"]].head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Top performers
    ax1.barh(top_perf["employee_id"], top_perf["goal_achievement_pct"],
             color=PALETTE["High"], alpha=0.80)
    ax1.set_title("🏆 Top Performers (Predicted High)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Goal Achievement %")

    # Low performers (at-risk)
    ax2.barh(high_risk["employee_id"], high_risk["goal_achievement_pct"],
             color=PALETTE["Low"], alpha=0.80)
    ax2.set_title("⚠️  At-Risk Employees (Predicted Low)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Goal Achievement %")

    plt.suptitle("Performance Extremes – Candidate Lists for HR Action", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "11_top_bottom_performers.png", output_dir)


# ── Run all plots ─────────────────────────────────────────
def generate_all_eda_plots(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    print("\n[Visualizations] Generating EDA plots …")
    plot_class_distribution(df, output_dir)
    plot_feature_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_quarterly_trend(df, output_dir)
    plot_dept_performance(df, output_dir)
    plot_training_vs_performance(df, output_dir)
    print("  ✅ EDA plots complete.")


def generate_all_model_plots(results: dict, fi_df: pd.DataFrame, best_name: str, output_dir=OUTPUT_DIR):
    print("\n[Visualizations] Generating model evaluation plots …")
    plot_confusion_matrices(results, output_dir)
    plot_model_comparison_bar(results, output_dir)
    if fi_df is not None and len(fi_df):
        plot_feature_importance(fi_df, best_name, output_dir)
    print("  ✅ Model plots complete.")


if __name__ == "__main__":
    print("Visualization module loaded. Run via main.py.")
