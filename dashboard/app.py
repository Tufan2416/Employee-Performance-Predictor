"""
=============================================================
 app.py  (Streamlit Dashboard)
 Employee Performance Predictor – Interactive HR Dashboard
=============================================================
 Run with:  streamlit run dashboard/app.py

 Features:
   • Upload employee CSV
   • Single employee prediction with explanation
   • Batch prediction with download
   • Business insights charts
   • Fairness audit view
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.hr_decision_engine import decide, batch_decisions
from src.explainability     import explain_single_prediction, HR_MESSAGES

# ── Load artefacts ────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models")

@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_encoders():
    path = os.path.join(MODELS_DIR, "encoders.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ── Constants ─────────────────────────────────────────────
FEATURE_COLS = [
    "age", "experience_years", "salary",
    "training_hours", "attendance_rate", "overtime_hours",
    "projects_completed", "manager_rating", "peer_score",
    "goal_achievement_pct",
    "department_enc", "education_enc", "role_enc",
]
DEPARTMENTS = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]
EDUCATIONS  = ["High School", "Bachelor's", "Master's", "PhD"]
ROLES       = ["Junior Dev", "Senior Dev", "Tech Lead", "Sales Rep", "Sales Manager",
               "HR Associate", "HR Manager", "Recruiter", "Marketing Analyst",
               "Financial Analyst", "Accountant", "Ops Analyst", "Process Lead"]
BAND_MAP    = {0: "Low", 1: "Medium", 2: "High"}
BAND_COLOR  = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
CLASS_NAMES = ["Low", "Medium", "High"]


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Employee Performance Predictor",
    page_icon  = "📊",
    layout     = "wide",
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    .metric-card {
        background: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .band-high   { color: #27ae60; font-size: 28px; font-weight: bold; }
    .band-medium { color: #e67e22; font-size: 28px; font-weight: bold; }
    .band-low    { color: #e74c3c; font-size: 28px; font-weight: bold; }
    .section-title { font-size: 18px; font-weight: 600; margin-top: 12px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.title("📊 Employee Performance Predictor")
st.markdown(
    "**Industry-Grade HR Analytics Dashboard** | Powered by Machine Learning + Explainable AI"
)
st.markdown("---")

model    = load_model()
scaler   = load_scaler()
encoders = load_encoders()

if model is None:
    st.error(
        "⚠️ No trained model found. Please run `python main.py` first to train the model."
    )
    st.stop()

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🔮 Single Prediction", "📂 Batch Prediction", "📈 Business Insights", "⚖️ Fairness Audit"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**How to use:**\n"
    "1. Run `python main.py` to train\n"
    "2. Use Single Prediction to test one employee\n"
    "3. Upload a CSV for batch results\n"
    "4. Explore insights and fairness tabs"
)


# ══════════════════════════════════════════════════════════
# PAGE 1: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════
def encode_value(col: str, val: str) -> int:
    """Encode a categorical using the saved LabelEncoder."""
    if encoders and col in encoders:
        try:
            return int(encoders[col].transform([val])[0])
        except Exception:
            return 0
    return 0


if page == "🔮 Single Prediction":
    st.subheader("🔮 Single Employee Performance Prediction")
    st.markdown("Enter employee details below to get an instant performance prediction with AI explanation.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Employee Info**")
        emp_id     = st.text_input("Employee ID", value="EMP0001")
        department = st.selectbox("Department", DEPARTMENTS)
        education  = st.selectbox("Education", EDUCATIONS)
        role       = st.selectbox("Role", ROLES)
        age        = st.slider("Age", 22, 60, 30)
        experience = st.slider("Experience (years)", 0, 35, 5)
        salary     = st.number_input("Salary (₹ / $)", min_value=20000, max_value=250000, value=60000, step=1000)

    with col2:
        st.markdown("**📊 Quarterly Performance Metrics**")
        training_hours    = st.slider("Training Hours", 0, 50, 15)
        attendance_rate   = st.slider("Attendance Rate", 0.50, 1.00, 0.90, step=0.01)
        overtime_hours    = st.slider("Overtime Hours", 0, 80, 10)
        projects_done     = st.slider("Projects Completed", 0, 10, 3)
        goal_achievement  = st.slider("Goal Achievement (%)", 0, 100, 65)

    with col3:
        st.markdown("**⭐ Ratings (out of 5)**")
        manager_rating = st.slider("Manager Rating", 1.0, 5.0, 3.5, step=0.1)
        peer_score     = st.slider("Peer Collaboration Score", 1.0, 5.0, 3.5, step=0.1)

    if st.button("🚀 Predict Performance", type="primary"):
        dept_enc = encode_value("department", department)
        edu_enc  = encode_value("education",  education)
        role_enc = encode_value("role",       role)

        raw_features = {
            "age":                  age,
            "experience_years":     experience,
            "salary":               salary,
            "training_hours":       training_hours,
            "attendance_rate":      attendance_rate,
            "overtime_hours":       overtime_hours,
            "projects_completed":   projects_done,
            "manager_rating":       manager_rating,
            "peer_score":           peer_score,
            "goal_achievement_pct": goal_achievement,
            "department_enc":       dept_enc,
            "education_enc":        edu_enc,
            "role_enc":             role_enc,
        }

        feat_arr = np.array([[raw_features[c] for c in FEATURE_COLS]])

        if scaler:
            feat_scaled = scaler.transform(feat_arr)
        else:
            feat_scaled = feat_arr

        # ── Prediction ──────────────────────────────────
        proba      = model.predict_proba(feat_scaled)[0]
        pred_class = int(np.argmax(proba))
        pred_band  = BAND_MAP[pred_class]

        # ── Explanation ─────────────────────────────────
        explanation = explain_single_prediction(
            model            = model,
            employee_features= feat_scaled[0],
            feature_names    = FEATURE_COLS,
            feature_values_raw = raw_features,
            sample_idx       = 0,
        )

        # ── HR Decision ─────────────────────────────────
        hr_dec = decide(emp_id, pred_band, float(proba[pred_class]), raw_features)

        st.markdown("---")
        st.subheader("📋 Prediction Results")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Band", pred_band)
        with m2:
            st.metric("Confidence", f"{proba[pred_class]*100:.1f}%")
        with m3:
            st.metric("Retention Risk", hr_dec.retention_risk)
        with m4:
            st.metric("Priority", hr_dec.priority)

        # Probability bar chart
        st.markdown("**Probability Breakdown:**")
        # Fix probabilities safely
        low  = proba[0] if len(proba) > 0 else 0
        med  = proba[1] if len(proba) > 1 else 0
        high = proba[2] if len(proba) > 2 else 0

        prob_df = pd.DataFrame({
            "Band": ["Low", "Medium", "High"],
            "Probability": [low, med, high],
        })
        fig, ax = plt.subplots(figsize=(6, 2.5))
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        ax.barh(prob_df["Band"], prob_df["Probability"], color=colors, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.01, i, f"{v:.1%}", va="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Explanation
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**✅ Key Strengths:**")
            for msg in explanation["top3_positive_drivers"]:
                st.success(msg)
        with col_b:
            st.markdown("**⚠️ Areas of Concern:**")
            for msg in explanation["top3_negative_drivers"]:
                st.warning(msg)

        # HR recommendations
        st.markdown("**📌 Recommended HR Actions:**")
        for action in hr_dec.recommended_actions:
            st.info(action)

        if hr_dec.pip_required:
            with st.expander("📄 Performance Improvement Plan (PIP) Goals"):
                for goal in hr_dec.pip_goals:
                    st.write(f"• {goal}")

        if hr_dec.promotion_ready:
            st.balloons()
            st.success("🚀 This employee is a PROMOTION CANDIDATE!")


# ══════════════════════════════════════════════════════════
# PAGE 2: BATCH PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.subheader("📂 Batch Employee Prediction")
    st.markdown("Upload a CSV file with employee data to predict performance for all employees.")

    sample_cols = ["employee_id", "age", "experience_years", "salary",
                   "training_hours", "attendance_rate", "overtime_hours",
                   "projects_completed", "manager_rating", "peer_score",
                   "goal_achievement_pct", "department", "education", "role"]
    st.info(f"📋 Required columns: `{', '.join(sample_cols)}`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**Loaded:** {len(df)} employees")
        st.dataframe(df.head(5))

        if st.button("🔮 Run Batch Predictions", type="primary"):
            with st.spinner("Running predictions …"):
                # Encode
                for col, ecol in [("department","department_enc"),
                                   ("education","education_enc"),
                                   ("role","role_enc")]:
                    if col in df.columns and encoders and col in encoders:
                        df[ecol] = encoders[col].transform(df[col].astype(str))
                    else:
                        df[ecol] = 0

                avail     = [c for c in FEATURE_COLS if c in df.columns]
                X         = df[avail].fillna(0).values
                X_scaled  = scaler.transform(X) if scaler else X

                preds     = model.predict(X_scaled)
                probas    = model.predict_proba(X_scaled)
                df["Predicted_Band"] = [BAND_MAP[int(p)] for p in preds]
                df["Confidence_%"]   = (probas.max(axis=1) * 100).round(1)

                hr_recs = batch_decisions(df, preds, probas)
                result  = df.merge(hr_recs[["Employee ID","Training Needed","Promotion Ready",
                                             "PIP Required","Retention Risk","Priority"]],
                                    left_on="employee_id", right_on="Employee ID", how="left")

            st.success("✅ Predictions complete!")
            st.dataframe(result[["employee_id","department","Predicted_Band",
                                  "Confidence_%","Retention Risk","Priority"]].head(20))

            csv = result.to_csv(index=False)
            st.download_button(
                label    = "⬇️  Download Full Results CSV",
                data     = csv,
                file_name= "batch_predictions.csv",
                mime     = "text/csv",
            )

            # Summary
            st.subheader("Summary")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🟢 High Performers",  (df["Predicted_Band"]=="High").sum())
            with c2: st.metric("🟡 Medium Performers",(df["Predicted_Band"]=="Medium").sum())
            with c3: st.metric("🔴 Low Performers",   (df["Predicted_Band"]=="Low").sum())


# ══════════════════════════════════════════════════════════
# PAGE 3: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "📈 Business Insights":
    st.subheader("📈 Business Insights Dashboard")

    graphs_dir = os.path.join(ROOT, "outputs", "graphs")
    graphs = {
        "Class Distribution":      "01_class_distribution.png",
        "Feature Distributions":   "02_feature_distributions.png",
        "Correlation Heatmap":     "03_correlation_heatmap.png",
        "Quarterly Trend":         "04_quarterly_trend.png",
        "Department Performance":  "05_dept_performance.png",
        "Training vs Performance": "06_training_vs_performance.png",
        "Confusion Matrices":      "07_confusion_matrices.png",
        "Model Comparison":        "08_model_comparison.png",
        "Feature Importance":      "09_feature_importance.png",
        "Top/Bottom Performers":   "11_top_bottom_performers.png",
    }

    available = {k: v for k, v in graphs.items()
                 if os.path.exists(os.path.join(graphs_dir, v))}

    if not available:
        st.warning("No graphs found. Run `python main.py` first to generate all plots.")
    else:
        selected = st.selectbox("Select Chart", list(available.keys()))
        img_path = os.path.join(graphs_dir, available[selected])
        st.image(img_path, use_column_width=True)

        descriptions = {
            "Class Distribution":     "Shows how many employees fall into each performance band. Useful for spotting class imbalance.",
            "Feature Distributions":  "Compares distributions of key features across performance bands to spot patterns.",
            "Correlation Heatmap":    "Highlights which features are strongly correlated — useful for feature selection.",
            "Quarterly Trend":        "Tracks average performance score over time — detects seasonal dips or improvements.",
            "Department Performance": "Reveals which departments have the most High/Medium/Low performers.",
            "Training vs Performance":"Visualises the ROI of training investment on performance outcomes.",
            "Confusion Matrices":     "Shows model prediction accuracy per class for all trained models.",
            "Model Comparison":       "Side-by-side accuracy and F1 comparison to justify final model selection.",
            "Feature Importance":     "Top features driving model predictions — useful for HR focus areas.",
            "Top/Bottom Performers":  "Candidate lists: employees most eligible for promotion vs those needing intervention.",
        }
        st.info(f"📊 **HR Insight:** {descriptions.get(selected, '')}")


# ══════════════════════════════════════════════════════════
# PAGE 4: FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════
elif page == "⚖️ Fairness Audit":
    st.subheader("⚖️ Model Fairness & Bias Audit")
    st.markdown(
        "Ensuring the model does not systematically disadvantage any department or experience group "
        "is a legal and ethical requirement in real HR AI systems."
    )

    outputs_dir = os.path.join(ROOT, "outputs")

    tabs = st.tabs(["Department Audit", "Experience Audit", "Statistical Parity"])

    with tabs[0]:
        dept_path = os.path.join(outputs_dir, "fairness_department.csv")
        if os.path.exists(dept_path):
            dept_df = pd.read_csv(dept_path)
            st.dataframe(dept_df, use_container_width=True)

            fig_path = os.path.join(outputs_dir, "graphs", "10_fairness_dept.png")
            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)

            bias_found = dept_df["Bias_Flag"].str.contains("INVESTIGATE").any()
            if bias_found:
                st.error("⚠️ Bias detected in one or more departments. See mitigation strategies.")
                with st.expander("💡 Mitigation Strategies"):
                    st.write("1. Review training hours distribution across departments")
                    st.write("2. Audit goal-setting processes — are goals calibrated fairly?")
                    st.write("3. Consider fairness constraints in model retraining")
                    st.write("4. Engage HR leadership to investigate structural inequities")
            else:
                st.success("✅ No significant departmental bias detected.")
        else:
            st.warning("Run `python main.py` to generate fairness audit results.")

    with tabs[1]:
        exp_path = os.path.join(outputs_dir, "fairness_experience.csv")
        if os.path.exists(exp_path):
            exp_df = pd.read_csv(exp_path)
            st.dataframe(exp_df, use_container_width=True)
        else:
            st.warning("Run `python main.py` first.")

    with tabs[2]:
        spd_path = os.path.join(outputs_dir, "fairness_spd.csv")
        if os.path.exists(spd_path):
            spd_df = pd.read_csv(spd_path)
            st.dataframe(spd_df, use_container_width=True)
            st.info(
                "**Statistical Parity Difference (SPD):** Ideal value = 0. "
                "|SPD| > 0.10 indicates potential bias that warrants investigation."
            )
        else:
            st.warning("Run `python main.py` first.")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Employee Performance Predictor v2.0 | Built for placement portfolio | "
    "Powered by scikit-learn, XGBoost, Streamlit</small>",
    unsafe_allow_html=True,
)
