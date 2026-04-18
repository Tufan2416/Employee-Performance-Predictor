# 📊 Employee Performance Predictor
### Industry-Grade HR Analytics System | Built for Placement Portfolio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Overview

A **production-ready HR analytics system** that predicts employee performance bands (High / Medium / Low) using Machine Learning, delivers **Explainable AI (XAI)** insights, generates **actionable HR recommendations**, and includes a full **Streamlit dashboard** — exactly how it would be built at a real company.

> **Why this project stands out:** It uses time-based train/test splitting (no data leakage), SHAP explainability, a rule-based HR Decision Engine, and a fairness audit — features typically seen in enterprise ML systems, not student projects.

---

## 🏢 Business Problem

HR departments struggle to identify:
- Which employees are at risk of low performance **before** it impacts business
- Who is ready for **promotion** vs needs a **Performance Improvement Plan (PIP)**
- Whether model predictions are **fair** across departments and experience levels

This system solves all three.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  Synthetic HR Dataset (500 employees × 12 quarters)         │
│  Time-series: Q1-2022 → Q4-2024 (6,000+ review records)    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 PREPROCESSING LAYER                          │
│  Clean → Impute → Encode → TIME-BASED SPLIT → Scale         │
│  (Train: Q1-2022 → Q4-2023 | Test: Q1-2024 → Q4-2024)      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   ML LAYER                                   │
│  ┌─────────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │ Logistic Regr.  │  │ Random Forest │  │   XGBoost    │  │
│  └────────┬────────┘  └───────┬───────┘  └──────┬───────┘  │
│           └───────────────────┼──────────────────┘          │
│                    Best Model Selected (F1-Macro)            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               INTELLIGENCE LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  XAI / SHAP  │  │ HR Decision  │  │  Fairness Audit   │  │
│  │ Explainability│  │   Engine     │  │  (Bias Check)     │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                PRESENTATION LAYER                            │
│          Streamlit Dashboard + Visualization Suite           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP, Permutation Importance |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Development | VS Code / Jupyter |

---

## 📁 Folder Structure

```
Employee-Performance-Predictor/
│
├── data/                    # Auto-generated HR dataset (CSV)
├── notebooks/               # Jupyter EDA notebook
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic HR data with employee lifecycle
│   ├── preprocessing.py     # Cleaning, encoding, TIME-BASED split
│   ├── model_training.py    # Train + compare 3 ML models
│   ├── explainability.py    # SHAP + permutation importance
│   ├── hr_decision_engine.py# Rule-based HR recommendations
│   ├── fairness_check.py    # Bias audit across departments
│   └── visualizations.py   # All charts and graphs
│
├── models/                  # Saved model artifacts (.pkl)
├── outputs/
│   ├── graphs/              # All generated PNG charts
│   ├── hr_decisions.csv     # Batch HR recommendations
│   ├── feature_importance.csv
│   └── fairness_*.csv       # Fairness audit results
│
├── dashboard/
│   └── app.py              # Streamlit interactive dashboard
│
├── main.py                 # Master pipeline script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

This will:
- Generate 500 synthetic employees × 12 quarters of data
- Train and compare Logistic Regression, Random Forest, XGBoost
- Generate all visualization charts
- Run the HR Decision Engine on all test employees
- Perform fairness audit
- Print a real-world case study

### 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

### 4. Custom Employee Count

```bash
python main.py --employees 1000
```

---

## 🔑 Key Features

### ⏱️ Time-Based Train/Test Split (No Data Leakage)
Unlike typical ML projects using random splits, this system trains on past quarters and tests on future quarters — exactly how a production HR system would be deployed.

```
Train: Q1-2022 → Q4-2023  (8 quarters)
Test:  Q1-2024 → Q4-2024  (4 quarters)
```

### 🧠 Explainable AI
Every prediction comes with:
- Top 3 positive performance drivers
- Top 3 negative/risk factors
- HR-friendly plain English explanations

Example:
> *"Low training hours (6 hrs) are limiting growth — enroll in L&D programs [value: 6]"*

### 📋 HR Decision Engine
Converts model output into concrete HR actions:
- **High performers** → Promotion readiness check
- **Medium performers** → Stretch project recommendations
- **Low performers** → PIP goals + coaching plan
- **Burnout signals** → Wellbeing intervention

### ⚖️ Fairness Audit
- Adverse Impact Ratio (4/5ths rule) by department
- Statistical Parity Difference across groups
- Automatic bias flag with mitigation suggestions

---

## 📊 Model Results

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.68 | ~0.71 |
| Random Forest | ~0.81 | ~0.79 | ~0.80 |
| **XGBoost** *(selected)* | **~0.83** | **~0.81** | **~0.82** |

*Results vary slightly with random seed. XGBoost selected based on F1-Macro.*

---

## 📋 Case Study Example

**Employee: EMP0042** | At-Risk Profile

| Input Feature | Value |
|---|---|
| Training Hours | 6 hrs/quarter |
| Attendance Rate | 74% |
| Goal Achievement | 38% |
| Manager Rating | 2.0/5 |
| Overtime Hours | 40 hrs |

**Model Output:** 🔴 LOW (confidence: 72%)

**HR Actions Generated:**
1. ⚠️ Schedule PIP review meeting within 2 weeks
2. 🏥 Overtime (40 hrs/qtr) exceeds healthy limit — review workload
3. 🏥 Attendance (74%) is critically low — wellbeing session recommended
4. 🔴 High attrition risk — escalate to HRBP immediately

---

## 🔮 Future Improvements

- [ ] **Real HR Dataset** — IBM HR Analytics (Kaggle), SHRM datasets
- [ ] **Deep Learning** — LSTM for temporal performance forecasting
- [ ] **Attrition Prediction** — Predict who will resign in next 6 months
- [ ] **Real-Time System** — FastAPI + PostgreSQL backend
- [ ] **MLOps** — MLflow experiment tracking, automated monthly retraining
- [ ] **Advanced Fairness** — Equalized odds constraints during training

---

## 🎓 Interview Preparation

### Top 10 Questions + Answers

**Q1: Why did you use a time-based split instead of random?**
> Random split allows future data to "leak" into training, creating artificially high accuracy that won't hold in deployment. In real HR systems, the model is trained on past quarters and must predict future performance — time-based split correctly simulates this.

**Q2: What is SHAP and why is it important?**
> SHAP (SHapley Additive exPlanations) measures each feature's contribution to a specific prediction. It's important in HR because decisions affecting employees must be explainable — HR managers need to know WHY the model flagged someone, not just THAT it did.

**Q3: How does the HR Decision Engine work?**
> It's a rule-based system that maps model predictions + feature values to specific HR actions. For example, if predicted_band = "Low" AND attendance < 0.80, it triggers a wellbeing check recommendation. This bridges the gap between an ML prediction and a business decision.

**Q4: How did you handle class imbalance?**
> Used `class_weight='balanced'` in models, which inversely weights classes by frequency. Also evaluated with F1-Macro (treats all classes equally) rather than accuracy alone.

**Q5: What is the fairness audit checking?**
> It checks if the model systematically predicts higher/lower performance for certain departments (Adverse Impact Ratio) and computes Statistical Parity Difference. If |SPD| > 0.10, we flag it for human review.

---

## 📸 Screenshots

*Run `python main.py` then check `outputs/graphs/` for all generated charts.*

Key outputs:
- `01_class_distribution.png` — Band distribution
- `05_dept_performance.png` — Department analysis
- `07_confusion_matrices.png` — Model comparison
- `08_model_comparison.png` — Accuracy/F1 bar chart
- `09_feature_importance.png` — Top features
- `11_top_bottom_performers.png` — HR candidate lists

---

## 🤝 Contributing

This is a portfolio project. Feel free to fork and build on it!

---

## 📄 License

MIT License — free to use for learning and portfolio purposes.

---

*Built by [YOUR NAME] | Data Science Enthusiast | [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)*
