# app.py — Student Dropout Risk Prediction (Random Forest)
from pathlib import Path
import os

import numpy as np
import pandas as pd
import streamlit as st

# Prefer joblib for sklearn models; fall back to pickle if unavailable
try:
    import joblib
    LOADER = "joblib"
except Exception:
    import pickle
    joblib = None
    LOADER = "pickle"

# =========================
# Page config
# =========================
st.set_page_config(page_title="Student Dropout Risk Prediction", page_icon="🎓", layout="centered")
st.title("🎓 Student Dropout Risk Prediction (Random Forest)")
st.write("Provide the student's academic, behavioral, and demographic data to predict dropout risk.")

# =========================
# Model loading
# =========================
HERE = Path(__file__).parent
MODEL_PATH = HERE / "random_forest_model.pkl"

with st.expander("🔎 Debug info (hide before sharing)"):
    st.write("Working dir:", os.getcwd())
    st.write("App folder:", str(HERE))
    try:
        st.write("Files in app folder:", os.listdir(HERE))
    except Exception as e:
        st.write("Could not list files:", e)
    # Show runtime versions if available
    try:
        import sklearn, numpy
        st.write("Versions →", {"sklearn": sklearn.__version__, "numpy": numpy.__version__})
    except Exception:
        pass

if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

# Load the model once
try:
    if LOADER == "joblib":
        model = joblib.load(MODEL_PATH)
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH} using {LOADER}. Details: {e}")
    st.stop()

# Helper to extract positive-class probability robustly
def get_positive_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(proba[0, 0])
    if hasattr(estimator, "decision_function"):
        val = float(estimator.decision_function(X)[0])
        return 1.0 / (1.0 + np.exp(-val))
    # last resort
    pred = estimator.predict(X)[0]
    return float(pred)

# Try to recover the model's expected feature names (works if trained on a DataFrame or in a Pipeline)
def get_feature_names(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "steps"):  # sklearn Pipeline
        # walk from the last step backwards and return the first that exposes names
        for _, step in reversed(est.steps):
            names = get_feature_names(step)
            if names:
                return names
    return None

FEATURE_NAMES = get_feature_names(model)

with st.expander("ℹ️ Model input shape", expanded=False):
    st.write("Model expects n_features_in_:", getattr(model, "n_features_in_", "unknown"))
    st.write("Model feature_names_in_ (if available):", FEATURE_NAMES if FEATURE_NAMES else "(not provided)")

# =========================
# UI — Inputs
# =========================
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=15, max_value=40, value=20)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
    midterm = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=70.0)
    final = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=70.0)
    assignments = st.number_input("Assignments Avg", min_value=0.0, max_value=100.0, value=70.0)
    quizzes = st.number_input("Quizzes Avg", min_value=0.0, max_value=100.0, value=70.0)
with col2:
    participation = st.number_input("Participation Score", min_value=0.0, max_value=100.0, value=70.0)
    projects = st.number_input("Projects Score", min_value=0.0, max_value=100.0, value=70.0)
    total_score = st.number_input("Total Score", min_value=0.0, max_value=100.0, value=70.0)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=100.0, value=15.0)
    stress = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    sleep = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0)

gender = st.selectbox("Gender", ["Male", "Female"])
dept = st.selectbox("Department", ["CS", "Engineering", "Mathematics", "Business"])
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
parent_edu = st.selectbox("Parent_Education_Level", ["None", "High School", "Bachelor's", "Master's", "PhD"])
income = st.selectbox("Family_Income_Level", ["Low", "Medium", "High"])

# =========================
# Build feature dictionary (superset of possible dummies)
# =========================
features = {
    # numeric
    "age": age,
    "attendance": attendance,
    "midterm": midterm,
    "final": final,
    "assignments": assignments,
    "quizzes": quizzes,
    "participation": participation,
    "projects": projects,
    "total_score": total_score,
    "study_hours": study_hours,
    "stress": stress,
    "sleep": sleep,

    # gender (include both in case model used both or only one)
    "gender_male": 1 if gender == "Male" else 0,
    "gender_female": 1 if gender == "Female" else 0,

    # department (include all four; many models use 3 with Business as baseline)
    "dept_cs": 1 if dept == "CS" else 0,
    "dept_eng": 1 if dept == "Engineering" else 0,
    "dept_math": 1 if dept == "Mathematics" else 0,
    "dept_business": 1 if dept == "Business" else 0,

    # other binaries
    "extra_yes": 1 if extra == "Yes" else 0,
    "internet_yes": 1 if internet == "Yes" else 0,

    # parent education (include 'None' and the rest; Bachelor's often baseline)
    "parent_none": 1 if parent_edu == "None" else 0,
    "parent_hs": 1 if parent_edu == "High School" else 0,
    "parent_master": 1 if parent_edu == "Master's" else 0,
    "parent_phd": 1 if parent_edu == "PhD" else 0,

    # income (include all; some models use Low/Medium with High as baseline)
    "income_low": 1 if income == "Low" else 0,
    "income_medium": 1 if income == "Medium" else 0,
    "income_high": 1 if income == "High" else 0,
}

# =========================
# Align to model's expected input
# =========================
if FEATURE_NAMES:
    # exact name/order alignment: fill any missing feature with 0.0
    row = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
else:
    # fallback to a fixed order — MUST match training order exactly
    # This vector totals 24 features (12 numeric + 12 dummies)
    gender_male = features["gender_male"]
    dept_cs = features["dept_cs"]; dept_eng = features["dept_eng"]; dept_math = features["dept_math"]
    extra_yes = features["extra_yes"]; internet_yes = features["internet_yes"]
    parent_hs = features["parent_hs"]; parent_master = features["parent_master"]; parent_phd = features["parent_phd"]; parent_none = features["parent_none"]
    income_low = features["income_low"]; income_medium = features["income_medium"]
    X = np.array([[
        features["age"], features["attendance"], features["midterm"], features["final"],
        features["assignments"], features["quizzes"], features["participation"],
        features["projects"], features["total_score"], features["study_hours"],
        features["stress"], features["sleep"], gender_male,
        dept_cs, dept_eng, dept_math,           # Business treated as baseline here
        extra_yes, internet_yes,
        parent_hs, parent_master, parent_phd, parent_none,
        income_low, income_medium               # High treated as baseline here
    ]], dtype=float)

with st.expander("🧮 Input vector check"):
    try:
        shape = X.shape if hasattr(X, "shape") else "(unknown)"
        st.write("X shape:", shape, " | model expects:", getattr(model, "n_features_in_", "unknown"))
        if isinstance(X, pd.DataFrame):
            st.write("Columns passed to model:", list(X.columns))
    except Exception as e:
        st.write("Could not display X shape/columns:", e)

# =========================
# Predict
# =========================
if st.button("Predict Dropout Risk"):
    try:
        y_pred = int(model.predict(X)[0])
        prob_pos = get_positive_proba(model, X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if y_pred == 1:
        st.error(f"⚠️ Student is AT RISK of dropping out (Risk Probability: {prob_pos:.2%})")
    else:
        st.success(f"✅ Student is NOT at high risk (Risk Probability: {prob_pos:.2%})")
