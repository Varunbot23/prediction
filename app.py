# app.py — Student Dropout Risk Prediction (Random Forest)

from pathlib import Path
import os
import numpy as np
import streamlit as st


# Use joblib if available; otherwise fall back to pickle so the app still starts
try:
    import joblib
    LOADER = "joblib"
except Exception:
    import pickle
    joblib = None
    LOADER = "pickle"

HERE = Path(__file__).parent
MODEL_PATH = HERE / "random_forest_model.pkl"

with st.expander("🔎 Debug info (hide before sharing)"):
    st.write("Working dir:", os.getcwd())
    st.write("App folder:", str(HERE))
    try:
        st.write("Files in app folder:", os.listdir(HERE))
    except Exception as e:
        st.write("Could not list files:", e)

if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

# Load model once
try:
    if LOADER == "joblib":
        model = joblib.load(MODEL_PATH)
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH} using {LOADER}. Details: {e}")
    st.stop()

# Small helper: safe probability extraction even if estimator lacks predict_proba
def get_positive_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        # Assume positive class is column 1 if present
        if proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(proba[0, 0])
    # Fallback: use decision_function and squash roughly to [0,1]
    if hasattr(estimator, "decision_function"):
        val = float(estimator.decision_function(X)[0])
        return 1.0 / (1.0 + np.exp(-val))
    # Last resort: cast prediction to 0/1
    return float(estimator.predict(X)[0])

# =========================
# UI
# =========================
st.set_page_config(page_title="Student Dropout Risk Prediction", page_icon="🎓", layout="centered")
st.title("🎓 Student Dropout Risk Prediction (Random Forest)")
st.write("Provide the student's academic, behavioral, and demographic data to predict dropout risk.")

# ===== Numeric inputs =====
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=15, max_value=40, value=20)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
    midterm = st.number_input("Midterm_Score", min_value=0.0, max_value=100.0, value=70.0)
    final = st.number_input("Final_Score", min_value=0.0, max_value=100.0, value=70.0)
    assignments = st.number_input("Assignments_Avg", min_value=0.0, max_value=100.0, value=70.0)
    quizzes = st.number_input("Quizzes_Avg", min_value=0.0, max_value=100.0, value=70.0)
with col2:
    participation = st.number_input("Participation_Score", min_value=0.0, max_value=100.0, value=70.0)
    projects = st.number_input("Projects_Score", min_value=0.0, max_value=100.0, value=70.0)
    total_score = st.number_input("Total_Score", min_value=0.0, max_value=100.0, value=70.0)
    study_hours = st.number_input("Study_Hours_per_Week", min_value=0.0, max_value=100.0, value=15.0)
    stress = st.number_input("Stress_Level (1-10)", min_value=1, max_value=10, value=5)
    sleep = st.number_input("Sleep_Hours_per_Night", min_value=0.0, max_value=24.0, value=7.0)

# ===== Categoricals (one-hot) — must match training order =====
gender = st.selectbox("Gender", ["Male", "Female"])
gender_male = 1 if gender == "Male" else 0

dept = st.selectbox("Department", ["CS", "Engineering", "Mathematics", "Business"])
dept_cs = 1 if dept == "CS" else 0
dept_eng = 1 if dept == "Engineering" else 0
dept_math = 1 if dept == "Mathematics" else 0
# (Business is the implicit baseline)

extra = st.selectbox("Extracurricular_Activities", ["Yes", "No"])
extra_yes = 1 if extra == "Yes" else 0

internet = st.selectbox("Internet_Access_at_Home", ["Yes", "No"])
internet_yes = 1 if internet == "Yes" else 0

parent_edu = st.selectbox("Parent_Education_Level", ["None", "High School", "Bachelor's", "Master's", "PhD"])
parent_hs = 1 if parent_edu == "High School" else 0
parent_master = 1 if parent_edu == "Master's" else 0
parent_phd = 1 if parent_edu == "PhD" else 0
# (None/Bachelor's are part of the baseline in this vector)

income = st.selectbox("Family_Income_Level", ["Low", "Medium", "High"])
income_low = 1 if income == "Low" else 0
income_medium = 1 if income == "Medium" else 0
# (High is baseline)

# =========================
# Build feature vector
# IMPORTANT: keep EXACT order used in training
# =========================
input_features = np.array([[
    age, attendance, midterm, final, assignments, quizzes, participation,
    projects, total_score, study_hours, stress, sleep, gender_male,
    dept_cs, dept_eng, dept_math, extra_yes, internet_yes,
    parent_hs, parent_master, parent_phd, income_low, income_medium
]], dtype=float)

# =========================
# Predict
# =========================
if st.button("Predict Dropout Risk"):
    try:
        y_pred = int(model.predict(input_features)[0])
        prob_pos = get_positive_proba(model, input_features)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if y_pred == 1:
        st.error(f"⚠️ Student is AT RISK of dropping out "
                 f"(Risk Probability: {prob_pos:.2%})")
    else:
        st.success(f"✅ Student is NOT at high risk "
                   f"(Risk Probability: {prob_pos:.2%})")

    with st.expander("See input features as vector"):
        st.write(input_features.tolist())



