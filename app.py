import streamlit as st
import numpy as np
import pickle
from pathlib import Path
import streamlit as st
import joblib, os

HERE = Path(app.py).parent
MODEL_PATH = HERE / "random_forest_model.pkl"

# Debug panel in the app (remove later if you like)
with st.expander("Debug info"):
    st.write("Working dir:", os.getcwd())
    st.write("App folder:", str(HERE))
    st.write("Files in app folder:", os.listdir(HERE))

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = joblib.load()

# ===== Load trained model =====
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== Page Title =====
st.title("🎓 Student Dropout Risk Prediction (Random Forest)")

st.write("""
Provide the student's academic, behavioral, and demographic data to predict dropout risk.
""")

# ===== User Inputs =====
age = st.number_input("Age", min_value=15, max_value=40, value=20)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
midterm = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=70.0)
final = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=70.0)
assignments = st.number_input("Assignments Avg", min_value=0.0, max_value=100.0, value=70.0)
quizzes = st.number_input("Quizzes Avg", min_value=0.0, max_value=100.0, value=70.0)
participation = st.number_input("Participation Score", min_value=0.0, max_value=100.0, value=70.0)
projects = st.number_input("Projects Score", min_value=0.0, max_value=100.0, value=70.0)
total_score = st.number_input("Total Score", min_value=0.0, max_value=100.0, value=70.0)
study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=100.0, value=15.0)
stress = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5)
sleep = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0)

# One-hot categorical variables
gender_male = st.selectbox("Gender", ["Male", "Female"])
gender_male = 1 if gender_male == "Male" else 0

dept = st.selectbox("Department", ["CS", "Engineering", "Mathematics", "Business"])
dept_cs = 1 if dept == "CS" else 0
dept_eng = 1 if dept == "Engineering" else 0
dept_math = 1 if dept == "Mathematics" else 0

extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
extra_yes = 1 if extra == "Yes" else 0

internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
internet_yes = 1 if internet == "Yes" else 0

parent_edu = st.selectbox("Parent Education Level", ["None", "High School", "Bachelor's", "Master's", "PhD"])
parent_hs = 1 if parent_edu == "High School" else 0
parent_master = 1 if parent_edu == "Master's" else 0
parent_phd = 1 if parent_edu == "PhD" else 0
parent_none = 1 if parent_edu == "None" else 0

income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
income_low = 1 if income == "Low" else 0
income_medium = 1 if income == "Medium" else 0

# ===== Arrange Inputs in EXACT same order as training =====
input_features = np.array([[
    age, attendance, midterm, final, assignments, quizzes, participation,
    projects, total_score, study_hours, stress, sleep, gender_male,
    dept_cs, dept_eng, dept_math, extra_yes, internet_yes,
    parent_hs, parent_master, parent_phd, income_low, income_medium
]])

# ===== Prediction =====
if st.button("Predict Dropout Risk"):
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Student is AT RISK of dropping out (Risk Probability: {probability:.2%})")
    else:
        st.success(f"✅ Student is NOT at high risk (Risk Probability: {probability:.2%})")

