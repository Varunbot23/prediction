# app.py — Student Dropout Risk Prediction (Random Forest) + Interactive UI + Review

from pathlib import Path
import os, re, copy

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
st.title("🎓 Student Dropout Risk Prediction (Logstic Regression)")
st.write("Provide the student's academic, behavioral, and demographic data to predict dropout risk.")

# =========================
# Model loading + debug
# =========================
HERE = Path(__file__).parent
MODEL_PATH = HERE / "logistic_regression_pipeline.pkl"

with st.expander("🔎 Debug info (hide before sharing)"):
    st.write("Working dir:", os.getcwd())
    st.write("App folder:", str(HERE))
    try:
        st.write("Files in app folder:", os.listdir(HERE))
    except Exception as e:
        st.write("Could not list files:", e)
    try:
        import sklearn, numpy
        st.write("Versions →", {"sklearn": sklearn.__version__, "numpy": numpy.__version__})
    except Exception:
        pass

if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    if LOADER == "joblib":
        model = joblib.load(MODEL_PATH)
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH} using {LOADER}. Details: {e}")
    st.stop()

# =========================
# Helpers
# =========================
def get_positive_proba(estimator, X):
    """Return positive-class probability robustly."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
    if hasattr(estimator, "decision_function"):
        val = float(estimator.decision_function(X)[0])
        return 1.0 / (1.0 + np.exp(-val))
    return float(estimator.predict(X)[0])

def get_feature_names(est):
    """Best-effort extraction of training column names."""
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "steps"):  # sklearn Pipeline
        for _, step in reversed(est.steps):
            names = get_feature_names(step)
            if names:
                return names
    return None

FEATURE_NAMES = get_feature_names(model)

with st.expander("ℹ️ Model input shape"):
    st.write("Model expects n_features_in_:", getattr(model, "n_features_in_", "unknown"))
    st.write("Model feature_names_in_ (if available):", FEATURE_NAMES if FEATURE_NAMES else "(not provided)")

# ---------- UI helpers ----------
def risk_level(prob):
    if prob >= 0.70: return "High", "🔴"
    if prob >= 0.40: return "Moderate", "🟠"
    return "Low", "🟢"

def show_risk_gauge(prob):
    lvl, dot = risk_level(prob)
    st.metric("Dropout Risk", f"{prob:.1%}", help=f"Level: {lvl}")
    st.progress(min(max(prob, 0.0), 1.0))

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

# ====== NEW: personalised review / suggestions ======
def build_recommendations(fdict: dict):
    """
    Return a list of (priority, title, tip) tuples based on input features.
    priority: 3 = critical, 2 = important, 1 = nice-to-have
    """
    recs = []

    # Helpers
    def add(priority, title, tip):
        recs.append((priority, title, tip))

    # Heuristic thresholds (adjust to your dataset norms if needed)
    attendance = float(fdict.get("attendance", 0))
    study_hours = float(fdict.get("study_hours", 0))
    total_score = float(fdict.get("total_score", 0))
    assignments = float(fdict.get("assignments", 0))
    quizzes = float(fdict.get("quizzes", 0))
    participation = float(fdict.get("participation", 0))
    projects = float(fdict.get("projects", 0))
    midterm = float(fdict.get("midterm", 0))
    final = float(fdict.get("final", 0))
    stress = float(fdict.get("stress", 5))
    sleep = float(fdict.get("sleep", 7))
    internet_yes = int(fdict.get("internet_yes", 1))
    extra_yes = int(fdict.get("extra_yes", 0))

    # Core academic levers
    if attendance < 75:
        add(3, "Attendance below 75%",
            "Target at least 85%. Block class times in a calendar and set reminders. Coordinate with your tutor for catch-up plans.")
    elif attendance < 85:
        add(2, "Attendance could be higher",
            "Aim for 85–90%+ to stabilise performance. Identify sessions you usually miss and remove obstacles.")

    if total_score < 60:
        add(3, "Low overall performance",
            "Focus on fundamentals first. Revisit lecture notes and attempt past questions. Book office hours weekly.")
    elif total_score < 75:
        add(2, "Improve overall score",
            "Add 30–60 mins of daily revision; use spaced repetition for key topics.")

    # Component scores
    if quizzes < 60: add(2, "Quizzes under 60", "Do short daily quizzes; track errors and review weak topics.")
    if assignments < 60: add(2, "Assignments under 60", "Start earlier; split into milestones; use rubric as a checklist.")
    if participation < 60: add(1, "Low participation", "Ask one question per class; summarise a concept to a peer.")
    if projects < 60: add(1, "Project score low", "Create a mini-plan with deliverables; pair up for peer review.")
    if midterm < 60: add(2, "Midterm needs work", "Rebuild a formula sheet / concept map; practise timed sections.")
    if final < 60: add(2, "Final score weak", "Simulate exam conditions weekly; focus on high-weight topics.")

    # Study habits
    if study_hours < 8:
        add(3, "Very low study time",
            "Increase to ~12–15 hrs/week. Use Pomodoro (25/5). Protect a fixed daily slot.")
    elif study_hours < 12:
        add(2, "Study time can increase",
            "Add +3–5 hrs/week. Replace passive reading with active recall (flashcards).")

    # Wellbeing
    if stress >= 8:
        add(3, "Stress very high",
            "Use time-blocking, weekly planning, and 10-minute wind-down before sleep. Consider speaking to counselling services.")
    elif stress >= 6:
        add(2, "Moderate stress",
            "Add two 10-minute breaks per study hour and light exercise 3x/week.")

    if sleep < 6:
        add(2, "Insufficient sleep",
            "Aim for 7–9 hours. Keep consistent sleep/wake time; avoid screens 1 hour before bed.")
    elif sleep > 10:
        add(1, "Oversleeping",
            "Align to 7–9 hours; investigate daytime fatigue causes.")

    # Access & balance
    if internet_yes == 0:
        add(2, "Unreliable home internet",
            "Download materials offline; plan campus/library sessions for assessments.")
    if extra_yes == 1 and (total_score < 70 or attendance < 85):
        add(1, "Balance extracurriculars",
            "Reduce load during assessment weeks; prioritise core modules.")

    # Derived metric: study efficiency
    if study_hours > 0:
        efficiency = total_score / max(study_hours, 1e-6)
        if efficiency < 4.0:  # arbitrary reference: <4 pts per hour
            add(2, "Low study efficiency",
                "Switch to active techniques: past papers, self-explanations, teaching a peer. Review after-action notes.")
    return sorted(recs, key=lambda x: -x[0])

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

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    auto_predict = st.toggle("🔄 Auto-predict", value=True, help="Recompute as you change inputs")

# =========================
# Build a superset of possible dummies
# =========================
features = {
    # numeric
    "age": age, "attendance": attendance, "midterm": midterm, "final": final,
    "assignments": assignments, "quizzes": quizzes, "participation": participation,
    "projects": projects, "total_score": total_score, "study_hours": study_hours,
    "stress": stress, "sleep": sleep,

    # gender (include both)
    "gender_male": 1 if gender == "Male" else 0,
    "gender_female": 1 if gender == "Female" else 0,

    # department (include all four)
    "dept_cs": 1 if dept == "CS" else 0,
    "dept_eng": 1 if dept == "Engineering" else 0,
    "dept_math": 1 if dept == "Mathematics" else 0,
    "dept_business": 1 if dept == "Business" else 0,

    # other binaries
    "extra_yes": 1 if extra == "Yes" else 0,
    "internet_yes": 1 if internet == "Yes" else 0,

    # parent education (include Bachelor's explicitly + aliases)
    "parent_none": 1 if parent_edu == "None" else 0,
    "parent_hs": 1 if parent_edu == "High School" else 0,
    "parent_bachelor": 1 if parent_edu == "Bachelor's" else 0,
    "parent_bachelors": 1 if parent_edu == "Bachelor's" else 0,  # alias
    "parent_master": 1 if parent_edu == "Master's" else 0,
    "parent_phd": 1 if parent_edu == "PhD" else 0,

    # income (include all three)
    "income_low": 1 if income == "Low" else 0,
    "income_medium": 1 if income == "Medium" else 0,
    "income_high": 1 if income == "High" else 0,
}

# =========================
# Align UI features to model's expected names
# =========================
# Normalized key -> value
norm_map = {norm(k): float(v) for k, v in features.items()}

# Common aliases: normalized model names -> our internal normalized key names
alias = {
    # numeric aliases
    "midtermscore": "midterm", "finalscore": "final",
    "assignmentsavg": "assignments", "quizzesavg": "quizzes",
    "participationscore": "participation", "projectsscore": "projects",
    "studyhoursperweek": "studyhours", "stresslevel": "stress",
    "sleephourspernight": "sleep",

    # gender
    "gendermale": "gendermale", "genderfemale": "genderfemale",

    # department
    "departmentcs": "deptcs", "departmentengineering": "depteng",
    "departmentmathematics": "deptmath", "departmentbusiness": "deptbusiness",

    # booleans
    "extracurricularactivitiesyes": "extrayes",
    "internetaccessathomeyes": "internetyes",

    # parent ed
    "parenteducationlevelnone": "parentnone",
    "parenteducationlevelhighschool": "parenths",
    "parenteducationlevelbachelors": "parentbachelor",
    "parenteducationlevelmasters": "parentmaster",
    "parenteducationlevelphd": "parentphd",

    # income
    "familyincomelevellow": "incomelow",
    "familyincomelevelmedium": "incomemedium",
    "familyincomelevelhigh": "incomehigh",
}

def value_for(model_name: str) -> float:
    n = norm(model_name)
    if n in norm_map:  # direct hit
        return norm_map[n]
    if n in alias and alias[n] in norm_map:  # alias hit
        return norm_map[alias[n]]
    # heuristics for small diffs
    if n.endswith("bachelors") and "parentbachelor" in norm_map:
        return norm_map["parentbachelor"]
    if n.endswith("bachelor") and "parentbachelors" in norm_map:
        return norm_map["parentbachelors"]
    if n.startswith("department") and n.replace("department", "dept") in norm_map:
        return norm_map[n.replace("department", "dept")]
    return 0.0

# Build X
FEATURE_NAMES = get_feature_names(model)  # refresh in case of pipeline
if FEATURE_NAMES:
    row = {name: value_for(name) for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
else:
    # Fallback to a fixed 24-feature order (must match training if names aren't available)
    X = np.array([[
        features["age"], features["attendance"], features["midterm"], features["final"],
        features["assignments"], features["quizzes"], features["participation"],
        features["projects"], features["total_score"], features["study_hours"],
        features["stress"], features["sleep"], features["gender_male"],
        features["dept_cs"], features["dept_eng"], features["dept_math"],  # Business = baseline
        features["extra_yes"], features["internet_yes"],
        features["parent_hs"], features["parent_master"], features["parent_phd"], features["parent_none"],
        features["income_low"], features["income_medium"]                  # High = baseline
    ]], dtype=float)

with st.expander("🧮 Input vector check"):
    try:
        st.write("X shape:", getattr(X, "shape", "(unknown)"),
                 "| model expects:", getattr(model, "n_features_in_", "unknown"))
        if isinstance(X, pd.DataFrame):
            non_zero = {k: float(v) for k, v in X.iloc[0].to_dict().items() if float(v) != 0.0}
            st.write("Non-zero features passed:", non_zero)
    except Exception as e:
        st.write("Could not display X:", e)

# =========================
# Predict (live or on click) + REVIEW
# =========================
def predict_once():
    try:
        y_pred = int(model.predict(X)[0])
        prob = get_positive_proba(model, X)
        return y_pred, prob
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

def render_review(prob, fdict):
    st.subheader("📝 Personalised Review & Suggestions")
    lvl, dot = risk_level(prob)
    st.caption(f"Current risk level: {dot} **{lvl}** ({prob:.1%})")
    recs = build_recommendations(fdict)
    if not recs:
        st.success("Looks solid — keep your current habits and monitor progress weekly.")
        return
    # Show top 5 by priority
    priority_badge = {3: "🔴 Critical", 2: "🟠 Important", 1: "🟢 Nice-to-have"}
    for i, (p, title, tip) in enumerate(recs[:5], start=1):
        st.markdown(f"**{i}. {priority_badge[p]} — {title}**  \n{tip}")

if auto_predict:
    y_pred, prob_pos = predict_once()
    if prob_pos is not None:
        lvl, dot = risk_level(prob_pos)
        st.subheader(f"{dot} {lvl} risk")
        show_risk_gauge(prob_pos)
        render_review(prob_pos, features)
else:
    if st.button("Predict Dropout Risk", type="primary"):
        y_pred, prob_pos = predict_once()
        if prob_pos is not None:
            lvl, dot = risk_level(prob_pos)
            st.subheader(f"{dot} {lvl} risk")
            show_risk_gauge(prob_pos)
            render_review(prob_pos, features)

# =========================
# Scenario Explorer (what-if)
# =========================
with st.expander("🧪 Scenario Explorer (what-if)"):
    sweepable = {
        "attendance": (0.0, 100.0, attendance),
        "study_hours": (0.0, 60.0, study_hours),
        "total_score": (0.0, 100.0, total_score),
        "stress": (1.0, 10.0, float(stress)),
        "sleep": (0.0, 12.0, sleep),
    }
    feat_name = st.selectbox("Feature to vary", list(sweepable.keys()))
    lo, hi, current = sweepable[feat_name]
    rng = st.slider("Range", float(lo), float(hi), (max(lo, current - 20), min(hi, current + 20)))
    steps = st.number_input("Steps", 5, 50, 20)

    vals = np.linspace(rng[0], rng[1], int(steps))
    probs = []

    if hasattr(X, "copy"):
        base = X.copy()
        for v in vals:
            Xv = base.copy()
            # If DF has the column, set directly; otherwise try loose match
            if feat_name in Xv.columns:
                Xv.loc[:, feat_name] = v
            else:
                target = feat_name.replace("_", "").lower()
                for c in Xv.columns:
                    if target in c.replace("_", "").lower():
                        Xv.loc[:, c] = v
            probs.append(get_positive_proba(model, Xv))
        # Plot
        import altair as alt
        chart = alt.Chart(pd.DataFrame({"value": vals, "risk": probs})).mark_line().encode(
            x="value:Q", y=alt.Y("risk:Q", axis=alt.Axis(format="%")),
            tooltip=["value", alt.Tooltip("risk:Q", format=".1%")]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Scenario Explorer needs column names (DataFrame). It’s unavailable for ndarray inputs.")

# =========================
# Batch predictions from CSV (optional)
# =========================
with st.expander("📥 Batch predictions from CSV"):
    uploaded = st.file_uploader("Upload CSV with columns matching model features", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        try:
            if FEATURE_NAMES:
                # Build an aligned frame with zeros then fill from df where possible
                aligned = pd.DataFrame(0.0, index=df.index, columns=FEATURE_NAMES)
                # Direct matches
                for col in FEATURE_NAMES:
                    if col in df.columns:
                        aligned[col] = df[col]
                # Normalised mapping
                incoming = {norm(c): c for c in df.columns}
                for col in FEATURE_NAMES:
                    if aligned[col].eq(0.0).all() and col not in df.columns:
                        n = norm(col)
                        if n in incoming:
                            aligned[col] = df[incoming[n]]
                Xb = aligned
            else:
                Xb = df  # hope order matches if names were not preserved

            # Predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xb)[:, 1]
            else:
                probs = model.predict(Xb).astype(float)

            out = df.copy()
            out["dropout_risk"] = probs
            st.dataframe(out.head(50))
            st.download_button("Download predictions CSV",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


