import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from src.preprocessing import build_preprocessor

st.set_page_config(page_title="AI Application for Score Prediction", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-size: 26px;
        font-weight: 800;
        color: #1e3a5f;
        margin: 0;
    }
    
    /* Big Score Card */
    .gpa-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 24px 30px;
        border: 1px solid #cbd5e1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .gpa-label {
        font-size: 18px;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 6px;
    }
    
    .gpa-number {
        font-size: 56px;
        font-weight: 900;
        color: #b45309;
        line-height: 1;
    }
    
    .gpa-sub-scores {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        font-size: 14px;
        color: #475569;
    }
    
    /* Slider Parameter Cards */
    .param-card-header {
        background: #33658a;
        color: white;
        padding: 10px 14px;
        font-weight: 600;
        font-size: 13.5px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        text-align: center;
    }
    
    .param-card-body {
        background: white;
        border: 1px solid #e2e8f0;
        border-top: none;
        padding: 14px;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        text-align: center;
    }
    
    .param-val {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    
    .limit-badge {
        font-size: 11px;
        color: #64748b;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar
c_title, c_share = st.columns([3, 1])
with c_title:
    st.markdown('<h1 class="main-title">AI Application<sup>🎯</sup> for Score Prediction</h1>', unsafe_allow_html=True)
with c_share:
    st.markdown("""
    <div style="text-align: right; padding-top: 5px;">
        <span style="background: #881337; color: white; padding: 8px 18px; border-radius: 8px; font-weight: 600; font-size: 13px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
            🔗 Share Model
        </span>
    </div>
    """, unsafe_allow_html=True)

# Model Selection Radio Pills matching image 4
st.write("##### Select Architecture:")
chosen_model_type = st.radio(
    "Algorithm Selection",
    ["Linear Regression", "Random Forest", "Gradient Boosting", "Decision Tree"],
    horizontal=True,
    label_visibility="collapsed"
)

# Load data to build or cache model pipeline for chosen model
@st.cache_resource
def get_pipelines():
    df = pd.read_csv("data/StudentsPerformance.csv")
    X = df.drop(columns=["math score", "reading score", "writing score"])
    
    model_objs = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    
    pipes = {}
    for name, m_obj in model_objs.items():
        pipes[name] = {
            "math": Pipeline([("prep", build_preprocessor()), ("reg", m_obj)]).fit(X, df["math score"]),
            "reading": Pipeline([("prep", build_preprocessor()), ("reg", m_obj)]).fit(X, df["reading score"]),
            "writing": Pipeline([("prep", build_preprocessor()), ("reg", m_obj)]).fit(X, df["writing score"])
        }
    return pipes

model_pipes = get_pipelines()
active_pipes = model_pipes[chosen_model_type]

# Session state for Run History
if "history" not in st.session_state:
    st.session_state.history = []

# Parameter mappings
edu_options = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
ethnicity_options = ["group A", "group B", "group C", "group D", "group E"]

st.write("##### Student Profile Parameters:")

p1, p2, p3, p4, p5 = st.columns(5)

with p1:
    st.markdown('<div class="param-card-header">Parental Education</div>', unsafe_allow_html=True)
    edu_idx = st.slider("Parental Education Level", 0, 5, 3, label_visibility="collapsed")
    parent_edu = edu_options[edu_idx]
    st.markdown(f'<div class="param-card-body"><div class="param-val">{parent_edu.title()}</div><div class="limit-badge">Min: 0 &nbsp;|&nbsp; Max: 5</div></div>', unsafe_allow_html=True)

with p2:
    st.markdown('<div class="param-card-header">Test Preparation</div>', unsafe_allow_html=True)
    prep_val = st.slider("Test Prep", 0, 1, 1, label_visibility="collapsed")
    test_prep = "completed" if prep_val == 1 else "none"
    st.markdown(f'<div class="param-card-body"><div class="param-val">{test_prep.capitalize()}</div><div class="limit-badge">0: None &nbsp;|&nbsp; 1: Completed</div></div>', unsafe_allow_html=True)

with p3:
    st.markdown('<div class="param-card-header">Lunch Type</div>', unsafe_allow_html=True)
    lunch_val = st.slider("Lunch", 0, 1, 0, label_visibility="collapsed")
    lunch = "standard" if lunch_val == 0 else "free/reduced"
    st.markdown(f'<div class="param-card-body"><div class="param-val">{lunch.capitalize()}</div><div class="limit-badge">0: Standard &nbsp;|&nbsp; 1: Reduced</div></div>', unsafe_allow_html=True)

with p4:
    st.markdown('<div class="param-card-header">Gender</div>', unsafe_allow_html=True)
    gender_val = st.slider("Gender", 0, 1, 0, label_visibility="collapsed")
    gender = "female" if gender_val == 0 else "male"
    st.markdown(f'<div class="param-card-body"><div class="param-val">{gender.capitalize()}</div><div class="limit-badge">0: Female &nbsp;|&nbsp; 1: Male</div></div>', unsafe_allow_html=True)

with p5:
    st.markdown('<div class="param-card-header">Ethnic Cohort</div>', unsafe_allow_html=True)
    eth_idx = st.slider("Ethnic Group", 0, 4, 2, label_visibility="collapsed")
    race = ethnicity_options[eth_idx]
    st.markdown(f'<div class="param-card-body"><div class="param-val">{race.upper()}</div><div class="limit-badge">Min: A &nbsp;|&nbsp; Max: E</div></div>', unsafe_allow_html=True)

# Generate Prediction
input_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "parental level of education": [parent_edu],
    "lunch": [lunch],
    "test preparation course": [test_prep]
})

math_p = float(np.clip(active_pipes["math"].predict(input_df)[0], 0, 100))
read_p = float(np.clip(active_pipes["reading"].predict(input_df)[0], 0, 100))
write_p = float(np.clip(active_pipes["writing"].predict(input_df)[0], 0, 100))
overall_p = (math_p + read_p + write_p) / 3

# Update History
run_entry = {
    "Model": chosen_model_type,
    "Parental Edu": parent_edu,
    "Test Prep": test_prep,
    "Math": round(math_p, 1),
    "Reading": round(read_p, 1),
    "Writing": round(write_p, 1),
    "Overall": round(overall_p, 1)
}

st.markdown("<br>", unsafe_allow_html=True)

# --- SCORE DISPLAY CARD WITH BUTTONS (MATCHING IMAGE 4) ---
col_score_display, col_actions = st.columns([2.5, 1])

with col_score_display:
    st.markdown(f"""
    <div class="gpa-card">
        <div>
            <div class="gpa-label">PREDICTED COMPOSITE SCORE</div>
            <div class="gpa-number">{overall_p:.1f}</div>
            <div class="gpa-sub-scores">
                <span>📐 <strong>Math:</strong> {math_p:.1f}</span>
                <span>📖 <strong>Reading:</strong> {read_p:.1f}</span>
                <span>✍️ <strong>Writing:</strong> {write_p:.1f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_actions:
    st.write("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    show_analysis = st.button("📊 Analysis", use_container_width=True)
    show_history = st.button("⏱️ Run History", use_container_width=True)
    if st.button("💾 Save Simulation", use_container_width=True):
        st.session_state.history.append(run_entry)
        st.success("Simulation saved to history!")

if show_analysis:
    st.markdown("### 🔍 Model Diagnostic Breakdown")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Mathematics", f"{math_p:.1f} / 100", f"{math_p - 66.1:+.1f} vs Cohort Mean")
    with a2:
        st.metric("Reading", f"{read_p:.1f} / 100", f"{read_p - 69.2:+.1f} vs Cohort Mean")
    with a3:
        st.metric("Writing", f"{write_p:.1f} / 100", f"{write_p - 68.1:+.1f} vs Cohort Mean")
        
    if test_prep == "none":
        st.warning("⚠️ **Academic Intervention:** Completing the test preparation course is projected to increase scores by +6 to +8 points on average.")
    else:
        st.info("✅ **Positive Indicator:** Active test preparation course participation provides a strong boost across all subject areas.")

if show_history:
    st.markdown("### 📋 Simulation Run History")
    if len(st.session_state.history) > 0:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.session_state.history.append(run_entry)
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
