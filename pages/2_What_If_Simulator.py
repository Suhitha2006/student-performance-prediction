import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="What-If Scenario Simulator", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .scenario-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ What-If Scenario Simulator")
st.caption("Compare how changing interventions (like test preparation or education background) directly impacts predicted scores.")

models_dir = Path("models")
try:
    math_model = joblib.load(models_dir / "math_model.joblib")
    reading_model = joblib.load(models_dir / "reading_model.joblib")
    writing_model = joblib.load(models_dir / "writing_model.joblib")
    models = {"math": math_model, "reading": reading_model, "writing": writing_model}
except:
    st.error("Models not found. Train models first using `python -m src.train_models`.")
    st.stop()

def predict_scenario(gender, race, parent_edu, lunch, prep):
    df = pd.DataFrame([{
        "gender": gender, "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch, "test preparation course": prep
    }])
    m = float(np.clip(models["math"].predict(df)[0], 0, 100))
    r = float(np.clip(models["reading"].predict(df)[0], 0, 100))
    w = float(np.clip(models["writing"].predict(df)[0], 0, 100))
    avg = (m + r + w) / 3
    return m, r, w, avg

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
    st.subheader("🔵 Scenario A (Current / Baseline)")
    a_gender = st.selectbox("Gender", ["female", "male"], key="ag")
    a_race = st.selectbox("Ethnic Group", ["group A", "group B", "group C", "group D", "group E"], index=2, key="ar")
    a_edu = st.selectbox("Parental Education", [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ], index=1, key="ae")
    a_lunch = st.selectbox("Lunch Program", ["standard", "free/reduced"], index=0, key="al")
    a_prep = st.selectbox("Test Prep Course", ["none", "completed"], index=0, key="ap")
    
    am, ar, aw, a_avg = predict_scenario(a_gender, a_race, a_edu, a_lunch, a_prep)
    st.metric("Projected Average Score", f"{a_avg:.1f} / 100")
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
    st.subheader("🟢 Scenario B (Modified Intervention)")
    b_gender = st.selectbox("Gender", ["female", "male"], index=["female", "male"].index(a_gender), key="bg")
    b_race = st.selectbox("Ethnic Group", ["group A", "group B", "group C", "group D", "group E"], index=["group A", "group B", "group C", "group D", "group E"].index(a_race), key="br")
    b_edu = st.selectbox("Parental Education", [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ], index=3, key="be")
    b_lunch = st.selectbox("Lunch Program", ["standard", "free/reduced"], index=0, key="bl")
    b_prep = st.selectbox("Test Prep Course", ["none", "completed"], index=1, key="bp")
    
    bm, br, bw, b_avg = predict_scenario(b_gender, b_race, b_edu, b_lunch, b_prep)
    delta_avg = b_avg - a_avg
    st.metric("Projected Average Score", f"{b_avg:.1f} / 100", f"{delta_avg:+.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Comparative Subject Chart
st.markdown("### 📊 Subject-by-Subject Delta Comparison")

categories = ['Mathematics', 'Reading Comprehension', 'Writing & Language', 'Overall Average']
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Scenario A',
    x=categories,
    y=[round(am, 1), round(ar, 1), round(aw, 1), round(a_avg, 1)],
    marker_color='#3b82f6',
    text=[f"{am:.1f}", f"{ar:.1f}", f"{aw:.1f}", f"{a_avg:.1f}"],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Scenario B',
    x=categories,
    y=[round(bm, 1), round(br, 1), round(bw, 1), round(b_avg, 1)],
    marker_color='#10b981',
    text=[f"{bm:.1f}", f"{br:.1f}", f"{bw:.1f}", f"{b_avg:.1f}"],
    textposition='outside'
))

fig.update_layout(
    barmode='group',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    yaxis=dict(range=[0, 105], gridcolor='#e2e8f0', title="Predicted Score"),
    font=dict(family='Inter', size=13),
    height=450
)

st.plotly_chart(fig, use_container_width=True)
