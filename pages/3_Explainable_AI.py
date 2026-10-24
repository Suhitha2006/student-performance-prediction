import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Explainable AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px;
    }
    .xai-panel {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
        height: 100%;
    }
    .panel-header-title {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Explainable AI: Feature Importance & Interpretability")
st.caption("Understanding the influence of demographics and background variables on student outcomes.")

models_dir = Path("models")
if not (models_dir / "math_model.joblib").exists():
    st.error("Models not found. Train models first using `python -m src.train_models`.")
    st.stop()

@st.cache_data
def get_feature_importances():
    df = pd.read_csv("data/StudentsPerformance.csv")
    X = df.drop(columns=["math score", "reading score", "writing score"])
    
    math_pipe = joblib.load(models_dir / "math_model.joblib")
    reading_pipe = joblib.load(models_dir / "reading_model.joblib")
    writing_pipe = joblib.load(models_dir / "writing_model.joblib")
    
    pipes = {
        "Mathematics": (math_pipe, df["math score"]),
        "Reading Comprehension": (reading_pipe, df["reading score"]),
        "Writing & Language": (writing_pipe, df["writing score"])
    }
    
    importance_dict = {}
    for sub, (pipe, y) in pipes.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        res = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=42, scoring="r2")
        importance_dict[sub] = pd.DataFrame({
            "Feature": [c.title() for c in X.columns],
            "Importance": np.maximum(0, res.importances_mean),
            "Std": res.importances_std
        }).sort_values(by="Importance", ascending=True)
        
    return importance_dict

importances = get_feature_importances()

subject_sel = st.radio(
    "Select Subject to Inspect Feature Importance:",
    ["Mathematics", "Reading Comprehension", "Writing & Language"],
    horizontal=True
)

imp_df = importances[subject_sel]

col_chart, col_insights = st.columns([1.5, 1.1], gap="large")

with col_chart:
    st.markdown(f'<div class="xai-panel"><div class="panel-header-title">Permutation Feature Importance — {subject_sel}</div>', unsafe_allow_html=True)
    
    # Modern Horizontal Bar Chart with ample margins and no overlapping text
    fig = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation='h',
        text=[f"  {val:.3f}" for val in imp_df["Importance"]],
        textposition='outside',
        textfont=dict(family='Plus Jakarta Sans', size=12, color='#1e293b'),
        marker=dict(
            color=imp_df["Importance"],
            colorscale='Blues',
            line=dict(color='#1e40af', width=1)
        )
    ))
    
    max_val = max(imp_df["Importance"].max() * 1.35, 0.25)
    
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=380,
        margin=dict(l=160, r=60, t=20, b=40),
        xaxis=dict(
            title=dict(text="<b>Importance (R² drop when shuffled)</b>", font=dict(family='Plus Jakarta Sans', size=12)),
            range=[0, max_val],
            gridcolor='#f1f5f9'
        ),
        yaxis=dict(
            tickfont=dict(family='Plus Jakarta Sans', size=13, color='#0f172a'),
            automargin=True
        ),
        font=dict(family="Plus Jakarta Sans", size=12)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption("ℹ️ Higher values mean that randomly shuffling this variable leads to a larger drop in prediction accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_insights:
    st.markdown('<div class="xai-panel">', unsafe_allow_html=True)
    st.markdown("### 💡 Interpretability Insights")
    
    top_feature = imp_df.iloc[-1]["Feature"]
    second_feature = imp_df.iloc[-2]["Feature"]
    
    st.markdown(f"""
    1. **Primary Driving Factor:** **{top_feature}** exhibits the highest relative importance for {subject_sel.lower()}.
    2. **Secondary Driver:** **{second_feature}** consistently contributes to variance explanation across cohorts.
    3. **Actionable Variables:** While demographics like race or gender are fixed attributes, **Test Preparation Course** is a fully modifiable intervention that yields tangible score shifts.
    """)
    st.markdown("""
    <div style="background: #fffbeb; border: 1px solid #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 14px; border-radius: 8px; font-size: 12.5px; color: #92400e; margin-top: 16px;">
        ⚠️ <b>Statistical Note:</b> Feature importance measures associative predictability within the model; it does not imply direct mechanistic causation.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
