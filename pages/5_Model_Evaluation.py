import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Regression Performance Comparison", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-top: 15px;
    }
    .legend-box {
        display: flex;
        justify-content: center;
        gap: 18px;
        margin-bottom: 12px;
    }
    .legend-tag {
        font-size: 13px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Regression Performance Comparison")
st.markdown("Comprehensive comparative evaluation of predictive regression architectures on the student performance test set.")

models_dir = Path("models")
metrics_file = models_dir / "evaluation_metrics.joblib"

if not metrics_file.exists():
    st.warning("Evaluation metrics not found. Please train models first using `python -m src.train_models`.")
    st.stop()

all_metrics = joblib.load(metrics_file)

# Subject selection tab/pills
subject_choice = st.radio(
    "Select Subject for Performance Analysis:",
    ["Overall / Composite", "Mathematics", "Reading Comprehension", "Writing & Language"],
    horizontal=True
)

subject_key_map = {
    "Overall / Composite": "math",
    "Mathematics": "math",
    "Reading Comprehension": "reading",
    "Writing & Language": "writing"
}

# Aggregate or single subject metrics
models_list = ["Linear Regression", "Random Forest", "Gradient Boosting", "Decision Tree"]

if subject_choice == "Overall / Composite":
    # Average metrics across all 3 subjects
    mae_vals = []
    rmse_vals = []
    r2_vals = []
    for m in models_list:
        mae_avg = np.mean([all_metrics[s]["all_results"][m]["MAE"] for s in ["math", "reading", "writing"]])
        rmse_avg = np.mean([all_metrics[s]["all_results"][m]["RMSE"] for s in ["math", "reading", "writing"]])
        r2_avg = np.mean([all_metrics[s]["all_results"][m]["R2"] for s in ["math", "reading", "writing"]])
        mae_vals.append(round(mae_avg, 2))
        rmse_vals.append(round(rmse_avg, 2))
        r2_vals.append(round(r2_avg, 3))
else:
    s_key = subject_key_map[subject_choice]
    sub_data = all_metrics[s_key]["all_results"]
    mae_vals = [round(sub_data[m]["MAE"], 2) for m in models_list]
    rmse_vals = [round(sub_data[m]["RMSE"], 2) for m in models_list]
    r2_vals = [round(sub_data[m]["R2"], 3) for m in models_list]

# --- RENDER GROUPED BAR CHART MATCHING USER SCREENSHOT ---
st.markdown("""
<div class="legend-box">
    <span class="legend-tag" style="background: #1e70bf; color: white;">MAE ↓ (Mean Absolute Error)</span>
    <span class="legend-tag" style="background: #d85a2b; color: white;">RMSE ↓ (Root Mean Squared Error)</span>
    <span class="legend-tag" style="background: #e5a920; color: white;">R² Score ↑ (Coefficient of Determination)</span>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Bar(
    name='MAE ↓',
    x=models_list,
    y=mae_vals,
    text=[str(v) for v in mae_vals],
    textposition='outside',
    marker_color='#1e70bf',
    marker_line_color='#0f4c81',
    marker_line_width=1.5
))

fig.add_trace(go.Bar(
    name='RMSE ↓',
    x=models_list,
    y=rmse_vals,
    text=[str(v) for v in rmse_vals],
    textposition='outside',
    marker_color='#d85a2b',
    marker_line_color='#9c3614',
    marker_line_width=1.5
))

fig.add_trace(go.Bar(
    name='R² Score ↑',
    x=models_list,
    y=r2_vals,
    text=[str(v) for v in r2_vals],
    textposition='outside',
    marker_color='#e5a920',
    marker_line_color='#b88012',
    marker_line_width=1.5
))

max_y = max(max(rmse_vals), 18) * 1.15

fig.update_layout(
    title=dict(
        text=f'<b>Regression Performance Comparison ({subject_choice})</b>',
        x=0.5,
        font=dict(size=18, color='#0f172a', family='Inter')
    ),
    barmode='group',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    yaxis=dict(
        title='<b>Metric Value</b>',
        gridcolor='#e2e8f0',
        range=[0, max_y],
        zeroline=True,
        zerolinecolor='#cbd5e1'
    ),
    xaxis=dict(
        tickfont=dict(size=13, family='Inter', color='#1e293b'),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12, family='Inter')
    ),
    margin=dict(l=40, r=40, t=80, b=40),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Metric Summary Cards
st.subheader("Model Comparison Summary")
c1, c2, c3, c4 = st.columns(4)
for idx, (col, m_name) in enumerate(zip([c1, c2, c3, c4], models_list)):
    with col:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 16px; border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-weight: 700; color: #1e293b; font-size: 15px; margin-bottom: 8px;">{m_name}</div>
            <div style="font-size: 12px; color: #64748b;">MAE: <strong style="color: #1e70bf;">{mae_vals[idx]}</strong></div>
            <div style="font-size: 12px; color: #64748b;">RMSE: <strong style="color: #d85a2b;">{rmse_vals[idx]}</strong></div>
            <div style="font-size: 13px; color: #64748b; margin-top: 4px;">R²: <strong style="color: #e5a920; font-size: 15px;">{r2_vals[idx]}</strong></div>
        </div>
        """, unsafe_allow_html=True)
