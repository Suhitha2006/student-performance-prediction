import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Student Performance Analytics", page_icon="🔍", layout="wide")

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
    .panel-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.05);
        margin-bottom: 24px;
        height: 100%;
    }
    .panel-title {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
    }
    .caption-box {
        font-size: 12px;
        color: #64748b;
        margin-top: 14px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Student Performance Analytics & Exploratory Data Analysis")
st.caption("Interactive behavioral and demographic analytics of assessment outcomes")

dataset_path = Path("data/StudentsPerformance.csv")
if not dataset_path.exists():
    st.error("Dataset not found in data/StudentsPerformance.csv.")
    st.stop()

df = pd.read_csv(dataset_path)
df["overall score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3

# --- TWO COLUMN ANALYTICS (RESPONSIVE & CLEAN) ---
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="panel-card"><div class="panel-title">Factor Analysis vs Exam Score</div>', unsafe_allow_html=True)
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        color_by = st.selectbox(
            "Color by Category:",
            ["gender", "lunch", "test preparation course", "race/ethnicity", "parental level of education"]
        )
    with c_f2:
        subject_y = st.selectbox(
            "Exam Subject:",
            ["overall score", "math score", "reading score", "writing score"]
        )
    
    score_range = st.slider("Filter Score Range:", 0, 100, (20, 100))
    
    # Filter data
    filtered_plot_df = df[(df[subject_y] >= score_range[0]) & (df[subject_y] <= score_range[1])]
    
    plot_data = filtered_plot_df.copy()
    categories = sorted(plot_data[color_by].unique())
    cat_map = {c: i for i, c in enumerate(categories)}
    plot_data["x_cat"] = plot_data[color_by].map(cat_map) + np.random.normal(0, 0.08, len(plot_data))
    
    fig_scatter = px.scatter(
        plot_data,
        x="x_cat",
        y=subject_y,
        color=color_by,
        hover_data=["gender", "race/ethnicity", "lunch", "test preparation course", subject_y],
        color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"],
        height=420
    )
    
    fig_scatter.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(categories))),
            ticktext=[c.title() for c in categories],
            title=dict(text=f"<b>{color_by.title()} Category</b>", font=dict(family="Plus Jakarta Sans", size=12)),
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title=dict(text=f"<b>{subject_y.title()}</b>", font=dict(family="Plus Jakarta Sans", size=12)),
            gridcolor="#f1f5f9",
            range=[max(0, score_range[0] - 5), min(100, score_range[1] + 5)]
        ),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(family="Plus Jakarta Sans", size=11)),
        font=dict(family="Plus Jakarta Sans", size=12),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
    st.markdown('<div class="caption-box">Scatter plot showing the distribution and relationship between student demographic categories and exam scores.</div></div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-card"><div class="panel-title">Demographic & Performance Correlation Heatmap</div>', unsafe_allow_html=True)
    
    df_encoded = df.copy()
    df_encoded["test prep (numeric)"] = (df_encoded["test preparation course"] == "completed").astype(int)
    df_encoded["lunch standard (numeric)"] = (df_encoded["lunch"] == "standard").astype(int)
    df_encoded["gender male (numeric)"] = (df_encoded["gender"] == "male").astype(int)
    
    edu_rank = {
        "some high school": 0, "high school": 1, "some college": 2,
        "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
    }
    df_encoded["parent education level"] = df_encoded["parental level of education"].map(edu_rank)
    
    corr_available_cols = [
        "math score", "reading score", "writing score", "overall score",
        "test prep (numeric)", "lunch standard (numeric)", "parent education level", "gender male (numeric)"
    ]
    
    selected_factors = st.multiselect(
        "Select Factors for Matrix:",
        corr_available_cols,
        default=["math score", "reading score", "writing score", "test prep (numeric)", "lunch standard (numeric)"]
    )
    
    show_corr_values = st.checkbox("Show Correlation Values Inside Cells", value=True)
    
    if len(selected_factors) > 1:
        corr_matrix = df_encoded[selected_factors].corr().round(3)
        clean_names = [c.replace(" (numeric)", "").title() for c in corr_matrix.columns]
        corr_matrix.columns = clean_names
        corr_matrix.index = clean_names
        
        # Robust, high-performance Heatmap with dedicated sizing and no label clipping
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=show_corr_values,
            color_continuous_scale="Blues",
            zmin=-1,
            zmax=1,
            aspect="auto",
            height=420
        )
        
        fig_heat.update_layout(
            font=dict(family="Plus Jakarta Sans", size=11),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=90, r=40, t=30, b=80),
            xaxis=dict(tickangle=-35, tickfont=dict(family="Plus Jakarta Sans", size=11, color="#0f172a")),
            yaxis=dict(tickfont=dict(family="Plus Jakarta Sans", size=11, color="#0f172a"))
        )
        
        st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Select at least 2 factors to view the correlation heatmap.")
        
    st.markdown('<div class="caption-box">Correlation heatmap showing statistical associations between demographic factors and exam subject performance.</div></div>', unsafe_allow_html=True)

# Additional Quick Statistical Insights (Well-spaced cards)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📊 Cohort Distribution Breakdown")

s1, s2, s3 = st.columns(3, gap="medium")

with s1:
    fig_box = px.box(
        df, x="test preparation course", y="overall score", color="test preparation course",
        title="<b>Impact of Test Preparation</b>",
        color_discrete_sequence=["#ef4444", "#10b981"],
        height=380
    )
    fig_box.update_layout(
        font=dict(family="Plus Jakarta Sans", size=11),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=30, r=20, t=50, b=40),
        xaxis=dict(title=dict(text="<b>Test Prep Status</b>")),
        yaxis=dict(title=dict(text="<b>Score</b>"), gridcolor='#f1f5f9'),
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})

with s2:
    fig_lunch = px.box(
        df, x="lunch", y="overall score", color="lunch",
        title="<b>Impact of Lunch Program</b>",
        color_discrete_sequence=["#3b82f6", "#f59e0b"],
        height=380
    )
    fig_lunch.update_layout(
        font=dict(family="Plus Jakarta Sans", size=11),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=30, r=20, t=50, b=40),
        xaxis=dict(title=dict(text="<b>Lunch Program</b>")),
        yaxis=dict(title=dict(text="<b>Score</b>"), gridcolor='#f1f5f9'),
        showlegend=False
    )
    st.plotly_chart(fig_lunch, use_container_width=True, config={'displayModeBar': False})

with s3:
    fig_edu = px.box(
        df, x="parental level of education", y="overall score",
        title="<b>Impact of Parental Education</b>",
        height=380,
        color_discrete_sequence=["#6366f1"]
    )
    fig_edu.update_layout(
        font=dict(family="Plus Jakarta Sans", size=10),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=30, r=20, t=50, b=70),
        xaxis=dict(tickangle=-35, title=dict(text="<b>Parent Education</b>")),
        yaxis=dict(title=dict(text="<b>Score</b>"), gridcolor='#f1f5f9'),
        showlegend=False
    )
    st.plotly_chart(fig_edu, use_container_width=True, config={'displayModeBar': False})
