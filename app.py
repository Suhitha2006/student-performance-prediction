import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
from src.auth import verify_credentials

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StudentPerformance ML • Analytics & Academic Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- AUTH STATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None

# Load external CSS file cleanly
def load_css():
    css_path = Path("styles.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.html(f"<style>{css_content}</style>")

load_css()

# Read 3D book illustration as base64
def get_image_base64():
    p = Path("login_books.jpg")
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

img_b64 = get_image_base64()

# ========================================================
# 1. TWO-TONE SPLIT LOGIN SCREEN
# ========================================================
if not st.session_state.authenticated:
    # Sidebar remains hidden on login screen using st.html
    st.html("<style>[data-testid='stSidebar'], [data-testid='stSidebarNav'], [data-testid='collapsedControl'] { display: none !important; } .main .block-container { max-width: 1040px !important; padding-top: 2rem !important; }</style>")

    c_left, c_right = st.columns([1, 1], gap="small")
    
    with c_left:
        img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" class="bottom-graphic-img" alt="Books and cap">' if img_b64 else '<div style="font-size: 58px; text-align: center;">📚 ✏️ 🎓</div>'
        left_panel_html = (
            '<div class="brand-hero-card">'
            '<div>'
            '<div class="brand-logo-text"><span>🎓</span> StudentPerformance<span class="accent">ML</span></div>'
            '<div class="brand-headline">Predict • Analyze • Support</div>'
            '<div class="brand-subhead">Turning data into better learning outcomes.</div>'
            '<div class="feat-row">'
            '<div class="feat-icon">📈</div>'
            '<div>'
            '<p class="feat-title">Predict Student Performance</p>'
            '<p class="feat-text">Use machine learning to forecast exam scores.</p>'
            '</div>'
            '</div>'
            '<div class="feat-row">'
            '<div class="feat-icon">🎯</div>'
            '<div>'
            '<p class="feat-title">Get Personalized Support</p>'
            '<p class="feat-text">Identify weak areas and receive tailored recommendations.</p>'
            '</div>'
            '</div>'
            '<div class="feat-row">'
            '<div class="feat-icon">📊</div>'
            '<div>'
            '<p class="feat-title">Explore Data & Insights</p>'
            '<p class="feat-text">Visualize trends, correlations and key factors.</p>'
            '</div>'
            '</div>'
            '</div>'
            f'<div>{img_tag}</div>'
            '</div>'
        )
        st.html(left_panel_html)

    with c_right:
        with st.form("signin_form"):
            st.html(
                '<div style="text-align: center; margin-bottom: 8px;">'
                '<span style="font-size: 26px;">🎓</span> '
                '<span style="font-size: 19px; font-weight: 800; color: #0f172a;">StudentPerformance<span style="color: #2563eb;">ML</span></span>'
                '</div>'
                '<h2 class="welcome-header">Welcome Back!</h2>'
                '<p class="welcome-sub">Sign in to access your dashboard</p>'
            )
            email_val = st.text_input("✉️ Email Address", placeholder="e.g. you@domain.com or admin")
            pass_val = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Remember me", value=True)
            submit_btn = st.form_submit_button("Sign In →", use_container_width=True)
            
            if submit_btn:
                user = verify_credentials(email_val, pass_val)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_info = user
                    st.rerun()
                else:
                    st.error("Invalid email/username or password.")

            st.html(
                '<div class="credentials-badge">'
                '<strong>🔑 Quick Demo Credentials:</strong><br>'
                '• <b>Admin:</b> <code>admin@edupredict.org</code> (or <code>admin</code>) &nbsp;|&nbsp; <code>admin123</code><br>'
                '• <b>Faculty:</b> <code>teacher@edupredict.org</code> (or <code>teacher</code>) &nbsp;|&nbsp; <code>teacher123</code><br>'
                '• <b>Student:</b> <code>student@edupredict.org</code> (or <code>student</code>) &nbsp;|&nbsp; <code>student123</code>'
                '</div>'
                '<div style="text-align: center; font-size: 12px; color: #64748b; margin-top: 14px;">'
                'Don\'t have an account? <span style="color: #2563eb; cursor: pointer; font-weight: 600;">Contact your administrator</span>'
                '</div>'
            )

    st.stop()

# ========================================================
# 2. AUTHENTICATED EXECUTIVE DASHBOARD (EXACT MATCH TO 2ND PIC)
# ========================================================
user_info = st.session_state.user_info

with st.sidebar:
    st.html("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
        <span style="font-size: 22px;">🎓</span>
        <span style="font-size: 16px; font-weight: 800; color: #ffffff;">StudentPerformance<span style="color: #38bdf8;">ML</span></span>
    </div>
    <div style="background: rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.12);">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Logged In As</div>
        <div style="font-size: 14px; font-weight: 700; color: #ffffff;">""" + user_info['name'] + """</div>
        <div style="font-size: 12px; color: #38bdf8;">""" + user_info['role'] + """</div>
    </div>
    """)
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.rerun()

# Load dataset
@st.cache_data
def load_dataset():
    data_path = Path("data/StudentsPerformance.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_dataset()
if df is None:
    st.error("Dataset not found in `data/StudentsPerformance.csv`.")
    st.stop()

# Top Dashboard Header
st.html(f"""
<div class="dash-topbar">
    <div>
        <h2 class="dash-header-title">Dashboard</h2>
        <p class="dash-header-sub">Overview of student performance and key insights</p>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="background: #eff6ff; color: #1d4ed8; padding: 6px 14px; border-radius: 20px; font-size: 12.5px; font-weight: 700; border: 1px solid #bfdbfe;">
            👤 Hello, {user_info['role']}
        </span>
    </div>
</div>
""")

# 4 Metric KPI Cards across top
total_students = len(df)
avg_math = df["math score"].mean()
avg_read = df["reading score"].mean()
avg_write = df["writing score"].mean()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.html(f"""
    <div class="metric-card" style="border-left: 4px solid #3b82f6;">
        <div class="metric-card-top">
            <span class="metric-card-label">👥 Total Students</span>
            <span class="metric-pill-green">↗ 100%</span>
        </div>
        <div class="metric-card-value">{total_students:,}</div>
        <div class="metric-card-sub">Active cohort size</div>
    </div>
    """)

with m2:
    st.html(f"""
    <div class="metric-card" style="border-left: 4px solid #06b6d4;">
        <div class="metric-card-top">
            <span class="metric-card-label">📐 Average Math Score</span>
            <span class="metric-pill-green">↗ 2.8%</span>
        </div>
        <div class="metric-card-value">{avg_math:.2f}</div>
        <div class="metric-card-sub">Out of 100</div>
    </div>
    """)

with m3:
    st.html(f"""
    <div class="metric-card" style="border-left: 4px solid #10b981;">
        <div class="metric-card-top">
            <span class="metric-card-label">📖 Average Reading Score</span>
            <span class="metric-pill-green">↗ 3.1%</span>
        </div>
        <div class="metric-card-value">{avg_read:.2f}</div>
        <div class="metric-card-sub">Out of 100</div>
    </div>
    """)

with m4:
    st.html(f"""
    <div class="metric-card" style="border-left: 4px solid #8b5cf6;">
        <div class="metric-card-top">
            <span class="metric-card-label">✍️ Average Writing Score</span>
            <span class="metric-pill-green">↗ 2.4%</span>
        </div>
        <div class="metric-card-value">{avg_write:.2f}</div>
        <div class="metric-card-sub">Out of 100</div>
    </div>
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Middle Section: Subject-wise Average Scores | Score Distribution | Quick Actions
c_mid1, c_mid2, c_mid3 = st.columns([1.2, 1.4, 1.2])

with c_mid1:
    st.html('<div class="dashboard-box"><div class="box-title">Subject-wise Average Scores</div>')
    fig_sub = go.Figure(data=[
        go.Bar(
            x=['Math', 'Reading', 'Writing'],
            y=[avg_math, avg_read, avg_write],
            text=[f"{avg_math:.2f}", f"{avg_read:.2f}", f"{avg_write:.2f}"],
            textposition='outside',
            marker_color=['#3b82f6', '#10b981', '#a855f7'],
            width=0.45
        )
    ])
    fig_sub.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=260,
        margin=dict(l=20, r=20, t=25, b=20),
        yaxis=dict(range=[0, 100], gridcolor='#f1f5f9'),
        font=dict(family="Plus Jakarta Sans", size=11)
    )
    st.plotly_chart(fig_sub, use_container_width=True, config={'displayModeBar': False})
    st.html('</div>')

with c_mid2:
    st.html('<div class="dashboard-box"><div class="box-title">Score Distribution</div>')
    x_range = np.linspace(20, 100, 100)
    fig_dist = go.Figure()
    
    for sub_name, col_name, clr in [('Math', 'math score', '#3b82f6'), ('Reading', 'reading score', '#10b981'), ('Writing', 'writing score', '#a855f7')]:
        vals = df[col_name]
        mean, std = vals.mean(), vals.std()
        density = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        fig_dist.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            fill='tozeroy',
            name=sub_name,
            line=dict(color=clr, width=2),
            opacity=0.6
        ))
        
    fig_dist.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=260,
        margin=dict(l=20, r=20, t=25, b=20),
        xaxis=dict(title="Score Range", gridcolor='#f1f5f9'),
        yaxis=dict(title="Density", gridcolor='#f1f5f9', showticklabels=False),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        font=dict(family="Plus Jakarta Sans", size=11)
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
    st.html('</div>')

with c_mid3:
    st.html("""
    <div class="dashboard-box">
        <div class="box-title">Quick Actions</div>
        <div class="quick-action-btn">
            <div class="action-icon" style="background: #eff6ff; color: #2563eb;">🎯</div>
            <div>
                <p class="action-title">Predict Performance</p>
                <p class="action-desc">Enter single student details</p>
            </div>
        </div>
        <div class="quick-action-btn">
            <div class="action-icon" style="background: #f0fdf4; color: #16a34a;">⚖️</div>
            <div>
                <p class="action-title">What-If Simulator</p>
                <p class="action-desc">Compare dynamic scenarios</p>
            </div>
        </div>
        <div class="quick-action-btn">
            <div class="action-icon" style="background: #faf5ff; color: #9333ea;">📁</div>
            <div>
                <p class="action-title">Upload CSV for Batch Prediction</p>
                <p class="action-desc">Get predictions for multiple students</p>
            </div>
        </div>
    </div>
    """)

# Lower Section: Top Performing Students | Performance by Race/Ethnicity (Donut) | Recent Insights
c_low1, c_low2, c_low3 = st.columns([1.3, 1.3, 1.2])

with c_low1:
    st.html('<div class="dashboard-box"><div class="box-title">Top Performing Students (Sample)</div>')
    top_df = df.copy()
    top_df["Average"] = ((top_df["math score"] + top_df["reading score"] + top_df["writing score"]) / 3).round(1)
    top_df = top_df.sort_values(by="Average", ascending=False).head(5)[["gender", "math score", "reading score", "writing score", "Average"]]
    top_df.columns = ["Gender", "Math", "Reading", "Writing", "Average"]
    st.dataframe(top_df, use_container_width=True, hide_index=True)
    st.html('</div>')

with c_low2:
    st.html('<div class="dashboard-box"><div class="box-title">Performance by Race/Ethnicity</div>')
    eth_df = df.groupby("race/ethnicity")[["math score", "reading score", "writing score"]].mean().mean(axis=1).reset_index()
    eth_df.columns = ["Group", "Score"]
    eth_df["Group_Clean"] = eth_df["Group"].str.title()
    
    fig_donut = px.pie(
        eth_df,
        names="Group_Clean",
        values="Score",
        hole=0.55,
        color_discrete_sequence=["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#8b5cf6"]
    )
    overall_mean_all = eth_df["Score"].mean()
    fig_donut.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Plus Jakarta Sans", size=11),
        annotations=[dict(text=f"<b>Average</b><br>{overall_mean_all:.2f}", x=0.5, y=0.5, font_size=12, showarrow=False)]
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
    st.html('</div>')

with c_low3:
    st.html("""
    <div class="dashboard-box">
        <div class="box-title">Recent Insights</div>
        <div class="insight-bullet">
            💡 <strong>Writing score shows the lowest average (68.05)</strong>. Consider additional practice in writing skills.
        </div>
        <div class="insight-bullet">
            🎓 <strong>Students who completed test preparation</strong> are likely to attain 7.5% higher total average.
        </div>
        <div class="insight-bullet">
            📊 <strong>Group E has the highest average score</strong> amongst all ethnicities, followed by Group D.
        </div>
    </div>
    """)
