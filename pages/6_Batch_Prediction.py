import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .instruction-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
        color: #166534;
        font-size: 13.5px;
    }
    .info-panel {
        background: #eff6ff;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        margin-bottom: 20px;
        font-size: 13.5px;
        line-height: 1.5;
    }
    .upload-container {
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        background: #f8fafc;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

models_dir = Path("models")
try:
    models = {
        "math": joblib.load(models_dir / "math_model.joblib"),
        "reading": joblib.load(models_dir / "reading_model.joblib"),
        "writing": joblib.load(models_dir / "writing_model.joblib")
    }
except:
    st.error("Models not found in models/ directory. Run `python -m src.train_models` first.")
    st.stop()

def get_performance_category(score):
    if score < 50: return "Needs Intensive Support (<50)"
    elif score < 65: return "Developing (50-64)"
    elif score < 80: return "Proficient (65-79)"
    else: return "Advanced Mastery (80-100)"

# --- TWO COLUMN LAYOUT MATCHING USER SCREENSHOT ---
col_side, col_main = st.columns([1, 2.3], gap="large")

with col_side:
    st.write("##### How would you like to predict?")
    pred_mode = st.selectbox("Prediction Mode", ["Batch", "Single Student"], label_visibility="collapsed")
    
    if pred_mode == "Single Student":
        st.info("👉 Switch to page **1_Student_Prediction** in the sidebar for single student input.")
    
    st.markdown("""
    <div class="info-panel">
        <strong>🎯 Student Performance Assessment</strong><br><br>
        This module is designed to predict complete student cohorts in batch.<br><br>
        The aim is to assist school boards and teachers in identifying students who need personalized academic support early, and keeping them on track towards graduation.
    </div>
    """, unsafe_allow_html=True)
    
    # Visual illustration card
    st.markdown("""
    <div style="text-align: center; background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="font-size: 48px;">👨‍🏫 📚 🎯</div>
        <div style="font-size: 12px; color: #64748b; margin-top: 8px;">Targeted Academic Intervention Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    load_sample = st.button("📥 Load Sample Cohort (100 Students)", use_container_width=True)

with col_main:
    st.title("Predicting Student Performance in Batch")
    st.caption("Kindly Follow The Instructions Below For The Best Experience While Using The Application")
    
    st.write("##### The Required Format For The CSV File To Be Uploaded Is Stated Below:")
    sample_preview_df = pd.DataFrame({
        "gender": ["female", "male", "female", "male", "female"],
        "race/ethnicity": ["group B", "group C", "group B", "group A", "group C"],
        "parental level of education": ["bachelor's degree", "some college", "master's degree", "associate's degree", "some college"],
        "lunch": ["standard", "free/reduced", "standard", "free/reduced", "standard"],
        "test preparation course": ["none", "completed", "none", "none", "completed"]
    })
    st.dataframe(sample_preview_df, use_container_width=True, hide_index=False)
    
    st.write("##### How would you like the results presented?")
    presentation_mode = st.selectbox(
        "Presentation Format",
        ["Pie-Chart", "Bar-Chart", "Table Preview", "Score Distribution Histogram"],
        label_visibility="collapsed"
    )
    
    st.write("##### Upload CSV File for Predictions")
    uploaded_file = st.file_uploader(
        "Drag and drop file here (Limit 200MB per file • CSV)",
        type=["csv"],
        label_visibility="visible"
    )
    
    data_to_predict = None
    if uploaded_file is not None:
        try:
            data_to_predict = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_sample:
        # Load sample from dataset
        try:
            raw_df = pd.read_csv("data/StudentsPerformance.csv")
            data_to_predict = raw_df[["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]].sample(100, random_state=42).reset_index(drop=True)
        except Exception as e:
            st.error(f"Could not load sample: {e}")

    if data_to_predict is not None:
        required_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
        if not all(col in data_to_predict.columns for col in required_cols):
            st.error(f"Missing required columns! Required: {required_cols}")
        else:
            with st.spinner("Executing machine learning batch prediction..."):
                results_df = data_to_predict.copy()
                results_df["predicted_math"] = np.round(models["math"].predict(results_df).clip(0, 100), 1)
                results_df["predicted_reading"] = np.round(models["reading"].predict(results_df).clip(0, 100), 1)
                results_df["predicted_writing"] = np.round(models["writing"].predict(results_df).clip(0, 100), 1)
                results_df["predicted_average"] = np.round((results_df["predicted_math"] + results_df["predicted_reading"] + results_df["predicted_writing"]) / 3, 1)
                results_df["performance_category"] = results_df["predicted_average"].apply(get_performance_category)
            
            st.success(f"Successfully processed {len(results_df)} student records!")
            
            # Render according to Presentation Format chosen
            if presentation_mode == "Pie-Chart":
                cat_counts = results_df["performance_category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_pie = px.pie(
                    cat_counts,
                    names="Category",
                    values="Count",
                    title="Student Performance Category Distribution",
                    color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b", "#ef4444"],
                    hole=0.4
                )
                fig_pie.update_layout(font=dict(family="Inter", size=13))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            elif presentation_mode == "Bar-Chart":
                cat_counts = results_df["performance_category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_bar = px.bar(
                    cat_counts,
                    x="Category",
                    y="Count",
                    color="Category",
                    title="Cohort Count by Performance Tier",
                    color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
                )
                fig_bar.update_layout(font=dict(family="Inter"))
                st.plotly_chart(fig_bar, use_container_width=True)
                
            elif presentation_mode == "Score Distribution Histogram":
                fig_hist = px.histogram(
                    results_df,
                    x="predicted_average",
                    nbins=20,
                    title="Predicted Average Score Distribution",
                    color_discrete_sequence=["#0284c7"]
                )
                fig_hist.update_layout(font=dict(family="Inter"))
                st.plotly_chart(fig_hist, use_container_width=True)
                
            # Table display
            st.write("##### Results Preview:")
            st.dataframe(results_df.head(15), use_container_width=True)
            
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Predictions (CSV)",
                data=csv_data,
                file_name="student_predictions_batch.csv",
                mime="text/csv",
                use_container_width=True
            )
