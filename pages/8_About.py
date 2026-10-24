import streamlit as st
import sys

st.set_page_config(page_title="About & Documentation", page_icon="ℹ️")
st.title("About the System")

st.markdown("""
### Student Performance Prediction & Personalized Academic Support System

**Version:** 1.0.0

This application was developed as a college mini-project to demonstrate the application of explainable machine learning in the educational domain.

#### Objectives
- Analyze student performance data.
- Train regression algorithms to predict math, reading, and writing scores.
- Provide interpretability and what-if simulation features.
- Offer actionable academic support recommendations based on model outputs.

#### Methodology
- **Data:** Kaggle's "Students Performance in Exams" dataset.
- **Preprocessing:** One-hot encoding for categorical variables.
- **Algorithms:** Cross-validated models including Linear Regression, Decision Trees, Random Forests, and Gradient Boosting. The best model per subject is selected based on $R^2$ score.
- **Evaluation:** Models are evaluated on a held-out test set using MAE, RMSE, and $R^2$.

#### Limitations & Ethical Considerations
- Predictions are estimates with inherent uncertainty.
- Feature importance identifies correlations, not causal relationships.
- Demographic categories (e.g., race/ethnicity, lunch status) reflect societal and systemic associations in the specific dataset, not inherent student abilities.

#### Environment
""")
st.write(f"Python Version: {sys.version}")
