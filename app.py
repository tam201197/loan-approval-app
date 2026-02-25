import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from utils import load_data  # assuming your CSV helper from previous code

st.set_page_config(page_title="Loan Risk App", layout="wide", initial_sidebar_state="expanded")

st.title("🏦 Loan Risk Management App")

st.markdown("""
Welcome to the Loan Risk Application.

A complete Machine Learning web application that predicts whether a loan
application is likely to be approved or rejected.

This project demonstrates a full ML lifecycle including:
- Data exploration
- Model validation
- Hyperparameter tuning
- Best model tracking
- Real-time prediction
""")

st.divider()

# -----------------------------------
# PROJECT OVERVIEW
# -----------------------------------
st.header("📌 About This Project")

st.markdown("""
This application is built from **Tam Truong** using:

- **Streamlit** for the web interface  
- **Scikit-learn & XGBoost** for Machine Learning  
- **Plotly**, **Seaborn** for interactive visualizations  
- A **public Kaggle dataset** for loan approval classification  

The goal is to simulate a real-world credit risk evaluation pipeline.
""")

st.divider()

# -----------------------------------
# APP STRUCTURE
# -----------------------------------
st.header("🗂 App Structure")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 1. Dashboard")
    st.markdown("""
    - Explore dataset statistics  
    - Visualize numeric & categorical features  
    - Compare loan approval distribution  
    - Correlation heatmap  
    - Interactive charts  
    """)

    st.subheader("📝 2. Data Entry")
    st.markdown("""
    - Add a new loan application  
    - Validate user inputs  
    - Append new entry to dataset  
    """)

with col2:
    st.subheader("🔮 3. Prediction")
    st.markdown("""
    - Load best validated model  
    - Predict loan approval from the last data entry 
    - Display model metadata  
    """)

    st.subheader("🧠 4. Model Validation")
    st.markdown("""
    - Compare multiple ML models  
    - Cross-validation  
    - GridSearchCV tuning  
    - ROC curve comparison  
    - Confusion matrix visualization  
    - Auto-save best model  
    """)

st.divider()

# -----------------------------------
# ML PIPELINE VISUAL
# -----------------------------------
st.header("⚙ Machine Learning Workflow")

st.markdown("""
1. Load and clean dataset  
2. Preprocess features  
3. Train multiple models  
4. Perform cross-validation  
5. Tune hyperparameters  
6. Compare ROC-AUC  
7. Select and save best model  
8. Use best model for prediction  
""")

st.divider()

# -----------------------------------
# CALL TO ACTION
# -----------------------------------
st.success("👉 Use the sidebar to navigate through the app pages.")

st.caption("Built for Machine Learning Engineering portfolio demonstration.")
