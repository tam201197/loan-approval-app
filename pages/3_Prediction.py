import streamlit as st
from utils import load_data
import os
import joblib
import json

st.set_page_config(page_title="🤖 Loan Risk Prediction", layout="wide")

model_path = "best_model/best_model.pkl" 
meta_path = "best_model/best_model_meta.json"

if not os.path.exists(model_path):
    st.error("⚠ No trained model found. Please run Model Validation first.")
    st.stop()

model = joblib.load(model_path)

if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        metadata = json.load(f)
else:
    metadata = None

st.subheader("🏆 Active Model Information")

if metadata:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", metadata["model_name"])

    with col2:
        st.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")

    with col3:
        st.metric("Training Samples", metadata["training_samples"])

    st.caption(f"Trained on: {metadata['training_time']}")

    with st.expander("⚙ Best Hyperparameters"):
        st.json(metadata["best_params"])
else:
    st.warning("No metadata file found.")

df = load_data()

if df.empty:
    st.warning("No data available.")
else:
    latest = df.iloc[[-1]]  # Keep DataFrame format
    input_data = latest.drop("loan_status", axis=1)

    y_pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", latest["person_age"].values[0])
        st.metric("Gender", latest["person_gender"].values[0])
        st.metric("Education", latest["person_education"].values[0])

    with col2:
        st.metric("Annual Income ($)", f'{latest["person_income"].values[0]:,.0f}')
        st.metric("Employment Experience (years)", latest["person_emp_exp"].values[0])
        st.metric("Home Ownership", latest["person_home_ownership"].values[0])

    with col3:
        #st.metric("Credit Score", latest["credit_score"].values[0])
        st.metric("Credit History Length", latest["cb_person_cred_hist_length"].values[0])
        st.metric("Previous Defaults", latest["previous_loan_defaults_on_file"].values[0])
    
    st.subheader("💰 Loan Details")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Loan Amount ($)", f'{latest["loan_amnt"].values[0]:,.0f}')
    with col5:
        st.metric("Interest Rate (%)", latest["loan_int_rate"].values[0])
    with col6:
        st.metric("Loan % of Income", f'{latest["loan_percent_income"].values[0]:,.3f}')

    y_pred = model.predict(latest.drop("loan_status", axis=1))

    st.subheader("🤖 Risk Assessment")

    if y_pred[0] == 1:
        st.success("✅ Low Risk Applicant")
        st.write("Prediction:", "Approved!")
    else:
        st.error("⚠ High Risk Applicant")
        st.write("Prediction:", "Rejected!")