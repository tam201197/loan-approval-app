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

Use the sidebar to navigate between:
- 📥 Data Entry
- 🤖 Prediction
- 📊 Model Validation
""")

COLUMN_LABELS = {
    "person_age": "Age of the Person",
    "person_gender": "Gender",
    "person_education": "Highest Education Level",
    "person_income": "Annual Income (€)",
    "person_emp_exp": "Years of Employment Experience",
    "person_home_ownership": "Home Ownership Status",
    "loan_amnt": "Loan Amount Requested (€)",
    "loan_intent": "Purpose of the Loan",
    "loan_int_rate": "Loan Interest Rate (%)",
    "loan_percent_income": "Loan % of Income",
    "cb_person_cred_hist_length": "Credit History Length (Years)",
    "credit_score": "Credit Score",
    "previous_loan_defaults_on_file": "Previous Loan Defaults"
}

df = load_data()

if df.empty:
    st.warning("No data available yet.")
    st.stop()

st.sidebar.header("Filters")

age_filter = st.sidebar.slider("Age Range", int(df['person_age'].min()), int(df['person_age'].max()),
                               (int(df['person_age'].min()), int(df['person_age'].max())))
income_filter = st.sidebar.slider("Income Range (€)", int(df['person_income'].min()), int(df['person_income'].max()),
                                  (int(df['person_income'].min()), int(df['person_income'].max())))
gender_filter = st.sidebar.multiselect("Gender", options=df['person_gender'].unique(),
                                       default=df['person_gender'].unique())
education_filter = st.sidebar.multiselect("Education", options=df['person_education'].unique(),
                                          default=df['person_education'].unique())

# Apply filters
filtered_df = df[
    (df['person_age'] >= age_filter[0]) & (df['person_age'] <= age_filter[1]) &
    (df['person_income'] >= income_filter[0]) & (df['person_income'] <= income_filter[1]) &
    (df['person_gender'].isin(gender_filter)) &
    (df['person_education'].isin(education_filter))
]
filtered_df["loan_status_label"] = filtered_df["loan_status"].map({
    0: "Rejected",
    1: "Approved"
})

# ---------------------------
# KPI Metrics
# ---------------------------
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Loans", len(filtered_df))
col2.metric(f"Average {COLUMN_LABELS['loan_amnt']}", round(filtered_df['loan_amnt'].mean(), 2))
col3.metric(f"Average {COLUMN_LABELS['credit_score']}", round(filtered_df['credit_score'].mean(), 1))
col4.metric(f"Average {COLUMN_LABELS['loan_int_rate']}", round(filtered_df['loan_int_rate'].mean(), 2))

# ---------------------------
# Numeric Feature Distributions
# ---------------------------
st.subheader("📈 Numeric Feature Distributions")
numeric_cols = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
]

for col in numeric_cols:
    fig = px.histogram(filtered_df, x=col, color="loan_status_label", nbins=20, marginal="box", hover_data=filtered_df.columns,
                       title=f"{COLUMN_LABELS[col]} Distribution")
    st.plotly_chart(fig, width='stretch')

# ---------------------------
# Categorical Feature Distributions
# ---------------------------
st.subheader("📊 Categorical Feature Distributions")
categorical_cols = [
    'person_gender', 'person_education', 'person_home_ownership',
    'loan_intent', 'previous_loan_defaults_on_file'
]

for col in categorical_cols:
    vc_df = filtered_df[col].value_counts().reset_index()
    vc_df.columns = [COLUMN_LABELS[col], "count_unique"]  # unique safe column name
    fig = px.bar(vc_df, x=COLUMN_LABELS[col], y="count_unique", text="count_unique",
                 title=f"{COLUMN_LABELS[col]} Distribution")
    st.plotly_chart(fig, width='stretch')

# ---------------------------
# Scatter Plot: Loan Amount vs Credit Score
# ---------------------------
st.subheader("💹 Loan Amount vs Credit Score")
fig = px.scatter(
    filtered_df, x='credit_score', y='loan_amnt',
    color='previous_loan_defaults_on_file',
    size='person_income',
    hover_data=['person_age', 'loan_intent'],
    labels={
        'credit_score': COLUMN_LABELS['credit_score'],
        'loan_amnt': COLUMN_LABELS['loan_amnt'],
        'previous_loan_defaults_on_file': COLUMN_LABELS['previous_loan_defaults_on_file'],
        'person_income': COLUMN_LABELS['person_income']
    },
    title="Loan Amount vs Credit Score (Bubble size = Income)"
)
st.plotly_chart(fig, width='stretch')

# ---------------------------
# Correlation Heatmap
# ---------------------------
st.subheader("🧮 Correlation Heatmap")
numeric_df = filtered_df[numeric_cols]
fig = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                labels={col: COLUMN_LABELS[col] for col in numeric_cols},
                title="Correlation Heatmap of Numeric Features")
st.plotly_chart(fig, width='stretch')

# ---------------------------
# Top Loans Table
# ---------------------------
st.subheader("🏦 Top 10 Largest Loans")
st.dataframe(
    filtered_df.sort_values(by='loan_amnt', ascending=False)
    .rename(columns=COLUMN_LABELS)
    .head(10),
    width='stretch'
)

# ---------------------------
# Optional: Download Filtered Data
# ---------------------------
st.download_button(
    "💾 Download Filtered Data",
    filtered_df.to_csv(index=False),
    file_name="filtered_loans.csv"
)