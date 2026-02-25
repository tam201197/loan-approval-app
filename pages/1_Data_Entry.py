import streamlit as st
import pandas as pd
from utils import load_data, save_data, COLUMNS

st.set_page_config(page_title="Loan Entry Form", layout="wide")
st.title("📝 Loan Data Entry Form")

# Load Data

df = load_data()

# Create Loan Entry Form

with st.form("loan_entry_form"):
    
    st.markdown("## 👤 Personal Information")
    with st.expander("Show/Hide Personal Info", expanded=True):
        col1, col2, col3 = st.columns([1.5, 1, 1])
        
        with col1:
            person_age = st.number_input("Age 🧓", min_value=18, max_value=100, step=1,
                                         help="Enter age in years")
            person_gender = st.selectbox("Gender ⚥", ["Male", "Female", "Other"])
            person_education = st.selectbox("Highest Education 🎓", ["High School", "Bachelor", "Master", "Doctorate", "Accosiate"])
        
        with col2:
            person_income = st.number_input("Annual Income (€) 💰", min_value=0.0, step=1000.0)
            person_emp_exp = st.number_input("Years of Employment Experience 👔", min_value=0, step=1)
        
        with col3:
            person_home_ownership = st.selectbox("Home Ownership 🏠", ["Rent", "Own", "Mortgage", "Other"])
    
    st.markdown("## 💸 Loan Information")
    with st.expander("Show/Hide Loan Info", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount Requested (€) 💵", min_value=0.0, step=500.0)
            loan_intent = st.selectbox("Purpose of Loan 🎯", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt consolidaion"])
            loan_int_rate = st.number_input("Interest Rate (%) 📈", min_value=0.0, max_value=100.0, step=0.1)
        
        with col2:
            cb_person_cred_hist_length = st.number_input("Credit History Length (Years) 📜", min_value=0.0, step=1.0)
            #credit_score = st.number_input("Credit Score 🏦", min_value=300, max_value=850, step=1)
            previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults ❌", ["No", "Yes"])
    
    submitted = st.form_submit_button("Submit ✅")


# Processing Form Submission
if submitted:
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0

    new_entry = {
        "person_age": int(person_age),
        "person_gender": person_gender.lower(),
        "person_education": person_education.upper(),
        "person_income": float(person_income),
        "person_emp_exp": int(person_emp_exp),
        "person_home_ownership": person_home_ownership.upper(),
        "loan_amnt": float(loan_amnt),
        "loan_intent": loan_intent.upper().replace(' ', ''),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
        "credit_score": 0,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    new_row_df = pd.DataFrame([new_entry], columns=COLUMNS)
    df = pd.concat([df, new_row_df], ignore_index=True)
    save_data(df)

    st.balloons()
    st.success(f"✅ Entry successfully saved!\nLoan % of Income: {round(loan_percent_income, 2)}")