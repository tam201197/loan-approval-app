import streamlit as st
import pandas as pd
from utils import load_data, save_data, COLUMNS

st.set_page_config(page_title="Loan Entry Form", layout="wide")
st.title("📝 Loan Data Entry Form")

df = load_data()

# -----------------------------
# Input Fields (NOT inside form yet)
# -----------------------------
st.markdown("## 👤 Personal Information")

col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    person_age = st.number_input("Age 🧓", 18, 100, 25)
    person_gender = st.selectbox("Gender ⚥", ["Male", "Female", "Other"])
    person_education = st.selectbox(
        "Highest Education 🎓",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )

with col2:
    person_income = st.number_input("Annual Income (€) 💰", 0.0, step=1000.0)
    person_emp_exp = st.number_input("Years of Employment 👔", 0, step=1)

with col3:
    person_home_ownership = st.selectbox(
        "Home Ownership 🏠",
        ["Rent", "Own", "Mortgage", "Other"]
    )

st.markdown("## 💸 Loan Information")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount (€) 💵", 0.0, step=500.0)
    loan_intent = st.selectbox(
        "Loan Purpose 🎯",
        ["Personal", "Education", "Medical",
         "Venture", "Home Improvement", "Debt Consolidation"]
    )
    loan_int_rate = st.number_input("Interest Rate (%) 📈", 0.0, 100.0, step=0.1)

with col2:
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years) 📜",
        0.0, step=1.0
    )
    previous_loan_defaults_on_file = st.selectbox(
        "Previous Defaults ❌",
        ["No", "Yes"]
    )

# -----------------------------
# 🔥 REAL-TIME VALIDATION
# -----------------------------
errors = []

if person_income <= 0:
    errors.append("Income must be greater than 0.")

if loan_amnt <= 0:
    errors.append("Loan amount must be greater than 0.")

if loan_int_rate <= 0:
    errors.append("Interest rate must be greater than 0.")

if loan_amnt > person_income * 10 and person_income > 0:
    errors.append("Loan amount is too large compared to income.")

if cb_person_cred_hist_length < 0:
    errors.append("Credit history cannot be negative.")

# Show dynamic metrics
if person_income > 0:
    loan_percent_income = loan_amnt / person_income
    st.info(f"📊 Loan % of Income: {loan_percent_income:.2f}")
else:
    loan_percent_income = 0

# Display validation messages
if errors:
    st.error("⚠ Please fix the following issues:")
    for e in errors:
        st.write(f"- {e}")
    form_valid = False
else:
    st.success("✅ All inputs look good.")
    form_valid = True

# -----------------------------
# Submit Button (Disabled if invalid)
# -----------------------------
submitted = st.button("Submit Application ✅", disabled=not form_valid)

# -----------------------------
# Save if Valid
# -----------------------------
if submitted:

    new_entry = {
        "person_age": int(person_age),
        "person_gender": person_gender.lower(),
        "person_education": person_education.upper().replace(" ", "_"),
        "person_income": float(person_income),
        "person_emp_exp": int(person_emp_exp),
        "person_home_ownership": person_home_ownership.upper(),
        "loan_amnt": float(loan_amnt),
        "loan_intent": loan_intent.upper().replace(" ", "_"),
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
    st.success("🎉 Entry successfully saved!")