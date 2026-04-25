# app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

# =========================
# Page config
# =========================
st.set_page_config(page_title="Bucknell Lending AI", layout="centered")

st.markdown("""
    <style>
    h1 { color: #FF6600; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Bucknell Lending Decision Engine")

st.markdown("Evaluate loan applications using AI models.")

# =========================
# Inputs
# =========================
st.subheader("📋 Applicant Information")

loan_amnt = st.number_input("Loan Amount", 1000, 50000, 10000)
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
term = st.selectbox("Term", ["36 months", "60 months"])
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
fico = st.slider("FICO Score", 300, 850, 650)
annual_inc = st.number_input("Annual Income", 20000, 500000, 80000)

# =========================
# Call API
# =========================
if st.button("🚀 Evaluate Loan"):

    payload = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "term": term,
        "dti": dti,
        "fico": fico,
        "annual_inc": annual_inc
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.subheader("📊 Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("💰 Return", f"{result['predicted_return']:.3f}")
        col2.metric("✅ Fully Paid", f"{result['prob_fully_paid']:.2%}")
        col3.metric("⚠️ Default Risk", f"{result['prob_default']:.2%}")

        st.markdown("---")

        decision = result["decision"]

        if decision == "APPROVE":
            st.success("✅ APPROVE LOAN")
        else:
            st.error("❌ REJECT LOAN")

        st.markdown("### 🧠 Risk-Adjusted Score")
        st.write(result["score"])

    else:
        st.error("API error")