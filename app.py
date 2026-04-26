
import streamlit as st
import requests
import numpy as np
import pandas as pd
import base64

API_URL = "https://ml-api-project-1-kfrb.onrender.com/predict"

def get_base64(img_file):
    with open(img_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("bucknell_bg.jpg")

# =========================
# Page config
# =========================
st.set_page_config(page_title="Bucknell Lending AI", layout="centered")

st.markdown(f"""
<style>

/* Background layer */
.stApp {{
    background: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    
}}

/* White overlay */
.stApp::after {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background: rgba(255,255,255,0.65);
    z-index: -1;
}}

.main .block-container {{
    background: rgba(255,255,255,0.95);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.15);
}}

.stMarkdown {{
    font-size: 16px;
    line-height: 1.6;
}}

/* Titles */
h1 {{
    color: #003366;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
}}

h2, h3 {{
    color: #FF6600;
    front-weight: 700;
}}

/* General text */
p, label, div {{
    color: #111111;
    font-weight: 500;
}}

/* Buttons */
.stButton>button {{
    background-color: #003366;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}}

.stButton>button:hover {{
    background-color: #FF6600;
    color: white;
}}

/* Metrics cards */
[data-testid="stMetric"] {{
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #003366;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}}

/* Recommendation box */
.decision-box {{
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}}

/* Divider */
hr {{
    border: 1px solid #003366;
}}

</style>
""", unsafe_allow_html=True)

st.title("🏦 Bucknell Lending Decision Engine")

st.markdown("""
Evaluate loan applications using machine learning.

!! This system is designed to **rank loans and improve selection**, not perfectly predict returns !!
""")

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

# Additional required fields (important for realism)
grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
home_ownership = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE"])
purpose = st.selectbox("Purpose", ["debt_consolidation","credit_card","home_improvement","other"])

revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 30.0)

# =========================
# Validation
# =========================
errors = []
warnings = []

# Hard validation
if loan_amnt <= 0:
    errors.append("Loan amount must be positive.")

# Soft warnings
if annual_inc < 10000:
    warnings.append("Low income → limited repayment capacity")

if dti > 45:
    warnings.append("High DTI → borrower may struggle with payments")

if fico < 500:
    warnings.append("Very low credit score → significantly elevated default risk")
    

# =========================
# Show validation warnings
# =========================
# Display warnings
for w in warnings:
    st.warning(w)

# Display errors
if errors:
    for e in errors:
        st.error(e)

# =========================
# Call API
# =========================
if st.button("🚀 Evaluate Loan", disabled=len(errors) > 0):

    payload = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "term": term,
        "dti": dti,
        "fico": fico,
        "annual_inc": annual_inc,
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "revol_util": revol_util
    }
    
    try:

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("💰 Return", f"{result['predicted_return']:.3f}")
            col2.metric("✅ Fully Paid", f"{result['prob_fully_paid']:.2%}")
            col3.metric("⚠️ Default Risk", f"{result['prob_default']:.2%}")
            
            st.subheader("📉 Expected Return Range")

            st.write(
                f"{result['predicted_return']:.2f} "
                f"(range: {result['return_lower']:.2f} to {result['return_upper']:.2f})"
            )
            
            st.markdown("---")
            
            st.subheader("🔍 Model Confidence")

            confidence = result["confidence"]

            st.progress(confidence)

            if confidence < 0.3:
                st.warning("Low confidence prediction")
            elif confidence < 0.6:
                st.info("Moderate confidence")
            else:
                st.success("High confidence")

            st.markdown("---")

            decision = result["decision"]
            
            st.subheader("🏁 Recommendation")

            if decision == "APPROVE":
                st.success("✅ APPROVE LOAN")
            else:
                st.error("❌ REJECT LOAN")

            st.markdown("### 🧠 Risk-Adjusted Score")
            st.write(f"{result['score']:.3f}")
            st.caption("""
            This score combines expected return and default risk.
            Higher values indicate better lending opportunities.
            """)

        else:
            st.error(f"API error: {response.status_code}")
            st.write(response.text)
    
    except requests.exceptions.RequestException:
        st.error("Cannot reach API")
        
        
# =========================
# =========================
# MODEL PERFORMANCE SECTION
# =========================
# =========================

st.markdown("---")
st.header("📈 Model Performance Insights")

st.markdown("""
This model is evaluated based on **loan selection quality**, not raw prediction accuracy.
""")

# =========================
# TOP-K PERFORMANCE GRAPH
# =========================
st.subheader("🏆 Top Loan Selection Performance")

percentiles = ["Top 10%", "Top 20%", "Top 30%", "Top 50%"]

model_returns = [-0.685, -0.843, -0.906, -0.996]
random_returns = [-1.30, -1.15, -1.42, -1.29]

chart_df = pd.DataFrame({
    "Model": model_returns,
    "Random": random_returns
}, index=percentiles)

st.line_chart(chart_df)

st.caption("""
The model consistently selects loans with **better returns than random selection**, 
especially in the top 20%.
""")

# =========================
# THRESHOLD GRAPH
# =========================
st.subheader("⚖️ Threshold vs Return Tradeoff")

thresholds = [0.2, 0.3, 0.4]
returns = [-0.41, -0.81, -1.06]

df_thresh = pd.DataFrame({
    "Avg Return": returns
}, index=thresholds)

st.bar_chart(df_thresh)

st.caption("""
While absolute returns are negative, the model **significantly outperforms random selection**.
This indicates the model is effectively ranking loans by risk-adjusted return.
""")

# =========================
# FINAL MESSAGE
# =========================
st.info("""
This system is designed to assist decision-making by prioritizing better loan opportunities.
It is not intended to perfectly predict outcomes, but to improve investment selection.
""")

st.markdown("---")
st.markdown(
    "**Developed by Odilon Ligan & Nick Snyder**  |  Bucknell University"
)