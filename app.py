
import streamlit as st
import requests
import numpy as np
import pandas as pd
import base64

API_URL = "https://ml-api-project-1-kfrb.onrender.com/predict"

def get_base64(img_file):
    with open(img_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("bucknell_2.jpg")

# =========================
# Page config
# =========================
st.set_page_config(page_title="Bucknell Lending AI", layout="centered")

st.markdown(f"""
<style>

/* ===== BACKGROUND ===== */
.stApp {{
    background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
    background-size: cover;
}}

/* Soft dark overlay (better than white for readability) */
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 30, 60, 0.55); /* Bucknell blue tint */
    z-index: -1;
}}

/* ===== MAIN CONTAINER (GLASS EFFECT) ===== */
.main .block-container {{
    background: rgba(255, 255, 255, 0.92);
    padding: 2.5rem;
    border-radius: 18px;
    max-width: 900px;
    margin: auto;
    
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}}

/* ===== TEXT ===== */
h1 {{
    color: white;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 1px;
}}

h2, h3 {{
    color: #FF6600; /* Bucknell orange */
    font-weight: 700;
}}

p {{
    color: #333;
    font-size: 16px;
}}

/* Labels */
label {{
    color: #003366; /* Bucknell blue */
    font-weight: 600;
}}

/* ===== BUTTON ===== */
.stButton>button {{
    background: linear-gradient(90deg, #003366, #0055A4);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1rem;
}}

.stButton>button:hover {{
    background: #FF6600;
}}

/* ===== METRICS ===== */
[data-testid="stMetric"] {{
    background: white;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #FF6600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}

</style>
""", unsafe_allow_html=True)

st.title("🏦 Bucknell Lending Decision Engine")

st.markdown("""
<p style='text-align:center; color:white; font-size:16px;'>
Evaluate loan applications using machine learning.<br>
<span style='opacity:0.8'>
This system ranks loans to improve selection, not perfectly predict returns.
</span>
</p>
""", unsafe_allow_html=True)

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
<p style='text-align:center; color:white; font-size:16px;'>
This model is evaluated based on **loan selection quality**, not raw prediction accuracy.<br>
</p>
""", unsafe_allow_html=True)

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

st.markdown("""
<p style='text-align:center; color:white; font-size:16px;'>
"**Developed by Odilon Ligan & Nick Snyder**  |  Bucknell University"<br>
</p>
""", unsafe_allow_html=True)
