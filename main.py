
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Bucknell Lending API")

# =========================
# Load models
# =========================
reg_model = joblib.load("loan_model.pkl")
clf_model = joblib.load("cl_loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

BEST_THRESHOLD = 0.3

# =========================
# Health check
# =========================
@app.post("/predict")
def predict(data: dict):

    # =========================
    # Base features
    # =========================
    term_num = 36 if data["term"] == "36 months" else 60

    # =========================
    # Start with ALL zeros
    # =========================
    input_dict = {col: 0 for col in feature_names}

    # =========================
    # Fill numeric features
    # =========================
    input_dict['int_rate'] = data["int_rate"]
    input_dict['term_num'] = term_num
    input_dict['dti'] = data["dti"]
    input_dict['fico_avg'] = data["fico"]
    input_dict['annual_inc'] = np.log1p(data["annual_inc"])
    input_dict['loan_amnt'] = data["loan_amnt"]
    input_dict['revol_util'] = data.get("revol_util", 0)  # optional

    # =========================
    # One-hot encoding
    # =========================

    # Grade
    grade = data["grade"]
    if f"grade_{grade}" in input_dict:
        input_dict[f"grade_{grade}"] = 1

    # Home ownership
    home = data["home_ownership"]
    if f"home_ownership_{home}" in input_dict:
        input_dict[f"home_ownership_{home}"] = 1

    # Purpose
    if data["purpose"] == "debt_consolidation":
        input_dict["purpose_debt_consolidation"] = 1

    # =========================
    # Convert to DataFrame
    # =========================
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names] 

    # =========================
    # Predictions
    # =========================
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns, index=input_df.index)
    
    
    pred_return = float(reg_model.predict(input_scaled)[0])
    prob_default = float(clf_model.predict_proba(input_scaled)[0][1])
    prob_fully_paid = 1 - prob_default

    # =========================
    # Decision
    # =========================
    if prob_default < BEST_THRESHOLD:
        decision = "APPROVE"
    else:
        decision = "REJECT"

    score = pred_return * (1 - prob_default)

    return {
        "predicted_return": round(pred_return, 4),
        "prob_default": round(prob_default, 4),
        "prob_fully_paid": round(prob_fully_paid, 4),
        "decision": decision,
        "score": round(score, 4)
    }