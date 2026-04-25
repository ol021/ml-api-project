import joblib
import pandas as pd
import numpy as np

print("RUNNING SCRIPT")
# Load models
reg_model = joblib.load("loan_model.pkl")
clf_model = joblib.load("cl_loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

BEST_THRESHOLD = 0.3

# =========================
# Simulated input (same as API)
# =========================
data = {
    "loan_amnt": 8000,
    "int_rate": 8.5,
    "term": "36 months",
    "dti": 10,
    "fico": 750,
    "annual_inc": 120000,
    "grade": "A",
    "home_ownership": "OWN",
    "purpose": "credit_card",
    "revol_util": 20
}

# =========================
# Feature engineering
# =========================
term_num = 36 if data["term"] == "36 months" else 60

input_dict = {col: 0 for col in feature_names}

# numeric
input_dict['int_rate'] = data["int_rate"]
input_dict['term_num'] = term_num
input_dict['dti'] = data["dti"]
input_dict['fico_avg'] = data["fico"]
input_dict['annual_inc'] = np.log1p(data["annual_inc"])
input_dict['loan_amnt'] = data["loan_amnt"]
input_dict['revol_util'] = data.get("revol_util", 0)

# categorical
if f"grade_{data['grade']}" in input_dict:
    input_dict[f"grade_{data['grade']}"] = 1

if f"home_ownership_{data['home_ownership']}" in input_dict:
    input_dict[f"home_ownership_{data['home_ownership']}"] = 1

if data["purpose"] == "debt_consolidation":
    input_dict["purpose_debt_consolidation"] = 1

# =========================
# DataFrame + scaling
# =========================
input_df = pd.DataFrame([input_dict])
input_df = input_df[feature_names]

print("\nRAW INPUT:")
print(input_df)
print("\nScaler was trained on order:")
print(feature_names)

#input_scaled = scaler.transform(input_df)
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns, index=input_df.index)
# input_scaled.to_csv("fname.csv", index=False)
# =========================
# DEBUG PRINTS
# =========================
print("\nSCALED INPUT:")
print(scaler.mean_)
print(input_scaled)

# =========================
# Predictions
# =========================
pred_return = float(reg_model.predict(input_scaled)[0])
prob_default = float(clf_model.predict_proba(input_scaled)[0][1])
prob_fully_paid = 1 - prob_default

decision = "APPROVE" if prob_default < BEST_THRESHOLD else "REJECT"
score = pred_return * (1 - prob_default)

# =========================
# Output
# =========================
print("\nRESULTS:")
print({
    "predicted_return": pred_return,
    "prob_default": prob_default,
    "prob_fully_paid": prob_fully_paid,
    "decision": decision,
    "score": score
})