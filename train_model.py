import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# --- Sample dataset (you can swap in your own) ---
data = pd.DataFrame({
    "monthly_spend":  [50, 60, 70, 120, 150, 200, 80, 90, 40, 30],
    "tenure_months":  [ 2,  3,  6,  12,  24,  36,  4,  5,  1,  1],
    "support_calls":  [ 3,  1,  0,   1,   0,   0,  2,  1,  4,  3],
    "churn":          [ 1,  1,  0,   0,   0,   0,  1,  0,  1,  1]
})

# --- Prepare features and target ---
X = data[["monthly_spend", "tenure_months", "support_calls"]]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Train the model ---
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# IN PRACTICE (or final projects), create the best performing model you can (balancing bias-variance) using all your skills. 

# --- NEW STEP: Save the trained model to a file ---
# --- This is the DEPLOYED model ---
joblib.dump(model, "churn_model.pkl")

print("Model saved as churn_model.pkl")