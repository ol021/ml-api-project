# Bucknell Lending Club Loan Selection Model

A machine learning framework for selecting and ranking loans based on expected return and default risk using historical Lending Club data.

---

## Live Application

🔗 **Streamlit App:**  
[https://ml-api-project-6whizlofx9lcnhhvwtpvkm.streamlit.app/]

---

## Project Overview

This project develops a data-driven loan selection strategy that helps lenders allocate capital more effectively.

Instead of simply predicting whether a borrower will default, the system:

1. Predicts expected loan return
2. Estimates probability of default
3. Combines both predictions into a risk-adjusted score
4. Ranks loans from most attractive to least attractive

The goal is to maximize expected returns while controlling lending risk.

---

## Business Problem

Loan approvals often lack a systematic prioritization strategy, leading to:

- Avoidable losses
- Inefficient capital allocation
- Increased exposure to risky borrowers

This project answers the question:

> Given limited capital, which loans should be approved to maximize return while managing risk?

---

## Dataset

**Source:** Lending Club Historical Loan Data

The dataset contains approximately **50,000 loans** and includes borrower characteristics such as:

- Loan Amount
- Interest Rate
- Loan Grade
- Annual Income
- Debt-to-Income Ratio (DTI)
- Credit Score
- Revolving Utilization
- Home Ownership
- Loan Purpose

---

## Data Preparation

To ensure realistic deployment conditions and avoid data leakage, the following preprocessing steps were performed:

### Missing Values
- Imputed missing employment length values

### Feature Engineering
- Created credit age feature
- Encoded categorical variables

### Transformations
- Applied log transformation to income

### Leakage Prevention
- Removed variables unavailable at application time

### Validation Strategy
- Time-based train/test split
  - Training: Past loans
  - Testing: Future loans

This better reflects how the model would perform in production.

---

## Modeling Approach

### Return Prediction (Regression)

Models Evaluated:
- Linear Regression
- Lasso Regression
- Polynomial Regression
- Random Forest Regression

Metrics:
- MAE
- RMSE
- R²

#### Selected Model: Lasso Regression

Reasons:
- Comparable performance to complex models
- Automatic feature selection
- Easy deployment and interpretation

---

### Default Risk Prediction (Classification)

Models Evaluated:
- Logistic Regression
- Decision Tree Classifier

Metrics:
- ROC-AUC
- Lift

#### Selected Model: Logistic Regression

Reasons:
- Stable performance
- Reduced overfitting
- Reliable probability estimates

---

## Risk-Adjusted Scoring Framework

The final ranking score combines return and risk:

\[
Score = Expected\ Return \times (1 - Default\ Probability)
\]

Loans are ranked according to this score, allowing capital to be allocated toward the most attractive opportunities.

---

## Model Performance

### Regression

| Metric | Result |
|----------|----------|
| RMSE | ~9.7 |
| MAE | ~5.7 |
| R² | Near 0 |

Returns are inherently noisy, making precise prediction difficult.

### Classification

| Metric | Logistic Regression |
|----------|----------|
| ROC-AUC | ~0.69–0.72 |
| Lift (Top 20%) | ~1.17 |

Interpretation:

- The model distinguishes risky and safe borrowers moderately well.
- Loans selected by the model are approximately **16–17% less likely to default** than randomly selected loans.

---

## Key Insights

Important predictors identified by Lasso Regression:

- Interest Rate
- Loan Grade
- Debt-to-Income Ratio
- Revolving Utilization
- Credit Score
- Loan Purpose

Key findings:

- Higher interest rates increase return but also increase risk.
- Credit quality strongly influences loan outcomes.
- Higher debt burdens are associated with poorer performance.
- Ranking loans improves decision quality compared to random selection.

---

## Streamlit Application

The project includes an interactive Streamlit dashboard where users can enter applicant information and receive:

- Predicted Return
- Expected Return Range
- Default Risk Estimate
- Risk-Adjusted Evaluation

### Example Inputs

- Loan Amount
- Interest Rate
- Loan Term
- Debt-to-Income Ratio
- Credit Score
- Income
- Loan Grade
- Home Ownership
- Loan Purpose
- Revolving Utilization

---

## Future Improvements

Potential enhancements include:

- Regular model retraining
- Additional feature engineering
- Threshold optimization
- Portfolio-level optimization
- Segment-specific lending strategies
- Model drift monitoring

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit

---

## Author

**Odilon Ligan**  

Bucknell University

---

## Running Locally

```bash
git clone 

pip install -r requirements.txt

streamlit run app.py
```

---

## License

This project was developed for academic and educational purposes.