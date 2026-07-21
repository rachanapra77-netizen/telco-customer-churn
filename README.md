# Telco Customer Churn — Segmentation & Prediction

## Overview
Predict which telecom customers will churn, and segment the customer base for targeted retention. Combines unsupervised (clustering) and supervised (classification) learning on the same dataset.

## Dataset
7043 telecom customers, 21 features (demographics, services, account info). Target: Churn (27% Yes — imbalanced). Source: Kaggle (blastchar/telco-customer-churn).

## Approach
- **Cleaning:** Fixed TotalCharges (was text due to 11 blank values from new customers).
- **Encoding:** get_dummies for 18 categorical columns (one-hot, no fake ordering).
- **Segmentation:** K-Means (k=3 via elbow method) → 3 customer segments.
- **Visualization:** PCA to 2D (45% variance retained).
- **Prediction:** 3 models with leakage-safe pipeline, stratified split, imbalance handling.

## Key Insight
New customers churn at 44% — 6× the rate of budget users (7%). Retention should focus on the first 16 months of a customer's lifecycle.

## Results

| Model | ROC-AUC | Churn Recall |
|---|---|---|
| Logistic Regression | 0.84 | 0.79 |
| Random Forest | 0.82 | 0.49 |
| XGBoost | 0.82 | 0.68 |

**Logistic Regression won** — the simplest model. Complex models don't always win; the best model depends on the data. For churn detection, recall matters most (catch churners), and Logistic had the highest.

## Key Learnings
- Used ROC-AUC + recall instead of accuracy (imbalanced data — accuracy misleads).
- Leakage-safe: scaling fit only on training data, after the split.
- Scaling only for distance-based steps (K-Means); tree models used raw data.

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, Jupyter

## How to Run
1. Download the dataset from Kaggle
2. `pip install -r requirements.txt`
3. Open `Telco_Customer_Churn.ipynb` and Run All