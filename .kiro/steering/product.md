# Product Overview

## Project: Employee Attrition Predictor

This is a final-year AI & Data Science mini project.

### Goal
Predict whether an employee will leave the organisation (binary classification: attrition = Yes / No), explain *why* the model made that prediction using SHAP values, and give HR managers an interactive what-if analysis tool.

### Core deliverables
1. **Trained model** — a well-tuned classifier (XGBoost or LightGBM) that handles class imbalance and generalises to unseen employees.
2. **Explainability** — SHAP summary plots, waterfall charts, and force plots that surface the top drivers of attrition for any individual employee.
3. **Streamlit app** — a what-if UI with sliders/dropdowns for every input feature; changing a value instantly reruns inference and updates the predicted probability and SHAP explanation.

### Target users
- HR managers and business analysts (non-technical).
- Final-year project evaluators looking for end-to-end ML best practices.

### Success criteria
- AUC-ROC ≥ 0.85 on the held-out test set.
- SHAP explanations consistent with domain knowledge (e.g., overtime and job satisfaction are strong predictors).
- Streamlit app runs locally with a single `streamlit run app/app.py` command.
