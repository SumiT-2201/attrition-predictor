# Requirements — Employee Attrition Predictor
**Format:** EARS (Easy Approach to Requirements Syntax)  
**Project:** Final-year AI & DS Mini Project  
**Dataset:** IBM HR Analytics Employee Attrition & Performance

---

## 1. Data Ingestion & Validation

**REQ-01** — Ubiquitous  
The system shall load the raw CSV file from `data/raw/` into a pandas DataFrame on every pipeline run.

**REQ-02** — Ubiquitous  
The system shall validate that all 35 expected IBM HR Analytics columns are present before any processing begins.

**REQ-03** — Event-driven  
When a required column is missing from the raw data, the system shall raise a `ValueError` naming the missing columns and halt execution.

**REQ-04** — Ubiquitous  
The system shall confirm that the raw dataset contains no missing values and log a warning if any are found.

---

## 2. Exploratory Data Analysis

**REQ-05** — Ubiquitous  
The system shall produce a documented EDA script (`notebooks/01_eda.py`) containing univariate, bivariate, and correlation analyses of all 35 features.

**REQ-06** — Ubiquitous  
The EDA notebook shall visualise the class imbalance in the target column `Attrition` with a labelled bar chart showing absolute counts and percentages.

**REQ-07** — Ubiquitous  
The EDA notebook shall save all generated figures as PNG files to `reports/figures/`.

---

## 3. Preprocessing

**REQ-08** — Ubiquitous  
The system shall drop the four constant/identifier columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`) before any modelling step.

**REQ-09** — Ubiquitous  
The system shall encode the binary columns `Attrition` and `OverTime` as integers (Yes→1, No→0).

**REQ-10** — Ubiquitous  
The system shall apply `StandardScaler` to numerical features, `OrdinalEncoder` to ordinal features, and `OneHotEncoder` to nominal features via a single `ColumnTransformer` pipeline.

**REQ-11** — Ubiquitous  
The system shall split the dataset into 80% train and 20% test using stratified sampling to preserve the ~16% attrition rate in both splits.

**REQ-12** — Ubiquitous  
The fitted preprocessing pipeline shall be serialised to `models/preprocessing_pipeline.joblib` for reuse by the Streamlit app.

**REQ-13** — Ubiquitous  
The cleaned and encoded train/test sets shall be saved as CSV files to `data/processed/`.

---

## 4. Class Imbalance Handling

**REQ-14** — Ubiquitous  
The system shall apply SMOTE exclusively to the training set after the train/test split to prevent data leakage into the test set.

**REQ-15** — Ubiquitous  
The system shall log the class distribution before and after SMOTE so the resampling effect is auditable.

---

## 5. Model Training & Comparison

**REQ-16** — Ubiquitous  
The system shall train and evaluate four classifiers: Logistic Regression, Random Forest, XGBoost, and LightGBM.

**REQ-17** — Ubiquitous  
The system shall evaluate every classifier using stratified 5-fold cross-validation on the SMOTE-resampled training set.

**REQ-18** — Ubiquitous  
The system shall report cross-validation mean and standard deviation for F1, Precision, Recall, and ROC-AUC for each model.

**REQ-19** — Ubiquitous  
The system shall perform hyperparameter tuning on the best-performing model using `RandomizedSearchCV` with at least 30 iterations.

**REQ-20** — Ubiquitous  
The final tuned model shall be serialised to `models/` with a timestamp-based filename (e.g. `xgboost_20240101_120000.joblib`).

---

## 6. Evaluation

**REQ-21** — Ubiquitous  
The system shall evaluate the final model on the held-out test set and report F1, Precision, Recall, ROC-AUC, and PR-AUC.

**REQ-22** — Ubiquitous  
The system shall generate and save a labelled confusion matrix heatmap to `reports/figures/confusion_matrix.png`.

**REQ-23** — Ubiquitous  
The system shall generate and save an ROC curve plot to `reports/figures/roc_curve.png`.

**REQ-24** — Ubiquitous  
The system shall generate and save a Precision-Recall curve plot to `reports/figures/pr_curve.png`.

**REQ-25** — Ubiquitous  
The system shall tune the decision threshold using Youden's J statistic and report the optimal threshold alongside F1 at that threshold.

**REQ-26** — Constraint  
The final model shall achieve a ROC-AUC of at least 0.85 on the held-out test set; if this is not met, the training script shall print a warning.

---

## 7. Explainability (SHAP)

**REQ-27** — Ubiquitous  
The system shall compute SHAP values for the full test set using `shap.TreeExplainer`.

**REQ-28** — Ubiquitous  
The system shall generate and save a global SHAP beeswarm summary plot to `reports/figures/shap_summary.png`.

**REQ-29** — Event-driven  
When a single employee's record is provided, the system shall generate a SHAP waterfall plot showing that employee's top contributing features.

**REQ-30** — Event-driven  
When a single employee's record is provided, the system shall generate a SHAP force plot for that employee.

**REQ-31** — Ubiquitous  
SHAP explanations shall use the same preprocessed feature representation as the model input so values are consistent.

---

## 8. Streamlit Application

**REQ-32** — Ubiquitous  
The application shall be launchable with a single command: `streamlit run app/app.py`.

**REQ-33** — Ubiquitous  
The application shall load the preprocessing pipeline and the latest trained model from `models/` at startup and cache them for the session.

**REQ-34** — Event-driven  
When the user adjusts any slider or dropdown in the sidebar, the application shall immediately rerun inference and update the displayed attrition probability without requiring a manual submit action.

**REQ-35** — Ubiquitous  
The application shall display the predicted attrition probability as a visual gauge (or equivalent metric display) with a colour-coded risk label (green = low risk, red = high risk).

**REQ-36** — Ubiquitous  
The application shall display a SHAP bar chart of the top contributing features for the current employee profile.

**REQ-37** — Ubiquitous  
The application shall provide input controls (sliders and dropdowns) covering all features used during model training.

**REQ-38** — Event-driven  
When no trained model artefact is found in `models/`, the application shall display a clear error message instructing the user to run the training script first.

---

## 9. Code Quality & Structure

**REQ-39** — Ubiquitous  
All Python source files shall conform to PEP 8 with a maximum line length of 88 characters.

**REQ-40** — Ubiquitous  
Every function and class shall have a Google-style docstring with Args and Returns sections.

**REQ-41** — Ubiquitous  
All functions shall include type hints on parameters and return values.

**REQ-42** — Ubiquitous  
No file paths or numeric constants shall be hard-coded; all shall be imported from `config.py`.

**REQ-43** — Ubiquitous  
`random_state=42` (sourced from `config.RANDOM_STATE`) shall be passed to every stochastic operation.

---

## 10. Testing

**REQ-44** — Ubiquitous  
The repository shall include `pytest` unit tests for every public function in `src/`.

**REQ-45** — Ubiquitous  
Tests shall use `pytest` fixtures and `tmp_path` for any file I/O; they shall not write to `data/raw/` or `models/`.

**REQ-46** — Event-driven  
When `pytest` is run from the project root, all tests shall pass without requiring external network access or a GPU.

---

## 11. Documentation

**REQ-47** — Ubiquitous  
The repository shall contain a `README.md` with sections: Project Overview, Setup, Usage (EDA → Train → Evaluate → App), Results Summary, and Project Structure.

**REQ-48** — Ubiquitous  
`requirements.txt` shall contain pinned package versions and shall be kept in sync with the virtual environment.

---

## 12. Additions

**REQ-49** — Ubiquitous  
The system shall provide a `get_feature_names(fitted_pipeline)` helper that returns the full list of output feature names from a fitted `ColumnTransformer` for use by the Streamlit app and SHAP plots.

**REQ-50** — Ubiquitous  
The Streamlit app shall call `encode_binary_columns` on the input DataFrame before passing it to `pipeline.transform` so that binary string features (OverTime) are correctly converted to integers.

**REQ-51** — Ubiquitous  
The Streamlit app shall display a semicircle gauge chart coloured green/orange/red based on the predicted attrition probability thresholds of 0.3 and 0.6.

**REQ-52** — Ubiquitous  
The system shall support training and comparing four classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM) via a `compare_models()` function that returns a DataFrame of CV metrics.

**REQ-53** — Ubiquitous  
`src/pipeline.py` shall orchestrate the full end-to-end training workflow (load → preprocess → SMOTE → compare → tune → evaluate → SHAP → save) and be runnable as a standalone script.
