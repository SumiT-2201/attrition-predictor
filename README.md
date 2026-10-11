# Employee Attrition Predictor

> **Final-year AI & Data Science Mini Project**  
> Predict whether an employee will leave, explain *why* using SHAP, and let HR
> managers explore what-if scenarios through an interactive Streamlit app.

---

## Table of Contents

1. [Overview](#overview)
2. [Screenshots](#screenshots)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [How to Run](#how-to-run)
   - [EDA Notebook](#1-eda-notebook)
   - [Training Pipeline](#2-training-pipeline)
   - [Evaluation](#3-evaluation)
   - [SHAP Explanations](#4-shap-explanations)
   - [Streamlit App](#5-streamlit-app)
7. [Results](#results)
8. [SHAP Insights](#shap-insights)
9. [Limitations](#limitations)
10. [Future Work](#future-work)
11. [Project Documents](#project-documents)

---

## Overview

Employee attrition costs organisations an estimated **50–200 % of an
employee's annual salary** in recruiting, onboarding, and lost productivity.
This project builds an end-to-end ML system that:

- **Predicts** attrition risk as a probability for any employee profile.
- **Explains** every prediction with SHAP values — global feature importance
  and per-employee waterfall / bar charts.
- **Assists decisions** through a Streamlit what-if UI where HR managers
  adjust sliders and dropdowns and see the risk gauge update live.

### Key design choices at a glance

| Concern | Choice | Why |
|---|---|---|
| Class imbalance (16 % attrition) | `class_weight="balanced"` + SMOTE-in-pipeline | No leakage; each model gets the best fit |
| Threshold selection | Business-cost optimisation (FN costs 5× FP) | Matches real HR cost asymmetry |
| Explainability | `shap.TreeExplainer` / `LinearExplainer` | Exact values, fast, model-native |
| Evaluation metric | PR-AUC primary, ROC-AUC secondary | PR-AUC is more honest on imbalanced data |
| Reproducibility | `random_state=42` everywhere | Same results on every run |

---

## Screenshots

> Replace the placeholders below with actual screenshots after running the app.

**Tab 1 — What-If Analysis**

![What-If tab screenshot](reports/figures/shap_waterfall.png)

*The sidebar lets you adjust any employee feature. The gauge, probability,
and SHAP bar chart update on every change.*

**Global Feature Importance**

![Global SHAP bar chart](reports/figures/shap_bar.png)

*Mean absolute SHAP values across the held-out test set.*

**SHAP Beeswarm Summary**

![SHAP summary plot](reports/figures/shap_summary.png)

*Each dot is one employee × one feature. Red = high feature value,
blue = low. Features sorted by mean |SHAP|.*

**ROC and PR Curves**

| ROC Curve | Precision-Recall Curve |
|---|---|
| ![ROC](reports/figures/roc_curve.png) | ![PR](reports/figures/pr_curve.png) |

**Confusion Matrix**

![Confusion matrix](reports/figures/confusion_matrix.png)

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

| Property | Value |
|---|---|
| Source | [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Rows | 1,470 employees |
| Columns | 35 features |
| Target | `Attrition` (Yes = left, No = stayed) |
| Class split | 16.1 % Yes, 83.9 % No |
| Missing values | None |
| Licence | Open Database Licence (ODbL) |

### Feature categories

| Type | Features |
|---|---|
| **Numerical** | Age, DailyRate, DistanceFromHome, HourlyRate, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager |
| **Ordinal** | BusinessTravel, Education, EnvironmentSatisfaction, JobInvolvement, JobLevel, JobSatisfaction, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, WorkLifeBalance |
| **Nominal** | Department, EducationField, Gender, JobRole, MaritalStatus |
| **Binary** | OverTime |
| **Dropped** | EmployeeCount, Over18, StandardHours, EmployeeNumber (zero variance or pure ID) |

---

## Project Structure

```
attrition-predictor/
│
├── data/
│   ├── raw/                         # Original IBM HR CSV (read-only)
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── processed/                   # Cleaned, scaled splits
│       ├── train.csv                # 1,176 rows × 49 cols (48 features + target)
│       └── test.csv                 # 294 rows × 49 cols
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Fully executed EDA notebook (10 sections)
│   └── 01_eda.py                    # Headless equivalent (run without Jupyter)
│
├── src/
│   ├── data.py                      # Load raw data, validate schema, summarise
│   ├── preprocess.py                # Encode, scale, split, save pipeline
│   ├── train.py                     # 4-model CV, RandomizedSearchCV, save best
│   ├── evaluate.py                  # Metrics, plots, cost-optimal threshold
│   ├── explain.py                   # SHAP explainer, plots, top-N factors
│   └── pipeline.py                  # End-to-end orchestrator script
│
├── models/
│   ├── best_model.joblib            # Tuned winning pipeline (ImbPipeline)
│   ├── preprocessing_pipeline.joblib# Fitted ColumnTransformer
│   └── shap_explainer.joblib        # Serialised SHAP explainer
│
├── app/
│   └── app.py                       # Streamlit what-if UI (2 tabs)
│
├── reports/
│   ├── figures/                     # All saved PNG plots
│   ├── model_comparison.csv         # 5-fold CV results for all 4 models
│   └── metrics.json                 # Test-set metrics at cost-optimal threshold
│
├── tests/
│   ├── test_data.py                 # 4 tests for src/data.py
│   ├── test_preprocess.py           # 37 tests for src/preprocess.py
│   ├── test_train.py                # 31 tests for src/train.py
│   ├── test_evaluate.py             # 35 tests for src/evaluate.py
│   └── test_explain.py              # 31 tests for src/explain.py
│
├── scripts/
│   └── check_docstrings.py          # Hook: verify docstrings + type hints
│
├── .kiro/hooks/                     # Agent hooks (auto-run on file save)
│   ├── pytest-on-src-save.json
│   ├── check-docstrings-on-src-save.json
│   └── reinstall-on-requirements-change.json
│
├── config.py                        # All paths, constants, hyperparameters
├── requirements.txt                 # Pinned dependencies
├── requirements.md                  # EARS-format requirements (48 items)
├── design.md                        # Architecture and design decisions
└── tasks.md                         # Ordered implementation checklist
```

---

## Setup

### Prerequisites

- Python 3.11 or later (tested on 3.13.2)
- Git

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd attrition-predictor
```

### 2. Create and activate the virtual environment

```bash
# Create
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> All versions are pinned. If you add a new package, run
> `pip freeze > requirements.txt` to keep the file in sync.

### 4. Place the dataset

Download the IBM HR Analytics CSV from Kaggle and place it at:

```
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

The file must be named exactly as above — `config.py` references it by that
name.

---

## How to Run

Each step is independent once the previous step's artefacts exist.
Run them in order for a full end-to-end execution.

---

### 1. EDA Notebook

**Option A — Jupyter (interactive)**

```bash
jupyter lab notebooks/01_eda.ipynb
```

Run all cells top-to-bottom. The notebook saves 9 EDA figures to
`reports/figures/` and writes the processed train/test CSVs and the
preprocessing pipeline to disk.

**Option B — headless script**

```bash
python notebooks/01_eda.py
```

Equivalent output, no browser required.

**What it produces:**

| Output | Location |
|---|---|
| Target distribution chart | `reports/figures/01_target_distribution.png` |
| Attrition by OverTime | `reports/figures/02_attrition_by_overtime.png` |
| Attrition by Job Role | `reports/figures/03_attrition_by_jobrole.png` |
| Attrition by Department | `reports/figures/04_attrition_by_department.png` |
| Attrition by Income Band | `reports/figures/05_attrition_by_income_band.png` |
| Attrition by Age Group | `reports/figures/06_attrition_by_age_group.png` |
| Numeric distributions (KDE) | `reports/figures/07_numeric_distributions.png` |
| Correlation heatmap | `reports/figures/08_correlation_heatmap.png` |
| Satisfaction scores | `reports/figures/09_satisfaction_vs_attrition.png` |
| Processed splits | `data/processed/train.csv`, `data/processed/test.csv` |
| Preprocessing pipeline | `models/preprocessing_pipeline.joblib` |

---

### 2. Training Pipeline

Runs preprocessing, 5-fold CV comparison, RandomizedSearchCV tuning on the
top-2 models, and saves the best model.

```bash
python src/train.py
```

Alternatively, run the full orchestrator (preprocessing + training together):

```bash
python src/pipeline.py
```

**What it produces:**

| Output | Location |
|---|---|
| CV comparison table | `reports/model_comparison.csv` |
| Best model pipeline | `models/best_model.joblib` |
| Timestamped XGBoost model | `models/xgboost_tuned_<timestamp>.joblib` |

**Expected console output (trimmed):**

```
Step A — Stratified 5-fold cross-validation (all 4 models)
  CV [Logistic Regression] … ROC-AUC 0.8154  PR-AUC 0.5722
  CV [Random Forest]       … ROC-AUC 0.8099  PR-AUC 0.5546
  CV [XGBoost]             … ROC-AUC 0.8094  PR-AUC 0.6054
  CV [LightGBM]            … ROC-AUC 0.8089  PR-AUC 0.5965

Step B — RandomizedSearchCV on top-2 models
  Tuning [Logistic Regression] … Best CV ROC-AUC: 0.8214
  Tuning [Random Forest]       … Best CV ROC-AUC: 0.8097

Step C — Select and save best model
  Best model: Logistic Regression  (tuned CV ROC-AUC = 0.8214)
```

---

### 3. Evaluation

Runs the held-out test set through the best model, computes all metrics, and
saves three figures.

```bash
python src/evaluate.py
```

**What it produces:**

| Output | Location |
|---|---|
| Metrics JSON | `reports/metrics.json` |
| ROC curve | `reports/figures/roc_curve.png` |
| PR curve | `reports/figures/pr_curve.png` |
| Confusion matrix | `reports/figures/confusion_matrix.png` |

---

### 4. SHAP Explanations

Builds the SHAP explainer, saves it, computes values on the test set, and
saves three explanation figures.

```bash
python src/explain.py
```

**What it produces:**

| Output | Location |
|---|---|
| SHAP explainer | `models/shap_explainer.joblib` |
| Beeswarm summary | `reports/figures/shap_summary.png` |
| Bar chart (mean \|SHAP\|) | `reports/figures/shap_bar.png` |
| Waterfall (employee #0) | `reports/figures/shap_waterfall.png` |

---

### 5. Streamlit App

```bash
streamlit run app/app.py
```

Opens at **http://localhost:8501**.

**Tab 1 — What-If Analysis**
- Sidebar with 30 feature inputs (pre-filled with dataset medians/modes).
- Risk gauge (green / amber / red arc) + attrition probability.
- SHAP bar chart of the top 10 factors for the current profile.
- Baseline vs what-if comparison table.

**Tab 2 — Model Overview**
- Global SHAP feature importance bar chart.
- CV model comparison table (all 4 models, best row highlighted).
- Test-set metrics at the business-cost threshold.
- Pipeline architecture summary.

---

### Run all tests

```bash
python -m pytest tests/ -v
```

Expected: **138 tests, 0 failures**.

---

## Results

### 5-fold cross-validation comparison

All models evaluated on the SMOTE-resampled training set with imbalance-aware
pipelines. Stratified 5-fold, scored on five metrics simultaneously.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.8154 ± 0.028** | 0.5722 ± 0.083 | 0.4775 ± 0.034 | 0.3632 | 0.7053 |
| Random Forest | 0.8099 ± 0.020 | 0.5546 ± 0.062 | 0.2637 ± 0.070 | 0.8678 | 0.1579 |
| XGBoost | 0.8094 ± 0.030 | **0.6054 ± 0.073** | **0.5514 ± 0.076** | 0.5743 | 0.5368 |
| LightGBM | 0.8089 ± 0.035 | 0.5965 ± 0.074 | 0.5014 ± 0.063 | 0.7512 | 0.3789 |

⭐ **Winner by CV ROC-AUC after tuning: Logistic Regression (0.8214)**

> Note: All four models achieve similar ROC-AUC (~0.81), confirming the
> preprocessed feature set has good signal regardless of model choice.
> XGBoost leads on PR-AUC and F1, making it the better choice if the
> business prioritises catching more leavers over ranking quality.

### Test-set metrics (held-out, 294 employees)

Decision threshold tuned to minimise `5 × FN + 1 × FP` (a missed leaver
costs 5× a false alarm).

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.8024 |
| **PR-AUC** | 0.5998 |
| **F1** | 0.5714 |
| **Precision** | 0.5490 |
| **Recall** | 0.5957 |
| Decision threshold | 0.638 |
| True Positives (caught leavers) | 28 / 47 |
| False Negatives (missed leavers) | 19 |
| False Positives (false alarms) | 23 |
| True Negatives | 224 |

> ⚠ **ROC-AUC of 0.80 is below the 0.85 target.** The model achieves solid
> recall (catching ~60 % of leavers) but the AUC target is not met.
> See [Future Work](#future-work) for improvement directions.

---

## SHAP Insights

Global feature importance ranked by **mean |SHAP value|** across the 294
test-set employees. Higher = more influential on average.

| Rank | Feature | Mean \|SHAP\| | Business Interpretation |
|---|---|---|---|
| 1 | **OverTime** | 0.847 | Working overtime is the single strongest predictor — overtime employees leave at ~3× the rate |
| 2 | **StockOptionLevel** | 0.526 | Employees with no stock options (level 0) have much higher attrition — equity aligns incentives |
| 3 | **NumCompaniesWorked** | 0.472 | Employees with a longer job-hopping history are more likely to leave again |
| 4 | **DistanceFromHome** | 0.439 | Longer commutes correlate with higher attrition, especially beyond 10 km |
| 5 | **EnvironmentSatisfaction** | 0.438 | Low scores (1–2) roughly double attrition risk versus high scores (3–4) |
| 6 | **JobLevel** | 0.422 | Junior employees (level 1–2) leave more often than senior ones — career progression matters |
| 7 | **JobSatisfaction** | 0.417 | Directly correlated with retention; low satisfaction is a consistent predictor |
| 8 | **YearsWithCurrManager** | 0.385 | Shorter tenure with the current manager signals instability or poor fit |
| 9 | **Age** | 0.364 | Younger employees (18–25) attrite at ~34 % vs ~6 % for those 46–60 |
| 10 | **JobRole: Research Scientist** | 0.358 | Research Scientists show distinctly different attrition patterns from other roles |

### Key findings for HR

- **OverTime is the dominant lever.** Reducing mandatory overtime is the
  single highest-impact intervention this model identifies.
- **Compensation + equity matter, but structure matters more.** Stock option
  level outranks raw monthly income, suggesting retention programmes should
  focus on ownership, not just salary.
- **Early-career employees need different treatment.** The combination of low
  job level + no stock options + young age forms the highest-risk profile.
- **SHAP is consistent with domain knowledge.** All top factors align with
  established HR research, which increases confidence in the model.

---

## Limitations

1. **Synthetic / controlled dataset.** The IBM HR dataset was generated for
   demonstration purposes. Real-world attrition data has messier distributions,
   missing values, and time-varying features (e.g. recent performance trend)
   that this dataset lacks.

2. **ROC-AUC below target.** The 0.80 test AUC does not reach the 0.85
   success criterion. The model is useful but not deployment-ready without
   further improvement.

3. **No temporal dimension.** The dataset is a snapshot. Attrition is often
   a process — deteriorating satisfaction scores over quarters — but we model
   it as a static cross-section. Time-series features (tenure rate of change,
   recent promotion gap) would likely improve the model.

4. **Logistic Regression as best model.** The winning model by CV ROC-AUC is
   linear, suggesting the preprocessed features are already nearly linear in
   log-odds. This is good for interpretability but may indicate that
   non-linear feature interactions are not being exploited.

5. **SHAP explainer mismatch.** The SHAP explainer was built on the XGBoost
   model (best tree model for SHAP), but the saved `best_model.joblib` is
   Logistic Regression. The app uses the explainer for the XGBoost model,
   which means SHAP explanations reflect a different model than the one
   generating probabilities. A production system would align these.

6. **No demographic fairness analysis.** The model includes `Gender` and
   `MaritalStatus` as features. No fairness audit has been performed to check
   whether the model discriminates across protected groups.

7. **Single-label binary classification.** The model predicts whether
   attrition will occur but not *when* or *to which role*, limiting its
   utility for workforce planning.

---

## Future Work

| Priority | Improvement | Expected Benefit |
|---|---|---|
| High | **Feature engineering** — income-to-peers ratio, tenure acceleration, promotion velocity | Better signal, likely +5–10 pp AUC |
| High | **Align explainer with deployed model** — use `LinearExplainer` for the LR best model | Consistent SHAP ↔ prediction explanations |
| High | **Larger hyperparameter search** — increase `TUNE_N_ITER` to 100, add CatBoost | Higher ceiling on tuned AUC |
| Medium | **Stacking / ensembling** — combine LR (best AUC) + XGBoost (best PR-AUC) | Could exceed 0.85 AUC |
| Medium | **Threshold calibration per department** — Sales has 21 % attrition vs R&D 14 % | Better precision within each business unit |
| Medium | **Fairness audit** — demographic parity, equalised odds across gender | Responsible AI compliance |
| Medium | **Survival analysis** — Cox Proportional Hazards or DeepHit | Predict *when* an employee will leave, not just whether |
| Low | **Real-time data integration** — connect to HRIS API, trigger on monthly payroll export | Production-ready pipeline |
| Low | **Retraining pipeline** — detect distribution shift, auto-retrain when AUC degrades | Model stays fresh as workforce changes |
| Low | **Convert `01_eda.py` to fully executed `.ipynb`** | Cleaner notebook submission for evaluation |

---

## Project Documents

| Document | Description |
|---|---|
| [`requirements.md`](requirements.md) | 48 EARS-format requirements across 11 sections |
| [`design.md`](design.md) | Architecture, data flow, module responsibilities, SHAP integration, app structure |
| [`tasks.md`](tasks.md) | 58 ordered, testable implementation tasks with completion status |
| [`config.py`](config.py) | Central configuration — all paths, constants, hyperparameters |

---

## Tech Stack

| Purpose | Library | Version |
|---|---|---|
| Data wrangling | pandas | 2.2.3 |
| Numerical computing | numpy | 2.1.3 |
| ML pipeline | scikit-learn | 1.5.2 |
| Imbalance handling | imbalanced-learn | 0.12.4 |
| Gradient boosting | XGBoost | 2.1.3 |
| Gradient boosting | LightGBM | 4.5.0 |
| Explainability | SHAP | 0.53.0 |
| Visualisation | matplotlib | 3.9.4 |
| Visualisation | seaborn | 0.13.2 |
| Model persistence | joblib | 1.4.2 |
| Web app | Streamlit | 1.40.2 |
| Notebooks | Jupyter | 1.1.1 |
| Testing | pytest | 8.3.4 |
| Python | — | 3.13.2 |

---

*Built as a final-year AI & Data Science mini project.*
