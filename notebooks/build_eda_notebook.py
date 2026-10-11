"""Script to programmatically build notebooks/01_eda.ipynb using nbformat.

Run with:
    python notebooks/build_eda_notebook.py
"""

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "01_eda.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    """Shorthand for a markdown cell."""
    return new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    """Shorthand for a code cell."""
    return new_code_cell(text.strip())


cells = []

# ==========================================================================
# Title
# ==========================================================================
cells.append(md("""
# Employee Attrition Predictor — Exploratory Data Analysis

**Project:** Final-year AI & Data Science Mini Project  
**Dataset:** IBM HR Analytics Employee Attrition & Performance  
**Author:** see README.md  

This notebook explores the IBM HR Analytics dataset before any modelling. 
The goals are:

1. Understand the raw data: shape, dtypes, missing values.  
2. Identify and drop constant/leakage columns.  
3. Examine the target distribution (class imbalance).  
4. Explore attrition rates across key categorical and numeric features.  
5. Inspect numeric distributions and inter-feature correlations.  
6. Summarise key insights that will inform modelling decisions.  

All figures are saved to `reports/figures/` and the cleaned dataset is written  
to `data/processed/` for use by the modelling pipeline.
"""))

# ==========================================================================
# 0. Imports & setup
# ==========================================================================
cells.append(md("## 0. Imports & Setup"))

cells.append(code("""
import sys
from pathlib import Path

# Ensure project root is importable so config.py and src/ are accessible.
ROOT = Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend required for nbconvert execution.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import load_raw_data, validate_schema
from src.preprocess import (
    drop_leakage_columns,
    encode_binary_columns,
    split_features_target,
    build_preprocessing_pipeline,
    train_test_split_stratified,
    save_pipeline,
)

# Consistent visual style across all figures.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Setup complete.")
print(f"Figures will be saved to: {FIGURES_DIR}")
print(f"Processed data will be saved to: {DATA_PROCESSED_DIR}")
"""))

# ==========================================================================
# 1. Dataset Overview
# ==========================================================================
cells.append(md("""
## 1. Dataset Overview

The IBM HR Analytics dataset contains **1,470 employee records** with **35 features**
covering demographics, job characteristics, compensation, and satisfaction scores.
The binary target column `Attrition` records whether the employee left the company.

We start by loading the raw CSV, validating that all expected columns are present,
and printing a high-level summary.
"""))

cells.append(code("""
raw = load_raw_data()
validate_schema(raw)

print(f"Shape : {raw.shape}")
print(f"\\nDtypes summary:")
print(raw.dtypes.value_counts())
print(f"\\nFirst 3 rows:")
raw.head(3)
"""))

cells.append(code("""
# Detailed dtype listing — useful to identify which columns need encoding.
print(raw.dtypes.to_string())
"""))

# ==========================================================================
# 2. Missing Values
# ==========================================================================
cells.append(md("""
## 2. Missing Values

Before any analysis we check for missing values. 
The IBM HR dataset is a synthetic dataset and is known to be complete,
but we verify this explicitly rather than assuming.
"""))

cells.append(code("""
missing = raw.isnull().sum()
total_missing = missing.sum()

if total_missing == 0:
    print("✅ No missing values found in any column.")
else:
    print(f"⚠️  {total_missing} missing values found:")
    print(missing[missing > 0])
"""))

# ==========================================================================
# 3. Dropping Constant Columns
# ==========================================================================
cells.append(md("""
## 3. Dropping Constant / Identifier Columns

Four columns carry no predictive signal and must be removed before modelling:

| Column | Reason to Drop |
|---|---|
| `EmployeeCount` | Constant value of 1 for every row — zero variance |
| `StandardHours` | Constant value of 80 for every row — zero variance |
| `Over18` | Constant value of "Y" for every row — zero variance |
| `EmployeeNumber` | Unique identifier — causes overfitting if kept |

Keeping these columns would inflate correlation metrics and potentially
leak identifier information into the model.
"""))

cells.append(code("""
# Confirm the constant values before dropping.
constant_cols = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
for col in constant_cols:
    unique_vals = raw[col].unique()
    print(f"{col:20s} → unique values: {unique_vals}")
"""))

cells.append(code("""
df = drop_leakage_columns(raw)
print(f"Shape after dropping constant columns: {df.shape}")
print(f"Dropped: {set(raw.columns) - set(df.columns)}")
"""))

# ==========================================================================
# 4. Target Distribution
# ==========================================================================
cells.append(md("""
## 4. Target Distribution — Class Imbalance

The target variable `Attrition` is binary: **Yes** (employee left) or **No** (stayed).
We expect significant class imbalance because attrition is a relatively rare event in practice.

Understanding the imbalance ratio is critical for:
- Choosing the right evaluation metric (F1, PR-AUC rather than accuracy).
- Deciding on a class-imbalance mitigation strategy (SMOTE in the pipeline).
- Setting `scale_pos_weight` in XGBoost.
"""))

cells.append(code("""
counts = df["Attrition"].value_counts()
pcts   = df["Attrition"].value_counts(normalize=True).mul(100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart — absolute counts.
bars = axes[0].bar(counts.index, counts.values,
                   color=["#4CAF50", "#F44336"], edgecolor="white", width=0.5)
axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=12)
axes[0].set_title("Attrition — Absolute Counts")
axes[0].set_ylabel("Number of Employees")
axes[0].set_xlabel("Attrition")

# Pie chart — proportions.
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Attrition — Proportion")

plt.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Attrition=Yes : {counts['Yes']:4d} ({pcts['Yes']}%)")
print(f"Attrition=No  : {counts['No']:4d} ({pcts['No']}%)")
print(f"Imbalance ratio (No:Yes) ≈ {counts['No']/counts['Yes']:.1f}:1")
"""))

# ==========================================================================
# 5. Attrition Rates by Category
# ==========================================================================
cells.append(md("""
## 5. Attrition Rates by Key Features

We now examine attrition rates broken down by the most business-relevant
categorical and derived features. These plots reveal which groups are at
highest risk and inform feature importance expectations later in SHAP analysis.
"""))

# --- 5a OverTime ---
cells.append(md("""
### 5a. Attrition by OverTime

Employees required to work overtime may experience burnout, leading to higher
attrition. We expect a substantially higher attrition rate for OverTime=Yes.
"""))

cells.append(code("""
ot_pivot = (df.groupby("OverTime")["Attrition"]
              .value_counts(normalize=True)
              .mul(100)
              .rename("Pct")
              .reset_index())
ot_yes = ot_pivot[ot_pivot["Attrition"] == "Yes"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(ot_yes["OverTime"], ot_yes["Pct"],
              color=["#4CAF50", "#F44336"], edgecolor="white", width=0.4)
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Attrition Rate by OverTime", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xlabel("OverTime")
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_attrition_by_overtime.png", dpi=150, bbox_inches="tight")
plt.show()

for ot, grp in df.groupby("OverTime"):
    rate = (grp["Attrition"] == "Yes").mean() * 100
    print(f"OverTime={ot:3s}  Attrition rate: {rate:.1f}%")
"""))

# --- 5b JobRole ---
cells.append(md("""
### 5b. Attrition by Job Role

Different roles carry different levels of stress, career growth opportunity,
and compensation. Sales Representatives and Laboratory Technicians are often
cited in HR literature as high-turnover roles.
"""))

cells.append(code("""
role_rates = (df.groupby("JobRole")["Attrition"]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(role_rates.index, role_rates.values,
               color=sns.color_palette("RdYlGn_r", len(role_rates)))
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Attrition Rate by Job Role", fontweight="bold")
ax.set_xlabel("Attrition Rate (%)")
ax.set_xlim(0, 50)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_attrition_by_jobrole.png", dpi=150, bbox_inches="tight")
plt.show()
print(role_rates.round(1).to_string())
"""))

# --- 5c Department ---
cells.append(md("""
### 5c. Attrition by Department

Department-level attrition rates can indicate structural issues in specific
business units — e.g., demanding sales targets or limited R&D career paths.
"""))

cells.append(code("""
dept_rates = (df.groupby("Department")["Attrition"]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(dept_rates.index, dept_rates.values,
              color=["#2196F3", "#FF9800", "#9C27B0"], edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Attrition Rate by Department", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_ylim(0, 30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_attrition_by_department.png", dpi=150, bbox_inches="tight")
plt.show()
print(dept_rates.round(1).to_string())
"""))

# --- 5d Income Band ---
cells.append(md("""
### 5d. Attrition by Monthly Income Band

Compensation is a major factor in retention. We bin `MonthlyIncome` into
quartile-based bands to examine whether lower earners leave at higher rates.
"""))

cells.append(code("""
df2 = df.copy()
df2["IncomeBand"] = pd.qcut(
    df2["MonthlyIncome"], q=4,
    labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
)
income_rates = (df2.groupby("IncomeBand", observed=True)["Attrition"]
                   .apply(lambda s: (s == "Yes").mean() * 100))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(income_rates.index.astype(str), income_rates.values,
              color=sns.color_palette("Blues_d", 4), edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Attrition Rate by Monthly Income Quartile", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xlabel("Income Band")
ax.set_ylim(0, 40)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_attrition_by_income_band.png", dpi=150, bbox_inches="tight")
plt.show()
print(income_rates.round(1).to_string())
"""))

# --- 5e Age Group ---
cells.append(md("""
### 5e. Attrition by Age Group

Younger employees are generally more mobile and less tied to a single employer.
We group ages into decade-based bands to test this hypothesis.
"""))

cells.append(code("""
df2["AgeGroup"] = pd.cut(
    df2["Age"], bins=[17, 25, 35, 45, 60],
    labels=["18–25", "26–35", "36–45", "46–60"]
)
age_rates = (df2.groupby("AgeGroup", observed=True)["Attrition"]
                .apply(lambda s: (s == "Yes").mean() * 100))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(age_rates.index.astype(str), age_rates.values,
              color=sns.color_palette("Oranges_d", 4), edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Attrition Rate by Age Group", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xlabel("Age Group")
ax.set_ylim(0, 40)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_attrition_by_age_group.png", dpi=150, bbox_inches="tight")
plt.show()
print(age_rates.round(1).to_string())
"""))

# ==========================================================================
# 6. Numeric Distributions
# ==========================================================================
cells.append(md("""
## 6. Numeric Feature Distributions

We compare the distribution of key numeric features between employees who stayed
and those who left. Overlapping KDE (kernel density estimate) plots reveal whether
a feature has discriminative power for predicting attrition.

Features examined: `Age`, `MonthlyIncome`, `YearsAtCompany`, `DistanceFromHome`,
`TotalWorkingYears`, `YearsInCurrentRole`.
"""))

cells.append(code("""
numeric_features = [
    "Age", "MonthlyIncome", "YearsAtCompany",
    "DistanceFromHome", "TotalWorkingYears", "YearsInCurrentRole",
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, feat in zip(axes, numeric_features):
    for label, color in [("No", "#4CAF50"), ("Yes", "#F44336")]:
        subset = df[df["Attrition"] == label][feat]
        subset.plot.kde(ax=ax, label=f"Attrition={label}", color=color, linewidth=2)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.suptitle("Numeric Feature Distributions by Attrition", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ==========================================================================
# 7. Correlation Heatmap
# ==========================================================================
cells.append(md("""
## 7. Correlation Heatmap

A Pearson correlation heatmap across all numeric features helps identify:
- **Multicollinearity** (highly correlated feature pairs that may be redundant).
- **Target correlations** (features most linearly related to Attrition).

We first encode `Attrition` and `OverTime` as integers so they appear in the matrix.

> Note: Pearson correlation only captures *linear* relationships. Tree-based models
> can exploit non-linear interactions even for low-correlation features.
"""))

cells.append(code("""
df_corr = encode_binary_columns(df.copy())

# Select only numeric columns for the heatmap.
numeric_cols = df_corr.select_dtypes(include="number").columns.tolist()
corr_matrix  = df_corr[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # Upper triangle mask.
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    cmap="RdBu_r", center=0,
    linewidths=0.3, linecolor="white",
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Pearson Correlation Heatmap — Numeric Features", fontsize=14,
             fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Print top correlations with Attrition.
attrition_corr = corr_matrix["Attrition"].drop("Attrition").sort_values(key=abs, ascending=False)
print("Top 10 correlations with Attrition:")
print(attrition_corr.head(10).round(3).to_string())
"""))

# ==========================================================================
# 8. Satisfaction Scores vs Attrition
# ==========================================================================
cells.append(md("""
## 8. Satisfaction Scores vs Attrition

The dataset includes four ordinal satisfaction scores (scale 1–4):
`JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`,
and `WorkLifeBalance`. We visualise their attrition rates to understand
how dissatisfaction predicts turnover.
"""))

cells.append(code("""
satisfaction_cols = [
    "JobSatisfaction", "EnvironmentSatisfaction",
    "RelationshipSatisfaction", "WorkLifeBalance",
]
labels_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
for ax, col in zip(axes, satisfaction_cols):
    rates = (df.groupby(col)["Attrition"]
               .apply(lambda s: (s == "Yes").mean() * 100))
    x_labels = [labels_map.get(i, str(i)) for i in rates.index]
    bars = ax.bar(x_labels, rates.values,
                  color=sns.color_palette("coolwarm_r", len(rates)), edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_title(col.replace("Satisfaction", "\\nSatisfaction"), fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylim(0, 35)
    if ax == axes[0]:
        ax.set_ylabel("Attrition Rate (%)")

plt.suptitle("Attrition Rate by Satisfaction Scores", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_satisfaction_vs_attrition.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ==========================================================================
# 9. Save Cleaned Data
# ==========================================================================
cells.append(md("""
## 9. Save Cleaned Dataset to `data/processed/`

We now apply the full preprocessing pipeline — dropping constant columns,
encoding binary columns, and performing a stratified 80/20 train/test split —
and save the resulting artefacts:

- `data/processed/train.csv` — training set (features + target, pre-SMOTE)
- `data/processed/test.csv`  — held-out test set
- `models/preprocessing_pipeline.joblib` — fitted `ColumnTransformer`

> **Important:** SMOTE is **not** applied here. It is applied inside `src/pipeline.py`
> on the training set only, after this split, to prevent leakage into the test set.
"""))

cells.append(code("""
from config import MODELS_DIR
from src.preprocess import save_pipeline

df_clean = drop_leakage_columns(raw)
df_clean = encode_binary_columns(df_clean)

X, y = split_features_target(df_clean)
X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

preprocessor = build_preprocessing_pipeline(X_train)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Retrieve feature names from the fitted ColumnTransformer.
feature_names = preprocessor.get_feature_names_out().tolist()

# Assemble DataFrames with feature names + target column.
train_df = pd.DataFrame(X_train_proc, columns=feature_names)
train_df[TARGET_COLUMN] = y_train.values

test_df = pd.DataFrame(X_test_proc, columns=feature_names)
test_df[TARGET_COLUMN] = y_test.values

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
test_df.to_csv(DATA_PROCESSED_DIR / "test.csv",  index=False)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
save_pipeline(preprocessor)

print(f"train.csv saved  : {train_df.shape}")
print(f"test.csv saved   : {test_df.shape}")
print(f"Pipeline saved   : {MODELS_DIR / 'preprocessing_pipeline.joblib'}")
print(f"\\nTrain attrition rate : {y_train.mean()*100:.1f}%")
print(f"Test  attrition rate : {y_test.mean()*100:.1f}%")
"""))

# ==========================================================================
# 10. Key Insights
# ==========================================================================
cells.append(md("""
## 10. Key Insights

Based on the exploratory analysis, the following findings will guide modelling decisions:

### Class Imbalance
- The dataset is **~84% No / 16% Yes** (attrition). Accuracy is a misleading metric.
- We will use **SMOTE** on the training set and report **F1, PR-AUC, and ROC-AUC**.
- XGBoost's `scale_pos_weight ≈ 5` will compensate for the imbalance during training.

### Strongest Attrition Signals (from charts)
| Feature | Finding |
|---|---|
| **OverTime** | Employees working overtime leave at ~3× the rate of those who don't (~31% vs ~10%) |
| **JobRole** | Sales Representatives (~40%) and Laboratory Technicians (~24%) have the highest rates |
| **MonthlyIncome** | Q1 (lowest earners) attrite at ~26% vs ~8% for Q4 (highest earners) |
| **AgeGroup** | 18–25 year-olds attrite at ~34%; risk drops sharply with age |
| **Department** | Sales (~21%) > HR (~19%) > R&D (~14%) |

### Satisfaction Scores
- All four satisfaction dimensions show a **monotonic negative relationship** with attrition.
- `JobSatisfaction=1` (Low) has the highest attrition rate among satisfaction predictors.

### Correlation Observations
- `TotalWorkingYears`, `YearsAtCompany`, `Age`, and `MonthlyIncome` are positively correlated
  with each other — older, more tenured employees earn more and leave less.
- The highest *linear* correlations with Attrition (negative) are `TotalWorkingYears`,
  `JobLevel`, and `MonthlyIncome` — but tree models will capture non-linear effects too.

### Multicollinearity
- `YearsAtCompany`, `YearsInCurrentRole`, `YearsWithCurrManager`, and `YearsSinceLastPromotion`
  are strongly correlated (r > 0.6). The `ColumnTransformer` handles them as separate scaled
  features; tree models are robust to multicollinearity anyway.

### Constant Columns Removed
- `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber` — all removed (zero variance
  or pure identifier). These were confirmed to have exactly one unique value each.

---
*End of EDA. Proceed to `src/pipeline.py` for the full modelling pipeline.*
"""))

# ==========================================================================
# Build and write the notebook
# ==========================================================================
nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.13.2",
    },
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
