# Design Document — Employee Attrition Predictor

**Version:** 1.0  
**Project:** Final-year AI & DS Mini Project  
**Dataset:** IBM HR Analytics Employee Attrition & Performance (1,470 rows, 35 columns)

---

## 1. Architecture Overview

The system is a three-layer pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  data/raw/  ──►  src/data.py  ──►  src/preprocess.py               │
│  (immutable CSV)   (load + validate)  (encode, scale, split)        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ data/processed/{train,test}.csv
                                 │ models/preprocessing_pipeline.joblib
┌────────────────────────────────▼────────────────────────────────────┐
│                        MODEL LAYER                                  │
│  src/train.py  ──►  src/evaluate.py  ──►  src/explain.py           │
│  (SMOTE + 4 models  (F1/AUC/PR-AUC    (SHAP values,               │
│   CV + tuning)       + threshold)       summary/waterfall)          │
│                                                                     │
│  models/xgboost_tuned_<timestamp>.joblib                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ loads .joblib artefacts
┌────────────────────────────────▼────────────────────────────────────┐
│                    PRESENTATION LAYER                               │
│  app/app.py  (Streamlit what-if UI)                                 │
│  ├── Sidebar: 30 input sliders/dropdowns                            │
│  ├── Gauge: live attrition probability                              │
│  └── SHAP bar chart: top 10 feature contributions                  │
└─────────────────────────────────────────────────────────────────────┘
```

All layers share a single source of truth for paths and constants via `config.py`.

---

## 2. Data Flow

```
CSV (data/raw/)
    │
    ▼ src/data.load_raw_data()
    │   validate_schema()          — assert 35 expected columns present
    │
    ▼ src/preprocess.drop_leakage_columns()
    │   — removes EmployeeCount, Over18, StandardHours, EmployeeNumber
    │
    ▼ src/preprocess.encode_binary_columns()
    │   — Attrition: Yes→1 / No→0
    │   — OverTime:  Yes→1 / No→0
    │
    ▼ src/preprocess.split_features_target()
    │   — X (30 features), y (binary Attrition)
    │
    ▼ src/preprocess.train_test_split_stratified()
    │   — 80/20 stratified split (random_state=42)
    │   — preserves ~16% attrition rate in both splits
    │
    ├──► X_train / y_train ──► ColumnTransformer.fit_transform()
    │                               ├── StandardScaler  (numerical)
    │                               ├── OrdinalEncoder  (ordinal cats)
    │                               └── OneHotEncoder   (nominal cats)
    │                          ──► apply_smote()
    │                               — minority class oversampled on train only
    │                               — prevents leakage into test set
    │
    └──► X_test / y_test ──► ColumnTransformer.transform()
                             (pipeline fitted on train only)
    │
    ▼
    X_train_resampled, y_train_resampled  →  model training
    X_test_processed                      →  evaluation
    │
    ▼ Save artefacts
    models/preprocessing_pipeline.joblib
    models/<model>_<timestamp>.joblib
    data/processed/train.csv
    data/processed/test.csv
```

---

## 3. Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all paths, constants, and hyperparameter defaults. Never import paths from anywhere else. |
| `src/data.py` | Load raw CSV, validate schema (35 expected columns), print summary statistics. Read-only with respect to `data/raw/`. |
| `src/preprocess.py` | Drop constant columns, encode binary columns, build `ColumnTransformer`, stratified split, save/load pipeline. `build_preprocessing_pipeline()` returns an **unfitted** transformer; the caller fits it. `get_feature_names()` extracts output column names post-fit. |
| `src/train.py` | Train all 4 models (`train_logistic_regression`, `train_random_forest`, `train_xgboost`, `train_lightgbm`). `compare_models()` runs 5-fold CV with F1/Precision/Recall/ROC-AUC/**PR-AUC**. `tune_xgboost()` runs `RandomizedSearchCV`. `save_model()` / `load_model()` handle persistence. |
| `src/evaluate.py` | Compute and plot ROC curve, PR curve, confusion matrix. `find_optimal_threshold()` uses Youden's J. `compute_pr_auc()` for imbalanced-class reporting. |
| `src/explain.py` | Wrap `shap.TreeExplainer`. `compute_shap_values()` handles shap 0.53 ndarray returns. `plot_summary()` global beeswarm. `plot_waterfall()` per-employee (uses module-level `plt.savefig` — no pre-allocated figure). `plot_force()` for force plot. |
| `src/pipeline.py` | End-to-end orchestrator script. Runs tasks 1–9 in order: load → preprocess → SMOTE → compare → tune → evaluate → SHAP → save. Entry point for the full training run. |
| `notebooks/01_eda.py` | Headless EDA script (runnable without Jupyter). Generates 5 figures, saves processed CSVs and the fitted pipeline. |
| `app/app.py` | Streamlit UI. Loads artefacts once via `@st.cache_resource`. Builds input row from sidebar, runs `encode_binary_columns` then `pipeline.transform`, calls `model.predict_proba`, renders gauge + SHAP chart. |
| `tests/` | pytest unit tests using synthetic DataFrames only. No dependency on real CSV or models/. Uses `tmp_path` for file I/O. |

---

## 4. Model Selection Strategy

### Candidate models
Four models are trained to allow principled comparison:

| Model | Rationale |
|---|---|
| Logistic Regression | Fast linear baseline; interpretable coefficients; good with scaled features |
| Random Forest | Non-linear ensemble; handles feature interactions; less prone to overfitting than single trees |
| XGBoost | Gradient-boosted trees with L1/L2 regularisation; strong on tabular data; `scale_pos_weight` for imbalance |
| LightGBM | Faster XGBoost alternative; leaf-wise growth; `class_weight="balanced"` for imbalance |

### Comparison protocol
- **Stratified 5-fold CV** on the SMOTE-resampled training set
- Metrics reported: F1 (primary), Precision, Recall, ROC-AUC, **PR-AUC** (most informative for imbalanced data)
- PR-AUC is the tiebreaker when ROC-AUC scores are close — it penalises false positives harder

### Hyperparameter tuning
- Best model (XGBoost in practice) is tuned with `RandomizedSearchCV` (n_iter=30, scoring=`roc_auc`)
- Search space covers: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`
- All runs use `random_state=42` for reproducibility

### Decision threshold
- Default 0.5 threshold is suboptimal for imbalanced classes
- Optimal threshold selected by maximising **Youden's J = TPR − FPR** on the test set
- Business cost logic: missing a true attritor (false negative) is more costly than a false alarm (false positive), so the threshold is tuned to favour recall

---

## 5. SHAP Integration Approach

### Why TreeExplainer
`shap.TreeExplainer` is used (not `KernelExplainer`) because:
- It is exact for tree-based models (no approximation error)
- It runs in O(TLD) time (T=trees, L=leaves, D=depth) vs O(N²) for Kernel
- It supports both XGBoost and LightGBM natively

### shap 0.53 compatibility
shap 0.53 returns a plain `ndarray` from `shap_values()` for binary XGBoost models (older versions returned a list). `compute_shap_values()` handles all three return formats:
```python
if isinstance(sv, list):       # shap < 0.46 — take class 1
    return sv[1]
if hasattr(sv, "values"):      # Explanation object — unwrap .values
    ...
return sv                      # shap 0.53+ ndarray — use directly
```

### Waterfall plot fix
`shap.plots.waterfall()` creates its own internal matplotlib figure. Pre-allocating `fig, ax = plt.subplots()` before calling it produces a **blank duplicate figure**. The fix is to call `shap.plots.waterfall(..., show=False)` then `plt.savefig()` at module level — this captures the figure SHAP created internally.

### Global vs per-employee explanations
| Type | Function | Output |
|---|---|---|
| Global importance | `plot_summary()` | Beeswarm plot — each dot is one employee; colour = feature value |
| Per-employee | `plot_waterfall()` | Waterfall — how each feature pushes probability up/down from base rate |
| Per-employee | `plot_force()` | Force plot — single-row horizontal view of feature contributions |

---

## 6. Streamlit App Structure

```
app/app.py
│
├── @st.cache_resource load_artefacts()
│       ├── load_pipeline()   → models/preprocessing_pipeline.joblib
│       └── load_model()      → models/<latest>.joblib
│           (raises st.error if missing)
│
├── SIDEBAR — st.sidebar
│   ├── Numerical sliders (Age, MonthlyIncome, YearsAtCompany, …)  [20 sliders]
│   └── Categorical dropdowns (Department, JobRole, MaritalStatus, …) [10 dropdowns]
│
├── build_input_df()
│       └── assembles sidebar values → 1-row pd.DataFrame (30 features)
│
├── INFERENCE (runs on every widget change)
│   ├── encode_binary_columns(input_df)   ← OverTime Yes/No → 0/1
│   ├── pipeline.transform(input_df)      ← scale + encode
│   └── model.predict_proba(X_proc)[0][1] ← attrition probability
│
├── COLUMN 1 — Prediction
│   ├── st.metric("Attrition Probability", f"{prob:.1%}")
│   ├── Colour-coded risk label (🟢 Low / 🔴 High)
│   └── Gauge chart (matplotlib arc or plotly indicator)
│
├── COLUMN 2 — SHAP Explanation
│   ├── shap.TreeExplainer (cached in st.session_state)
│   ├── shap_values = explainer.shap_values(X_proc)
│   └── shap.summary_plot(plot_type="bar", top 10 features)
│
└── EXPANDER — Raw Feature Values
        └── st.dataframe(input_df)
```

### State management
- `@st.cache_resource` — pipeline and model loaded once per session (not per rerun)
- `st.session_state.explainer` — SHAP explainer built once, reused across reruns
- Every slider/dropdown change triggers a full Streamlit rerun automatically

---

## 7. Architecture Diagram

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                        config.py                                 │
 │  PROJECT_ROOT, DATA_RAW_DIR, MODELS_DIR, RANDOM_STATE=42, …     │
 └──────────┬───────────────────────────────────────────────────────┘
            │ imported by all modules
            │
 ┌──────────▼───────────┐     ┌──────────────────────────────────┐
 │     src/data.py      │────►│        data/raw/                 │
 │  load_raw_data()     │     │  WA_Fn-UseC_-HR-Employee-        │
 │  validate_schema()   │     │  Attrition.csv  (read-only)      │
 └──────────┬───────────┘     └──────────────────────────────────┘
            │ raw DataFrame
 ┌──────────▼───────────┐
 │  src/preprocess.py   │────► data/processed/train.csv
 │  drop_leakage_cols() │────► data/processed/test.csv
 │  encode_binary()     │────► models/preprocessing_pipeline.joblib
 │  ColumnTransformer   │
 │  SMOTE (train only)  │
 └──────────┬───────────┘
            │ X_train_res, y_train_res
            │ X_test_proc, y_test
 ┌──────────▼──────────────────────────────────────────┐
 │                   src/train.py                      │
 │  compare_models()  ← 5-fold CV (F1/AUC/PR-AUC)     │
 │  tune_xgboost()   ← RandomizedSearchCV (n_iter=30) │
 │  save_model()     ────────────────────────────────► models/*.joblib
 └──────────┬──────────────────────────────────────────┘
            │ fitted model + test data
 ┌──────────▼───────────┐     ┌──────────────────────────────────┐
 │   src/evaluate.py    │────►│      reports/figures/            │
 │  plot_roc_curve()    │     │  roc_curve.png                   │
 │  plot_pr_curve()     │     │  pr_curve.png                    │
 │  plot_confusion_mx() │     │  confusion_matrix.png            │
 │  find_opt_threshold()│     └──────────────────────────────────┘
 └──────────┬───────────┘
            │ fitted model + X_test
 ┌──────────▼───────────┐     ┌──────────────────────────────────┐
 │    src/explain.py    │────►│      reports/figures/            │
 │  TreeExplainer       │     │  shap_summary.png                │
 │  compute_shap_vals() │     │  shap_waterfall.png              │
 │  plot_summary()      │     │  shap_force.png                  │
 │  plot_waterfall()    │     └──────────────────────────────────┘
 └──────────────────────┘
            
 ┌────────────────────────────────────────────────────────────────┐
 │                       app/app.py                               │
 │                                                                │
 │  ┌─────────────┐    ┌──────────────────────────────────────┐  │
 │  │   SIDEBAR   │    │            MAIN PANEL                │  │
 │  │             │    │                                      │  │
 │  │ 20 sliders  │───►│  ┌─────────────┐  ┌──────────────┐  │  │
 │  │ 10 dropdowns│    │  │  PREDICTION │  │     SHAP     │  │  │
 │  │             │    │  │   GAUGE     │  │  BAR CHART   │  │  │
 │  │             │    │  │  prob: X%   │  │  top 10 feats│  │  │
 │  │             │    │  │  🔴/🟢 risk │  │              │  │  │
 │  └─────────────┘    │  └─────────────┘  └──────────────┘  │  │
 │                     │                                      │  │
 │                     │  ▼ Raw Feature Values (expander)     │  │
 │                     └──────────────────────────────────────┘  │
 └────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|---|---|
| SMOTE on train-only | Applying SMOTE before splitting would let synthetic minority samples appear in the test set, inflating performance metrics |
| ColumnTransformer over manual encoding | Single sklearn-compatible pipeline artefact; `get_feature_names_out()` gives consistent feature names to SHAP |
| Youden's J for threshold | Maximises sensitivity + specificity jointly; better than F1-maximisation for HR cost asymmetry |
| PR-AUC added to CV scoring | ROC-AUC is overly optimistic on imbalanced data (large TN pool inflates specificity); PR-AUC focuses on the minority class |
| `@st.cache_resource` for artefacts | Models are large; reloading on every Streamlit rerun would be too slow |
| Module-level `plt.savefig` in SHAP plots | SHAP creates its own figure internally; pre-allocating `plt.subplots()` creates a blank duplicate |
| `random_state=42` everywhere | Full reproducibility across split, SMOTE, model init, CV shuffle, and hyperparameter search |
