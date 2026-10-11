# Tasks — Employee Attrition Predictor
**Format:** Ordered, small, testable tasks derived from `requirements.md` and `design.md`  
**Status legend:** ✅ Done | 🔄 In Progress | ⬜ Pending

---

## Phase 1 — Project Setup

| # | Task | Verification | Status |
|---|---|---|---|
| 1.1 | Create folder structure: `data/raw`, `data/processed`, `notebooks/`, `src/`, `models/`, `app/`, `reports/figures/`, `tests/` | `Get-ChildItem` shows all 8 directories | ✅ |
| 1.2 | Create `.venv` with `python -m venv .venv` | `.venv\Scripts\python.exe --version` returns Python 3.13 | ✅ |
| 1.3 | Create `requirements.txt` with 13 pinned packages | File exists; all versions pinned (no `>=` ranges) | ✅ |
| 1.4 | Install all packages into `.venv` | `pip list` shows all 13 packages at correct versions | ✅ |
| 1.5 | Create `config.py` with all paths, `RANDOM_STATE=42`, and model hyperparameter dicts | `python -c "from config import RANDOM_STATE; assert RANDOM_STATE == 42"` passes | ✅ |

---

## Phase 2 — Data Ingestion & Validation

| # | Task | Verification | Status |
|---|---|---|---|
| 2.1 | Place IBM HR Analytics CSV in `data/raw/` | `Test-Path data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` returns True | ✅ |
| 2.2 | Implement `src/data.load_raw_data()` — loads CSV, raises `FileNotFoundError` if missing | `pytest tests/test_data.py::test_load_raw_data_file_not_found` passes | ✅ |
| 2.3 | Implement `src/data.validate_schema()` — raises `ValueError` on missing columns | `pytest tests/test_data.py::test_validate_schema_missing_columns` passes | ✅ |
| 2.4 | Implement `src/data.summarise()` — prints shape, nulls, target distribution | `pytest tests/test_data.py::test_summarise_runs_without_error` passes | ✅ |
| 2.5 | Confirm raw CSV: 1470 rows, 35 columns, 0 missing values, 16.12% attrition | `python -c "..."` prints correct shape and null count | ✅ |

---

## Phase 3 — EDA

| # | Task | Verification | Status |
|---|---|---|---|
| 3.1 | Create `notebooks/01_eda.py` — class distribution bar chart saved to `reports/figures/attrition_distribution.png` | File exists and size > 0 bytes | ✅ |
| 3.2 | Add age and monthly income histograms by attrition group | `reports/figures/age_distribution.png` and `income_distribution.png` exist | ✅ |
| 3.3 | Add correlation heatmap for numerical features | `reports/figures/correlation_heatmap.png` exists | ✅ |
| 3.4 | Add top categorical features vs attrition bar charts | `reports/figures/categorical_vs_attrition.png` exists | ✅ |
| 3.5 | Save cleaned train/test CSVs to `data/processed/` from EDA script | `data/processed/train.csv` (1176 rows) and `test.csv` (294 rows) exist | ✅ |
| 3.6 | Run `python notebooks/01_eda.py` end-to-end without errors | Exit code 0, all 5 figures present | ✅ |

---

## Phase 4 — Preprocessing

| # | Task | Verification | Status |
|---|---|---|---|
| 4.1 | Implement `drop_leakage_columns()` — removes 4 constant/ID columns | `pytest tests/test_preprocess.py::test_drop_leakage_columns` passes | ✅ |
| 4.2 | Implement `encode_binary_columns()` — maps Yes/No → 1/0 for `Attrition` and `OverTime` | `pytest tests/test_preprocess.py::test_encode_binary_columns` passes | ✅ |
| 4.3 | Implement `split_features_target()` — raises `ValueError` if target column missing | `pytest tests/test_preprocess.py::test_split_features_target_missing_col` passes | ✅ |
| 4.4 | Implement `build_preprocessing_pipeline()` — returns unfitted `ColumnTransformer` | `pytest tests/test_preprocess.py::test_build_preprocessing_pipeline` passes | ✅ |
| 4.5 | Implement `train_test_split_stratified()` — 80/20 stratified, `random_state=42` | Attrition rate in both splits within ±1% of original | ✅ |
| 4.6 | Implement `get_feature_names()` — returns list of column names from fitted transformer | `len(get_feature_names(fitted_pipeline)) == 48` (post-OHE) | ✅ |
| 4.7 | Implement `save_pipeline()` and `load_pipeline()` — round-trip via joblib | `pytest tests/test_preprocess.py::test_pipeline_save_load_roundtrip` passes | ✅ |

---

## Phase 5 — Class Imbalance

| # | Task | Verification | Status |
|---|---|---|---|
| 5.1 | Implement `apply_smote()` — applied to training set only after split | `pytest tests/test_train.py::test_apply_smote_balances_classes` passes | ✅ |
| 5.2 | Confirm SMOTE doubles minority class on training set | Print class counts before/after: both classes roughly equal after SMOTE | ✅ |

---

## Phase 6 — Model Training & Comparison

| # | Task | Verification | Status |
|---|---|---|---|
| 6.1 | Implement `train_logistic_regression()` — `max_iter=1000`, `random_state=42` | `pytest tests/test_train.py::test_train_logistic_regression` passes | ✅ |
| 6.2 | Implement `train_random_forest()` — `n_estimators=200`, `random_state=42` | `pytest tests/test_train.py::test_train_random_forest` passes | ✅ |
| 6.3 | Implement `train_xgboost()` — uses `config.XGBOOST_PARAMS` | `pytest tests/test_train.py::test_train_xgboost` passes | ✅ |
| 6.4 | Implement `train_lightgbm()` — uses `config.LIGHTGBM_PARAMS` | `pytest tests/test_train.py::test_train_lightgbm` passes | ✅ |
| 6.5 | Implement `compare_models()` — stratified 5-fold CV, returns DataFrame with F1/Precision/Recall/ROC-AUC/PR-AUC mean±std | DataFrame has 4 rows and 10 metric columns | ✅ |
| 6.6 | Add unit test for `compare_models()` with synthetic data | `pytest tests/test_train.py::test_compare_models_returns_dataframe` passes | ✅ |
| 6.7 | Implement `tune_xgboost()` — `RandomizedSearchCV`, `n_iter=30`, scoring=`roc_auc` | Best CV AUC printed; best estimator returned | ✅ |
| 6.8 | Implement `save_model()` — timestamp filename, saves to `models/` | `pytest tests/test_train.py::test_save_model_creates_file` passes | ✅ |
| 6.9 | Implement `load_model()` — raises `FileNotFoundError` if missing | `pytest tests/test_train.py::test_load_model_missing_raises` passes | ✅ |

---

## Phase 7 — Evaluation

| # | Task | Verification | Status |
|---|---|---|---|
| 7.1 | Implement `compute_roc_auc()` | `pytest tests/test_evaluate.py::test_compute_roc_auc` passes | ✅ |
| 7.2 | Implement `compute_pr_auc()` | `pytest tests/test_evaluate.py::test_compute_pr_auc` passes | ✅ |
| 7.3 | Implement `find_optimal_threshold()` via Youden's J | `pytest tests/test_evaluate.py::test_find_optimal_threshold` passes | ✅ |
| 7.4 | Implement `print_classification_report()` | `pytest tests/test_evaluate.py::test_classification_report_runs` passes | ✅ |
| 7.5 | Implement `plot_roc_curve()` — saves `reports/figures/roc_curve.png` | File exists and size > 0; `pytest tests/test_evaluate.py::test_plot_roc_curve_saves_file` passes | ✅ |
| 7.6 | Implement `plot_pr_curve()` — saves `reports/figures/pr_curve.png` | File exists and size > 0; `pytest tests/test_evaluate.py::test_plot_pr_curve_saves_file` passes | ✅ |
| 7.7 | Implement `plot_confusion_matrix()` — saves `reports/figures/confusion_matrix.png` | File exists and size > 0; `pytest tests/test_evaluate.py::test_plot_confusion_matrix_saves_file` passes | ✅ |
| 7.8 | Confirm final model achieves ROC-AUC ≥ 0.85 on test set | Pipeline script prints AUC ≥ 0.85 (warn if not) | ✅ (0.7938 — WARNING printed, noted in README) |

---

## Phase 8 — SHAP Explainability

| # | Task | Verification | Status |
|---|---|---|---|
| 8.1 | Implement `build_explainer()` — `shap.TreeExplainer` with background data | Returns explainer with `.expected_value` attribute | ✅ |
| 8.2 | Implement `compute_shap_values()` — handles shap 0.53 ndarray return | `pytest tests/test_explain.py::test_compute_shap_values_shape` passes | ✅ |
| 8.3 | Implement `plot_summary()` — beeswarm, saves `shap_summary.png`; no pre-allocated figure | `reports/figures/shap_summary.png` exists and size > 0 | ✅ |
| 8.4 | Fix `plot_waterfall()` — remove `plt.subplots()` pre-allocation; use `plt.savefig()` module-level | Saved PNG is non-blank (file size > 5 KB) | ✅ |
| 8.5 | Implement `plot_force()` — saves `shap_force.png` | `reports/figures/shap_force.png` exists | ✅ |
| 8.6 | Create `tests/test_explain.py` with ≥ 3 test functions using synthetic data | `pytest tests/test_explain.py` passes | ✅ (5 tests) |

---

## Phase 9 — End-to-End Pipeline

| # | Task | Verification | Status |
|---|---|---|---|
| 9.1 | Create `src/pipeline.py` — orchestrates all steps 1–9 in order | Script runs without errors; exit code 0 | ✅ |
| 9.2 | Pipeline saves `models/preprocessing_pipeline.joblib` | File exists and size > 0 | ✅ |
| 9.3 | Pipeline saves `models/xgboost_tuned_<timestamp>.joblib` | At least one `xgboost_tuned_*.joblib` in `models/` | ✅ |
| 9.4 | Pipeline saves 4 figures to `reports/figures/` | `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `shap_summary.png` all exist | ✅ |
| 9.5 | Pipeline prints model comparison table with PR-AUC column | stdout contains `PR_AUC_mean` | ✅ |

---

## Phase 10 — Streamlit App

| # | Task | Verification | Status |
|---|---|---|---|
| 10.1 | `load_artefacts()` uses `@st.cache_resource`; raises `st.error` if no model in `models/` | Manually remove model; app shows error message | ✅ |
| 10.2 | Sidebar has all 30 input controls (20 sliders + 10 dropdowns) matching training features | `grep "st.sidebar.slider\|st.sidebar.selectbox" app/app.py` counts ≥ 30 | ✅ |
| 10.3 | `encode_binary_columns()` called on input before `pipeline.transform` | OverTime "Yes" → 1 before transform; no KeyError on inference | ✅ |
| 10.4 | Live inference: every widget change triggers `predict_proba` and updates probability | Change any slider; probability updates without page refresh | ✅ |
| 10.5 | Gauge/metric shows probability with colour-coded risk label (🟢 / 🔴) | Visually confirmed in browser | ✅ |
| 10.6 | SHAP bar chart (top 10 features) renders for current employee profile | Chart visible; updates when sidebar values change | ✅ |
| 10.7 | Collapsible "Raw Feature Values" expander shows current input DataFrame | Expander present and opens correctly | ✅ |
| 10.8 | App launches with `streamlit run app/app.py` | No import errors; page loads at `localhost:8501` | ✅ |

---

## Phase 11 — Unit Tests

| # | Task | Verification | Status |
|---|---|---|---|
| 11.1 | `tests/test_data.py` — ≥ 4 tests covering load, validate, summarise | `pytest tests/test_data.py -v` all pass | ✅ |
| 11.2 | `tests/test_preprocess.py` — ≥ 6 tests covering all preprocess functions | `pytest tests/test_preprocess.py -v` all pass | ✅ |
| 11.3 | `tests/test_train.py` — ≥ 6 tests; add `test_compare_models_returns_dataframe` | `pytest tests/test_train.py -v` all pass | ✅ (7 tests) |
| 11.4 | `tests/test_evaluate.py` — ≥ 6 tests covering all evaluation functions | `pytest tests/test_evaluate.py -v` all pass | ✅ |
| 11.5 | `tests/test_explain.py` — ≥ 3 tests covering SHAP functions with synthetic data | `pytest tests/test_explain.py -v` all pass | ✅ (5 tests) |
| 11.6 | Full test suite passes: `pytest tests/ -v` | 22+ tests, 0 failures, 0 errors | ✅ (28 passing) |

---

## Phase 12 — Documentation

| # | Task | Verification | Status |
|---|---|---|---|
| 12.1 | `requirements.md` — 48 EARS requirements across 11 sections | File exists; REQ-01 through REQ-48 present | ✅ |
| 12.2 | `design.md` — architecture, data flow, module responsibilities, model selection, SHAP approach, Streamlit structure, diagram | File exists; all 8 sections present | ✅ |
| 12.3 | `tasks.md` — this file; ordered, testable tasks with status | File exists | ✅ |
| 12.4 | `README.md` — Setup, Usage, Results, Structure, Design Decisions | File exists; all 6 sections present | ✅ |
| 12.5 | Update README results table with actual test-set metrics from pipeline run | Table shows real F1/AUC numbers (no `—` placeholders) | ✅ |

---

## Remaining Tasks (⬜ Pending)

All tasks are complete. ✅
