"""End-to-end training orchestrator for the Employee Attrition Predictor.

Run from the project root:
    python src/pipeline.py

Steps executed:
  1. Load and validate raw data.
  2. Preprocess (drop leakage columns, encode binary columns, split).
  3. Build, fit, and save the ColumnTransformer preprocessing pipeline.
  4. Apply SMOTE to the training set only (no leakage).
  5. Compare four classifiers via 5-fold stratified CV.
  6. Tune XGBoost with RandomizedSearchCV.
  7. Evaluate on the held-out test set (AUC-ROC, PR-AUC, threshold, report).
  8. Compute and plot SHAP values on a test sample.
  9. Save the final model.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — set before any pyplot import.

import sys
from pathlib import Path

# Allow imports from project root when the script is run as
# ``python src/pipeline.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.data import load_raw_data, summarise, validate_schema
from src.evaluate import (
    compute_pr_auc,
    compute_roc_auc,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_classification_report,
)
from src.explain import build_explainer, compute_shap_values, plot_summary
from src.preprocess import (
    build_preprocessing_pipeline,
    drop_leakage_columns,
    encode_binary_columns,
    get_feature_names,
    save_pipeline,
    split_features_target,
    train_test_split_stratified,
)
from src.train import (
    apply_smote,
    compare_models,
    save_model,
    tune_xgboost,
)


def run_pipeline() -> None:
    """Execute the full training pipeline from raw data to saved model.

    This function is the single entry point for a complete end-to-end run:
    load, preprocess, balance, compare, tune, evaluate, explain, and persist.
    """
    # ------------------------------------------------------------------
    # 1. Load and validate
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1 — Loading raw data")
    print("=" * 60)
    raw = load_raw_data()
    validate_schema(raw)
    summarise(raw)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2 — Preprocessing")
    print("=" * 60)
    df = drop_leakage_columns(raw)
    df = encode_binary_columns(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 3. Build and fit preprocessing pipeline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3 — Fitting ColumnTransformer pipeline")
    print("=" * 60)
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_pipeline(preprocessor)
    feature_names = get_feature_names(preprocessor)
    print(f"Feature count after preprocessing: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 4. SMOTE — on training data only to prevent leakage into test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4 — Applying SMOTE")
    print("=" * 60)
    print(f"Before SMOTE — class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    X_res, y_res = apply_smote(X_train_proc, y_train)
    print(f"After  SMOTE — class counts: {dict(zip(*np.unique(y_res, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 5. Model comparison (CV)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5 — Comparing models via 5-fold stratified CV")
    print("=" * 60)
    cv_results = compare_models(X_res, y_res)
    print(cv_results.to_string())

    # ------------------------------------------------------------------
    # 6. Hyperparameter tuning (XGBoost)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6 — Tuning XGBoost with RandomizedSearchCV (n_iter=30)")
    print("=" * 60)
    best_model = tune_xgboost(X_res, y_res, n_iter=30)

    # ------------------------------------------------------------------
    # 7. Evaluate on held-out test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7 — Evaluating on held-out test set")
    print("=" * 60)
    y_prob = best_model.predict_proba(X_test_proc)[:, 1]

    auc = compute_roc_auc(y_test, y_prob)
    print(f"Test AUC-ROC: {auc:.4f}")
    if auc < 0.85:
        print("WARNING: AUC-ROC below 0.85 target.")

    threshold = find_optimal_threshold(y_test, y_prob)
    print(f"Optimal threshold (Youden J): {threshold:.4f}")

    y_pred = (y_prob >= threshold).astype(int)
    print_classification_report(y_test, y_pred)

    pr_auc = compute_pr_auc(y_test, y_prob)
    print(f"PR-AUC: {pr_auc:.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # save=True; Agg backend makes plt.show() a no-op.
    plot_roc_curve(y_test, y_prob, save=True)
    plot_confusion_matrix(y_test, y_pred, save=True)
    plot_pr_curve(y_test, y_prob, save=True)

    # ------------------------------------------------------------------
    # 8. SHAP explanations — 200-sample subset of test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 8 — Computing SHAP values")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(
        len(X_train_proc),
        size=min(200, len(X_train_proc)),
        replace=False,
    )
    background = X_train_proc[idx]

    explainer = build_explainer(best_model, background)
    # Use at most 200 test samples to keep SHAP computation fast.
    shap_sample = X_test_proc[:200]
    shap_vals = compute_shap_values(explainer, shap_sample)
    plot_summary(shap_vals, shap_sample, feature_names, save=True)
    print(f"SHAP summary saved → {FIGURES_DIR / 'shap_summary.png'}")

    # ------------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 9 — Saving model")
    print("=" * 60)
    save_model(best_model, "xgboost_tuned")
    print("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
