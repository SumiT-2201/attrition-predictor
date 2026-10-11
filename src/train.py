"""Model training, class-imbalance handling, cross-validation, and tuning.

Design
------
Class imbalance (~16 % attrition) is handled in two complementary ways:

1. **Class-weight parametrisation** — LogisticRegression and RandomForest
   receive ``class_weight="balanced"`` so the optimiser up-weights minority
   samples without generating synthetic data.

2. **SMOTE inside an imbalanced-learn Pipeline** — XGBoost and LightGBM do
   not natively support ``class_weight="balanced"`` in the sklearn sense, so
   their pipelines use ``imblearn.pipeline.Pipeline`` + ``SMOTE`` as the first
   step.  Because the pipeline is the CV estimator, SMOTE is re-fit on each
   training fold and never sees the held-out validation fold — **no leakage**.

   XGBoost also uses ``scale_pos_weight`` (majority / minority ratio) which
   provides an additional signal during tree construction.

Flow
----
1. ``build_candidate_pipelines()``  — four imblearn Pipelines, one per model.
2. ``run_cv()``                     — stratified 5-fold CV, returns comparison df.
3. ``save_comparison_table()``      — writes reports/model_comparison.csv.
4. ``tune_top_two()``               — RandomizedSearchCV on the top-2 by ROC-AUC.
5. ``select_best_pipeline()``       — picks the winner.
6. ``save_best_model()``            — persists to models/best_model.joblib.
7. ``run_training()``               — convenience entry point that runs 1-6.

Legacy single-estimator helpers (``train_xgboost``, ``train_logistic_regression``,
etc.) are retained for backwards compatibility with ``src/pipeline.py``.
"""

from __future__ import annotations

import datetime
import sys
import warnings
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when the module is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    BEST_MODEL_FILE,
    LIGHTGBM_PARAMS,
    MODEL_COMPARISON_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    XGBOOST_PARAMS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metrics evaluated during cross-validation.
# PR-AUC (average_precision) is critical for imbalanced data because the
# large true-negative pool inflates ROC-AUC, masking poor minority-class recall.
CV_SCORING: list[str] = [
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "average_precision",
]

# Number of CV folds.
N_FOLDS: int = 5

# How many random parameter settings to try per model during tuning.
TUNE_N_ITER: int = 20


# ---------------------------------------------------------------------------
# 1. Build candidate pipelines
# ---------------------------------------------------------------------------

def build_candidate_pipelines() -> dict[str, ImbPipeline]:
    """Build one imbalanced-learn Pipeline per candidate model.

    Each pipeline has two steps:
    - ``"sampler"``   — SMOTE (XGBoost / LightGBM) or PassThrough placeholder
    - ``"classifier"`` — the model itself

    For Logistic Regression and Random Forest we use ``class_weight="balanced"``
    instead of SMOTE; they still share the same Pipeline structure so the CV
    loop is uniform.  ``SMOTE`` is embedded in the pipeline so it is re-fitted
    on each CV training fold and **never** applied to the validation fold.

    Returns:
        Dict mapping model name → unfitted ``ImbPipeline``.
    """
    # XGBoost params without any deprecated keys.
    xgb_params = {k: v for k, v in XGBOOST_PARAMS.items()
                  if k != "use_label_encoder"}

    # Approximate class-imbalance ratio for scale_pos_weight (≈ 84/16 ≈ 5).
    # This is a prior estimate; the tuning search will explore other values.
    smote = SMOTE(random_state=RANDOM_STATE)

    pipelines: dict[str, ImbPipeline] = {
        # class_weight="balanced" scales loss inversely with class frequency —
        # no synthetic data needed, but the pipeline structure is uniform.
        "Logistic Regression": ImbPipeline([
            ("sampler",     "passthrough"),
            ("classifier",  LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("sampler",     "passthrough"),
            ("classifier",  RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        # SMOTE inside the pipeline prevents leakage: synthetic minority samples
        # are generated only from the CV training fold, never the validation fold.
        "XGBoost": ImbPipeline([
            ("sampler",     SMOTE(random_state=RANDOM_STATE)),
            ("classifier",  XGBClassifier(**xgb_params)),
        ]),
        "LightGBM": ImbPipeline([
            ("sampler",     SMOTE(random_state=RANDOM_STATE)),
            ("classifier",  LGBMClassifier(**LIGHTGBM_PARAMS)),
        ]),
    }
    return pipelines


# ---------------------------------------------------------------------------
# 2. Cross-validation comparison
# ---------------------------------------------------------------------------

def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pipelines: dict[str, ImbPipeline] | None = None,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Evaluate all candidate pipelines with stratified k-fold CV.

    The CV estimator is the full imbalanced-learn Pipeline, so SMOTE is
    applied inside each fold's training partition only.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Binary target vector (0 = retained, 1 = attrition).
        pipelines: Dict of name → ImbPipeline.  Defaults to
            ``build_candidate_pipelines()``.
        n_folds: Number of stratified folds.

    Returns:
        DataFrame indexed by model name with mean and std for every CV metric.
        Columns: F1, Precision, Recall, ROC_AUC, PR_AUC (mean + std each).
    """
    if pipelines is None:
        pipelines = build_candidate_pipelines()

    # StratifiedKFold preserves the minority-class ratio in every fold.
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    rows: list[dict[str, Any]] = []
    for name, pipeline in pipelines.items():
        print(f"  CV [{name}] …", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=CV_SCORING,
                n_jobs=-1,
            )
        row = {"Model": name}
        for metric, key in [
            ("F1",        "test_f1"),
            ("Precision", "test_precision"),
            ("Recall",    "test_recall"),
            ("ROC_AUC",   "test_roc_auc"),
            ("PR_AUC",    "test_average_precision"),
        ]:
            row[f"{metric}_mean"] = round(float(scores[key].mean()), 4)
            row[f"{metric}_std"]  = round(float(scores[key].std()),  4)
        rows.append(row)
        print(f"ROC-AUC {row['ROC_AUC_mean']:.4f}  PR-AUC {row['PR_AUC_mean']:.4f}")

    df = pd.DataFrame(rows).set_index("Model")
    return df


# ---------------------------------------------------------------------------
# 3. Save comparison table
# ---------------------------------------------------------------------------

def save_comparison_table(
    df: pd.DataFrame,
    path: Path | str | None = None,
) -> Path:
    """Write the CV comparison DataFrame to a CSV file.

    Args:
        df: Comparison DataFrame from ``run_cv()``.
        path: Output path. Defaults to ``config.MODEL_COMPARISON_FILE``.

    Returns:
        Resolved path where the CSV was written.
    """
    resolved = Path(path) if path else MODEL_COMPARISON_FILE
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved)
    print(f"Comparison table saved → {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# 4. Hyperparameter tuning — top-2 models
# ---------------------------------------------------------------------------

def _param_grid_for(name: str) -> dict[str, list[Any]]:
    """Return a RandomizedSearchCV parameter grid for the given model name.

    Parameter keys are prefixed with ``classifier__`` to target the Pipeline's
    classifier step.

    Args:
        name: Model name string (must match ``build_candidate_pipelines`` keys).

    Returns:
        Dict of parameter name → list of candidate values.

    Raises:
        ValueError: If ``name`` has no defined search space.
    """
    grids: dict[str, dict[str, list[Any]]] = {
        "Logistic Regression": {
            "classifier__C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver":  ["lbfgs", "saga"],
            "classifier__penalty": ["l2"],
        },
        "Random Forest": {
            "classifier__n_estimators":      [100, 200, 300, 400],
            "classifier__max_depth":         [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf":  [1, 2, 4],
            "classifier__max_features":      ["sqrt", "log2"],
        },
        "XGBoost": {
            "classifier__n_estimators":    [100, 200, 300, 500],
            "classifier__max_depth":       [3, 4, 5, 6],
            "classifier__learning_rate":   [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample":       [0.6, 0.7, 0.8, 0.9],
            "classifier__colsample_bytree":[0.6, 0.7, 0.8, 0.9],
            "classifier__scale_pos_weight":[3, 5, 7],
        },
        "LightGBM": {
            "classifier__n_estimators":  [100, 200, 300, 500],
            "classifier__max_depth":     [3, 4, 5, 6, -1],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample":     [0.6, 0.7, 0.8, 0.9],
            "classifier__num_leaves":    [20, 31, 50, 80],
        },
    }
    if name not in grids:
        raise ValueError(
            f"No search grid defined for '{name}'. "
            f"Available: {list(grids.keys())}"
        )
    return grids[name]


def tune_top_two(
    X_train: np.ndarray,
    y_train: np.ndarray,
    comparison_df: pd.DataFrame,
    pipelines: dict[str, ImbPipeline] | None = None,
    n_iter: int = TUNE_N_ITER,
) -> dict[str, tuple[ImbPipeline, float]]:
    """Run RandomizedSearchCV on the two best models ranked by ROC-AUC.

    Uses the same StratifiedKFold as the initial CV so results are comparable.
    SMOTE inside the pipeline is re-applied per fold during tuning, preserving
    the no-leakage guarantee.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Binary target vector.
        comparison_df: DataFrame returned by ``run_cv()``; used to rank models.
        pipelines: Dict of name → ImbPipeline.  Defaults to
            ``build_candidate_pipelines()``.
        n_iter: Number of random configurations to try per model.

    Returns:
        Dict mapping model name → (best fitted ImbPipeline, best CV ROC-AUC).
        Always has exactly 2 entries (the top-2 by initial CV ROC-AUC).
    """
    if pipelines is None:
        pipelines = build_candidate_pipelines()

    # Rank by CV ROC-AUC (descending) and take the top 2.
    top_two = (
        comparison_df["ROC_AUC_mean"]
        .sort_values(ascending=False)
        .head(2)
        .index.tolist()
    )
    print(f"\nTuning top-2 models: {top_two}")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tuned: dict[str, tuple[ImbPipeline, float]] = {}

    for name in top_two:
        print(f"\n  Tuning [{name}] (n_iter={n_iter}) …")
        param_grid = _param_grid_for(name)
        search = RandomizedSearchCV(
            estimator=pipelines[name],
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,           # refit best estimator on full X_train
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        best_auc = round(search.best_score_, 4)
        print(f"    Best CV ROC-AUC: {best_auc}  params: {search.best_params_}")
        tuned[name] = (search.best_estimator_, best_auc)

    return tuned


# ---------------------------------------------------------------------------
# 5. Select best pipeline
# ---------------------------------------------------------------------------

def select_best_pipeline(
    tuned: dict[str, tuple[ImbPipeline, float]],
) -> tuple[str, ImbPipeline]:
    """Pick the pipeline with the highest tuned CV ROC-AUC.

    Args:
        tuned: Dict returned by ``tune_top_two()``:
               ``{name: (fitted_pipeline, best_cv_auc)}``.

    Returns:
        Tuple ``(model_name, best_fitted_pipeline)``.
    """
    best_name = max(tuned, key=lambda k: tuned[k][1])
    best_pipeline, best_auc = tuned[best_name]
    print(f"\nBest model: {best_name}  (tuned CV ROC-AUC = {best_auc})")
    return best_name, best_pipeline


# ---------------------------------------------------------------------------
# 6. Save / load best model
# ---------------------------------------------------------------------------

def save_best_model(
    pipeline: ImbPipeline,
    path: Path | str | None = None,
) -> Path:
    """Persist the best fitted pipeline to ``models/best_model.joblib``.

    Args:
        pipeline: Fitted ImbPipeline (the overall winner).
        path: Optional override path. Defaults to ``config.BEST_MODEL_FILE``.

    Returns:
        Path to the saved file.
    """
    resolved = Path(path) if path else BEST_MODEL_FILE
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, resolved)
    print(f"Best model saved → {resolved}")
    return resolved


def save_model(
    model: object,
    name: str,
    directory: Path | None = None,
) -> Path:
    """Persist any fitted model with a timestamp-based filename.

    Kept for backwards compatibility with ``src/pipeline.py``.

    Args:
        model: Any fitted sklearn-compatible estimator.
        name: Short label (e.g. ``"xgboost_tuned"``).
        directory: Target directory; defaults to ``config.MODELS_DIR``.

    Returns:
        Path to the saved ``.joblib`` file.
    """
    target_dir = directory or MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = target_dir / f"{name}_{timestamp}.joblib"
    joblib.dump(model, filepath)
    print(f"Model saved → {filepath}")
    return filepath


def load_model(path: Path | str) -> object:
    """Load a saved model artefact from disk.

    Args:
        path: Path to the ``.joblib`` file.

    Returns:
        Deserialised estimator object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")
    return joblib.load(resolved)


def load_best_model(path: Path | str | None = None) -> ImbPipeline:
    """Load the best model pipeline from ``models/best_model.joblib``.

    Args:
        path: Optional override path. Defaults to ``config.BEST_MODEL_FILE``.

    Returns:
        Deserialised ImbPipeline.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path) if path else BEST_MODEL_FILE
    if not resolved.exists():
        raise FileNotFoundError(
            f"Best model not found: {resolved}\n"
            "Run src/train.py first to generate it."
        )
    return joblib.load(resolved)


# ---------------------------------------------------------------------------
# 7. End-to-end convenience entry point
# ---------------------------------------------------------------------------

def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_n_iter: int = TUNE_N_ITER,
) -> tuple[str, ImbPipeline, pd.DataFrame]:
    """Run the full training workflow: CV → save table → tune top-2 → save best.

    Args:
        X_train: Preprocessed training feature matrix (output of
            ``src.preprocess.fit_transform_pipeline``).
        y_train: Binary target vector (0 / 1).
        tune_n_iter: Number of random configs per model in tuning.

    Returns:
        Tuple ``(best_model_name, best_pipeline, comparison_df)``.
    """
    print("\n" + "=" * 60)
    print("Step A — Stratified 5-fold cross-validation (all 4 models)")
    print("=" * 60)
    pipelines = build_candidate_pipelines()
    comparison_df = run_cv(X_train, y_train, pipelines=pipelines)

    print("\n--- CV Comparison ---")
    print(comparison_df.to_string())
    save_comparison_table(comparison_df)

    print("\n" + "=" * 60)
    print("Step B — RandomizedSearchCV on top-2 models")
    print("=" * 60)
    # Rebuild fresh pipelines for tuning so CV pipelines remain unfitted.
    tune_pipelines = build_candidate_pipelines()
    tuned = tune_top_two(
        X_train, y_train,
        comparison_df=comparison_df,
        pipelines=tune_pipelines,
        n_iter=tune_n_iter,
    )

    print("\n" + "=" * 60)
    print("Step C — Select and save best model")
    print("=" * 60)
    best_name, best_pipeline = select_best_pipeline(tuned)
    save_best_model(best_pipeline)

    return best_name, best_pipeline, comparison_df


# ---------------------------------------------------------------------------
# Legacy single-estimator helpers (kept for src/pipeline.py compatibility)
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with SMOTE on training data only.

    Prefer embedding SMOTE in an ImbPipeline via ``build_candidate_pipelines``
    for CV use-cases.  This standalone helper is retained for
    ``src/pipeline.py``'s manual pipeline construction.

    Args:
        X_train: Training feature matrix (already preprocessed).
        y_train: Training target vector.

    Returns:
        Tuple ``(X_resampled, y_resampled)`` as numpy arrays.
    """
    # Fit SMOTE only on training data to prevent leakage into the test set.
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs: Any,
) -> LogisticRegression:
    """Fit a class-weighted Logistic Regression classifier.

    Args:
        X_train: Preprocessed feature matrix.
        y_train: Target vector.
        **kwargs: Extra kwargs forwarded to ``LogisticRegression``.

    Returns:
        Fitted ``LogisticRegression`` instance.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Fit a class-weighted Random Forest classifier.

    Args:
        X_train: Preprocessed feature matrix.
        y_train: Target vector.
        **kwargs: Extra kwargs forwarded to ``RandomForestClassifier``.

    Returns:
        Fitted ``RandomForestClassifier`` instance.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
) -> XGBClassifier:
    """Fit an XGBoost classifier.

    Args:
        X_train: Preprocessed feature matrix.
        y_train: Target vector.
        params: Optional params dict; falls back to ``config.XGBOOST_PARAMS``.

    Returns:
        Fitted ``XGBClassifier`` instance.
    """
    effective = {k: v for k, v in (params or XGBOOST_PARAMS).items()
                 if k != "use_label_encoder"}
    model = XGBClassifier(**effective)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
) -> LGBMClassifier:
    """Fit a LightGBM classifier.

    Args:
        X_train: Preprocessed feature matrix.
        y_train: Target vector.
        params: Optional params dict; falls back to ``config.LIGHTGBM_PARAMS``.

    Returns:
        Fitted ``LGBMClassifier`` instance.
    """
    model = LGBMClassifier(**(params or LIGHTGBM_PARAMS))
    model.fit(X_train, y_train)
    return model


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 30,
) -> XGBClassifier:
    """Run RandomizedSearchCV on a bare XGBClassifier (legacy helper).

    Prefer ``tune_top_two`` for pipeline-based tuning.

    Args:
        X_train: Preprocessed feature matrix.
        y_train: Target vector.
        n_iter: Number of random configurations to try.

    Returns:
        Best-fitted ``XGBClassifier``.
    """
    param_dist = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 5, 6],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9],
        "scale_pos_weight":[3, 5, 7],
    }
    base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="auc")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter,
        scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    print(f"Best CV AUC-ROC: {search.best_score_:.4f}")
    print(f"Best params    : {search.best_params_}")
    return search.best_estimator_


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """Compare all four models with 5-fold CV (legacy entry point).

    Delegates to ``run_cv`` using default pipelines.  Kept for
    backwards compatibility with existing tests.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training target vector.

    Returns:
        CV comparison DataFrame (same schema as ``run_cv``).
    """
    return run_cv(X_train, y_train)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.preprocess import run_preprocessing

    print("Loading and preprocessing data …")
    X_train, X_test, y_train, y_test, feature_names = run_preprocessing()

    best_name, best_pipeline, comparison_df = run_training(X_train, y_train)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best model : {best_name}")
    print(f"Saved to   : {BEST_MODEL_FILE}")
    print(f"Comparison : {MODEL_COMPARISON_FILE}")
    print("=" * 60)
