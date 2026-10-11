"""Unit tests for src/train.py.

All tests use synthetic datasets generated with sklearn's make_classification.
No real CSV, no files in data/ or models/ are touched.
File I/O uses pytest's ``tmp_path`` fixture exclusively.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import (
    apply_smote,
    build_candidate_pipelines,
    compare_models,
    load_best_model,
    load_model,
    run_cv,
    run_training,
    save_best_model,
    save_comparison_table,
    save_model,
    select_best_pipeline,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    tune_top_two,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def imbalanced_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a small imbalanced dataset (200 samples, 15 features, ~20% positive).

    Returns:
        Tuple (X, y) with numpy arrays.
    """
    X, y = make_classification(
        n_samples=200,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        weights=[0.80, 0.20],
        random_state=42,
    )
    return X, y


@pytest.fixture()
def balanced_dataset(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a SMOTE-balanced version of the imbalanced dataset.

    Args:
        imbalanced_dataset: Imbalanced (X, y) fixture.

    Returns:
        Balanced (X_res, y_res).
    """
    X, y = imbalanced_dataset
    return apply_smote(X, y)


@pytest.fixture()
def tiny_cv_df() -> pd.DataFrame:
    """Return a minimal comparison DataFrame for select_best_pipeline tests.

    Returns:
        DataFrame with 4 model rows and a ROC_AUC_mean column.
    """
    return pd.DataFrame(
        {
            "ROC_AUC_mean": [0.70, 0.82, 0.85, 0.78],
            "PR_AUC_mean":  [0.40, 0.55, 0.60, 0.50],
        },
        index=["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
    )


# ===========================================================================
# 1. apply_smote
# ===========================================================================


def test_apply_smote_balances_classes(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """SMOTE must produce equal class counts.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    X_res, y_res = apply_smote(X, y)
    _, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1], f"Classes not balanced: {counts}"


def test_apply_smote_returns_numpy(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """apply_smote must return numpy arrays.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    X_res, y_res = apply_smote(X, y)
    assert isinstance(X_res, np.ndarray)
    assert isinstance(y_res, np.ndarray)


def test_apply_smote_increases_minority(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Minority class count must increase after SMOTE.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    minority_before = int((y == 1).sum())
    _, y_res = apply_smote(X, y)
    minority_after = int((y_res == 1).sum())
    assert minority_after > minority_before


# ===========================================================================
# 2. build_candidate_pipelines
# ===========================================================================


def test_build_candidate_pipelines_returns_four(
) -> None:
    """build_candidate_pipelines must return exactly 4 pipelines."""
    pipelines = build_candidate_pipelines()
    assert len(pipelines) == 4


def test_build_candidate_pipelines_all_imbpipelines() -> None:
    """Every value must be an imbalanced-learn Pipeline instance."""
    pipelines = build_candidate_pipelines()
    for name, pipe in pipelines.items():
        assert isinstance(pipe, ImbPipeline), f"{name} is not an ImbPipeline"


def test_build_candidate_pipelines_have_sampler_and_classifier() -> None:
    """Each pipeline must have 'sampler' and 'classifier' steps."""
    pipelines = build_candidate_pipelines()
    for name, pipe in pipelines.items():
        steps = dict(pipe.steps)
        assert "sampler"    in steps, f"{name} missing 'sampler' step"
        assert "classifier" in steps, f"{name} missing 'classifier' step"


def test_build_candidate_pipelines_can_fit_and_predict(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every pipeline must fit and predict_proba without error.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    pipelines = build_candidate_pipelines()
    for name, pipe in pipelines.items():
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2), f"{name}: wrong proba shape"


# ===========================================================================
# 3. Individual legacy train helpers
# ===========================================================================


def test_train_logistic_regression_predict_proba(
    balanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_logistic_regression returns a model with correct predict_proba shape.

    Args:
        balanced_dataset: Balanced (X, y).
    """
    X, y = balanced_dataset
    model = train_logistic_regression(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_train_random_forest_predict_proba(
    balanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_random_forest returns a model with correct predict_proba shape.

    Args:
        balanced_dataset: Balanced (X, y).
    """
    X, y = balanced_dataset
    model = train_random_forest(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_train_xgboost_predict_proba(
    balanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_xgboost returns a model with correct predict_proba shape.

    Args:
        balanced_dataset: Balanced (X, y).
    """
    X, y = balanced_dataset
    model = train_xgboost(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_train_lightgbm_predict_proba(
    balanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_lightgbm returns a model with correct predict_proba shape.

    Args:
        balanced_dataset: Balanced (X, y).
    """
    X, y = balanced_dataset
    model = train_lightgbm(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)


# ===========================================================================
# 4. run_cv / compare_models
# ===========================================================================


def test_run_cv_returns_dataframe(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """run_cv must return a pandas DataFrame.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    result = run_cv(X, y, n_folds=2)
    assert isinstance(result, pd.DataFrame)


def test_run_cv_four_rows(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """run_cv must have exactly one row per model (4 total).

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    result = run_cv(X, y, n_folds=2)
    assert len(result) == 4, f"Expected 4 rows, got {len(result)}"


def test_run_cv_required_columns(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """run_cv DataFrame must include all required mean/std columns.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    result = run_cv(X, y, n_folds=2)
    required = [
        "F1_mean", "F1_std",
        "Precision_mean", "Precision_std",
        "Recall_mean", "Recall_std",
        "ROC_AUC_mean", "ROC_AUC_std",
        "PR_AUC_mean", "PR_AUC_std",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_run_cv_metrics_in_unit_range(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """All mean metric values must be in [0, 1].

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    result = run_cv(X, y, n_folds=2)
    mean_cols = [c for c in result.columns if c.endswith("_mean")]
    for col in mean_cols:
        assert result[col].between(0, 1).all(), (
            f"{col} has out-of-range values: {result[col].tolist()}"
        )


def test_compare_models_delegates_to_run_cv(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """compare_models (legacy) must return same schema as run_cv.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
    """
    X, y = imbalanced_dataset
    result = compare_models(X, y)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert "ROC_AUC_mean" in result.columns
    assert "PR_AUC_mean"  in result.columns


# ===========================================================================
# 5. save_comparison_table
# ===========================================================================


def test_save_comparison_table_creates_csv(
    tiny_cv_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """save_comparison_table must create a non-empty CSV file.

    Args:
        tiny_cv_df: Minimal comparison DataFrame.
        tmp_path: pytest temporary directory.
    """
    out = tmp_path / "model_comparison.csv"
    save_comparison_table(tiny_cv_df, path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_comparison_table_round_trips(
    tiny_cv_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """CSV reloaded must have same shape as the original DataFrame.

    Args:
        tiny_cv_df: Minimal comparison DataFrame.
        tmp_path: pytest temporary directory.
    """
    out = tmp_path / "model_comparison.csv"
    save_comparison_table(tiny_cv_df, path=out)
    reloaded = pd.read_csv(out, index_col=0)
    assert reloaded.shape == tiny_cv_df.shape


# ===========================================================================
# 6. tune_top_two
# ===========================================================================


def test_tune_top_two_returns_two_entries(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tiny_cv_df: pd.DataFrame,
) -> None:
    """tune_top_two must return exactly 2 entries.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tiny_cv_df: Comparison DataFrame (XGBoost + Random Forest on top).
    """
    X, y = imbalanced_dataset
    tuned = tune_top_two(X, y, comparison_df=tiny_cv_df, n_iter=3)
    assert len(tuned) == 2


def test_tune_top_two_are_imbpipelines(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tiny_cv_df: pd.DataFrame,
) -> None:
    """Tuned estimators must be ImbPipeline instances.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tiny_cv_df: Comparison DataFrame.
    """
    X, y = imbalanced_dataset
    tuned = tune_top_two(X, y, comparison_df=tiny_cv_df, n_iter=3)
    for name, (pipe, score) in tuned.items():
        assert isinstance(pipe, ImbPipeline), f"{name} is not an ImbPipeline"


def test_tune_top_two_scores_in_unit_range(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tiny_cv_df: pd.DataFrame,
) -> None:
    """Best CV AUC scores from tuning must be in (0, 1].

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tiny_cv_df: Comparison DataFrame.
    """
    X, y = imbalanced_dataset
    tuned = tune_top_two(X, y, comparison_df=tiny_cv_df, n_iter=3)
    for name, (_, score) in tuned.items():
        assert 0 < score <= 1.0, f"{name} score out of range: {score}"


def test_tune_top_two_selects_highest_auc_models(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tiny_cv_df: pd.DataFrame,
) -> None:
    """tune_top_two must select the top-2 models by ROC_AUC_mean.

    tiny_cv_df has XGBoost (0.85) and Random Forest (0.82) on top.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tiny_cv_df: Comparison DataFrame with known rankings.
    """
    X, y = imbalanced_dataset
    tuned = tune_top_two(X, y, comparison_df=tiny_cv_df, n_iter=3)
    assert set(tuned.keys()) == {"XGBoost", "Random Forest"}


# ===========================================================================
# 7. select_best_pipeline
# ===========================================================================


def test_select_best_pipeline_picks_highest_auc() -> None:
    """select_best_pipeline must return the name with the highest CV AUC."""
    from imblearn.pipeline import Pipeline as ImbP
    from sklearn.linear_model import LogisticRegression
    dummy_pipe = ImbP([("clf", LogisticRegression())])
    tuned = {
        "XGBoost":       (dummy_pipe, 0.85),
        "Random Forest": (dummy_pipe, 0.82),
    }
    best_name, _ = select_best_pipeline(tuned)
    assert best_name == "XGBoost"


def test_select_best_pipeline_returns_imbpipeline() -> None:
    """select_best_pipeline must return an ImbPipeline as the second element."""
    from imblearn.pipeline import Pipeline as ImbP
    from sklearn.linear_model import LogisticRegression
    pipe = ImbP([("clf", LogisticRegression())])
    tuned = {"ModelA": (pipe, 0.90), "ModelB": (pipe, 0.75)}
    _, best_pipe = select_best_pipeline(tuned)
    assert isinstance(best_pipe, ImbPipeline)


# ===========================================================================
# 8. save_model / load_model / save_best_model / load_best_model
# ===========================================================================


def test_save_model_creates_timestamped_file(
    balanced_dataset: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """save_model must create a file with a timestamp in its name.

    Args:
        balanced_dataset: Balanced (X, y).
        tmp_path: pytest temporary directory.
    """
    X, y = balanced_dataset
    model = train_xgboost(X, y)
    saved = save_model(model, "xgb_test", directory=tmp_path)
    assert saved.exists()
    assert "xgb_test_" in saved.name


def test_load_model_round_trips(
    balanced_dataset: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """Loaded model must be the same type and give identical predictions.

    Args:
        balanced_dataset: Balanced (X, y).
        tmp_path: pytest temporary directory.
    """
    X, y = balanced_dataset
    model = train_xgboost(X, y)
    saved = save_model(model, "xgb_rt", directory=tmp_path)
    loaded = load_model(saved)
    assert type(loaded) is type(model)
    np.testing.assert_array_equal(
        model.predict(X), loaded.predict(X)
    )


def test_load_model_raises_on_missing(tmp_path: Path) -> None:
    """load_model must raise FileNotFoundError for a non-existent path.

    Args:
        tmp_path: pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(tmp_path / "ghost.joblib")


def test_save_best_model_creates_file(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """save_best_model must create a non-empty file at the given path.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tmp_path: pytest temporary directory.
    """
    X, y = imbalanced_dataset
    pipe = build_candidate_pipelines()["XGBoost"]
    pipe.fit(X, y)
    out = tmp_path / "best_model.joblib"
    save_best_model(pipe, path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_load_best_model_round_trips(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """load_best_model must return an ImbPipeline with identical predict output.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tmp_path: pytest temporary directory.
    """
    X, y = imbalanced_dataset
    pipe = build_candidate_pipelines()["XGBoost"]
    pipe.fit(X, y)
    out = tmp_path / "best_model.joblib"
    save_best_model(pipe, path=out)
    loaded = load_best_model(path=out)
    assert isinstance(loaded, ImbPipeline)
    np.testing.assert_array_equal(
        pipe.predict(X), loaded.predict(X)
    )


def test_load_best_model_raises_on_missing(tmp_path: Path) -> None:
    """load_best_model must raise FileNotFoundError when file is absent.

    Args:
        tmp_path: pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError):
        load_best_model(path=tmp_path / "missing_best.joblib")


# ===========================================================================
# 9. run_training (integration smoke test)
# ===========================================================================


def test_run_training_returns_expected_types(
    imbalanced_dataset: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_training must return (str, ImbPipeline, DataFrame) and save files.

    Uses monkeypatch to redirect BEST_MODEL_FILE and MODEL_COMPARISON_FILE
    so no files are written to the real models/ or reports/ directories.

    Args:
        imbalanced_dataset: Imbalanced (X, y).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.train as train_module

    monkeypatch.setattr(train_module, "BEST_MODEL_FILE",      tmp_path / "best.joblib")
    monkeypatch.setattr(train_module, "MODEL_COMPARISON_FILE", tmp_path / "cmp.csv")

    X, y = imbalanced_dataset
    best_name, best_pipe, cmp_df = run_training(X, y, tune_n_iter=2)

    assert isinstance(best_name, str)
    assert isinstance(best_pipe, ImbPipeline)
    assert isinstance(cmp_df, pd.DataFrame)

    assert (tmp_path / "best.joblib").exists(), "best_model.joblib not created"
    assert (tmp_path / "cmp.csv").exists(),     "model_comparison.csv not created"
