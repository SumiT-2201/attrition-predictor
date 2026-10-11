"""Unit tests for src/evaluate.py.

All tests use synthetic predictions generated in-memory.
File I/O uses pytest's ``tmp_path`` fixture — nothing is written to
``reports/`` or ``models/`` during test execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must precede pyplot import.

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (
    compute_all_metrics,
    compute_pr_auc,
    compute_roc_auc,
    find_cost_optimal_threshold,
    find_optimal_threshold,
    load_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    run_evaluation,
    save_metrics,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Return y_true and y_prob for a perfect classifier (AUC = 1.0).

    Returns:
        Tuple (y_true, y_prob) where probs are 0.0 for class 0 and 1.0 for 1.
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.05, 0.1, 0.9, 0.95, 1.0])
    return y_true, y_prob


@pytest.fixture()
def correlated_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic y_true and y_prob for general evaluation tests.

    Probabilities are correlated with labels so AUC > 0.5.

    Returns:
        Tuple (y_true, y_prob) as 1-D numpy arrays of length 200.
    """
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    noise  = rng.random(size=200) * 0.3
    y_prob = np.clip(y_true * 0.7 + noise, 0.0, 1.0)
    return y_true, y_prob


@pytest.fixture()
def imbalanced_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Return y_true / y_prob mimicking the ~16 % attrition rate.

    Returns:
        Tuple (y_true, y_prob) with ~16 % positive class.
    """
    rng = np.random.default_rng(0)
    n = 300
    y_true = (rng.random(n) < 0.16).astype(int)
    noise  = rng.random(n) * 0.4
    y_prob = np.clip(y_true * 0.65 + noise, 0.0, 1.0)
    return y_true, y_prob


# ===========================================================================
# 1. compute_roc_auc
# ===========================================================================


def test_compute_roc_auc_perfect(
    perfect_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Perfect classifier must give ROC-AUC = 1.0.

    Args:
        perfect_predictions: Perfect (y_true, y_prob).
    """
    y_true, y_prob = perfect_predictions
    assert compute_roc_auc(y_true, y_prob) == pytest.approx(1.0)


def test_compute_roc_auc_returns_float(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_roc_auc must return a Python float.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    result = compute_roc_auc(y_true, y_prob)
    assert isinstance(result, float)


def test_compute_roc_auc_in_unit_range(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """ROC-AUC must be in [0, 1].

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    assert 0.0 <= compute_roc_auc(y_true, y_prob) <= 1.0


def test_compute_roc_auc_above_random(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Correlated predictions must yield AUC > 0.5 (better than random).

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    assert compute_roc_auc(y_true, y_prob) > 0.5


# ===========================================================================
# 2. compute_pr_auc
# ===========================================================================


def test_compute_pr_auc_perfect(
    perfect_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Perfect classifier must give PR-AUC = 1.0.

    Args:
        perfect_predictions: Perfect (y_true, y_prob).
    """
    y_true, y_prob = perfect_predictions
    assert compute_pr_auc(y_true, y_prob) == pytest.approx(1.0, abs=1e-9)


def test_compute_pr_auc_in_unit_range(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """PR-AUC must be in [0, 1] (with floating-point tolerance).

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    result = compute_pr_auc(y_true, y_prob)
    assert -1e-9 <= result <= 1.0 + 1e-9


def test_compute_pr_auc_returns_float(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_pr_auc must return a Python float.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    assert isinstance(compute_pr_auc(y_true, y_prob), float)


# ===========================================================================
# 3. compute_all_metrics
# ===========================================================================


def test_compute_all_metrics_keys(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_all_metrics must return all five required keys.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
    for key in ("f1", "precision", "recall", "roc_auc", "pr_auc"):
        assert key in metrics, f"Missing key: {key}"


def test_compute_all_metrics_values_in_range(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """All metric values must be in [0, 1].

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
    for key, val in metrics.items():
        assert -1e-9 <= val <= 1.0 + 1e-9, f"{key} = {val} out of range"


def test_compute_all_metrics_low_threshold_high_recall(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """A very low threshold should yield recall close to 1.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    metrics = compute_all_metrics(y_true, y_prob, threshold=0.01)
    assert metrics["recall"] >= 0.95


def test_compute_all_metrics_high_threshold_high_precision(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """A very high threshold should yield precision close to 1 (or 0 if no positives).

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    metrics = compute_all_metrics(y_true, y_prob, threshold=0.99)
    # With a 0.99 threshold either precision is high or there are no positives.
    assert metrics["precision"] >= 0.8 or metrics["precision"] == 0.0


# ===========================================================================
# 4. find_optimal_threshold (Youden's J)
# ===========================================================================


def test_find_optimal_threshold_in_range(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """find_optimal_threshold must return a value in [0, 1].

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    t = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= t <= 1.0


def test_find_optimal_threshold_returns_float(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """find_optimal_threshold must return a Python float.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
    """
    y_true, y_prob = correlated_predictions
    assert isinstance(find_optimal_threshold(y_true, y_prob), float)


def test_find_optimal_threshold_perfect_gives_separator(
    perfect_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """For a perfect classifier the threshold should lie between 0.1 and 0.9.

    Args:
        perfect_predictions: Perfect (y_true, y_prob).
    """
    y_true, y_prob = perfect_predictions
    t = find_optimal_threshold(y_true, y_prob)
    # The separator between classes 0 (max 0.1) and 1 (min 0.9).
    assert 0.1 <= t <= 0.9


# ===========================================================================
# 5. find_cost_optimal_threshold
# ===========================================================================


def test_find_cost_optimal_threshold_in_range(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Cost-optimal threshold must be in [0, 1].

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
    """
    y_true, y_prob = imbalanced_predictions
    t = find_cost_optimal_threshold(y_true, y_prob, fn_fp_ratio=5.0)
    assert 0.0 <= t <= 1.0


def test_find_cost_optimal_threshold_higher_ratio_lower_threshold(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """A higher FN/FP cost ratio must push the threshold lower (more recall).

    With a high penalty on false negatives the model should flag positives
    more aggressively, meaning the threshold shifts downward.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
    """
    y_true, y_prob = imbalanced_predictions
    t_low_cost  = find_cost_optimal_threshold(y_true, y_prob, fn_fp_ratio=1.0)
    t_high_cost = find_cost_optimal_threshold(y_true, y_prob, fn_fp_ratio=10.0)
    # Higher penalty on FN → threshold should be equal or lower.
    assert t_high_cost <= t_low_cost + 0.05  # allow small tolerance


def test_find_cost_optimal_threshold_returns_float(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """find_cost_optimal_threshold must return a Python float.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
    """
    y_true, y_prob = imbalanced_predictions
    assert isinstance(find_cost_optimal_threshold(y_true, y_prob), float)


def test_find_cost_threshold_ratio5_recall_ge_youden(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Cost threshold (ratio=5) should give equal or higher recall than Youden.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
    """
    y_true, y_prob = imbalanced_predictions
    t_youden = find_optimal_threshold(y_true, y_prob)
    t_cost   = find_cost_optimal_threshold(y_true, y_prob, fn_fp_ratio=5.0)

    from sklearn.metrics import recall_score as rs
    recall_youden = rs(y_true, (y_prob >= t_youden).astype(int), zero_division=0)
    recall_cost   = rs(y_true, (y_prob >= t_cost).astype(int),   zero_division=0)
    # Cost threshold (more FN-averse) should achieve at least as much recall.
    assert recall_cost >= recall_youden - 0.05  # allow 5 pp tolerance


# ===========================================================================
# 6. plot_roc_curve
# ===========================================================================


def test_plot_roc_curve_saves_file(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_roc_curve must save roc_curve.png when save=True.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    plot_roc_curve(y_true, y_prob, save=True)
    assert (tmp_path / "roc_curve.png").exists()
    assert (tmp_path / "roc_curve.png").stat().st_size > 1000


def test_plot_roc_curve_with_threshold_marker(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_roc_curve must not error when a threshold marker is requested.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    t = find_optimal_threshold(y_true, y_prob)
    plot_roc_curve(y_true, y_prob, threshold=t, save=True)
    assert (tmp_path / "roc_curve.png").exists()


def test_plot_roc_curve_no_save_does_not_create_file(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_roc_curve must not write any file when save=False.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    plot_roc_curve(y_true, y_prob, save=False)
    assert not (tmp_path / "roc_curve.png").exists()


# ===========================================================================
# 7. plot_pr_curve
# ===========================================================================


def test_plot_pr_curve_saves_file(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_pr_curve must save pr_curve.png when save=True.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    plot_pr_curve(y_true, y_prob, save=True)
    assert (tmp_path / "pr_curve.png").exists()
    assert (tmp_path / "pr_curve.png").stat().st_size > 1000


def test_plot_pr_curve_with_threshold_marker(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_pr_curve must not error when a threshold marker is requested.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    plot_pr_curve(y_true, y_prob, threshold=0.4, save=True)
    assert (tmp_path / "pr_curve.png").exists()


# ===========================================================================
# 8. plot_confusion_matrix
# ===========================================================================


def test_plot_confusion_matrix_saves_file(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_confusion_matrix must save confusion_matrix.png when save=True.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, save=True)
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 1000


def test_plot_confusion_matrix_custom_filename(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_confusion_matrix must respect a custom filename argument.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, save=True, filename="cm_custom.png")
    assert (tmp_path / "cm_custom.png").exists()


def test_plot_confusion_matrix_with_threshold_in_title(
    correlated_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a threshold should not cause an error.

    Args:
        correlated_predictions: Correlated (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    y_true, y_prob = correlated_predictions
    y_pred = (y_prob >= 0.3).astype(int)
    plot_confusion_matrix(y_true, y_pred, threshold=0.3, save=True)
    assert (tmp_path / "confusion_matrix.png").exists()


# ===========================================================================
# 9. save_metrics / load_metrics
# ===========================================================================


def test_save_metrics_creates_json(tmp_path: Path) -> None:
    """save_metrics must create a non-empty JSON file."""
    data = {"f1": 0.72, "roc_auc": 0.85, "threshold": 0.3}
    out = tmp_path / "metrics.json"
    save_metrics(data, path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_metrics_valid_json(tmp_path: Path) -> None:
    """The written file must be valid, parseable JSON."""
    data = {"f1": 0.72, "roc_auc": 0.85}
    out = tmp_path / "metrics.json"
    save_metrics(data, path=out)
    with open(out) as fh:
        loaded = json.load(fh)
    assert loaded == data


def test_load_metrics_round_trips(tmp_path: Path) -> None:
    """load_metrics must recover the exact dict written by save_metrics."""
    data = {"f1": 0.65, "pr_auc": 0.42, "threshold_cost": 0.28}
    out = tmp_path / "metrics.json"
    save_metrics(data, path=out)
    recovered = load_metrics(path=out)
    assert recovered == data


def test_load_metrics_raises_on_missing(tmp_path: Path) -> None:
    """load_metrics must raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Metrics file not found"):
        load_metrics(path=tmp_path / "ghost_metrics.json")


# ===========================================================================
# 10. run_evaluation (integration smoke test)
# ===========================================================================


def test_run_evaluation_returns_dict(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_evaluation must return a dictionary.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    monkeypatch.setattr("src.evaluate.METRICS_FILE", tmp_path / "metrics.json")
    y_true, y_prob = imbalanced_predictions
    result = run_evaluation(y_true, y_prob)
    assert isinstance(result, dict)


def test_run_evaluation_required_keys(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_evaluation result must contain all required metric keys.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    monkeypatch.setattr("src.evaluate.METRICS_FILE", tmp_path / "metrics.json")
    y_true, y_prob = imbalanced_predictions
    result = run_evaluation(y_true, y_prob)
    for key in ("threshold_youden", "threshold_cost", "fn_fp_ratio",
                "f1", "precision", "recall", "roc_auc", "pr_auc"):
        assert key in result, f"Missing key: {key}"


def test_run_evaluation_saves_three_figures(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_evaluation must save roc_curve.png, pr_curve.png, confusion_matrix.png.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    monkeypatch.setattr("src.evaluate.METRICS_FILE", tmp_path / "metrics.json")
    y_true, y_prob = imbalanced_predictions
    run_evaluation(y_true, y_prob, save_figures=True, save_json=True)
    for fname in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png"):
        assert (tmp_path / fname).exists(), f"Missing figure: {fname}"
        assert (tmp_path / fname).stat().st_size > 1000


def test_run_evaluation_saves_metrics_json(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_evaluation must write a valid metrics.json file.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    monkeypatch.setattr("src.evaluate.METRICS_FILE", metrics_path)
    y_true, y_prob = imbalanced_predictions
    run_evaluation(y_true, y_prob, save_json=True)
    assert metrics_path.exists()
    with open(metrics_path) as fh:
        data = json.load(fh)
    assert "roc_auc" in data
    assert "threshold_cost" in data


def test_run_evaluation_fn_fp_ratio_respected(
    imbalanced_predictions: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fn_fp_ratio stored in metrics must match the argument passed.

    Args:
        imbalanced_predictions: Imbalanced (y_true, y_prob).
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    monkeypatch.setattr("src.evaluate.METRICS_FILE", tmp_path / "m.json")
    y_true, y_prob = imbalanced_predictions
    result = run_evaluation(y_true, y_prob, fn_fp_ratio=3.0, save_json=False)
    assert result["fn_fp_ratio"] == 3.0
