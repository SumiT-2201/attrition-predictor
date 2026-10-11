"""Unit tests for src/explain.py.

All tests use synthetic datasets — no real CSV, no files in models/ or
reports/ are touched.  File I/O uses pytest ``tmp_path`` or monkeypatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — must precede pyplot import.

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import (
    _is_tree_model,
    _unwrap_classifier,
    build_explainer,
    compute_shap_values,
    load_explainer,
    plot_bar,
    plot_summary,
    plot_waterfall,
    run_explanation,
    save_explainer,
    top_factors,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return synthetic (X_train, X_test, y_train, feature_names).

    Returns:
        Tuple of (X_train, X_test, y_train, feature_names).
    """
    X, y = make_classification(
        n_samples=150,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test = X[:100], X[100:]
    y_train = y[:100]
    names = [f"feat_{i}" for i in range(X.shape[1])]
    return X_train, X_test, y_train, names


@pytest.fixture()
def xgb_model(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
) -> XGBClassifier:
    """Return a fitted XGBClassifier.

    Args:
        data: Synthetic data fixture.

    Returns:
        Fitted XGBClassifier.
    """
    X_train, _, y_train, _ = data
    m = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    m.fit(X_train, y_train)
    return m


@pytest.fixture()
def lr_model(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
) -> LogisticRegression:
    """Return a fitted LogisticRegression.

    Args:
        data: Synthetic data fixture.

    Returns:
        Fitted LogisticRegression.
    """
    X_train, _, y_train, _ = data
    m = LogisticRegression(max_iter=500, random_state=42)
    m.fit(X_train, y_train)
    return m


@pytest.fixture()
def rf_model(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
) -> RandomForestClassifier:
    """Return a fitted RandomForestClassifier.

    Args:
        data: Synthetic data fixture.

    Returns:
        Fitted RandomForestClassifier.
    """
    X_train, _, y_train, _ = data
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X_train, y_train)
    return m


@pytest.fixture()
def imbpipeline_lr(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
) -> ImbPipeline:
    """Return a fitted ImbPipeline wrapping LogisticRegression (no SMOTE).

    Args:
        data: Synthetic data fixture.

    Returns:
        Fitted ImbPipeline with 'sampler'='passthrough' and 'classifier'=LR.
    """
    X_train, _, y_train, _ = data
    pipe = ImbPipeline([
        ("sampler",     "passthrough"),
        ("classifier",  LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


@pytest.fixture()
def xgb_explainer(
    xgb_model: XGBClassifier,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
):
    """Return a TreeExplainer built from the XGB model.

    Args:
        xgb_model: Fitted XGBClassifier fixture.
        data: Synthetic data fixture.

    Returns:
        Fitted shap.TreeExplainer.
    """
    X_train, _, _, _ = data
    return build_explainer(xgb_model, X_train)


@pytest.fixture()
def lr_explainer(
    lr_model: LogisticRegression,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
):
    """Return a LinearExplainer built from the LR model.

    Args:
        lr_model: Fitted LogisticRegression fixture.
        data: Synthetic data fixture.

    Returns:
        Fitted shap.LinearExplainer.
    """
    X_train, _, _, _ = data
    return build_explainer(lr_model, X_train)


# ===========================================================================
# 1. Internal helpers
# ===========================================================================


def test_is_tree_model_xgboost(xgb_model: XGBClassifier) -> None:
    """XGBClassifier must be identified as a tree model."""
    assert _is_tree_model(xgb_model) is True


def test_is_tree_model_random_forest(rf_model: RandomForestClassifier) -> None:
    """RandomForestClassifier must be identified as a tree model."""
    assert _is_tree_model(rf_model) is True


def test_is_tree_model_logistic_regression(lr_model: LogisticRegression) -> None:
    """LogisticRegression must NOT be identified as a tree model."""
    assert _is_tree_model(lr_model) is False


def test_unwrap_classifier_bare_model(xgb_model: XGBClassifier) -> None:
    """_unwrap_classifier must return the same object for a non-pipeline model."""
    assert _unwrap_classifier(xgb_model) is xgb_model


def test_unwrap_classifier_imbpipeline(imbpipeline_lr: ImbPipeline) -> None:
    """_unwrap_classifier must extract the 'classifier' step from an ImbPipeline."""
    clf = _unwrap_classifier(imbpipeline_lr)
    assert isinstance(clf, LogisticRegression)


# ===========================================================================
# 2. build_explainer
# ===========================================================================


def test_build_explainer_tree_has_expected_value(
    xgb_model: XGBClassifier,
    data: tuple,
) -> None:
    """TreeExplainer must have an expected_value attribute.

    Args:
        xgb_model: Fitted XGBClassifier.
        data: Synthetic data fixture.
    """
    X_train, _, _, _ = data
    explainer = build_explainer(xgb_model, X_train)
    assert hasattr(explainer, "expected_value")


def test_build_explainer_linear_has_expected_value(
    lr_model: LogisticRegression,
    data: tuple,
) -> None:
    """LinearExplainer must have an expected_value attribute.

    Args:
        lr_model: Fitted LogisticRegression.
        data: Synthetic data fixture.
    """
    X_train, _, _, _ = data
    explainer = build_explainer(lr_model, X_train)
    assert hasattr(explainer, "expected_value")


def test_build_explainer_from_imbpipeline(
    imbpipeline_lr: ImbPipeline,
    data: tuple,
) -> None:
    """build_explainer must work transparently with an ImbPipeline.

    Args:
        imbpipeline_lr: Fitted ImbPipeline wrapping LR.
        data: Synthetic data fixture.
    """
    X_train, _, _, _ = data
    explainer = build_explainer(imbpipeline_lr, X_train)
    assert hasattr(explainer, "expected_value")


def test_build_explainer_rf(rf_model: RandomForestClassifier, data: tuple) -> None:
    """build_explainer must produce a TreeExplainer for RandomForest.

    Args:
        rf_model: Fitted RandomForestClassifier.
        data: Synthetic data fixture.
    """
    X_train, _, _, _ = data
    import shap
    explainer = build_explainer(rf_model, X_train)
    assert isinstance(explainer, shap.TreeExplainer)


# ===========================================================================
# 3. compute_shap_values
# ===========================================================================


def test_compute_shap_values_shape_xgb(
    xgb_explainer,
    data: tuple,
) -> None:
    """SHAP values for XGB must have shape (n_test, n_features).

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, _ = data
    sv = compute_shap_values(xgb_explainer, X_test)
    assert sv.shape == X_test.shape


def test_compute_shap_values_shape_lr(
    lr_explainer,
    data: tuple,
) -> None:
    """SHAP values for LR must have shape (n_test, n_features).

    Args:
        lr_explainer: Fitted LinearExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, _ = data
    sv = compute_shap_values(lr_explainer, X_test)
    assert sv.shape == X_test.shape


def test_compute_shap_values_returns_ndarray(
    xgb_explainer,
    data: tuple,
) -> None:
    """compute_shap_values must always return a numpy ndarray.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, _ = data
    sv = compute_shap_values(xgb_explainer, X_test)
    assert isinstance(sv, np.ndarray)


def test_compute_shap_values_finite(
    xgb_explainer,
    data: tuple,
) -> None:
    """All SHAP values must be finite — no NaN or Inf allowed.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, _ = data
    sv = compute_shap_values(xgb_explainer, X_test)
    assert np.all(np.isfinite(sv)), "SHAP values contain NaN or Inf"


def test_compute_shap_values_2d(xgb_explainer, data: tuple) -> None:
    """Output must always be 2-D regardless of shap version return format.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, _ = data
    sv = compute_shap_values(xgb_explainer, X_test)
    assert sv.ndim == 2


# ===========================================================================
# 4. plot_summary
# ===========================================================================


def test_plot_summary_saves_file(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_summary must save a non-empty PNG to FIGURES_DIR.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    sv = compute_shap_values(xgb_explainer, X_test)
    mod.plot_summary(sv, X_test, names, save=True)
    out = tmp_path / "shap_summary.png"
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_summary_custom_filename(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_summary must respect a custom filename argument.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    sv = compute_shap_values(xgb_explainer, X_test)
    mod.plot_summary(sv, X_test, names, save=True, filename="custom_summary.png")
    assert (tmp_path / "custom_summary.png").exists()


def test_plot_summary_no_save_creates_no_file(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_summary must not create any file when save=False.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    sv = compute_shap_values(xgb_explainer, X_test)
    mod.plot_summary(sv, X_test, names, save=False)
    assert not (tmp_path / "shap_summary.png").exists()


# ===========================================================================
# 5. plot_bar
# ===========================================================================


def test_plot_bar_saves_file(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_bar must save a non-empty PNG to FIGURES_DIR.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    sv = compute_shap_values(xgb_explainer, X_test)
    mod.plot_bar(sv, names, save=True)
    out = tmp_path / "shap_bar.png"
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_bar_max_display_respected(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_bar must run without error when max_display < n_features.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    sv = compute_shap_values(xgb_explainer, X_test)
    # X_test has 12 features; requesting only 5 should not error.
    mod.plot_bar(sv, names, max_display=5, save=True)
    assert (tmp_path / "shap_bar.png").exists()


# ===========================================================================
# 6. plot_waterfall
# ===========================================================================


def test_plot_waterfall_saves_file(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_waterfall must save a non-empty PNG.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    mod.plot_waterfall(xgb_explainer, X_test[0:1], names, save=True)
    out = tmp_path / "shap_waterfall.png"
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_waterfall_custom_filename(
    xgb_explainer,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_waterfall must respect a custom filename.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR", tmp_path)
    _, X_test, _, names = data
    mod.plot_waterfall(xgb_explainer, X_test[0:1], names,
                       save=True, filename="wf_custom.png")
    assert (tmp_path / "wf_custom.png").exists()


# ===========================================================================
# 7. top_factors
# ===========================================================================


def test_top_factors_returns_n_entries(xgb_explainer, data: tuple) -> None:
    """top_factors must return exactly n entries.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, names = data
    factors = top_factors(xgb_explainer, X_test[0:1], names, n=5)
    assert len(factors) == 5


def test_top_factors_required_keys(xgb_explainer, data: tuple) -> None:
    """Each factor dict must contain all four required keys.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, names = data
    factors = top_factors(xgb_explainer, X_test[0:1], names, n=3)
    for f in factors:
        for key in ("feature", "shap_value", "direction", "feature_value"):
            assert key in f, f"Missing key '{key}' in factor: {f}"


def test_top_factors_direction_consistent_with_shap_sign(
    xgb_explainer, data: tuple,
) -> None:
    """'↑ risk' must correspond to positive SHAP values, '↓ risk' to negative.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, names = data
    factors = top_factors(xgb_explainer, X_test[0:1], names, n=5)
    for f in factors:
        if f["shap_value"] > 0:
            assert f["direction"] == "↑ risk", (
                f"Positive SHAP but direction='{f['direction']}'"
            )
        elif f["shap_value"] < 0:
            assert f["direction"] == "↓ risk", (
                f"Negative SHAP but direction='{f['direction']}'"
            )


def test_top_factors_sorted_by_magnitude(xgb_explainer, data: tuple) -> None:
    """Factors must be sorted by |SHAP value| descending.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, names = data
    factors = top_factors(xgb_explainer, X_test[0:1], names, n=6)
    magnitudes = [abs(f["shap_value"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_factors_features_are_valid_names(xgb_explainer, data: tuple) -> None:
    """Every feature name returned must exist in the original feature list.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, names = data
    factors = top_factors(xgb_explainer, X_test[0:1], names, n=5)
    for f in factors:
        assert f["feature"] in names, (
            f"Feature '{f['feature']}' not in feature_names"
        )


def test_top_factors_works_with_lr_explainer(lr_explainer, data: tuple) -> None:
    """top_factors must work with a LinearExplainer (LogisticRegression).

    Args:
        lr_explainer: Fitted LinearExplainer fixture.
        data: Synthetic data fixture.
    """
    _, X_test, _, names = data
    factors = top_factors(lr_explainer, X_test[0:1], names, n=3)
    assert len(factors) == 3


# ===========================================================================
# 8. save_explainer / load_explainer
# ===========================================================================


def test_save_explainer_creates_file(
    xgb_explainer, tmp_path: Path,
) -> None:
    """save_explainer must create a non-empty joblib file.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        tmp_path: pytest temporary directory.
    """
    out = tmp_path / "explainer.joblib"
    save_explainer(xgb_explainer, path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_load_explainer_round_trips(xgb_explainer, data: tuple, tmp_path: Path) -> None:
    """Loaded explainer must produce identical SHAP values to the original.

    Args:
        xgb_explainer: Fitted TreeExplainer fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
    """
    _, X_test, _, _ = data
    out = tmp_path / "explainer.joblib"
    save_explainer(xgb_explainer, path=out)
    loaded = load_explainer(path=out)
    sv_orig   = compute_shap_values(xgb_explainer, X_test)
    sv_loaded = compute_shap_values(loaded, X_test)
    np.testing.assert_array_almost_equal(sv_orig, sv_loaded)


def test_load_explainer_raises_on_missing(tmp_path: Path) -> None:
    """load_explainer must raise FileNotFoundError for non-existent path.

    Args:
        tmp_path: pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError, match="Explainer file not found"):
        load_explainer(path=tmp_path / "ghost_explainer.joblib")


# ===========================================================================
# 9. run_explanation (integration smoke test)
# ===========================================================================


def test_run_explanation_returns_explainer_and_shap(
    xgb_model: XGBClassifier,
    data: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_explanation must return (explainer, shap_values) and save artefacts.

    Args:
        xgb_model: Fitted XGBClassifier fixture.
        data: Synthetic data fixture.
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    import shap as shap_mod
    import src.explain as mod
    monkeypatch.setattr(mod, "FIGURES_DIR",    tmp_path)
    monkeypatch.setattr(mod, "EXPLAINER_FILE", tmp_path / "explainer.joblib")

    X_train, X_test, _, names = data
    explainer, sv = run_explanation(xgb_model, X_train, X_test, names)

    # Return types.
    assert isinstance(explainer, shap_mod.TreeExplainer)
    assert isinstance(sv, np.ndarray)
    assert sv.shape == X_test.shape

    # All three figures saved.
    for fname in ("shap_summary.png", "shap_bar.png", "shap_waterfall.png"):
        assert (tmp_path / fname).exists(), f"Missing: {fname}"

    # Explainer saved.
    assert (tmp_path / "explainer.joblib").exists()
