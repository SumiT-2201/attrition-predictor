"""SHAP-based model explainability for the Employee Attrition Predictor.

Public API
----------
``build_explainer(model, X_background)``
    Build a SHAP explainer appropriate for the model type.
    - Tree-based models  (XGBoost, LightGBM, RandomForest)  → TreeExplainer
    - Linear models      (LogisticRegression)                → LinearExplainer
    - ImbPipeline wrappers are unwrapped automatically.

``compute_shap_values(explainer, X)``
    Return SHAP values as a 2-D ndarray (n_samples, n_features),
    normalised across shap versions (0.46–0.53+).

``plot_summary(shap_values, X, feature_names)``
    Global beeswarm summary — one dot per employee per feature,
    coloured by feature value.  Saved to ``reports/figures/shap_summary.png``.

``plot_bar(shap_values, feature_names)``
    Global bar chart of mean |SHAP| values — easier to read than the beeswarm
    for presentations.  Saved to ``reports/figures/shap_bar.png``.

``plot_waterfall(explainer, X_row, feature_names)``
    Per-employee waterfall — shows how each feature pushes a single
    prediction away from the base rate.

``top_factors(explainer, X_row, feature_names, n)``
    Return a list of the top ``n`` factors driving one employee's risk up or
    down, as human-readable dicts with keys ``feature``, ``shap_value``,
    and ``direction`` ("↑ risk" / "↓ risk").

``save_explainer(explainer)`` / ``load_explainer()``
    Persist / reload the explainer with joblib.

``run_explanation(model, X_train, X_test, feature_names)``
    End-to-end entry point: builds explainer, saves it, computes SHAP values
    on X_test, saves both plots, and prints a sample top-5 explanation.

Design notes
------------
* ``shap.TreeExplainer`` is exact and fast for gradient-boosted trees.
  No approximation error, O(TLD) per sample.
* ``shap.LinearExplainer`` is used for LogisticRegression.  It computes
  exact Shapley values via the linear formula (coefficients × feature values).
* SMOTE inside an ImbPipeline means the pipeline's ``sampler`` step is
  skipped at inference time — we unwrap the ``classifier`` step before
  passing it to SHAP so the explainer works on the same feature space as
  the preprocessed input array.
* shap 0.53 returns a plain ndarray from ``shap_values()`` for binary
  classifiers; older versions returned a list.  ``compute_shap_values``
  handles all three return shapes.
* ``plot_summary`` and ``plot_bar`` must NOT pre-allocate ``plt.subplots()``
  before calling shap plot functions — shap creates its own figure internally.
  We call ``plt.savefig()`` at module level to capture the figure shap drew.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — must precede pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EXPLAINER_FILE, FIGURES_DIR, RANDOM_STATE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_classifier(model: Any) -> Any:
    """Extract the bare classifier from an ImbPipeline / sklearn Pipeline.

    SHAP explainers must receive the classifier directly, not the whole
    pipeline, because the pipeline's transform steps operate on the raw
    DataFrame, whereas SHAP operates on the already-preprocessed array.

    Args:
        model: Any sklearn-compatible estimator or Pipeline.

    Returns:
        The ``"classifier"`` step if ``model`` is a Pipeline,
        otherwise ``model`` unchanged.
    """
    if hasattr(model, "named_steps"):
        # Support both sklearn Pipeline and imblearn Pipeline.
        if "classifier" in model.named_steps:
            return model.named_steps["classifier"]
        # Fall back to the last step for generic pipelines.
        return list(model.named_steps.values())[-1]
    return model


def _is_tree_model(clf: Any) -> bool:
    """Return True if clf is a tree-based model supported by TreeExplainer.

    Args:
        clf: A fitted sklearn-compatible classifier.

    Returns:
        True for XGBClassifier, LGBMClassifier, RandomForestClassifier, etc.
    """
    tree_types = (
        "XGBClassifier", "LGBMClassifier",
        "RandomForestClassifier", "GradientBoostingClassifier",
        "ExtraTreesClassifier", "DecisionTreeClassifier",
    )
    return type(clf).__name__ in tree_types


# ---------------------------------------------------------------------------
# 1. Build explainer
# ---------------------------------------------------------------------------

def build_explainer(
    model: Any,
    X_background: np.ndarray,
) -> shap.Explainer:
    """Build a SHAP explainer appropriate for the model type.

    Automatically selects:
    - ``shap.TreeExplainer``   for XGBoost, LightGBM, RandomForest
    - ``shap.LinearExplainer`` for LogisticRegression and other linear models

    ImbPipeline / sklearn Pipeline wrappers are transparently unwrapped so
    the explainer always receives a bare classifier.

    Args:
        model: Fitted model or Pipeline.  Pipelines are unwrapped to extract
            the ``"classifier"`` step.
        X_background: Representative sample of the training data used to
            compute the expected (base) SHAP value.  For TreeExplainer this
            is optional (can be ``None`` to use the tree's built-in mean);
            for LinearExplainer it sets the background distribution mean.

    Returns:
        Configured but not yet called SHAP explainer instance.
    """
    clf = _unwrap_classifier(model)

    if _is_tree_model(clf):
        # TreeExplainer is exact and fast for gradient-boosted / ensemble trees.
        # Passing X_background computes E[f(X)] from data rather than tree leafs.
        return shap.TreeExplainer(clf, data=X_background)

    # Linear models: exact Shapley values via coefficient × (x − E[x]).
    # interventional_perturbation gives correct values for correlated features.
    return shap.LinearExplainer(
        clf,
        masker=shap.maskers.Independent(X_background),
        feature_perturbation="interventional",
    )


# ---------------------------------------------------------------------------
# 2. Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values for a feature matrix, returning a clean 2-D ndarray.

    Handles three return formats across shap versions:
    - Plain ndarray 2-D         (shap ≥ 0.46, XGBoost binary)  → use directly
    - Plain ndarray 3-D         (n, feats, classes)             → take class 1
    - List [class_0, class_1]   (shap < 0.46)                  → take index 1
    - Explanation object        (.values attribute)             → unwrap

    Args:
        explainer: Fitted SHAP explainer.
        X: Preprocessed feature matrix (n_samples, n_features).

    Returns:
        SHAP values array of shape (n_samples, n_features) for the positive
        (attrition) class.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X)

    # Explanation object (shap ≥ 0.46 Explanation class).
    if hasattr(sv, "values"):
        vals = sv.values
        if vals.ndim == 3:
            # (n_samples, n_features, n_classes) → positive class.
            return vals[:, :, 1]
        return vals

    # List of arrays — older shap binary format.
    if isinstance(sv, list):
        return np.asarray(sv[1])

    # Plain ndarray.
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


# ---------------------------------------------------------------------------
# 3. Global summary (beeswarm)
# ---------------------------------------------------------------------------

def plot_summary(
    shap_values: np.ndarray,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    max_display: int = 20,
    save: bool = True,
    filename: str = "shap_summary.png",
) -> None:
    """Plot a SHAP beeswarm summary showing global feature importance.

    Each dot represents one employee × one feature.  Colour encodes the
    feature value (red = high, blue = low).  Features are sorted by mean |SHAP|.

    shap.summary_plot creates its own internal matplotlib figure.
    Pre-allocating ``plt.subplots()`` before calling it produces a blank
    duplicate.  We use ``plt.savefig()`` at module level to capture the
    figure shap drew.

    Args:
        shap_values: 2-D SHAP array (n_samples, n_features).
        X: Feature matrix used to colour each point.
        feature_names: Column names matching the features axis of X.
        max_display: Maximum number of features to show (sorted by importance).
        save: Whether to save the figure to ``FIGURES_DIR``.
        filename: Output filename inside ``FIGURES_DIR``.
    """
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# 4. Global bar chart (mean |SHAP|)
# ---------------------------------------------------------------------------

def plot_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    save: bool = True,
    filename: str = "shap_bar.png",
) -> None:
    """Plot a bar chart of mean absolute SHAP values (global feature importance).

    Easier to read than the beeswarm for slide decks and reports.
    Features are sorted descending by mean |SHAP|.

    Args:
        shap_values: 2-D SHAP array (n_samples, n_features).
        feature_names: Column names matching the feature axis.
        max_display: Maximum features to show.
        save: Whether to save the figure to ``FIGURES_DIR``.
        filename: Output filename inside ``FIGURES_DIR``.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Pair features with their importance and sort descending.
    pairs = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    names, vals = zip(*pairs[:max_display])
    # Reverse so the most important feature appears at the top of the chart.
    names = list(reversed(names))
    vals  = list(reversed(vals))

    fig, ax = plt.subplots(figsize=(9, max(4, max_display * 0.35)))
    bars = ax.barh(names, vals, color="#1565C0", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)",
                  fontsize=11)
    ax.set_title("Global Feature Importance — Mean |SHAP|",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.15)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Per-employee waterfall
# ---------------------------------------------------------------------------

def plot_waterfall(
    explainer: shap.Explainer,
    X_row: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    save: bool = True,
    filename: str = "shap_waterfall.png",
) -> None:
    """Plot a SHAP waterfall for a single employee prediction.

    Shows how each feature pushes the predicted probability away from the
    model's base rate (expected value).  Red bars push risk up, blue push down.

    shap.plots.waterfall creates its own figure — we must NOT pre-allocate
    a figure with plt.subplots(), or we get a blank duplicate.

    Args:
        explainer: Fitted SHAP explainer.
        X_row: Single-row feature array of shape (1, n_features).
        feature_names: Feature names aligned with X_row columns.
        max_display: Max features to display on the waterfall.
        save: Whether to save the figure to ``FIGURES_DIR``.
        filename: Output filename inside ``FIGURES_DIR``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = explainer(X_row)

    # Handle (n_samples, n_features, n_classes) → pick positive class.
    if explanation.values.ndim == 3:
        explanation = explanation[..., 1]

    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# 6. Top-N factors for a single employee
# ---------------------------------------------------------------------------

def top_factors(
    explainer: shap.Explainer,
    X_row: np.ndarray,
    feature_names: list[str],
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top ``n`` SHAP drivers for one employee's attrition risk.

    Each entry in the returned list describes a feature that most strongly
    pushes the predicted probability up or down from the base rate.

    Args:
        explainer: Fitted SHAP explainer.
        X_row: Single-row feature array of shape (1, n_features).
        feature_names: Feature names aligned with X_row columns.
        n: Number of top factors to return (default 5).

    Returns:
        List of dicts, sorted by |SHAP value| descending, each containing:
        ``feature``    — feature name string  
        ``shap_value`` — raw SHAP value (positive = increases risk,
                         negative = decreases risk)  
        ``direction``  — human-readable ``"↑ risk"`` or ``"↓ risk"``  
        ``feature_value`` — the actual feature value for this employee

    Example::

        [
            {"feature": "num__OverTime",        "shap_value":  0.312,
             "direction": "↑ risk", "feature_value": 1.22},
            {"feature": "num__MonthlyIncome",   "shap_value": -0.198,
             "direction": "↓ risk", "feature_value": -1.05},
            ...
        ]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_row)

    # Normalise to 1-D array of shape (n_features,) for the positive class.
    if isinstance(sv, list):
        row_shap = np.asarray(sv[1])[0]
    elif hasattr(sv, "values"):
        vals = sv.values
        if vals.ndim == 3:
            row_shap = vals[0, :, 1]
        else:
            row_shap = vals[0]
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            row_shap = arr[0, :, 1]
        else:
            row_shap = arr[0]

    row_features = X_row[0]   # actual feature values for this employee

    # Sort by absolute SHAP value descending.
    sorted_idx = np.argsort(np.abs(row_shap))[::-1]
    result: list[dict[str, Any]] = []
    for idx in sorted_idx[:n]:
        sv_val = float(row_shap[idx])
        result.append({
            "feature":       feature_names[idx],
            "shap_value":    round(sv_val, 4),
            "direction":     "↑ risk" if sv_val > 0 else "↓ risk",
            "feature_value": round(float(row_features[idx]), 4),
        })
    return result


# ---------------------------------------------------------------------------
# 7. Persist / reload explainer
# ---------------------------------------------------------------------------

def save_explainer(
    explainer: shap.Explainer,
    path: Path | str | None = None,
) -> Path:
    """Persist the fitted SHAP explainer to disk with joblib.

    Args:
        explainer: Fitted SHAP explainer instance.
        path: Override path. Defaults to ``config.EXPLAINER_FILE``.

    Returns:
        Path where the explainer was saved.
    """
    resolved = Path(path) if path else EXPLAINER_FILE
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, resolved)
    print(f"Explainer saved → {resolved}")
    return resolved


def load_explainer(path: Path | str | None = None) -> shap.Explainer:
    """Load a previously saved SHAP explainer from disk.

    Args:
        path: Override path. Defaults to ``config.EXPLAINER_FILE``.

    Returns:
        Deserialised SHAP explainer.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path) if path else EXPLAINER_FILE
    if not resolved.exists():
        raise FileNotFoundError(
            f"Explainer file not found: {resolved}\n"
            "Run src/explain.py first to generate it."
        )
    return joblib.load(resolved)


# ---------------------------------------------------------------------------
# 8. End-to-end entry point
# ---------------------------------------------------------------------------

def run_explanation(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    n_background: int = 200,
    sample_idx: int = 0,
) -> tuple[shap.Explainer, np.ndarray]:
    """Run the full SHAP explanation workflow.

    Steps
    -----
    1. Build the appropriate explainer using a background sample of X_train.
    2. Save the explainer to ``models/shap_explainer.joblib``.
    3. Compute SHAP values on X_test.
    4. Save global beeswarm summary to ``reports/figures/shap_summary.png``.
    5. Save global bar chart to ``reports/figures/shap_bar.png``.
    6. Save waterfall for employee at ``sample_idx`` to
       ``reports/figures/shap_waterfall.png``.
    7. Print the top-5 factors for that employee.

    Args:
        model: Fitted model or ImbPipeline.
        X_train: Preprocessed training feature matrix.
        X_test: Preprocessed test feature matrix.
        feature_names: Feature names aligned with both matrices.
        n_background: Number of training samples used as SHAP background.
            Larger values are slower but more accurate for LinearExplainer.
        sample_idx: Index of the test-set employee to explain individually.

    Returns:
        Tuple ``(explainer, shap_values)`` where ``shap_values`` has shape
        ``(len(X_test), len(feature_names))``.
    """
    print("\n" + "=" * 60)
    print("Step 1 — Building SHAP explainer")
    print("=" * 60)
    clf = _unwrap_classifier(model)
    explainer_type = "TreeExplainer" if _is_tree_model(clf) else "LinearExplainer"
    print(f"  Classifier : {type(clf).__name__}")
    print(f"  Explainer  : {explainer_type}")

    # Subsample background to keep explainer construction fast.
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)),
                        replace=False)
    X_background = X_train[bg_idx]
    explainer = build_explainer(model, X_background)

    print("\nStep 2 — Saving explainer")
    save_explainer(explainer)

    print("\nStep 3 — Computing SHAP values on test set …")
    shap_values = compute_shap_values(explainer, X_test)
    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  Features         : {shap_values.shape[1]}")
    print(f"  Mean |SHAP|      : {np.abs(shap_values).mean():.4f}")

    print("\nStep 4 — Saving global beeswarm summary plot")
    plot_summary(shap_values, X_test, feature_names, save=True)
    print(f"  → {FIGURES_DIR / 'shap_summary.png'}")

    print("\nStep 5 — Saving global bar chart")
    plot_bar(shap_values, feature_names, save=True)
    print(f"  → {FIGURES_DIR / 'shap_bar.png'}")

    print(f"\nStep 6 — Saving waterfall for employee #{sample_idx}")
    X_sample = X_test[sample_idx : sample_idx + 1]
    plot_waterfall(explainer, X_sample, feature_names, save=True)
    print(f"  → {FIGURES_DIR / 'shap_waterfall.png'}")

    print(f"\nStep 7 — Top 5 factors for employee #{sample_idx}")
    factors = top_factors(explainer, X_sample, feature_names, n=5)
    print(f"  {'Feature':<35} {'SHAP':>8}  Direction")
    print(f"  {'-'*35} {'------':>8}  ---------")
    for f in factors:
        print(f"  {f['feature']:<35} {f['shap_value']:>8.4f}  {f['direction']}")

    return explainer, shap_values


# ---------------------------------------------------------------------------
# Script entry point — verifies against saved pipeline + best model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from src.preprocess import load_pipeline, get_feature_names
    from src.train import load_best_model
    from config import PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE, TARGET_COLUMN, MODELS_DIR

    print("Loading artefacts …")

    # Prefer the XGBoost model for richer SHAP explanations; fall back to best_model.
    import glob as _glob
    xgb_files = sorted(_glob.glob(str(MODELS_DIR / "xgboost_tuned_*.joblib")))
    if xgb_files:
        model = joblib.load(xgb_files[-1])
        print(f"Using XGBoost model: {xgb_files[-1].split(chr(92))[-1]}")
    else:
        model = load_best_model()
        print("Using best_model.joblib")

    # Load processed splits — they already have the transformed feature columns.
    train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
    test_df  = pd.read_csv(PROCESSED_TEST_FILE)

    y_train = train_df[TARGET_COLUMN].values
    y_test  = test_df[TARGET_COLUMN].values
    X_train = train_df.drop(columns=[TARGET_COLUMN]).values
    X_test  = test_df.drop(columns=[TARGET_COLUMN]).values
    feature_names = [c for c in train_df.columns if c != TARGET_COLUMN]

    print(f"Train : {X_train.shape}  Test : {X_test.shape}")
    print(f"Features : {len(feature_names)}")

    # Verify the model predicts correctly on the preprocessed array.
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Sample proba[0:5]: {y_prob[:5].round(3)}")

    explainer, shap_values = run_explanation(
        model, X_train, X_test, feature_names,
        n_background=200, sample_idx=0,
    )

    print("\n✅  explain.py completed successfully.")
    print(f"   Explainer   : {EXPLAINER_FILE}")
    print(f"   SHAP summary: {FIGURES_DIR / 'shap_summary.png'}")
    print(f"   SHAP bar    : {FIGURES_DIR / 'shap_bar.png'}")
    print(f"   Waterfall   : {FIGURES_DIR / 'shap_waterfall.png'}")
