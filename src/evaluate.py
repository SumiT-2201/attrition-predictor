"""Model evaluation on the held-out test set.

Responsibilities
----------------
* Compute all scalar metrics: F1, Precision, Recall, ROC-AUC, PR-AUC.
* Find the optimal decision threshold two ways:
    1. Youden's J  — maximises (TPR − FPR), good general baseline.
    2. Business cost — minimises expected cost when a missed leaver (FN)
       costs ``FN_FP_COST_RATIO`` × more than a false alarm (FP).
       With the default ratio of 5 the cost function is:
           cost(t) = 5 × FN(t) + 1 × FP(t)
* Generate and save three figures:
    - ROC curve with AUC annotation and optimal threshold point.
    - Precision-Recall curve with AP annotation and cost-optimal point.
    - Labelled confusion matrix heatmap at the business-cost threshold.
* Serialise all scalar metrics + both thresholds to ``reports/metrics.json``.
* ``run_evaluation()`` is the single entry point that runs everything and
  returns the full metrics dict.

All file paths come from ``config.py`` — nothing is hard-coded here.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must precede pyplot import.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FIGURES_DIR,
    FN_FP_COST_RATIO,
    METRICS_FILE,
    REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Scalar metric helpers
# ---------------------------------------------------------------------------


def compute_roc_auc(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> float:
    """Compute the Area Under the ROC Curve.

    Args:
        y_true: Ground-truth binary labels (0 / 1).
        y_prob: Predicted probabilities for the positive (attrition) class.

    Returns:
        ROC-AUC score in [0, 1].
    """
    return float(roc_auc_score(y_true, y_prob))


def compute_pr_auc(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> float:
    """Compute the Average Precision (area under Precision-Recall curve).

    Average Precision is more informative than ROC-AUC for imbalanced
    datasets because it focuses on the minority class rather than the
    large true-negative pool.

    Args:
        y_true: Ground-truth binary labels (0 / 1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        PR-AUC (Average Precision) score in [0, 1].
    """
    return float(average_precision_score(y_true, y_prob))


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute F1, Precision, Recall, ROC-AUC, and PR-AUC at a given threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision threshold used to binarise ``y_prob``.

    Returns:
        Dict with keys: ``f1``, ``precision``, ``recall``,
        ``roc_auc``, ``pr_auc``.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(compute_roc_auc(y_true, y_prob), 4),
        "pr_auc":    round(compute_pr_auc(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> float:
    """Find the decision threshold that maximises Youden's J statistic.

    Youden's J = TPR − FPR.  Maximising J provides a balanced threshold
    that accounts for both sensitivity and specificity.  It is a good
    default when the relative cost of FN vs FP is unknown.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold in [0, 1].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Youden's J: maximise TPR − FPR across all candidate thresholds.
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    return float(thresholds[optimal_idx])


def find_cost_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    fn_fp_ratio: float = FN_FP_COST_RATIO,
) -> float:
    """Find the threshold that minimises expected business cost.

    The cost model assigns:
      - Cost of 1 to each False Positive  (false alarm — unnecessary HR action)
      - Cost of ``fn_fp_ratio`` to each False Negative (missed leaver)

    Total cost at threshold t:
        cost(t) = fn_fp_ratio × FN(t) + 1 × FP(t)

    With the default ``fn_fp_ratio = 5`` this means missing a real attritor
    is five times more expensive than a false alarm, so the threshold is
    shifted toward higher recall at the expense of precision.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        fn_fp_ratio: Cost multiplier for false negatives vs false positives.
            Defaults to ``config.FN_FP_COST_RATIO`` (5.0).

    Returns:
        Cost-optimal threshold in [0, 1].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    y_arr = np.asarray(y_true)
    n_pos = int(y_arr.sum())
    n_neg = int(len(y_arr) - n_pos)

    # FN(t) = n_pos × (1 − TPR(t))
    # FP(t) = n_neg × FPR(t)
    fn_cost = fn_fp_ratio * n_pos * (1.0 - tpr)
    fp_cost = 1.0           * n_neg * fpr
    total_cost = fn_cost + fp_cost

    optimal_idx = int(np.argmin(total_cost))
    return float(thresholds[optimal_idx])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float | None = None,
    save: bool = True,
    filename: str = "roc_curve.png",
) -> None:
    """Plot the ROC curve, optionally marking a chosen threshold point.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: If provided, marks the (FPR, TPR) point on the curve
            corresponding to this threshold.
        save: Whether to save the figure to ``reports/figures/``.
        filename: Output filename inside ``FIGURES_DIR``.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2,
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

    if threshold is not None:
        # Find the ROC point closest to the requested threshold.
        # thresholds from roc_curve are in descending order; locate nearest.
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(
            fpr[idx], tpr[idx],
            color="#E53935", zorder=5, s=80,
            label=f"Threshold = {threshold:.3f}",
        )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Employee Attrition", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float | None = None,
    save: bool = True,
    filename: str = "pr_curve.png",
) -> None:
    """Plot the Precision-Recall curve, optionally marking a threshold point.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: If provided, marks the operating point on the curve.
        save: Whether to save the figure to ``reports/figures/``.
        filename: Output filename inside ``FIGURES_DIR``.
    """
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
        y_true, y_prob
    )
    ap = average_precision_score(y_true, y_prob)
    # Baseline = random classifier precision = prevalence rate.
    baseline = float(np.asarray(y_true).mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, color="#2E7D32", linewidth=2,
            label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1,
               label=f"Random baseline ({baseline:.2f})")

    if threshold is not None:
        # pr_thresholds has one fewer element than precision/recall arrays.
        idx = int(np.argmin(np.abs(pr_thresholds - threshold)))
        ax.scatter(
            recall_vals[idx], precision_vals[idx],
            color="#E53935", zorder=5, s=80,
            label=f"Threshold = {threshold:.3f}",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Employee Attrition",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    threshold: float | None = None,
    save: bool = True,
    filename: str = "confusion_matrix.png",
) -> None:
    """Plot a labelled confusion matrix heatmap with count and percentage cells.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (already thresholded).
        threshold: If provided, annotates the title with the threshold used.
        save: Whether to save the figure to ``reports/figures/``.
        filename: Output filename inside ``FIGURES_DIR``.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100
    labels = ["Retained", "Attrition"]

    # Build annotation strings: "N\n(P%)" in each cell.
    annot = np.array([
        [f"{cm[r, c]}\n({cm_pct[r, c]:.1f}%)"
         for c in range(cm.shape[1])]
        for r in range(cm.shape[0])
    ])

    title = "Confusion Matrix"
    if threshold is not None:
        title += f"  (threshold = {threshold:.3f})"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Classification report (stdout)
# ---------------------------------------------------------------------------


def print_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Print a full sklearn classification report to stdout.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
    """
    print(classification_report(
        y_true, y_pred,
        target_names=["Retained (0)", "Attrition (1)"],
    ))


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------


def save_metrics(
    metrics: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Serialise the metrics dictionary to a JSON file.

    Args:
        metrics: Dict of metric name → scalar value (float or str).
        path: Output path. Defaults to ``config.METRICS_FILE``.

    Returns:
        Resolved path where the file was written.
    """
    resolved = Path(path) if path else METRICS_FILE
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved → {resolved}")
    return resolved


def load_metrics(path: Path | str | None = None) -> dict[str, Any]:
    """Load a previously saved metrics JSON file.

    Args:
        path: Path to the JSON file. Defaults to ``config.METRICS_FILE``.

    Returns:
        Dict of metric name → value.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path) if path else METRICS_FILE
    if not resolved.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {resolved}\n"
            "Run src/evaluate.py first to generate it."
        )
    with open(resolved, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# End-to-end evaluation entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    fn_fp_ratio: float = FN_FP_COST_RATIO,
    save_figures: bool = True,
    save_json: bool = True,
    metrics_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the full evaluation suite on the held-out test set.

    Steps
    -----
    1. Find the Youden-J optimal threshold.
    2. Find the business cost-optimal threshold (``fn_fp_ratio`` × FN cost).
    3. Compute all scalar metrics at the cost-optimal threshold.
    4. Plot and save: ROC curve, PR curve, confusion matrix.
    5. Save all metrics + both thresholds to ``reports/metrics.json``.

    Args:
        y_true: Ground-truth binary labels on the held-out test set.
        y_prob: Predicted probabilities for the positive class.
        fn_fp_ratio: False-negative to false-positive cost multiplier.
            Defaults to ``config.FN_FP_COST_RATIO`` (5.0).
        save_figures: Whether to persist the three figure files.
        save_json: Whether to persist ``reports/metrics.json``.
        metrics_path: Optional override path for the JSON output.

    Returns:
        Dict with keys:
            ``threshold_youden``, ``threshold_cost``, ``fn_fp_ratio``,
            ``f1``, ``precision``, ``recall``, ``roc_auc``, ``pr_auc``,
            plus ``@default_0.5`` sub-dict with metrics at 0.5.
    """
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_prob)

    # ---- 1. Thresholds ----
    t_youden = find_optimal_threshold(y_arr, p_arr)
    t_cost   = find_cost_optimal_threshold(y_arr, p_arr, fn_fp_ratio=fn_fp_ratio)

    # ---- 2. Metrics at cost-optimal threshold (primary reporting threshold) ----
    metrics_cost    = compute_all_metrics(y_arr, p_arr, threshold=t_cost)
    metrics_default = compute_all_metrics(y_arr, p_arr, threshold=0.5)
    metrics_youden  = compute_all_metrics(y_arr, p_arr, threshold=t_youden)

    # ---- 3. Print summary ----
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Youden threshold   : {t_youden:.4f}")
    print(f"  Cost threshold     : {t_cost:.4f}  "
          f"(FN:FP cost ratio = {fn_fp_ratio}:1)")
    print()
    print("  --- Metrics at cost-optimal threshold ---")
    for k, v in metrics_cost.items():
        print(f"    {k:<12}: {v:.4f}")
    print()
    print("  --- Classification report at cost threshold ---")
    y_pred_cost = (p_arr >= t_cost).astype(int)
    print_classification_report(y_arr, y_pred_cost)

    if metrics_cost["roc_auc"] < 0.85:
        print("  ⚠  WARNING: ROC-AUC is below the 0.85 target.")

    # ---- 4. Figures ----
    plot_roc_curve(y_arr, p_arr, threshold=t_cost,
                   save=save_figures, filename="roc_curve.png")
    plot_pr_curve(y_arr, p_arr,  threshold=t_cost,
                  save=save_figures, filename="pr_curve.png")
    plot_confusion_matrix(y_arr, y_pred_cost, threshold=t_cost,
                          save=save_figures, filename="confusion_matrix.png")

    if save_figures:
        print(f"\n  Figures saved to: {FIGURES_DIR}")

    # ---- 5. Assemble full metrics dict ----
    full_metrics: dict[str, Any] = {
        "threshold_youden":  round(t_youden, 4),
        "threshold_cost":    round(t_cost, 4),
        "fn_fp_ratio":       fn_fp_ratio,
        **metrics_cost,
        "@threshold_youden": metrics_youden,
        "@default_0.5":      metrics_default,
    }

    # ---- 6. Persist ----
    if save_json:
        save_metrics(full_metrics, path=metrics_path)

    return full_metrics


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    from src.preprocess import load_pipeline, get_feature_names
    from src.train import load_best_model
    from config import PROCESSED_TEST_FILE, TARGET_COLUMN

    print("Loading test set and model artefacts …")
    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    y_test  = test_df[TARGET_COLUMN].values
    X_test  = test_df.drop(columns=[TARGET_COLUMN]).values

    model = load_best_model()
    y_prob = model.predict_proba(X_test)[:, 1]

    run_evaluation(y_test, y_prob)
