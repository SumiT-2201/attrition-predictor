"""Central configuration for the Employee Attrition Predictor.

All file paths, directory references, and tunable constants live here.
Import from this module instead of hard-coding values anywhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
RAW_DATA_FILE: Path = DATA_RAW_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_TRAIN_FILE: Path = DATA_PROCESSED_DIR / "train.csv"
PROCESSED_TEST_FILE: Path = DATA_PROCESSED_DIR / "test.csv"
PIPELINE_FILE: Path = MODELS_DIR / "preprocessing_pipeline.joblib"
BEST_MODEL_FILE: Path = MODELS_DIR / "best_model.joblib"
MODEL_COMPARISON_FILE: Path = REPORTS_DIR / "model_comparison.csv"
METRICS_FILE: Path = REPORTS_DIR / "metrics.json"
EXPLAINER_FILE: Path = MODELS_DIR / "shap_explainer.joblib"

# ---------------------------------------------------------------------------
# Business cost ratio for threshold optimisation
# A missed leaver (false negative) is assumed to cost this many times more
# than a false alarm (false positive).  Used by find_cost_optimal_threshold.
# ---------------------------------------------------------------------------
FN_FP_COST_RATIO: float = 5.0

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Model hyperparameter defaults
# (override via CLI args or hyperparameter search in train.py)
# ---------------------------------------------------------------------------
XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,  # rough class-imbalance ratio; tuned during training
    "random_state": RANDOM_STATE,
    "eval_metric": "auc",
}

LIGHTGBM_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Attrition"
