"""Feature engineering, encoding, scaling, and train/test splitting.

Pipeline steps
--------------
1. Drop constant / leakage columns (zero-variance or pure identifiers).
2. Encode binary string columns (Attrition, OverTime) as 0 / 1 integers.
3. Separate the feature matrix X from the target vector y.
4. Build a full sklearn Pipeline containing a ColumnTransformer that:
   - StandardScaler   → numerical features
   - OrdinalEncoder   → ordinal categorical features (order matters)
   - OneHotEncoder    → nominal categorical features (no inherent order)
5. Perform a stratified 80/20 train / test split.
6. Fit the Pipeline on the training set only; transform both splits.
7. Save the processed splits as CSV and the Pipeline as a joblib artefact.

All paths and constants are imported from config.py — nothing is hard-coded here.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from config import (
    DATA_PROCESSED_DIR,
    PIPELINE_FILE,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

# Zero-variance columns or pure identifiers — contain no predictive signal.
COLUMNS_TO_DROP: list[str] = [
    "EmployeeCount",   # always 1
    "Over18",          # always "Y"
    "StandardHours",   # always 80
    "EmployeeNumber",  # unique row ID — causes overfitting if kept
]

# Ordinal features: categories carry a meaningful rank (1 < 2 < 3 …).
# OrdinalEncoder preserves this ordering during encoding.
ORDINAL_FEATURES: list[str] = [
    "BusinessTravel",           # Non-Travel < Travel_Rarely < Travel_Frequently
    "Education",                # 1=Below College … 5=Doctor
    "EnvironmentSatisfaction",  # 1=Low … 4=Very High
    "JobInvolvement",           # 1=Low … 4=Very High
    "JobLevel",                 # 1=Entry … 5=C-Suite
    "JobSatisfaction",          # 1=Low … 4=Very High
    "PerformanceRating",        # 3=Excellent, 4=Outstanding
    "RelationshipSatisfaction", # 1=Low … 4=Very High
    "StockOptionLevel",         # 0=None … 3=High
    "WorkLifeBalance",          # 1=Bad … 4=Best
]

# Nominal features: categories have no inherent order → one-hot encoded.
NOMINAL_FEATURES: list[str] = [
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
]


# ---------------------------------------------------------------------------
# Step 1 — Drop leakage columns
# ---------------------------------------------------------------------------

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove constant and identifier columns that carry no predictive signal.

    Args:
        df: Raw or partially processed DataFrame.

    Returns:
        New DataFrame with the four leakage columns removed.
        Columns not present are silently ignored (errors="ignore").
    """
    return df.drop(columns=COLUMNS_TO_DROP, errors="ignore")


# ---------------------------------------------------------------------------
# Step 2 — Encode binary string columns
# ---------------------------------------------------------------------------

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No string columns to integer 0 / 1.

    Only `Attrition` and `OverTime` are binary string columns in this dataset.
    The mapping is applied in-place on a copy so the original is not mutated.

    Args:
        df: DataFrame that may contain "Attrition" and/or "OverTime" columns
            with string values "Yes" / "No".

    Returns:
        DataFrame with the binary columns replaced by integer 0 / 1 values.
    """
    binary_map: dict[str, int] = {"Yes": 1, "No": 0}
    df = df.copy()
    for col in [TARGET_COLUMN, "OverTime"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Separate features from target
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the binary target vector y.

    Args:
        df: Preprocessed DataFrame that includes the target column.

    Returns:
        Tuple ``(X, y)`` where X is the feature DataFrame and y is the
        integer-typed target Series named ``config.TARGET_COLUMN``.

    Raises:
        ValueError: If ``config.TARGET_COLUMN`` is not present in ``df``.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Step 4 — Build the sklearn Pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build an unfitted sklearn Pipeline wrapping a ColumnTransformer.

    The Pipeline contains a single ``"preprocessor"`` step that applies:
    - ``StandardScaler``  to all numerical columns
    - ``OrdinalEncoder``  to ordinal categorical columns (order preserved)
    - ``OneHotEncoder``   to nominal categorical columns (order-agnostic)

    Numerical columns are inferred as every column in X that is not listed
    in ``ORDINAL_FEATURES`` or ``NOMINAL_FEATURES``.

    Args:
        X: Feature matrix used **only** to infer numerical column names.
           The Pipeline returned is unfitted; call ``.fit(X_train)`` separately.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline`` instance ready to be fitted
        on a training set.
    """
    # Derive numerical columns: everything that is not an explicit categorical.
    numerical_features: list[str] = [
        c for c in X.columns
        if c not in ORDINAL_FEATURES and c not in NOMINAL_FEATURES
    ]

    # Only include feature groups that are actually present in X to avoid
    # ColumnTransformer errors when testing with subsets of columns.
    ordinal_present  = [c for c in ORDINAL_FEATURES  if c in X.columns]
    nominal_present  = [c for c in NOMINAL_FEATURES   if c in X.columns]

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "ord",
                # unknown_value=-1 handles unseen categories at inference time
                # without raising an error — important for the Streamlit app.
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_present,
            ),
            (
                "nom",
                # sparse_output=False returns a dense array, required by XGBoost.
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                nominal_present,
            ),
        ],
        # Drop any columns not explicitly listed above (safety net).
        remainder="drop",
    )

    # Wrap in a Pipeline so future steps (e.g. feature selection) can be
    # appended without changing the calling code.
    pipeline = Pipeline(steps=[("preprocessor", column_transformer)])
    return pipeline


# ---------------------------------------------------------------------------
# Step 5 — Stratified train / test split
# ---------------------------------------------------------------------------

def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train / test split preserving the minority class ratio.

    Stratification ensures the ~16% attrition rate is preserved in both
    the training and test partitions, preventing evaluation bias.

    Args:
        X: Feature matrix (any number of columns).
        y: Binary target vector (0 = retained, 1 = attrition).

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
    """
    # stratify=y keeps the attrition ratio consistent across both splits.
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Step 6 — Fit pipeline and transform splits
# ---------------------------------------------------------------------------

def fit_transform_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the pipeline on X_train and transform both X_train and X_test.

    Fitting on X_train only prevents any test-set statistics (mean, std,
    encoder categories) from leaking into the training distribution.

    Args:
        pipeline: Unfitted sklearn Pipeline from ``build_preprocessing_pipeline``.
        X_train:  Training feature matrix (the pipeline is fitted on this).
        X_test:   Test feature matrix (only transformed, never fitted).

    Returns:
        Tuple ``(X_train_proc, X_test_proc)`` as dense numpy arrays.
    """
    # Fit only on training data — test data must remain unseen during fitting.
    X_train_proc: np.ndarray = pipeline.fit_transform(X_train)
    X_test_proc:  np.ndarray = pipeline.transform(X_test)
    return X_train_proc, X_test_proc


# ---------------------------------------------------------------------------
# Step 7 — Persist artefacts
# ---------------------------------------------------------------------------

def save_splits(
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    train_path: str | None = None,
    test_path: str | None = None,
) -> None:
    """Save the processed train and test splits as CSV files.

    Each CSV contains the transformed feature columns (with prefixed names
    from the ColumnTransformer) plus the integer target column appended
    as the last column.

    Args:
        X_train_proc: Processed training feature matrix (numpy array).
        X_test_proc:  Processed test feature matrix (numpy array).
        y_train:      Training target Series.
        y_test:       Test target Series.
        feature_names: Column names for the processed feature matrix.
        train_path:   Optional override path for train CSV.
                      Defaults to ``config.PROCESSED_TRAIN_FILE``.
        test_path:    Optional override path for test CSV.
                      Defaults to ``config.PROCESSED_TEST_FILE``.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    resolved_train = train_path or PROCESSED_TRAIN_FILE
    resolved_test  = test_path  or PROCESSED_TEST_FILE

    train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df[TARGET_COLUMN] = y_train.values
    train_df.to_csv(resolved_train, index=False)

    test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(resolved_test, index=False)


def save_pipeline(pipeline: Pipeline, path: str | None = None) -> None:
    """Persist the fitted preprocessing Pipeline to disk with joblib.

    Args:
        pipeline: Fitted sklearn ``Pipeline`` (or ``ColumnTransformer``).
        path:     Optional override path. Defaults to ``config.PIPELINE_FILE``.
    """
    resolved = path or PIPELINE_FILE
    joblib.dump(pipeline, resolved)


def load_pipeline(path: str | None = None) -> Pipeline:
    """Load a previously saved preprocessing Pipeline from disk.

    Args:
        path: Optional override path. Defaults to ``config.PIPELINE_FILE``.

    Returns:
        Deserialised Pipeline object ready for ``transform`` calls.

    Raises:
        FileNotFoundError: If no file exists at the resolved path.
    """
    resolved = path or PIPELINE_FILE
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Pipeline file not found: {resolved}\n"
            "Run the training pipeline first to generate this artefact."
        )
    return joblib.load(resolved)


# ---------------------------------------------------------------------------
# Helper — feature names
# ---------------------------------------------------------------------------

def get_feature_names(fitted_pipeline: Pipeline | ColumnTransformer) -> list[str]:
    """Extract output feature names from a fitted Pipeline or ColumnTransformer.

    Supports both a bare ``ColumnTransformer`` and a ``Pipeline`` whose first
    step is named ``"preprocessor"`` (the shape produced by
    ``build_preprocessing_pipeline``).

    The names are prefixed by transformer name (e.g. ``num__Age``,
    ``nom__JobRole_Sales Executive``) which guarantees uniqueness.

    Args:
        fitted_pipeline: A fitted ``Pipeline`` or ``ColumnTransformer``.

    Returns:
        List of output feature name strings in transformer order.
    """
    if isinstance(fitted_pipeline, Pipeline):
        # Unwrap the ColumnTransformer from inside the Pipeline.
        ct: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]
    else:
        ct = fitted_pipeline
    return list(ct.get_feature_names_out())


# ---------------------------------------------------------------------------
# Convenience — run the full preprocessing chain end-to-end
# ---------------------------------------------------------------------------

def run_preprocessing() -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str]]:
    """Execute the full preprocessing chain on the raw IBM HR dataset.

    Loads, cleans, encodes, splits, transforms, and persists all artefacts.
    Intended as a convenience entry point for ``src/pipeline.py``.

    Returns:
        Tuple ``(X_train_proc, X_test_proc, y_train, y_test, feature_names)``.

    Raises:
        FileNotFoundError: If the raw CSV is missing.
    """
    from src.data import load_raw_data, validate_schema

    raw = load_raw_data()
    validate_schema(raw)

    df = drop_leakage_columns(raw)
    df = encode_binary_columns(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    pipeline = build_preprocessing_pipeline(X_train)
    X_train_proc, X_test_proc = fit_transform_pipeline(pipeline, X_train, X_test)

    feature_names = get_feature_names(pipeline)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_splits(X_train_proc, X_test_proc, y_train, y_test, feature_names)
    save_pipeline(pipeline)

    print(f"Train : {X_train_proc.shape}  |  attrition rate {y_train.mean()*100:.1f}%")
    print(f"Test  : {X_test_proc.shape}  |  attrition rate {y_test.mean()*100:.1f}%")
    print(f"Features : {len(feature_names)}")
    print(f"Pipeline saved → {PIPELINE_FILE}")

    return X_train_proc, X_test_proc, y_train, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing()
