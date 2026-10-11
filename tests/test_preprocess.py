"""Unit tests for src/preprocess.py.

All tests use synthetic DataFrames — no dependency on the real CSV or
on any file in data/ or models/. File I/O uses pytest's ``tmp_path`` fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ensure the project root is importable so config.py and src/ resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    COLUMNS_TO_DROP,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    build_preprocessing_pipeline,
    drop_leakage_columns,
    encode_binary_columns,
    fit_transform_pipeline,
    get_feature_names,
    load_pipeline,
    run_preprocessing,
    save_pipeline,
    save_splits,
    split_features_target,
    train_test_split_stratified,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Return a 60-row synthetic DataFrame mirroring the IBM HR schema.

    Includes leakage columns (to be dropped), all required feature columns,
    and the Attrition target with 30 Yes / 30 No rows to support stratified
    splitting in downstream fixtures.

    Returns:
        Synthetic DataFrame ready for full preprocessing.
    """
    rng = np.random.default_rng(42)
    n = 60

    return pd.DataFrame(
        {
            # --- Leakage columns ---
            "EmployeeCount":  [1] * n,
            "EmployeeNumber": rng.integers(1, 9999, n),
            "Over18":         ["Y"] * n,
            "StandardHours":  [80] * n,
            # --- Numerical features ---
            "Age":                     rng.integers(18, 60, n),
            "DailyRate":               rng.integers(100, 1500, n),
            "DistanceFromHome":        rng.integers(1, 29, n),
            "HourlyRate":              rng.integers(30, 100, n),
            "MonthlyIncome":           rng.integers(1000, 20000, n),
            "MonthlyRate":             rng.integers(2000, 27000, n),
            "NumCompaniesWorked":      rng.integers(0, 10, n),
            "PercentSalaryHike":       rng.integers(11, 26, n),
            "TotalWorkingYears":       rng.integers(0, 40, n),
            "TrainingTimesLastYear":   rng.integers(0, 7, n),
            "YearsAtCompany":          rng.integers(0, 40, n),
            "YearsInCurrentRole":      rng.integers(0, 18, n),
            "YearsSinceLastPromotion": rng.integers(0, 15, n),
            "YearsWithCurrManager":    rng.integers(0, 17, n),
            # --- Ordinal features ---
            "BusinessTravel": rng.choice(
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n
            ),
            "Education":                rng.integers(1, 6, n),
            "EnvironmentSatisfaction":  rng.integers(1, 5, n),
            "JobInvolvement":           rng.integers(1, 5, n),
            "JobLevel":                 rng.integers(1, 6, n),
            "JobSatisfaction":          rng.integers(1, 5, n),
            "PerformanceRating":        rng.choice([3, 4], n),
            "RelationshipSatisfaction": rng.integers(1, 5, n),
            "StockOptionLevel":         rng.integers(0, 4, n),
            "WorkLifeBalance":          rng.integers(1, 5, n),
            # --- Nominal features ---
            "Department": rng.choice(
                ["Human Resources", "Research & Development", "Sales"], n
            ),
            "EducationField": rng.choice(
                ["Human Resources", "Life Sciences", "Marketing",
                 "Medical", "Other", "Technical Degree"], n
            ),
            "Gender":   rng.choice(["Female", "Male"], n),
            "JobRole":  rng.choice(
                ["Healthcare Representative", "Human Resources",
                 "Laboratory Technician", "Manager",
                 "Manufacturing Director", "Research Director",
                 "Research Scientist", "Sales Executive",
                 "Sales Representative"], n
            ),
            "MaritalStatus": rng.choice(["Divorced", "Married", "Single"], n),
            # --- Binary columns ---
            # Alternating Yes/No ensures encode_binary_columns covers both values.
            "OverTime":  (["Yes"] * 30 + ["No"] * 30),
            # Equal class split so stratified splitting never fails on small data.
            "Attrition": (["Yes"] * 30 + ["No"] * 30),
        }
    )


@pytest.fixture()
def clean_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Return the synthetic DataFrame after dropping leakage and encoding binary cols.

    Args:
        synthetic_df: Raw synthetic DataFrame.

    Returns:
        Cleaned DataFrame ready for split_features_target.
    """
    df = drop_leakage_columns(synthetic_df)
    return encode_binary_columns(df)


@pytest.fixture()
def X_y(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from the cleaned DataFrame.

    Args:
        clean_df: Cleaned DataFrame fixture.

    Returns:
        Tuple (X, y).
    """
    return split_features_target(clean_df)


@pytest.fixture()
def fitted_pipeline(X_y: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    """Return a Pipeline fitted on the full feature matrix X.

    Args:
        X_y: (X, y) fixture.

    Returns:
        Fitted sklearn Pipeline.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    pipeline.fit(X)
    return pipeline


# ===========================================================================
# 1. drop_leakage_columns
# ===========================================================================


def test_drop_leakage_columns_removes_all_four(synthetic_df: pd.DataFrame) -> None:
    """All four leakage column names must be absent after dropping.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    result = drop_leakage_columns(synthetic_df)
    for col in COLUMNS_TO_DROP:
        assert col not in result.columns, f"Leakage column still present: {col}"


def test_drop_leakage_columns_preserves_other_columns(
    synthetic_df: pd.DataFrame,
) -> None:
    """Non-leakage columns must be untouched after dropping.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    result = drop_leakage_columns(synthetic_df)
    kept = set(synthetic_df.columns) - set(COLUMNS_TO_DROP)
    assert kept == set(result.columns)


def test_drop_leakage_columns_idempotent(synthetic_df: pd.DataFrame) -> None:
    """Calling drop_leakage_columns twice gives the same result as once.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    once  = drop_leakage_columns(synthetic_df)
    twice = drop_leakage_columns(once)
    assert list(once.columns) == list(twice.columns)


# ===========================================================================
# 2. encode_binary_columns
# ===========================================================================


def test_encode_binary_attrition_dtype(synthetic_df: pd.DataFrame) -> None:
    """Attrition column must be integer dtype after encoding.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    result = encode_binary_columns(synthetic_df)
    assert pd.api.types.is_integer_dtype(result["Attrition"])


def test_encode_binary_overtime_dtype(synthetic_df: pd.DataFrame) -> None:
    """OverTime column must be integer dtype after encoding.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    result = encode_binary_columns(synthetic_df)
    assert pd.api.types.is_integer_dtype(result["OverTime"])


def test_encode_binary_values_are_zero_or_one(synthetic_df: pd.DataFrame) -> None:
    """Both binary columns must contain only 0 and 1 after encoding.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    result = encode_binary_columns(synthetic_df)
    assert set(result["Attrition"].unique()).issubset({0, 1})
    assert set(result["OverTime"].unique()).issubset({0, 1})


def test_encode_binary_does_not_mutate_original(synthetic_df: pd.DataFrame) -> None:
    """encode_binary_columns must not mutate the input DataFrame.

    Args:
        synthetic_df: Raw synthetic DataFrame.
    """
    original_attrition = synthetic_df["Attrition"].copy()
    encode_binary_columns(synthetic_df)
    assert (synthetic_df["Attrition"] == original_attrition).all()


def test_encode_binary_missing_columns_is_noop() -> None:
    """If neither Attrition nor OverTime is present, the DataFrame is unchanged."""
    df = pd.DataFrame({"Age": [30, 40], "MonthlyIncome": [5000, 8000]})
    result = encode_binary_columns(df)
    assert list(result.columns) == ["Age", "MonthlyIncome"]


# ===========================================================================
# 3. split_features_target
# ===========================================================================


def test_split_features_target_removes_target(clean_df: pd.DataFrame) -> None:
    """Attrition must not appear in X after splitting.

    Args:
        clean_df: Cleaned DataFrame fixture.
    """
    X, _ = split_features_target(clean_df)
    assert "Attrition" not in X.columns


def test_split_features_target_y_named_attrition(clean_df: pd.DataFrame) -> None:
    """The returned y Series must be named 'Attrition'.

    Args:
        clean_df: Cleaned DataFrame fixture.
    """
    _, y = split_features_target(clean_df)
    assert y.name == "Attrition"


def test_split_features_target_lengths_match(clean_df: pd.DataFrame) -> None:
    """X and y must have the same number of rows.

    Args:
        clean_df: Cleaned DataFrame fixture.
    """
    X, y = split_features_target(clean_df)
    assert len(X) == len(y) == len(clean_df)


def test_split_features_target_raises_on_missing_target(
    clean_df: pd.DataFrame,
) -> None:
    """ValueError must be raised when the target column is absent.

    Args:
        clean_df: Cleaned DataFrame fixture.
    """
    df_no_target = clean_df.drop(columns=["Attrition"])
    with pytest.raises(ValueError, match="Attrition"):
        split_features_target(df_no_target)


def test_split_features_target_y_is_integer(clean_df: pd.DataFrame) -> None:
    """y must be integer typed (0/1) after splitting.

    Args:
        clean_df: Cleaned DataFrame fixture.
    """
    _, y = split_features_target(clean_df)
    assert pd.api.types.is_integer_dtype(y)


# ===========================================================================
# 4. build_preprocessing_pipeline
# ===========================================================================


def test_build_pipeline_returns_pipeline_instance(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """build_preprocessing_pipeline must return a sklearn Pipeline.

    Args:
        X_y: (X, y) fixture.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_contains_preprocessor_step(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The Pipeline must have a step named 'preprocessor'.

    Args:
        X_y: (X, y) fixture.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    assert "preprocessor" in pipeline.named_steps


def test_build_pipeline_preprocessor_is_column_transformer(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The 'preprocessor' step must be a ColumnTransformer.

    Args:
        X_y: (X, y) fixture.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    ct = pipeline.named_steps["preprocessor"]
    assert isinstance(ct, ColumnTransformer)


def test_build_pipeline_transform_output_shape(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fitted Pipeline transform must return array with correct row count.

    Args:
        X_y: (X, y) fixture.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    X_proc = pipeline.fit_transform(X)
    assert X_proc.shape[0] == len(X)


def test_build_pipeline_output_is_numeric(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """All values in the transformed array must be finite floats.

    Args:
        X_y: (X, y) fixture.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    X_proc = pipeline.fit_transform(X)
    assert np.all(np.isfinite(X_proc)), "Transformed output contains NaN or Inf"


# ===========================================================================
# 5. train_test_split_stratified
# ===========================================================================


def test_stratified_split_sizes(X_y: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test set should be ~20% of the total dataset size.

    Args:
        X_y: (X, y) fixture.
    """
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.20) < 0.05


def test_stratified_split_preserves_class_ratio(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Attrition rate in train and test should be within 3 pp of the original.

    Args:
        X_y: (X, y) fixture.
    """
    X, y = X_y
    _, _, y_train, y_test = train_test_split_stratified(X, y)
    overall_rate = y.mean()
    assert abs(y_train.mean() - overall_rate) < 0.03
    assert abs(y_test.mean()  - overall_rate) < 0.03


def test_stratified_split_no_overlap(X_y: tuple[pd.DataFrame, pd.Series]) -> None:
    """Train and test index sets must be disjoint.

    Args:
        X_y: (X, y) fixture.
    """
    X, y = X_y
    X_train, X_test, _, _ = train_test_split_stratified(X, y)
    assert len(set(X_train.index) & set(X_test.index)) == 0


# ===========================================================================
# 6. fit_transform_pipeline
# ===========================================================================


def test_fit_transform_pipeline_returns_two_arrays(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """fit_transform_pipeline returns a tuple of two numpy arrays.

    Args:
        X_y: (X, y) fixture.
    """
    X, y = X_y
    X_train, X_test, y_train, _ = train_test_split_stratified(X, y)
    pipeline = build_preprocessing_pipeline(X_train)
    X_tr, X_te = fit_transform_pipeline(pipeline, X_train, X_test)
    assert isinstance(X_tr, np.ndarray)
    assert isinstance(X_te, np.ndarray)


def test_fit_transform_pipeline_row_counts(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Transformed arrays must have same row counts as their inputs.

    Args:
        X_y: (X, y) fixture.
    """
    X, y = X_y
    X_train, X_test, _, _ = train_test_split_stratified(X, y)
    pipeline = build_preprocessing_pipeline(X_train)
    X_tr, X_te = fit_transform_pipeline(pipeline, X_train, X_test)
    assert X_tr.shape[0] == len(X_train)
    assert X_te.shape[0] == len(X_test)


def test_fit_transform_pipeline_same_feature_count(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Train and test transformed arrays must have the same number of features.

    Args:
        X_y: (X, y) fixture.
    """
    X, y = X_y
    X_train, X_test, _, _ = train_test_split_stratified(X, y)
    pipeline = build_preprocessing_pipeline(X_train)
    X_tr, X_te = fit_transform_pipeline(pipeline, X_train, X_test)
    assert X_tr.shape[1] == X_te.shape[1]


# ===========================================================================
# 7. get_feature_names
# ===========================================================================


def test_get_feature_names_returns_list(fitted_pipeline: Pipeline) -> None:
    """get_feature_names must return a plain Python list.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
    """
    names = get_feature_names(fitted_pipeline)
    assert isinstance(names, list)


def test_get_feature_names_all_strings(fitted_pipeline: Pipeline) -> None:
    """Every element of the feature names list must be a string.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
    """
    names = get_feature_names(fitted_pipeline)
    assert all(isinstance(n, str) for n in names), "Non-string feature name found"


def test_get_feature_names_non_empty(fitted_pipeline: Pipeline) -> None:
    """Feature names list must not be empty after fitting.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
    """
    names = get_feature_names(fitted_pipeline)
    assert len(names) > 0


def test_get_feature_names_count_matches_transform_width(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Number of feature names must equal the number of columns after transform.

    Args:
        X_y: (X, y) fixture.
    """
    X, _ = X_y
    pipeline = build_preprocessing_pipeline(X)
    X_proc = pipeline.fit_transform(X)
    names = get_feature_names(pipeline)
    assert len(names) == X_proc.shape[1]


def test_get_feature_names_prefixed(fitted_pipeline: Pipeline) -> None:
    """Feature names from the ColumnTransformer must include step prefixes.

    The ColumnTransformer produces names like ``num__Age``, ``ord__Education``,
    ``nom__Department_Sales``.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
    """
    names = get_feature_names(fitted_pipeline)
    prefixes = {"num__", "ord__", "nom__"}
    assert any(any(n.startswith(p) for p in prefixes) for n in names)


def test_get_feature_names_unique(fitted_pipeline: Pipeline) -> None:
    """All feature names must be unique — no duplicates allowed.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
    """
    names = get_feature_names(fitted_pipeline)
    assert len(names) == len(set(names)), "Duplicate feature names detected"


# ===========================================================================
# 8. save_pipeline / load_pipeline
# ===========================================================================


def test_save_pipeline_creates_file(
    fitted_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """save_pipeline must create a non-empty file at the given path.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
        tmp_path: pytest temporary directory.
    """
    out = tmp_path / "test_pipeline.joblib"
    save_pipeline(fitted_pipeline, path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_load_pipeline_returns_pipeline(
    fitted_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """Loaded pipeline must be a sklearn Pipeline instance.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
        tmp_path: pytest temporary directory.
    """
    out = tmp_path / "test_pipeline.joblib"
    save_pipeline(fitted_pipeline, path=str(out))
    loaded = load_pipeline(path=str(out))
    assert isinstance(loaded, Pipeline)


def test_load_pipeline_transform_matches(
    fitted_pipeline: Pipeline,
    X_y: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Loaded pipeline must produce identical output to the original.

    Args:
        fitted_pipeline: Fitted Pipeline fixture.
        X_y: (X, y) fixture.
        tmp_path: pytest temporary directory.
    """
    X, _ = X_y
    out = tmp_path / "test_pipeline.joblib"
    save_pipeline(fitted_pipeline, path=str(out))
    loaded = load_pipeline(path=str(out))

    original = fitted_pipeline.transform(X)
    reloaded = loaded.transform(X)
    np.testing.assert_array_almost_equal(original, reloaded)


def test_load_pipeline_raises_on_missing_file(tmp_path: Path) -> None:
    """load_pipeline must raise FileNotFoundError for a non-existent path.

    Args:
        tmp_path: pytest temporary directory.
    """
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="Pipeline file not found"):
        load_pipeline(path=str(missing))


# ===========================================================================
# 9. save_splits
# ===========================================================================


def test_save_splits_creates_both_files(
    X_y: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """save_splits must create both train.csv and test.csv.

    Args:
        X_y: (X, y) fixture.
        tmp_path: pytest temporary directory.
    """
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    pipeline = build_preprocessing_pipeline(X_train)
    X_tr, X_te = fit_transform_pipeline(pipeline, X_train, X_test)
    feature_names = get_feature_names(pipeline)

    train_path = tmp_path / "train.csv"
    test_path  = tmp_path / "test.csv"
    save_splits(X_tr, X_te, y_train, y_test, feature_names,
                train_path=str(train_path), test_path=str(test_path))

    assert train_path.exists()
    assert test_path.exists()


def test_save_splits_correct_row_counts(
    X_y: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Saved CSVs must have the same row counts as the input arrays.

    Args:
        X_y: (X, y) fixture.
        tmp_path: pytest temporary directory.
    """
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    pipeline = build_preprocessing_pipeline(X_train)
    X_tr, X_te = fit_transform_pipeline(pipeline, X_train, X_test)
    feature_names = get_feature_names(pipeline)

    train_path = tmp_path / "train.csv"
    test_path  = tmp_path / "test.csv"
    save_splits(X_tr, X_te, y_train, y_test, feature_names,
                train_path=str(train_path), test_path=str(test_path))

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    assert len(train_df) == len(X_train)
    assert len(test_df)  == len(X_test)


def test_save_splits_contains_target_column(
    X_y: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Both saved CSVs must include the Attrition target column.

    Args:
        X_y: (X, y) fixture.
        tmp_path: pytest temporary directory.
    """
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    pipeline = build_preprocessing_pipeline(X_train)
    X_tr, X_te = fit_transform_pipeline(pipeline, X_train, X_test)
    feature_names = get_feature_names(pipeline)

    train_path = tmp_path / "train.csv"
    test_path  = tmp_path / "test.csv"
    save_splits(X_tr, X_te, y_train, y_test, feature_names,
                train_path=str(train_path), test_path=str(test_path))

    assert "Attrition" in pd.read_csv(train_path).columns
    assert "Attrition" in pd.read_csv(test_path).columns
