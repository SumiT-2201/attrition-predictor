"""Data loading and schema validation for the Employee Attrition Predictor.

Responsibilities:
- Load raw CSV from disk.
- Validate that expected columns are present.
- Perform basic sanity checks (shape, nulls, target distribution).
"""

import pandas as pd

from config import RAW_DATA_FILE, TARGET_COLUMN

# Columns the IBM HR Analytics dataset must contain.
EXPECTED_COLUMNS: list[str] = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw attrition CSV from disk.

    Args:
        path: Optional override path. Defaults to ``config.RAW_DATA_FILE``.

    Returns:
        Raw DataFrame exactly as read from CSV.

    Raises:
        FileNotFoundError: If the CSV does not exist at the resolved path.
    """
    resolved = path or RAW_DATA_FILE
    if not pd.io.common.file_exists(str(resolved)):
        raise FileNotFoundError(
            f"Raw data file not found: {resolved}\n"
            "Download the IBM HR Analytics dataset and place it in data/raw/."
        )
    return pd.read_csv(resolved)


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that all expected columns are present in the DataFrame.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If one or more expected columns are missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw data is missing expected columns: {sorted(missing)}"
        )


def summarise(df: pd.DataFrame) -> None:
    """Print a concise summary: shape, null counts, and target distribution.

    Args:
        df: DataFrame to summarise (typically the raw data).
    """
    print(f"Shape        : {df.shape}")
    print(f"Null counts  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    if TARGET_COLUMN in df.columns:
        dist = df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(1)
        print(f"Target dist  :\n{dist}")


if __name__ == "__main__":
    raw = load_raw_data()
    validate_schema(raw)
    summarise(raw)
