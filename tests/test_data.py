"""Unit tests for src/data.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import EXPECTED_COLUMNS, load_raw_data, summarise, validate_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Return a minimal 20-row DataFrame satisfying all EXPECTED_COLUMNS.

    Returns:
        Synthetic DataFrame with valid column names and representative values.
    """
    rng = np.random.default_rng(42)
    n = 20

    df = pd.DataFrame(
        {
            "Age": rng.integers(18, 60, n),
            "Attrition": rng.choice(["Yes", "No"], n),
            "BusinessTravel": rng.choice(
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n
            ),
            "DailyRate": rng.integers(100, 1500, n),
            "Department": rng.choice(
                ["Human Resources", "Research & Development", "Sales"], n
            ),
            "DistanceFromHome": rng.integers(1, 29, n),
            "Education": rng.integers(1, 6, n),
            "EducationField": rng.choice(
                [
                    "Human Resources",
                    "Life Sciences",
                    "Marketing",
                    "Medical",
                    "Other",
                    "Technical Degree",
                ],
                n,
            ),
            "EmployeeCount": [1] * n,
            "EmployeeNumber": rng.integers(1, 9999, n),
            "EnvironmentSatisfaction": rng.integers(1, 5, n),
            "Gender": rng.choice(["Female", "Male"], n),
            "HourlyRate": rng.integers(30, 100, n),
            "JobInvolvement": rng.integers(1, 5, n),
            "JobLevel": rng.integers(1, 6, n),
            "JobRole": rng.choice(
                [
                    "Healthcare Representative",
                    "Human Resources",
                    "Laboratory Technician",
                    "Manager",
                    "Manufacturing Director",
                    "Research Director",
                    "Research Scientist",
                    "Sales Executive",
                    "Sales Representative",
                ],
                n,
            ),
            "JobSatisfaction": rng.integers(1, 5, n),
            "MaritalStatus": rng.choice(["Divorced", "Married", "Single"], n),
            "MonthlyIncome": rng.integers(1000, 20000, n),
            "MonthlyRate": rng.integers(2000, 27000, n),
            "NumCompaniesWorked": rng.integers(0, 10, n),
            "Over18": ["Y"] * n,
            "OverTime": rng.choice(["Yes", "No"], n),
            "PercentSalaryHike": rng.integers(11, 26, n),
            "PerformanceRating": rng.choice([3, 4], n),
            "RelationshipSatisfaction": rng.integers(1, 5, n),
            "StandardHours": [80] * n,
            "StockOptionLevel": rng.integers(0, 4, n),
            "TotalWorkingYears": rng.integers(0, 40, n),
            "TrainingTimesLastYear": rng.integers(0, 7, n),
            "WorkLifeBalance": rng.integers(1, 5, n),
            "YearsAtCompany": rng.integers(0, 40, n),
            "YearsInCurrentRole": rng.integers(0, 18, n),
            "YearsSinceLastPromotion": rng.integers(0, 15, n),
            "YearsWithCurrManager": rng.integers(0, 17, n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_raw_data_file_not_found(tmp_path: Path) -> None:
    """load_raw_data raises FileNotFoundError for a non-existent path.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(path=str(missing))


def test_validate_schema_passes(synthetic_df: pd.DataFrame) -> None:
    """validate_schema raises no exception when all expected columns are present.

    Args:
        synthetic_df: Minimal valid synthetic DataFrame fixture.
    """
    # Should not raise.
    validate_schema(synthetic_df)


def test_validate_schema_missing_column(synthetic_df: pd.DataFrame) -> None:
    """validate_schema raises ValueError when a required column is absent.

    Args:
        synthetic_df: Minimal valid synthetic DataFrame fixture.
    """
    broken = synthetic_df.drop(columns=["Age"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_schema(broken)


def test_summarise_runs(synthetic_df: pd.DataFrame, capsys: pytest.CaptureFixture) -> None:
    """summarise executes without error and prints output.

    Args:
        synthetic_df: Minimal valid synthetic DataFrame fixture.
        capsys: pytest output capture fixture.
    """
    summarise(synthetic_df)
    captured = capsys.readouterr()
    # Shape line must appear in the output.
    assert "Shape" in captured.out
