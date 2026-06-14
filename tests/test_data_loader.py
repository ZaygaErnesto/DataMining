"""Tests for src.data_loader module."""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_dataset,
    create_target,
    prepare_features_and_target,
    validate_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


@pytest.fixture
def sample_df():
    """Create a small DataFrame matching the AI4I 2020 schema."""
    np.random.seed(42)
    n = 20
    data = {
        "UDI": range(1, n + 1),
        "Product ID": [f"L{i:05d}" for i in range(1, n + 1)],
        "Type": np.random.choice(["L", "M", "H"], size=n),
        "Air temperature [K]": np.random.uniform(295, 305, n),
        "Process temperature [K]": np.random.uniform(305, 315, n),
        "Rotational speed [rpm]": np.random.randint(1100, 2900, n),
        "Torque [Nm]": np.random.uniform(3.0, 80.0, n),
        "Tool wear [min]": np.random.randint(0, 250, n),
        "Machine failure": np.zeros(n, dtype=int),
        "TWF": np.zeros(n, dtype=int),
        "HDF": np.zeros(n, dtype=int),
        "PWF": np.zeros(n, dtype=int),
        "OSF": np.zeros(n, dtype=int),
        "RNF": np.zeros(n, dtype=int),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Write the sample DataFrame to a temporary CSV and return its path."""
    csv_path = tmp_path / "ai4i2020.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests — load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_validates_schema(sample_csv):
    """Loading a CSV with all required columns should succeed without error."""
    df = load_dataset(str(sample_csv))
    assert isinstance(df, pd.DataFrame)
    for col in SCHEMA_COLUMNS:
        assert col in df.columns, f"Missing expected column: {col}"


def test_load_dataset_raises_on_missing_columns(tmp_path):
    """Loading a CSV missing a required column should raise ValueError."""
    # Create a CSV missing the 'Torque [Nm]' column
    data = {
        "UDI": [1],
        "Product ID": ["L00001"],
        "Type": ["L"],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]": [1500],
        # "Torque [Nm]" intentionally omitted
        "Tool wear [min]": [50],
        "Machine failure": [0],
        "TWF": [0],
        "HDF": [0],
        "PWF": [0],
        "OSF": [0],
        "RNF": [0],
    }
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_dataset(str(csv_path))


# ---------------------------------------------------------------------------
# Tests — create_target
# ---------------------------------------------------------------------------


def test_create_target_no_failure(sample_df):
    """When all failure indicators are 0, target should be 'No Failure'."""
    result = create_target(sample_df)
    assert "Failure Type" in result.columns
    assert (result["Failure Type"] == "No Failure").all()


def test_create_target_single_failure(sample_df):
    """When exactly one indicator is 1, target should name that failure."""
    df = sample_df.copy()
    df.loc[0, "HDF"] = 1
    df.loc[0, "Machine failure"] = 1
    result = create_target(df)
    assert result.loc[0, "Failure Type"] == "Heat Dissipation Failure"


def test_create_target_multiple_failures(sample_df):
    """When multiple indicators are 1, target should be 'Multiple Failures'."""
    df = sample_df.copy()
    df.loc[0, "HDF"] = 1
    df.loc[0, "PWF"] = 1
    df.loc[0, "Machine failure"] = 1
    result = create_target(df)
    assert result.loc[0, "Failure Type"] == "Multiple Failures"


# ---------------------------------------------------------------------------
# Tests — prepare_features
# ---------------------------------------------------------------------------


COLUMNS_TO_DROP = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def test_prepare_features_drops_correct_columns(sample_df):
    """prepare_features should drop identifier and raw-indicator columns."""
    df = create_target(sample_df)
    X, y = prepare_features_and_target(df)
    for col in COLUMNS_TO_DROP:
        assert col not in X.columns, f"Column '{col}' should have been dropped"
    assert y.name == "Failure Type" or isinstance(y, (pd.Series, np.ndarray))


# ---------------------------------------------------------------------------
# Tests — validate_data
# ---------------------------------------------------------------------------


def test_validate_data_reports_missing_values(sample_df):
    """validate_data should detect and report rows with missing values."""
    df = sample_df.copy()
    df.loc[0, "Torque [Nm]"] = np.nan
    df.loc[3, "Air temperature [K]"] = np.nan
    report = validate_data(df)
    # The report should indicate there are missing values
    assert report["has_missing"] is True
    assert sum(report["missing_values"].values()) >= 2
