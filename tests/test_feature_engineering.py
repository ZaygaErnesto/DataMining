"""Tests for src.feature_engineering module."""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import engineer_features, get_feature_sets

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Create a basic raw dataframe matching the expected features."""
    np.random.seed(42)
    rows = 30
    return pd.DataFrame(
        {
            "Type": np.random.choice(["L", "M", "H"], size=rows),
            "Air temperature [K]": np.random.uniform(295.0, 300.0, size=rows),
            "Process temperature [K]": np.random.uniform(305.0, 310.0, size=rows),
            "Rotational speed [rpm]": np.random.uniform(1100, 2000, size=rows),
            "Torque [Nm]": np.random.uniform(10.0, 70.0, size=rows),
            "Tool wear [min]": np.random.randint(0, 250, size=rows),
        }
    )


@pytest.fixture
def sample_df_with_zero_rpm(sample_df):
    """Create a dataframe containing zero rotational speed to test division safety."""
    df = sample_df.copy()
    df.loc[0, "Rotational speed [rpm]"] = 0
    return df


# ---------------------------------------------------------------------------
# Expected New Columns
# ---------------------------------------------------------------------------

EXPECTED_NEW_COLUMNS = [
    "Temperature difference [K]",
    "Temperature ratio",
    "Torque per rpm",
    "Estimated power [W]",
    "Tool wear torque",
    "Tool wear speed",
    "Heat dissipation risk",
    "Low power risk",
    "High power risk",
    "Tool wear risk",
    "Overstrain risk",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_engineer_features_output_columns(sample_df):
    """engineer_features should add all the 11 expected new columns."""
    original_cols = set(sample_df.columns)
    result = engineer_features(sample_df)
    new_cols = set(result.columns) - original_cols
    assert len(new_cols) == 11, f"Expected 11 new columns, got {len(new_cols)}"
    for col in EXPECTED_NEW_COLUMNS:
        assert col in result.columns, f"Missing engineered column: {col}"


def test_temperature_difference(sample_df):
    """Temperature difference [K] should equal Process temp − Air temp."""
    result = engineer_features(sample_df)
    expected = sample_df["Process temperature [K]"] - sample_df["Air temperature [K]"]
    pd.testing.assert_series_equal(
        result["Temperature difference [K]"], expected, check_names=False
    )


def test_torque_per_rpm_handles_zero(sample_df_with_zero_rpm):
    """Torque per rpm should not raise division by zero error."""
    result = engineer_features(sample_df_with_zero_rpm)
    assert "Torque per rpm" in result.columns
    # Ensure row 0 with speed=0 has NaN for Torque per rpm
    assert pd.isna(result.loc[0, "Torque per rpm"])


def test_risk_flags(sample_df):
    """Binary risk flags should be 0 or 1."""
    result = engineer_features(sample_df)
    risk_cols = [
        "Heat dissipation risk",
        "Low power risk",
        "High power risk",
        "Tool wear risk",
        "Overstrain risk",
    ]
    for col in risk_cols:
        values = result[col].unique()
        assert set(values).issubset(
            {0, 1}
        ), f"Risk column '{col}' has unexpected values: {values}"


def test_get_feature_sets_returns_both(sample_df):
    """get_feature_sets should return both 'raw' and 'engineered' versions."""
    feature_sets = get_feature_sets(sample_df)
    assert isinstance(feature_sets, dict)
    assert "raw" in feature_sets
    assert "engineered" in feature_sets
    assert feature_sets["raw"].shape[1] == sample_df.shape[1]
    assert feature_sets["engineered"].shape[1] == sample_df.shape[1] + 11
