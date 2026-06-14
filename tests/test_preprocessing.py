"""Tests for src.preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from src.preprocessing import build_preprocessor, get_samplers, get_models

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Create a basic raw dataframe matching features needed for preprocessor."""
    np.random.seed(42)
    rows = 50
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


# ---------------------------------------------------------------------------
# Tests — build_preprocessor
# ---------------------------------------------------------------------------


def test_build_preprocessor_returns_column_transformer(sample_df):
    """build_preprocessor should return a ColumnTransformer instance."""
    preprocessor = build_preprocessor(sample_df)
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_transforms_data(sample_df):
    """fit_transform should produce a numpy array with the correct row count."""
    preprocessor = build_preprocessor(sample_df)
    X_transformed = preprocessor.fit_transform(sample_df)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(sample_df)
    # Columns: 5 numeric + one-hot encoded Type (L, M, H → 3 cols) = 8 columns
    assert X_transformed.shape[1] == 8


# ---------------------------------------------------------------------------
# Tests — get_samplers
# ---------------------------------------------------------------------------


def test_get_samplers_returns_dict():
    """get_samplers should return a dict with keys 'none', 'smote', 'smote_tomek'."""
    samplers = get_samplers()
    assert isinstance(samplers, dict)
    expected_keys = {"none", "smote", "borderline_smote", "smote_tomek"}
    assert (
        set(samplers.keys()) == expected_keys
    ), f"Expected keys {expected_keys}, got {set(samplers.keys())}"
    # 'none' should map to None (no resampling)
    assert samplers["none"] is None


# ---------------------------------------------------------------------------
# Tests — get_models
# ---------------------------------------------------------------------------


def test_get_models_returns_dict():
    """get_models should return a dict with the three expected model keys."""
    models = get_models()
    assert isinstance(models, dict)
    expected_keys = {"logistic_regression", "random_forest", "xgboost"}
    assert (
        set(models.keys()) == expected_keys
    ), f"Expected keys {expected_keys}, got {set(models.keys())}"
    # Each value should be an estimator-like object with fit/predict
    for name, model in models.items():
        assert hasattr(model, "fit"), f"Model '{name}' has no fit method"
        assert hasattr(model, "predict"), f"Model '{name}' has no predict method"
