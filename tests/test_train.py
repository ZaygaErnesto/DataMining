"""Tests for src.train module."""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.train import train_single, save_best_model
from src.predict import load_model
from src.preprocessing import build_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset():
    """Create a very small synthetic dataset (50 rows) for fast testing."""
    np.random.seed(42)
    rows = 50
    X = pd.DataFrame({
        "Type": np.random.choice(["L", "M", "H"], size=rows),
        "Air temperature [K]": np.random.uniform(295.0, 300.0, size=rows),
        "Process temperature [K]": np.random.uniform(305.0, 310.0, size=rows),
        "Rotational speed [rpm]": np.random.uniform(1100, 2000, size=rows),
        "Torque [Nm]": np.random.uniform(10.0, 70.0, size=rows),
        "Tool wear [min]": np.random.randint(0, 250, size=rows),
        "Temperature Difference": np.random.uniform(5.0, 10.0, size=rows),
    })
    y = np.random.choice([0, 1], size=rows)
    return X, y


@pytest.fixture
def dummy_pipeline():
    """A simple sklearn Pipeline for save/load testing."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )


# ---------------------------------------------------------------------------
# Tests — train_single
# ---------------------------------------------------------------------------


def test_train_single_returns_metrics(synthetic_dataset):
    """train_single should return a dict with standard metric keys."""
    X, y = synthetic_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(max_iter=300, random_state=42)
    class_names = ["No Failure", "Failure"]

    metrics, pipeline = train_single(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        sampler=None,
        model=model,
        class_names=class_names,
    )

    assert isinstance(metrics, dict)
    expected_keys = {"accuracy", "f1_macro", "balanced_accuracy"}
    assert expected_keys.issubset(set(metrics.keys())), (
        f"Missing metric keys. Expected at least {expected_keys}, "
        f"got {set(metrics.keys())}"
    )
    # All metric values should be between 0 and 1
    for key in expected_keys:
        assert 0.0 <= metrics[key] <= 1.0, (
            f"Metric '{key}' out of range: {metrics[key]}"
        )


# ---------------------------------------------------------------------------
# Tests — save / load model
# ---------------------------------------------------------------------------


def test_save_and_load_model(tmp_path, dummy_pipeline, synthetic_dataset):
    """Saving a pipeline and loading it back should produce an equivalent model."""
    X, y = synthetic_dataset
    X_numeric = X.drop(columns=["Type"])
    dummy_pipeline.fit(X_numeric, y)

    output_dir = str(tmp_path)
    metadata = {"class_names": ["No Failure", "Power Failure"]}
    save_best_model(dummy_pipeline, metadata, output_dir=output_dir)

    model_path = os.path.join(output_dir, "best_model.joblib")
    assert os.path.exists(model_path), "Model file was not created"

    loaded_pipeline, loaded_metadata = load_model(model_path)
    # Predictions from original and loaded model should match
    preds_original = dummy_pipeline.predict(X_numeric)
    preds_loaded = loaded_pipeline.predict(X_numeric)
    np.testing.assert_array_equal(preds_original, preds_loaded)
    assert loaded_metadata == metadata
