"""Tests for model quality gates (src.evaluate.check_quality_gates)."""

import pytest

from src.evaluate import check_quality_gates


# ---------------------------------------------------------------------------
# Tests — quality gate logic
# ---------------------------------------------------------------------------


def test_quality_gate_passes():
    """When all metrics exceed thresholds, the quality gate should pass."""
    metrics = {
        "accuracy": 0.92,
        "f1_macro": 0.85,
        "balanced_accuracy": 0.80,
    }
    passed, message = check_quality_gates(metrics)
    assert passed is True, f"Quality gate should have passed. Message: {message}"


def test_quality_gate_fails_f1():
    """When f1_macro is below threshold, the quality gate should fail."""
    metrics = {
        "accuracy": 0.92,
        "f1_macro": 0.30,  # well below any reasonable threshold
        "balanced_accuracy": 0.80,
    }
    passed, message = check_quality_gates(metrics)
    assert passed is False, "Quality gate should have failed on low f1_macro"
    joined_msg = " ".join(message).lower()
    assert "f1_macro" in joined_msg or "f1" in joined_msg, (
        f"Failure message should mention f1_macro. Got: {message}"
    )


def test_quality_gate_fails_balanced_accuracy():
    """When balanced_accuracy is below threshold, the quality gate should fail."""
    metrics = {
        "accuracy": 0.92,
        "f1_macro": 0.85,
        "balanced_accuracy": 0.35,  # well below any reasonable threshold
    }
    passed, message = check_quality_gates(metrics)
    assert passed is False, (
        "Quality gate should have failed on low balanced_accuracy"
    )
    joined_msg = " ".join(message).lower()
    assert "balanced_accuracy" in joined_msg or "balanced" in joined_msg, (
        f"Failure message should mention balanced_accuracy. Got: {message}"
    )
