"""
Evaluate — Model evaluation and quality gate checks.

This module provides functions to evaluate trained models, check quality gates,
and generate evaluation reports.
"""

import json
import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Evaluate a trained pipeline on the test set.

    Args:
        pipeline: Trained sklearn/imblearn pipeline.
        X_test: Test features DataFrame.
        y_test: Test target array (label-encoded).
        class_names: List of class name strings.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
    ).tolist()

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def check_quality_gates(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[bool, list[str]]:
    """Check whether model metrics pass the quality gate thresholds.

    Args:
        metrics: Dictionary of evaluation metrics.
        thresholds: Dictionary with keys like ``min_f1_macro`` and
            ``min_balanced_accuracy``.  Defaults are 0.60 / 0.65.

    Returns:
        Tuple of (passed: bool, messages: list of failure descriptions).
    """
    if thresholds is None:
        thresholds = {
            "min_f1_macro": 0.60,
            "min_balanced_accuracy": 0.65,
        }

    messages: list[str] = []

    # --- F1 macro gate ---
    min_f1 = thresholds.get("min_f1_macro", 0.60)
    actual_f1 = metrics.get("f1_macro", 0.0)
    if actual_f1 < min_f1:
        messages.append(
            f"FAIL: f1_macro={actual_f1:.4f} < threshold={min_f1:.4f}"
        )

    # --- Balanced accuracy gate ---
    min_ba = thresholds.get("min_balanced_accuracy", 0.65)
    actual_ba = metrics.get("balanced_accuracy", 0.0)
    if actual_ba < min_ba:
        messages.append(
            f"FAIL: balanced_accuracy={actual_ba:.4f} < threshold={min_ba:.4f}"
        )

    passed = len(messages) == 0
    return passed, messages


def generate_report(
    results: list[dict],
    best_result: dict,
    class_names: list[str],
    output_dir: str = "metrics",
) -> None:
    """Save evaluation results to CSV and JSON files.

    Args:
        results: List of metric dicts from all ablation experiments.
        best_result: Best result dictionary with ``metrics``,
            ``classification_report``, and ``confusion_matrix``.
        class_names: List of class name strings.
        output_dir: Directory to write reports into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- ablation_results.csv ---
    results_df = pd.DataFrame(results).sort_values(
        "f1_macro", ascending=False
    )
    results_df.to_csv(out / "ablation_results.csv", index=False)

    # --- per_class_metrics.csv ---
    per_class_rows: list[dict] = []
    for row in results:
        report = row.get("classification_report")
        if report is None:
            continue
        for cls in class_names:
            if cls in report:
                per_class_rows.append(
                    {
                        "feature_set": row.get("feature_set", ""),
                        "sampler": row.get("sampler", ""),
                        "model": row.get("model", ""),
                        "class": cls,
                        "precision": report[cls]["precision"],
                        "recall": report[cls]["recall"],
                        "f1_score": report[cls]["f1-score"],
                        "support": report[cls]["support"],
                    }
                )
    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(
            out / "per_class_metrics.csv", index=False
        )

    # --- best_confusion_matrix.csv ---
    cm = best_result.get("confusion_matrix", [])
    if cm:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(out / "best_confusion_matrix.csv")

    # --- metrics.json (DVC metrics file) ---
    best_metrics = {
        k: v
        for k, v in best_result.get("metrics", best_result).items()
        if k not in ("classification_report", "confusion_matrix")
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(best_metrics, fh, indent=2)

    # --- evaluation.json (extended) ---
    evaluation = {
        "class_names": class_names,
        "best_result": _make_serializable(best_result),
        "total_experiments": len(results),
    }
    with (out / "evaluation.json").open("w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=2)

    print(f"Reports saved to {out}/")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point:  python -m src.evaluate --config configs/config.yaml
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models_dir = Path(config.get("output", {}).get("models_dir", "models"))
    metrics_dir = Path(config.get("output", {}).get("metrics_dir", "metrics"))
    processed_dir = Path(
        config.get("paths", {}).get("processed_dir", "data/processed")
    )

    # Load test data
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").values.ravel()

    # Load model and metadata
    pipeline = joblib.load(models_dir / "best_model.joblib")
    with (models_dir / "best_model_metadata.json").open("r") as fh:
        metadata = json.load(fh)

    # Conditionally apply feature engineering
    feature_set_name = metadata.get("feature_set", "raw")
    if feature_set_name == "engineered":
        from src.feature_engineering import engineer_features
        print("Model trained on engineered features. Applying feature engineering to input data...")
        X_test = engineer_features(X_test)

    class_names = metadata.get("class_names", [])

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test, class_names)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        if k not in ("classification_report", "confusion_matrix"):
            print(f"  {k}: {v:.4f}")

    # Quality gates
    thresholds = config.get("quality_gates", {})
    passed, messages = check_quality_gates(metrics, thresholds)

    if passed:
        print("\n[SUCCESS] All quality gates PASSED!")
    else:
        print("\n[FAILED] Quality gates FAILED:")
        for msg in messages:
            print(f"  {msg}")

    # Save report
    best_result = {
        "metrics": metrics,
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }
    generate_report(
        results=[metrics],
        best_result=best_result,
        class_names=class_names,
        output_dir=str(metrics_dir),
    )

    if not passed:
        raise SystemExit(1)
