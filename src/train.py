"""
train — Ablation study and model training.
==========================================

Public helpers
--------------
run_ablation     Full grid search over feature sets × samplers × models.
train_single     Train a single configuration and return metrics.
save_best_model  Persist the best pipeline + metadata to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_loader import create_target, load_dataset, prepare_features_and_target
from src.evaluate import evaluate_model
from src.feature_engineering import get_feature_sets
from src.preprocessing import build_preprocessor, get_models, get_samplers

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
RANDOM_STATE: int = 42


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def run_ablation(
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Execute a full ablation study.

    The study iterates over every combination of:

    * **feature sets** — ``raw`` (6 features) and ``engineered``
      (17 features).
    * **samplers** — ``none``, ``smote``, ``smote_tomek``.
    * **models** — ``logistic_regression``, ``random_forest``,
      ``xgboost``.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Keys: ``results`` (list of per-run metric dicts),
        ``best_result`` (single best dict by balanced_accuracy),
        ``best_pipeline`` (fitted imblearn Pipeline),
        ``label_encoder`` (fitted LabelEncoder),
        ``class_names`` (list[str]),
        ``X_test``, ``y_test`` (held-out arrays).
    """
    # ---- Load config --------------------------------------------------
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    random_state = config["project"].get("random_state", RANDOM_STATE)
    test_size = config["project"].get("test_size", 0.2)

    # ---- Data preparation ---------------------------------------------
    df = load_dataset(config["paths"]["raw_data"])
    df = create_target(df)
    X_raw, y = prepare_features_and_target(df)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names: list[str] = le.classes_.tolist()
    logger.info("Classes: %s", class_names)

    # Train / test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Feature sets
    feature_sets_train = get_feature_sets(X_train_raw)
    feature_sets_test = get_feature_sets(X_test_raw)

    # Samplers & models
    samplers = get_samplers(random_state=random_state)
    model_dict = get_models(config=config.get("models"), random_state=random_state)

    # ---- Ablation grid ------------------------------------------------
    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_pipeline: ImbPipeline | None = None
    best_score: float = -1.0

    total = len(feature_sets_train) * len(samplers) * len(model_dict)
    run_idx = 0

    for fs_name, X_tr in feature_sets_train.items():
        X_te = feature_sets_test[fs_name]
        preprocessor = build_preprocessor(X_tr)

        for samp_name, sampler in samplers.items():
            for model_name, model in model_dict.items():
                run_idx += 1
                tag = f"[{run_idx}/{total}] {fs_name} | {samp_name} | {model_name}"
                logger.info("Running %s …", tag)

                # Clone model to avoid state leakage
                from sklearn.base import clone

                model_clone = clone(model)

                metrics, pipeline = train_single(
                    X_train=X_tr,
                    y_train=y_train,
                    X_test=X_te,
                    y_test=y_test,
                    preprocessor=preprocessor,
                    sampler=sampler,
                    model=model_clone,
                    class_names=class_names,
                )

                result = {
                    "feature_set": fs_name,
                    "sampler": samp_name,
                    "model": model_name,
                    **metrics,
                }
                results.append(result)

                # Track best by balanced accuracy
                score = metrics.get("balanced_accuracy", 0.0)
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_pipeline = pipeline

                logger.info(
                    "%s → balanced_accuracy=%.4f, f1_macro=%.4f",
                    tag,
                    metrics["balanced_accuracy"],
                    metrics["f1_macro"],
                )

    logger.info(
        "Best config: %s | %s | %s  (balanced_accuracy=%.4f)",
        best_result["feature_set"],
        best_result["sampler"],
        best_result["model"],
        best_result["balanced_accuracy"],
    )

    return {
        "results": results,
        "best_result": best_result,
        "best_pipeline": best_pipeline,
        "label_encoder": le,
        "class_names": class_names,
        "X_test": X_test_raw,
        "y_test": y_test,
    }


def train_single(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    preprocessor,
    sampler,
    model,
    class_names: list[str],
) -> tuple[dict[str, Any], ImbPipeline]:
    """Train a single configuration and return metrics + fitted pipeline.

    Parameters
    ----------
    X_train, y_train : training data.
    X_test, y_test : test data.
    preprocessor : sklearn ColumnTransformer.
    sampler : imblearn sampler or None.
    model : sklearn-compatible classifier.
    class_names : list of class label strings.

    Returns
    -------
    tuple[dict, ImbPipeline]
        ``(metrics_dict, fitted_pipeline)``.
    """
    start = time.time()

    # Build imblearn pipeline (supports optional sampler)
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", model))

    pipeline = ImbPipeline(steps=steps)

    fit_params = {}
    if type(model).__name__ == "XGBClassifier" and sampler is None:
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_params["classifier__sample_weight"] = sample_weights

    pipeline.fit(X_train, y_train, **fit_params)

    elapsed = time.time() - start

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test, class_names)
    metrics["train_time_seconds"] = round(elapsed, 3)

    return metrics, pipeline


def save_best_model(
    pipeline: ImbPipeline,
    metadata: dict[str, Any],
    output_dir: str = "models/",
) -> None:
    """Persist the best pipeline and its metadata to disk.

    Saves:
    * ``best_model.joblib`` — the fitted imblearn pipeline.
    * ``best_model_metadata.json`` — configuration and metrics.

    Parameters
    ----------
    pipeline : ImbPipeline
        The fitted pipeline to save.
    metadata : dict
        Arbitrary metadata (config, metrics, class names, etc.).
    output_dir : str, optional
        Directory to write into (created if missing).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "best_model.joblib"
    meta_path = out / "best_model_metadata.json"

    joblib.dump(pipeline, model_path)
    logger.info("Pipeline saved → %s", model_path)

    # Make metadata JSON-serialisable
    serialisable = _make_serialisable(metadata)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Metadata saved → %s", meta_path)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ------------------------------------------------------------------ #
# CLI entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the full ablation study.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    outcome = run_ablation(config_path=args.config)

    # Save best model
    best = outcome["best_result"]
    save_best_model(
        pipeline=outcome["best_pipeline"],
        metadata={
            "feature_set": best["feature_set"],
            "sampler": best["sampler"],
            "model": best["model"],
            "metrics": {
                k: v
                for k, v in best.items()
                if k not in ("feature_set", "sampler", "model")
            },
            "class_names": outcome["class_names"],
        },
    )

    # Save label encoder
    le_path = Path("models/label_encoder.joblib")
    joblib.dump(outcome["label_encoder"], le_path)
    logger.info("LabelEncoder saved → %s", le_path)

    # Generate reports
    from src.evaluate import generate_report

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    reports_dir = config["paths"].get("reports_dir", "reports/")

    generate_report(
        results=outcome["results"],
        best_result=outcome["best_result"],
        class_names=outcome["class_names"],
        output_dir=reports_dir,
    )

    print("\n[SUCCESS] Ablation study complete.")
    print(
        f"   Best: {best['feature_set']} | {best['sampler']} | "
        f"{best['model']}  ->  balanced_accuracy={best['balanced_accuracy']:.4f}"
    )
