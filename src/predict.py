"""
predict — Load a trained model and run inference.
===================================================

Public helpers
--------------
load_model       Load a joblib pipeline and its JSON metadata.
predict          Batch prediction on a DataFrame.
predict_single   Predict for a single observation dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def load_model(
    model_path: str = "models/best_model.joblib",
) -> tuple[Any, dict[str, Any]]:
    """Load a persisted pipeline and its metadata.

    Parameters
    ----------
    model_path : str
        Path to the ``.joblib`` file.

    Returns
    -------
    tuple[Any, dict]
        ``(pipeline, metadata)`` where *metadata* is the JSON dict
        saved alongside the model.

    Raises
    ------
    FileNotFoundError
        If the model or metadata file is missing.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    logger.info("Loaded pipeline from %s", model_path)

    meta_path = model_path.with_name("best_model_metadata.json")
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        logger.info("Loaded metadata from %s", meta_path)
    else:
        logger.warning("Metadata file not found: %s", meta_path)

    return pipeline, metadata


def predict(
    pipeline,
    input_data: pd.DataFrame,
    label_encoder=None,
) -> np.ndarray:
    """Run batch prediction.

    Parameters
    ----------
    pipeline : fitted sklearn/imblearn Pipeline.
    input_data : pd.DataFrame
        Feature matrix (must match the schema the pipeline was trained on).
    label_encoder : LabelEncoder, optional
        If provided, inverse-transforms integer predictions back to
        human-readable class labels.

    Returns
    -------
    np.ndarray
        Predicted class labels (strings if *label_encoder* supplied,
        else integer codes).
    """
    y_pred = pipeline.predict(input_data)

    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)

    logger.info("Predicted %d samples.", len(y_pred))
    return y_pred


def predict_single(
    pipeline,
    input_dict: dict[str, Any],
    label_encoder=None,
) -> str:
    """Predict the failure type for a single observation.

    Parameters
    ----------
    pipeline : fitted pipeline.
    input_dict : dict
        Feature name → value mapping for one observation.
    label_encoder : LabelEncoder, optional
        Inverse-transform to human-readable label.

    Returns
    -------
    str
        Predicted class label.
    """
    input_df = pd.DataFrame([input_dict])
    preds = predict(pipeline, input_df, label_encoder=label_encoder)
    return str(preds[0])


# ------------------------------------------------------------------ #
# CLI entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Predict failure type using the trained model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.joblib",
        help="Path to the saved model.",
    )
    parser.add_argument(
        "--label-encoder-path",
        type=str,
        default="models/label_encoder.joblib",
        help="Path to the saved LabelEncoder.",
    )
    args = parser.parse_args()

    # Load model
    pipeline, metadata = load_model(args.model_path)
    class_names = metadata.get("class_names", [])
    feature_set = metadata.get("feature_set", "raw")

    # Load label encoder
    le = None
    le_path = Path(args.label_encoder_path)
    if le_path.exists():
        le = joblib.load(le_path)
        logger.info("Loaded LabelEncoder from %s", le_path)

    # Demo prediction with a sample input
    sample = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }

    # Apply feature engineering if the model was trained with it
    if feature_set == "engineered":
        from src.feature_engineering import engineer_features

        sample_df = pd.DataFrame([sample])
        sample_df = engineer_features(sample_df)
        prediction = predict(pipeline, sample_df, label_encoder=le)
        prediction = str(prediction[0])
    else:
        prediction = predict_single(pipeline, sample, label_encoder=le)

    print("\n=== Single Prediction Demo ===")
    print(f"Input:      {sample}")
    print(f"Prediction: {prediction}")
    if class_names:
        print(f"Classes:    {class_names}")
