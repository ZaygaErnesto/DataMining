"""
data_loader — Load, validate, and prepare the AI4I 2020 dataset.
================================================================

Public helpers
--------------
load_dataset          Read the CSV and validate the expected schema.
create_target         Derive the multi-class *Failure Type* column.
prepare_features_and_target
                      Split the DataFrame into feature matrix X and target y.
validate_data         Quick sanity-check report (missing values, class balance).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
EXPECTED_COLUMNS: list[str] = [
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

FAILURE_MAP: dict[str, str] = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failures",
}

BINARY_FAILURE_COLS: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

DROP_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "Target",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Failure Type",
]


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and validate that it contains the expected columns.

    Parameters
    ----------
    path : str | Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded (unmodified) DataFrame.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any expected column is missing from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s from %s", df.shape, path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_cols)}"
        )

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the multi-class ``Failure Type`` column.

    Logic
    -----
    * ``failure_count == 0`` → ``'No Failure'``
    * ``failure_count == 1`` → specific failure name
    * ``failure_count > 1``  → ``'Multiple Failures'``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the binary indicator columns
        (TWF, HDF, PWF, OSF, RNF).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with a new ``Failure Type`` column appended.
    """
    df = df.copy()
    failure_count = df[BINARY_FAILURE_COLS].sum(axis=1)

    def _map_row(row: pd.Series, count: int) -> str:
        if count == 0:
            return "No Failure"
        if count > 1:
            return "Multiple Failures"
        # Exactly one failure — find which one
        for col, name in FAILURE_MAP.items():
            if row[col] == 1:
                return name
        return "No Failure"  # fallback (should not happen)

    df["Failure Type"] = [
        _map_row(df.iloc[i], failure_count.iloc[i])
        for i in range(len(df))
    ]

    logger.info(
        "Target distribution:\n%s",
        df["Failure Type"].value_counts().to_string(),
    )
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identifier / target columns and return (X, y).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already has a ``Failure Type`` column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where *X* contains only feature columns and *y* is
        the ``Failure Type`` series.
    """
    y = df["Failure Type"].copy()
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    logger.info("Feature matrix shape: %s", X.shape)
    return X, y


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Run basic data-quality checks.

    Returns a report dictionary with keys:

    * ``n_rows``, ``n_cols``
    * ``missing_values`` — per-column count of NaN
    * ``class_distribution`` — value-counts for ``Failure Type``
    * ``has_missing`` — boolean flag

    Parameters
    ----------
    df : pd.DataFrame
        The (potentially enriched) DataFrame.

    Returns
    -------
    dict[str, Any]
        Validation report.
    """
    report: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": df.isnull().sum().to_dict(),
        "has_missing": bool(df.isnull().any().any()),
    }

    if "Failure Type" in df.columns:
        report["class_distribution"] = (
            df["Failure Type"].value_counts().to_dict()
        )

    logger.info("Validation report: %s", json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Load, validate, and save processed dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    raw_path = config["paths"]["raw_data"]
    processed_dir = Path(config["paths"].get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    random_state = config["project"].get("random_state", 42)
    test_size = config["project"].get("test_size", 0.2)

    # 1. Load
    df = load_dataset(raw_path)

    # 2. Create target
    df = create_target(df)

    # 3. Validate
    report = validate_data(df)
    
    # Save validation report
    with open(processed_dir / "validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # 4. Feature and target separation
    X, y = prepare_features_and_target(df)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # 6. Save split files
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    pd.Series(y_train, name="Failure Type").to_csv(processed_dir / "y_train.csv", index=False)
    pd.Series(y_test, name="Failure Type").to_csv(processed_dir / "y_test.csv", index=False)

    print(f"\n[SUCCESS] Split data saved to {processed_dir}/")

