"""
feature_engineering — Domain-driven feature creation for predictive maintenance.
================================================================================

Public helpers
--------------
engineer_features   Create 11 engineered features from the 6 raw columns.
get_feature_sets    Return a dict with ``'raw'`` and ``'engineered'`` DataFrames.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
RAW_FEATURE_COLS: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create 11 engineered features from the raw feature columns.

    The returned DataFrame contains **all original columns plus** the
    new ones (17 columns total when the input has the 6 raw features).

    New features
    ------------
    +---------------------------------+---------------------------------------------+
    | Feature                         | Formula / logic                             |
    +=================================+=============================================+
    | Temperature difference [K]      | Process temp − Air temp                     |
    | Temperature ratio               | Process temp / Air temp                     |
    | Torque per rpm                  | Torque / Rotational speed                   |
    | Estimated power [W]             | Torque × Rotational speed / 9.5488          |
    | Tool wear torque                | Tool wear × Torque                          |
    | Tool wear speed                 | Tool wear × Rotational speed                |
    | Heat dissipation risk           | (temp_diff ≤ 8.6) & (speed < 1380)         |
    | Low power risk                  | power < 3500                                |
    | High power risk                 | power > 9000                                |
    | Tool wear risk                  | tool_wear ≥ 198                             |
    | Overstrain risk                 | type-dependent threshold on tool_wear_torque|
    +---------------------------------+---------------------------------------------+

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with at least the raw columns listed in
        ``RAW_FEATURE_COLS``.

    Returns
    -------
    pd.DataFrame
        A copy of *X* augmented with the 11 new columns.
    """
    X = X.copy()

    # --- Continuous features -------------------------------------------
    X["Temperature difference [K]"] = (
        X["Process temperature [K]"] - X["Air temperature [K]"]
    )
    X["Temperature ratio"] = X["Process temperature [K]"] / X["Air temperature [K]"]

    # Guard against zero rotational speed
    rot_speed = X["Rotational speed [rpm]"].replace(0, np.nan)
    X["Torque per rpm"] = X["Torque [Nm]"] / rot_speed

    power = X["Torque [Nm]"] * X["Rotational speed [rpm]"] / 9.5488
    X["Estimated power [W]"] = power

    tool_wear_torque = X["Tool wear [min]"] * X["Torque [Nm]"]
    X["Tool wear torque"] = tool_wear_torque

    X["Tool wear speed"] = X["Tool wear [min]"] * X["Rotational speed [rpm]"]

    # --- Binary risk flags ---------------------------------------------
    temp_diff = X["Temperature difference [K]"]
    speed = X["Rotational speed [rpm]"]
    tool_wear = X["Tool wear [min]"]

    X["Heat dissipation risk"] = ((temp_diff <= 8.6) & (speed < 1380)).astype(int)

    X["Low power risk"] = (power < 3500).astype(int)
    X["High power risk"] = (power > 9000).astype(int)
    X["Tool wear risk"] = (tool_wear >= 198).astype(int)

    # Overstrain risk: type-dependent thresholds
    type_col = X["Type"] if "Type" in X.columns else None
    if type_col is not None:
        overstrain = np.zeros(len(X), dtype=int)
        overstrain[(type_col == "L") & (tool_wear_torque > 11_000)] = 1
        overstrain[(type_col == "M") & (tool_wear_torque > 12_000)] = 1
        overstrain[(type_col == "H") & (tool_wear_torque > 13_000)] = 1
        X["Overstrain risk"] = overstrain
    else:
        # Fallback when Type column is absent (e.g. already encoded)
        X["Overstrain risk"] = (tool_wear_torque > 12_000).astype(int)
        logger.warning(
            "'Type' column not found — using single threshold for " "Overstrain risk."
        )

    logger.info("Engineered features added. Final feature count: %d", X.shape[1])
    return X


def get_feature_sets(
    X_raw: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return a dictionary of feature-set variants for ablation.

    Parameters
    ----------
    X_raw : pd.DataFrame
        The raw feature matrix (6 feature columns + possible ``Type``).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{'raw': X_raw, 'engineered': engineer_features(X_raw)}``.
    """
    return {
        "raw": X_raw.copy(),
        "engineered": engineer_features(X_raw),
    }
