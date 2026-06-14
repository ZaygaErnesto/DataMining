"""
preprocessing — Sklearn transformers, samplers, and model factories.
=====================================================================

Public helpers
--------------
build_preprocessor   Auto-detect numeric/categorical columns → ColumnTransformer.
get_samplers         Return a dict of imblearn samplers (none / SMOTE / SMOTETomek).
get_models           Return a dict of classifier instances.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
RANDOM_STATE: int = 42


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a :class:`ColumnTransformer` that handles numeric and
    categorical features automatically.

    * **Numeric pipeline**: ``SimpleImputer(strategy='median')`` →
      ``StandardScaler()``.
    * **Categorical pipeline**: ``SimpleImputer(strategy='most_frequent')``
      → ``OneHotEncoder(handle_unknown='ignore', sparse_output=False)``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix used to detect column dtypes.

    Returns
    -------
    ColumnTransformer
        A fitted-ready transformer.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    logger.info(
        "Preprocessor built — %d numeric, %d categorical columns.",
        len(numeric_cols),
        len(categorical_cols),
    )
    return preprocessor


def get_samplers(
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Return a dictionary of imblearn samplers for ablation.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        ``{'none': None, 'smote': SMOTE(...), 'borderline_smote': BorderlineSMOTE(...), 'smote_tomek': SMOTETomek(...)}``.
    """
    return {
        "none": None,
        "smote": SMOTE(random_state=random_state),
        "borderline_smote": BorderlineSMOTE(random_state=random_state),
        "smote_tomek": SMOTETomek(random_state=random_state),
    }


def get_models(
    config: dict[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Return a dictionary of classifier instances.

    Parameters
    ----------
    config : dict, optional
        The ``models`` section of the YAML config.  When provided, each
        model's hyper-parameters are taken from the corresponding key.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        ``{'logistic_regression': ..., 'random_forest': ..., 'xgboost': ...}``.
    """
    if config is None:
        config = {}

    lr_params = config.get("logistic_regression", {})
    rf_params = config.get("random_forest", {})
    xgb_params = config.get("xgboost", {})

    lr_kwargs: dict[str, Any] = {
        "max_iter": lr_params.get("max_iter", 1000),
        "solver": lr_params.get("solver", "lbfgs"),
        "class_weight": lr_params.get("class_weight", "balanced"),
        "random_state": random_state,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        lr_kwargs["multi_class"] = lr_params.get(
            "multi_class", "multinomial"
        )

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(**lr_kwargs),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", 10),
            min_samples_split=rf_params.get("min_samples_split", 5),
            min_samples_leaf=rf_params.get("min_samples_leaf", 2),
            class_weight=rf_params.get("class_weight", "balanced"),
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 200),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            use_label_encoder=xgb_params.get("use_label_encoder", False),
            eval_metric=xgb_params.get("eval_metric", "mlogloss"),
            random_state=random_state,
        ),
    }

    logger.info("Models instantiated: %s", list(models.keys()))
    return models
