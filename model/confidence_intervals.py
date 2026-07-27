"""
Residual-based confidence interval estimation.

Replaces XGBoost quantile regression for confidence intervals.
On small datasets (60-80 rows), quantile regression is unstable.
This approach uses the distribution of training residuals instead —
more reliable, statistically honest, and faster.

Method:
    1. Train the point forecast model on training data
    2. Compute in-sample residuals on the training set
    3. Use empirical percentiles of residuals as the CI offset
    4. Apply: lower = point - |residual_pXX|, upper = point + |residual_pXX|

This is equivalent to conformal prediction under the assumption that
future residuals are exchangeable with training residuals — a reasonable
assumption for a stationary commodity price series.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from utils.logger import get_logger

logger = get_logger(__name__)

SAVED_DIR = "model/saved"


def compute_residual_ci(
    model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lower_pct: float = 10.0,
    upper_pct: float = 90.0,
) -> dict:
    """
    Computes confidence interval offsets from training residuals.

    Args:
        model:      Trained XGBoost point forecast model
        X_train:    Training features
        y_train:    Training targets
        lower_pct:  Lower percentile for CI (default 10th = 80% CI)
        upper_pct:  Upper percentile for CI (default 90th = 80% CI)

    Returns:
        dict with keys: lower_offset, upper_offset, residual_std,
                        residual_mae, ci_width_median
    """
    X_filled = X_train.fillna(X_train.median())
    train_preds = model.predict(X_filled)
    residuals = y_train.values - train_preds

    lower_offset = float(np.percentile(residuals, lower_pct))
    upper_offset = float(np.percentile(residuals, upper_pct))

    ci_info = {
        "lower_offset":    lower_offset,
        "upper_offset":    upper_offset,
        "residual_std":    float(np.std(residuals)),
        "residual_mae":    float(np.mean(np.abs(residuals))),
        "ci_width_median": float(upper_offset - lower_offset),
        "lower_pct":       lower_pct,
        "upper_pct":       upper_pct,
        "n_residuals":     len(residuals),
    }

    logger.info(
        f"Residual CI: lower_offset={lower_offset:.1f}, "
        f"upper_offset={upper_offset:.1f}, "
        f"median_width={ci_info['ci_width_median']:.1f}, "
        f"n={len(residuals)}"
    )

    return ci_info


def apply_ci(point_forecast: float, ci_info: dict,
             price_floor: float = 150.0,
             price_ceiling: float = 1200.0) -> tuple[float, float]:
    """
    Applies residual CI offsets to a point forecast.

    Returns (lower_bound, upper_bound).
    """
    lower = point_forecast + ci_info["lower_offset"]
    upper = point_forecast + ci_info["upper_offset"]

    lower = max(lower, price_floor)
    upper = min(upper, price_ceiling)
    lower = min(lower, point_forecast)
    upper = max(upper, point_forecast)

    return round(lower, 1), round(upper, 1)


def save_ci_info(ci_info: dict, horizon: int):
    """Saves CI offsets to disk for use in forecast.py."""
    path = os.path.join(SAVED_DIR, f"ci_info_{horizon}m.json")
    with open(path, "w") as f:
        json.dump(ci_info, f, indent=2)
    logger.info(f"CI info saved: {path}")


def load_ci_info(horizon: int) -> dict | None:
    """Loads CI offsets from disk."""
    path = os.path.join(SAVED_DIR, f"ci_info_{horizon}m.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
