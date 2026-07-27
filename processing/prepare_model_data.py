"""
Model data preparation.

Loads the processed feature set and prepares train/test splits
for both the 1-month and 12-month forecast horizons.

Key principle: no lookahead bias.
    - All features must be lagged so that at prediction time t,
      every feature value was observable at time t-1 or earlier.
    - The test set must be a contiguous tail of the time series,
      not a random sample.

Run after build_features.py:
    py processing/prepare_model_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = "data/processed"

# ---------------------------------------------------------------------------
# Features to use in the model
# Ordered by expected importance. Exclude raw price levels where
# lagged/derived versions exist to reduce multicollinearity.
# ---------------------------------------------------------------------------

# Tier 1 — High signal, always include
TIER1_FEATURES = [
    "an_lag_1m",            # Last month's AN price — strongest single predictor
    "an_lag_2m",            # Two months ago
    "an_lag_3m",            # Three months ago
    "ttf_gas_gbp_lag1m",    # Gas price last month (primary cost driver)
    "ttf_gas_gbp_lag2m",    # Gas price two months ago (feedthrough lag)
    "an_roll_3m",           # 3-month rolling average (trend signal)
    "month_sin",            # Seasonality
    "month_cos",            # Seasonality
]

# Tier 2 — Good signal, include if available
TIER2_FEATURES = [
    "urea_gbp_lag1m",       # Global urea benchmark lagged
    "brent_gbp_bbl",        # Energy complex
    "an_roll_6m",           # 6-month trend
    "gas_roll_3m",          # Gas trend
    "an_mom_1m",            # Momentum
    "an_urea_spread",       # UK AN premium over global urea
    "cf_close_usd",         # CF Industries equity (production economics)
    "yara_close_nok",       # Yara equity proxy
]

# Tier 3 — Supplementary, use if non-null coverage > 50%
TIER3_FEATURES = [
    "eu_gas_storage_pct",   # Only useful once GIE history is complete
    "gas_storage_deviation",
    "gas_storage_lag1m",
    "wb_dap_usd_t",         # Cross-commodity demand signal
    "an_mom_3m",
]

TARGET = "an_price_gbp_t"


def prepare(horizon_months: int = 1) -> tuple[pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series]:
    """
    Loads processed features and returns train/test splits for the
    specified forecast horizon.

    Args:
        horizon_months: How many months ahead to forecast (1 or 12).
                        Creates a forward-shifted target variable.

    Returns:
        X_train, y_train, X_test, y_test as DataFrames/Series.
    """
    features_path = os.path.join(PROCESSED_DIR, "model_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Processed features not found at {features_path}. "
            "Run: py processing/build_features.py"
        )

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded processed features: {df.shape}")

    # --- Build forward target ---
    # y at time t = AN price at time t + horizon_months
    # This is what we're trying to predict
    target_col = f"an_price_target_{horizon_months}m"
    df[target_col] = df[TARGET].shift(-horizon_months)

    # --- Select features ---
    feature_cols = _select_features(df)
    logger.info(f"Selected {len(feature_cols)} features for horizon={horizon_months}m")
    logger.info(f"Features: {feature_cols}")

    # --- Drop rows where target or any Tier 1 feature is missing ---
    required_cols = [target_col] + [f for f in TIER1_FEATURES if f in df.columns]
    df_clean = df[feature_cols + [target_col]].dropna(subset=required_cols)
    logger.info(f"Clean rows after dropping missing required columns: {len(df_clean)}")

    if len(df_clean) < 24:
        logger.warning(
            f"Only {len(df_clean)} clean observations — model will have limited reliability. "
            "Minimum recommended: 36 months."
        )

    # --- Time series train/test split ---
    # Use last 12 months as test set, everything before as train.
    # Never random split a time series — that causes lookahead bias.
    test_cutoff = df_clean.index.max() - pd.DateOffset(months=12)
    train = df_clean[df_clean.index <= test_cutoff]
    test  = df_clean[df_clean.index >  test_cutoff]

    logger.info(f"Train: {len(train)} rows ({train.index.min().date()} to {train.index.max().date()})")
    logger.info(f"Test:  {len(test)} rows ({test.index.min().date()} to {test.index.max().date()})")

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test  = test[feature_cols]
    y_test  = test[target_col]

    # Save splits for reference
    train.to_csv(os.path.join(PROCESSED_DIR, f"train_h{horizon_months}.csv"))
    test.to_csv(os.path.join(PROCESSED_DIR, f"test_h{horizon_months}.csv"))
    logger.info(f"Train/test splits saved to {PROCESSED_DIR}/")

    return X_train, y_train, X_test, y_test, feature_cols


def _select_features(df: pd.DataFrame) -> list[str]:
    """
    Selects features based on availability and coverage thresholds.
    Tier 1 always included. Tier 2 included if present.
    Tier 3 included only if >50% non-null coverage.
    """
    selected = []

    for col in TIER1_FEATURES:
        if col in df.columns:
            selected.append(col)
        else:
            logger.warning(f"Tier 1 feature missing: {col}")

    for col in TIER2_FEATURES:
        if col in df.columns:
            selected.append(col)

    for col in TIER3_FEATURES:
        if col in df.columns:
            coverage = df[col].notna().mean()
            if coverage >= 0.5:
                selected.append(col)
            else:
                logger.info(f"Tier 3 feature excluded (coverage {coverage:.0%}): {col}")

    return selected


def get_prediction_row(horizon_months: int = 1) -> pd.DataFrame:
    """
    Returns the most recent row of features for making a live prediction.
    This is what you pass to model.predict() to get the next forecast.

    For horizon=1: predicts next month's AN price.
    For horizon=12: predicts AN price 12 months from now.
    """
    features_path = os.path.join(PROCESSED_DIR, "model_features.csv")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)

    _, _, _, _, feature_cols = prepare(horizon_months)

    # Most recent complete row
    latest = df[feature_cols].dropna().tail(1)

    if latest.empty:
        raise ValueError(
            "No complete feature row available for prediction. "
            "Check that ingestion has run recently and features are populated."
        )

    logger.info(f"Prediction row date: {latest.index[0].date()}")
    return latest


if __name__ == "__main__":
    for horizon in [1, 12]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Preparing data for {horizon}-month horizon")
        logger.info(f"{'='*40}")
        X_train, y_train, X_test, y_test, features = prepare(horizon)
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape:  {X_test.shape}")
