"""
Feature engineering pipeline.

Takes raw data from SQLite, aligns everything to monthly frequency,
engineers features, and outputs a clean model-ready DataFrame.

Run this after run_ingestion.py:
    py processing/build_features.py

Output: data/processed/model_features.csv
        data/processed/model_features_weekly.csv  (interpolated, 0-3 month window)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.db import read_table
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = "data/processed"


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def build():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Processing pipeline started")
    logger.info("=" * 60)

    # Step 1 — load and align all sources to monthly
    monthly = _build_monthly_frame()

    if monthly.empty:
        logger.error("Monthly frame is empty — check raw data tables.")
        return

    logger.info(f"Monthly frame: {len(monthly)} rows | {monthly.index.min()} to {monthly.index.max()}")
    logger.info(f"Columns: {list(monthly.columns)}")

    # Step 2 — engineer features
    featured = _engineer_features(monthly)
    logger.info(f"After feature engineering: {len(featured)} rows, {len(featured.columns)} columns")

    # Step 3 — drop rows where target is missing (can't train on those)
    featured = featured.dropna(subset=["an_price_gbp_t"])
    logger.info(f"After dropping missing target: {len(featured)} rows")

    # Step 4 — save monthly model-ready dataset
    monthly_path = os.path.join(PROCESSED_DIR, "model_features.csv")
    featured.to_csv(monthly_path)
    logger.info(f"Monthly features saved: {monthly_path}")

    # Step 5 — interpolate to weekly for short-term forecasting window
    weekly = _interpolate_to_weekly(featured)
    weekly_path = os.path.join(PROCESSED_DIR, "model_features_weekly.csv")
    weekly.to_csv(weekly_path)
    logger.info(f"Weekly interpolated features saved: {weekly_path}")

    # Step 6 — print summary
    _print_summary(featured)

    return featured, weekly


# ---------------------------------------------------------------------------
# Step 1 — Load and align to monthly
# ---------------------------------------------------------------------------

def _build_monthly_frame() -> pd.DataFrame:
    """
    Loads all raw tables, resamples to monthly frequency (MS = month start),
    and joins into a single wide DataFrame indexed by date.

    Missing values are forward-filled within a 3-month window — beyond that
    they stay NaN so the model doesn't train on stale data.
    """
    frames = {}

    # --- AHDB: UK AN spot price (target variable) ---
    # Already monthly but may have gaps
    try:
        ahdb = read_table("ahdb_raw")
        if not ahdb.empty:
            ahdb["data_date"] = pd.to_datetime(ahdb["data_date"])
            ahdb = ahdb.set_index("data_date")["price_gbp_t"]
            ahdb = ahdb.resample("MS").mean()
            ahdb.name = "an_price_gbp_t"
            frames["an_price_gbp_t"] = ahdb
            logger.info(f"AHDB: {len(ahdb)} monthly observations")
        else:
            logger.warning("AHDB table empty — target variable missing")
    except Exception as e:
        logger.error(f"Failed to load AHDB: {e}")

    # --- FRED: gas, FX, Brent ---
    try:
        fred = read_table("fred_raw")
        if not fred.empty:
            fred["data_date"] = pd.to_datetime(fred["data_date"])
            fred_pivot = fred.pivot_table(
                index="data_date", columns="series_name", values="value", aggfunc="mean"
            )
            fred_monthly = fred_pivot.resample("MS").mean()

            # Rename to clean feature names
            rename_map = {
                "ttf_gas_monthly":   "ttf_gas_usd_mmbtu",
                "hh_gas_monthly":    "hh_gas_usd_mmbtu",
                "brent_crude_daily": "brent_usd_bbl",
                "usd_gbp_daily":     "usd_gbp_fx",
            }
            fred_monthly = fred_monthly.rename(columns=rename_map)

            for col in fred_monthly.columns:
                frames[col] = fred_monthly[col]

            logger.info(f"FRED: {len(fred_monthly)} monthly rows, columns: {list(fred_monthly.columns)}")
    except Exception as e:
        logger.error(f"Failed to load FRED: {e}")

    # --- World Bank: urea, DAP, phosphate, European gas ---
    try:
        wb = read_table("worldbank_raw")
        if not wb.empty:
            wb["data_date"] = pd.to_datetime(wb["data_date"])
            wb_pivot = wb.pivot_table(
                index="data_date", columns="commodity", values="value", aggfunc="mean"
            )
            wb_monthly = wb_pivot.resample("MS").mean()

            rename_map = {
                "urea":           "wb_urea_usd_t",
                "dap":            "wb_dap_usd_t",
                "phosphate_rock": "wb_phosphate_usd_t",
                "gas_europe":     "wb_gas_europe_usd_mmbtu",
            }
            wb_monthly = wb_monthly.rename(columns=rename_map)

            for col in wb_monthly.columns:
                frames[col] = wb_monthly[col]

            logger.info(f"World Bank: {len(wb_monthly)} monthly rows, columns: {list(wb_monthly.columns)}")
    except Exception as e:
        logger.error(f"Failed to load World Bank: {e}")

    # --- yfinance: equity proxies (resample daily OHLCV to monthly close) ---
    try:
        yf = read_table("yfinance_raw")
        if not yf.empty:
            yf["data_date"] = pd.to_datetime(yf["data_date"])
            yf_pivot = yf.pivot_table(
                index="data_date", columns="ticker_name", values="close", aggfunc="mean"
            )
            yf_monthly = yf_pivot.resample("MS").mean()

            rename_map = {
                "cf_industries": "cf_close_usd",
                "yara":          "yara_close_nok",
                "soil_etf":      "soil_close_usd",
            }
            yf_monthly = yf_monthly.rename(columns=rename_map)

            for col in yf_monthly.columns:
                frames[col] = yf_monthly[col]

            logger.info(f"yfinance: {len(yf_monthly)} monthly rows, columns: {list(yf_monthly.columns)}")
    except Exception as e:
        logger.error(f"Failed to load yfinance: {e}")

    # --- GIE: gas storage (daily -> monthly mean % full) ---
    try:
        gie = read_table("gie_raw")
        if not gie.empty:
            gie["data_date"] = pd.to_datetime(gie["data_date"])
            gie = gie.set_index("data_date")

            gie_monthly = gie[["gas_in_storage", "full_pct"]].resample("MS").mean()
            gie_monthly.columns = ["eu_gas_storage_twh", "eu_gas_storage_pct"]

            for col in gie_monthly.columns:
                frames[col] = gie_monthly[col]

            logger.info(f"GIE: {len(gie_monthly)} monthly rows")
    except Exception as e:
        logger.error(f"Failed to load GIE: {e}")

    if not frames:
        return pd.DataFrame()

    # Join all series into one wide frame
    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    # Forward-fill gaps up to 3 months — beyond that leave as NaN
    # This handles months where AHDB or World Bank didn't publish
    combined = combined.ffill(limit=3)

    logger.info(f"Combined monthly frame: {combined.shape}")
    _log_coverage(combined)

    return combined


def _log_coverage(df: pd.DataFrame):
    """Logs % non-null per column so you can see data gaps at a glance."""
    logger.info("Data coverage per feature:")
    for col in df.columns:
        pct = df[col].notna().mean() * 100
        logger.info(f"  {col:<35} {pct:.0f}% complete")


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds model features from the aligned monthly frame.

    Feature categories:
    1. Lagged target (AN price 1, 2, 3 months ago)
    2. Gas-to-AN spread proxy (gas cost as % of AN price)
    3. Momentum (month-on-month % change)
    4. Seasonality (sine/cosine encoding of month)
    5. Gas storage deviation from seasonal norm
    6. FX-adjusted urea price (GBP)
    7. Rolling means (3-month, 6-month)
    """
    f = df.copy()

    # --- 1. Lagged target variable ---
    # Critical: these are the only lags that don't cause lookahead bias
    # Lag 1 = last month's price (available when forecasting next month)
    for lag in [1, 2, 3, 6]:
        f[f"an_lag_{lag}m"] = f["an_price_gbp_t"].shift(lag)

    # --- 2. Gas cost proxy in GBP ---
    # TTF gas is the primary AN cost driver; convert to GBP for relevance
    if "ttf_gas_usd_mmbtu" in f.columns and "usd_gbp_fx" in f.columns:
        f["ttf_gas_gbp_mmbtu"] = f["ttf_gas_usd_mmbtu"] / f["usd_gbp_fx"]
        # Lagged gas price — gas costs feed through to AN with ~1-2 month lag
        f["ttf_gas_gbp_lag1m"] = f["ttf_gas_gbp_mmbtu"].shift(1)
        f["ttf_gas_gbp_lag2m"] = f["ttf_gas_gbp_mmbtu"].shift(2)

    # --- 3. Brent in GBP ---
    if "brent_usd_bbl" in f.columns and "usd_gbp_fx" in f.columns:
        f["brent_gbp_bbl"] = f["brent_usd_bbl"] / f["usd_gbp_fx"]

    # --- 4. Urea price in GBP (global AN substitute anchor) ---
    if "wb_urea_usd_t" in f.columns and "usd_gbp_fx" in f.columns:
        f["urea_gbp_t"] = f["wb_urea_usd_t"] / f["usd_gbp_fx"]
        f["urea_gbp_lag1m"] = f["urea_gbp_t"].shift(1)

    # --- 5. Momentum features ---
    # Month-on-month % change in AN price
    f["an_mom_1m"] = f["an_price_gbp_t"].pct_change(1)
    f["an_mom_3m"] = f["an_price_gbp_t"].pct_change(3)

    # Gas price momentum
    if "ttf_gas_gbp_mmbtu" in f.columns:
        f["gas_mom_1m"] = f["ttf_gas_gbp_mmbtu"].pct_change(1)

    # --- 6. Rolling means ---
    f["an_roll_3m"]  = f["an_price_gbp_t"].shift(1).rolling(3).mean()
    f["an_roll_6m"]  = f["an_price_gbp_t"].shift(1).rolling(6).mean()

    if "ttf_gas_gbp_mmbtu" in f.columns:
        f["gas_roll_3m"] = f["ttf_gas_gbp_mmbtu"].shift(1).rolling(3).mean()

    # --- 7. Seasonality encoding ---
    # Sine/cosine captures cyclical nature of month without ordinal assumption
    # AN demand peaks: Feb-Apr (spring application) and Aug-Sep (autumn)
    f["month_sin"] = np.sin(2 * np.pi * f.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * f.index.month / 12)
    f["month"] = f.index.month  # Also keep raw month for tree-based models

    # --- 8. Gas storage deviation from seasonal norm ---
    # How full is EU storage vs typical for that time of year?
    # Low storage in autumn = upward gas price pressure = higher AN costs
    if "eu_gas_storage_pct" in f.columns:
        # Only compute if we have enough data (GIE currently limited to ~10 months)
        if f["eu_gas_storage_pct"].notna().sum() >= 6:
            monthly_mean = f.groupby(f.index.month)["eu_gas_storage_pct"].transform("mean")
            f["gas_storage_deviation"] = f["eu_gas_storage_pct"] - monthly_mean
            f["gas_storage_lag1m"] = f["eu_gas_storage_pct"].shift(1)
        else:
            logger.warning(
                "GIE storage data too limited for seasonal deviation feature "
                "(need 6+ months, have fewer). Feature excluded. "
                "Resolve GIE API key to unlock this feature."
            )

    # --- 9. AN-to-urea spread ---
    # Measures how UK AN price tracks global urea benchmark
    if "urea_gbp_t" in f.columns:
        f["an_urea_spread"] = f["an_price_gbp_t"] - f["urea_gbp_t"]

    # --- 10. Year (for trend capture) ---
    f["year"] = f.index.year

    logger.info("Feature engineering complete.")
    return f


# ---------------------------------------------------------------------------
# Step 3 — Interpolate to weekly
# ---------------------------------------------------------------------------

def _interpolate_to_weekly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the monthly feature DataFrame to weekly frequency.

    Method: cubic spline interpolation for continuous series (prices, rates),
    forward-fill for categorical/discrete features (month, year).

    This is used for the 0-3 month forecast window only.
    The interpolated weekly values are synthetic — label them clearly
    when presenting results to avoid implying false precision.
    """
    # Reindex to weekly frequency
    weekly_index = pd.date_range(
        start=monthly_df.index.min(),
        end=monthly_df.index.max(),
        freq="W-MON"  # Weekly on Mondays
    )

    # Reindex monthly to weekly, then interpolate
    weekly = monthly_df.reindex(monthly_df.index.union(weekly_index))
    weekly = weekly.sort_index()

    # Continuous series: cubic spline interpolation
    continuous_cols = [
        c for c in weekly.columns
        if c not in ["month", "year", "month_sin", "month_cos"]
    ]
    weekly[continuous_cols] = weekly[continuous_cols].interpolate(
        method="cubic", limit_direction="forward"
    )

    # Discrete/categorical: forward fill
    discrete_cols = ["month", "year"]
    for col in discrete_cols:
        if col in weekly.columns:
            weekly[col] = weekly[col].ffill()

    # Recompute sine/cosine from actual weekly dates (more accurate than interpolating)
    weekly["month_sin"] = np.sin(2 * np.pi * weekly.index.month / 12)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly.index.month / 12)

    # Keep only the weekly index rows (drop the original monthly anchor points
    # that aren't on Mondays, to avoid duplicates)
    weekly = weekly.reindex(weekly_index)

    logger.info(f"Weekly interpolation: {len(weekly)} weeks from {weekly.index.min().date()} to {weekly.index.max().date()}")
    return weekly


# ---------------------------------------------------------------------------
# Step 4 — Summary report
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame):
    """Prints a clean summary of the model-ready dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSED DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Date range:      {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"Observations:    {len(df)} monthly rows")
    logger.info(f"Features:        {len(df.columns)} columns")
    logger.info(f"Target (AN price): min={df['an_price_gbp_t'].min():.0f}, "
                f"max={df['an_price_gbp_t'].max():.0f}, "
                f"mean={df['an_price_gbp_t'].mean():.0f} GBP/tonne")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("\nColumns with missing values:")
        for col, count in missing.items():
            logger.info(f"  {col:<35} {count} missing ({count/len(df)*100:.0f}%)")
    else:
        logger.info("No missing values in processed dataset.")


if __name__ == "__main__":
    build()
