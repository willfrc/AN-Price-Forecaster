"""
Ember Climate ETS carbon price ingestion module.

Source: Ember Climate Carbon Price Viewer
URL: https://ember-climate.org/data/carbon-price-viewer/

IMPORTANT — manual step required:
    Ember does not provide a public API. Download the EU ETS carbon price
    CSV from the URL above and save as:
        data/raw/ember_ets_carbon_price.csv

    Expected CSV format:
        Date,Price_EUR_tonne
        2024-01-05,65.32
        2024-01-12,66.10
        ...

Why this matters for AN forecasting:
    EU ETS carbon costs affect the production economics of European
    nitrogen fertilizer producers (particularly those with older,
    less efficient plants). Higher carbon prices increase production
    costs and can reduce European supply, pushing up UK import prices.
    The effect is indirect and lagged — treat as a Tier 2 feature.
"""

import os
from datetime import datetime

import pandas as pd

from utils.config import EMBER_FILE_PATH, START_DATE
from utils.db import write_dataframe
from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    """Load Ember ETS carbon price CSV and write to ember_raw table."""
    if not os.path.exists(EMBER_FILE_PATH):
        logger.error(
            f"Ember ETS file not found at: {EMBER_FILE_PATH}\n"
            "Download the EU ETS carbon price data from:\n"
            "https://ember-climate.org/data/carbon-price-viewer/\n"
            "Save as data/raw/ember_ets_carbon_price.csv before running ingestion."
        )
        return

    retrieved_at = datetime.utcnow().isoformat()

    try:
        df = _load_ember_file(EMBER_FILE_PATH)
    except Exception as e:
        logger.error(f"Failed to load Ember file: {e}")
        return

    if df.empty:
        logger.warning("Ember file loaded but no valid rows found.")
        return

    df = df[df["data_date"] >= START_DATE]
    df["retrieved_at"] = retrieved_at

    write_dataframe(df, "ember_raw")
    logger.info(
        f"Ember ingestion complete — {len(df)} rows written "
        f"({df['data_date'].min()} to {df['data_date'].max()})"
    )


def _load_ember_file(filepath: str) -> pd.DataFrame:
    """
    Loads and normalises the Ember ETS price file.
    Handles common column name variations in Ember exports.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath, header=0)
    else:
        raw = pd.read_csv(filepath, header=0)

    logger.debug(f"Ember raw columns: {list(raw.columns)}")

    date_col = _detect_date_column(raw)
    price_col = _detect_price_column(raw)

    if date_col is None or price_col is None:
        raise ValueError(
            f"Could not detect required columns. Found: {list(raw.columns)}\n"
            "Expected: a date column and a EUR/tonne price column."
        )

    df = pd.DataFrame()
    df["data_date"] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    df["price_eur_t"] = pd.to_numeric(raw[price_col], errors="coerce")

    df = df.dropna(subset=["data_date", "price_eur_t"])
    df["data_date"] = df["data_date"].dt.strftime("%Y-%m-%d")

    return df


def _detect_date_column(df: pd.DataFrame) -> str | None:
    candidates = ["Date", "date", "DATE", "date_local", "Period", "Time"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.8:
                return col
        except Exception:
            continue
    return None


def _detect_price_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Price_EUR_tonne", "price", "Price", "EUA Price",
        "EUR/tonne", "€/tonne", "Carbon Price", "ETS Price",
        "EU ETS", "Close", "Value",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.select_dtypes(include=["number"]).columns:
        if "date" not in col.lower():
            return col
    return None


if __name__ == "__main__":
    from utils.db import initialise_tables
    initialise_tables()
    run()
