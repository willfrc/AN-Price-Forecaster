"""
AHDB UK AN price ingestion module.

Source: AHDB UK Fertiliser Price Tracker
URL: https://ahdb.org.uk/dairy/uk-fertiliser-price-tracker

IMPORTANT — manual step required:
    AHDB does not provide a public API. You must manually download the
    price data from the URL above and save it as:
        data/raw/ahdb_an_prices.csv

    Expected CSV format (adjust _load_ahdb_csv() if your download differs):
        Date,Price_GBP_tonne
        2024-01-05,320.00
        2024-01-12,318.50
        ...

    If AHDB provides an Excel file, rename to .xlsx and update the loader
    to use pd.read_excel() — comment is in the code below.

This is your TARGET VARIABLE — the series you are forecasting.
It's the most important data in the pipeline. If the format is wrong,
fix it here before touching anything else.
"""

import os
from datetime import datetime

import pandas as pd

from utils.config import AHDB_FILE_PATH, START_DATE
from utils.db import write_dataframe
from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    """Load AHDB AN price CSV and write to ahdb_raw table."""
    if not os.path.exists(AHDB_FILE_PATH):
        logger.error(
            f"AHDB file not found at: {AHDB_FILE_PATH}\n"
            "Download the UK fertiliser price tracker from:\n"
            "https://ahdb.org.uk/dairy/uk-fertiliser-price-tracker\n"
            "Save as data/raw/ahdb_an_prices.csv before running ingestion."
        )
        return

    retrieved_at = datetime.utcnow().isoformat()

    try:
        df = _load_ahdb_file(AHDB_FILE_PATH)
    except Exception as e:
        logger.error(f"Failed to load AHDB file: {e}")
        return

    if df.empty:
        logger.warning("AHDB file loaded but no valid rows found.")
        return

    df = df[df["data_date"] >= START_DATE]
    df["retrieved_at"] = retrieved_at
    df["product"] = "AN_34.5N_bulk"

    write_dataframe(df, "ahdb_raw")
    logger.info(
        f"AHDB ingestion complete — {len(df)} rows written "
        f"({df['data_date'].min()} to {df['data_date'].max()})"
    )


def _load_ahdb_file(filepath: str) -> pd.DataFrame:
    """
    Loads and normalises the AHDB price file.

    Confirmed structure:
    - Rows 0-12: metadata and blanks (skip)
    - Row 13:    header row (Month, AN UK produced, AN imported, Urea, ...)
    - Row 14+:   data, date in column 1 as 'Jan-17' format
    - Column 2:  AN UK produced 34.5% N — this is our target variable
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath, header=None)
    else:
        raw = pd.read_csv(filepath, header=None)

    logger.info(f"Loaded AHDB file: {filepath} — {len(raw)} raw rows")

    # Row 13 is the header, data starts row 14
    data = raw.iloc[14:].copy()
    data = data.reset_index(drop=True)

    # Column 1 = date, column 2 = AN UK produced price
    date_col   = 1
    price_col  = 2

    df = pd.DataFrame()
    df["data_date"]   = pd.to_datetime(
        data[date_col].astype(str).str.strip(),
        format="%b-%y",
        errors="coerce"
    )
    df["price_gbp_t"] = pd.to_numeric(data[price_col], errors="coerce")

    df = df.dropna(subset=["data_date", "price_gbp_t"])
    df["data_date"] = df["data_date"].dt.strftime("%Y-%m-%d")

    logger.info(f"AHDB: parsed {len(df)} valid rows from {df['data_date'].min()} to {df['data_date'].max()}")
    return df


if __name__ == "__main__":
    from utils.db import initialise_tables
    initialise_tables()
    run()
