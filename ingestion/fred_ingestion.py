"""
FRED ingestion module.

Pulls: NBP gas, TTF gas, Brent crude, USD/GBP FX, urea price.
Source: Federal Reserve Bank of St. Louis FRED API.
API key: free at https://fredaccount.stlouisfed.org/apikeys

Note on gas series:
- FRED's NBP and TTF series (PNGASUKUSDM, PNGASEUUSDM) are monthly.
- For higher-frequency gas signals, GIE AGSI+ storage data (daily) is a
  better proxy. We pull both and let the processing layer handle alignment.
"""

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from utils.config import FRED_SERIES, START_DATE, END_DATE
from utils.db import write_dataframe
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def run():
    """Pull all configured FRED series and write to fred_raw table."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error(
            "FRED_API_KEY not found in environment. "
            "Add it to your .env file. Get one free at: "
            "https://fredaccount.stlouisfed.org/apikeys"
        )
        return

    fred = Fred(api_key=api_key)
    retrieved_at = datetime.utcnow().isoformat()
    all_rows = []

    for series_name, series_id in FRED_SERIES.items():
        try:
            logger.info(f"Pulling FRED series: {series_id} ({series_name})")
            data = fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=END_DATE,
            )

            if data.empty:
                logger.warning(f"No data returned for {series_id}")
                continue

            # fred.get_series returns a pandas Series with DatetimeIndex
            series_df = data.reset_index()
            series_df.columns = ["data_date", "value"]
            series_df["series_id"] = series_id
            series_df["series_name"] = series_name
            series_df["retrieved_at"] = retrieved_at
            series_df["data_date"] = series_df["data_date"].dt.strftime("%Y-%m-%d")

            # Drop NaN values (FRED uses NaN for missing observations)
            series_df = series_df.dropna(subset=["value"])

            all_rows.append(series_df)
            logger.info(f"  -> {len(series_df)} observations from {series_df['data_date'].min()} to {series_df['data_date'].max()}")

        except Exception as e:
            # Log and continue — don't let one failed series kill the whole run
            logger.error(f"Failed to pull {series_id}: {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        write_dataframe(combined, "fred_raw")
        logger.info(f"FRED ingestion complete — {len(combined)} total rows written.")
    else:
        logger.warning("FRED ingestion completed with no data written.")


if __name__ == "__main__":
    from utils.db import initialise_tables
    initialise_tables()
    run()
