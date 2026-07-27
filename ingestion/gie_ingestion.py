"""
GIE AGSI+ ingestion module.

Pulls: EU aggregate natural gas storage levels (daily).
Source: Gas Infrastructure Europe AGSI+ platform.
API docs: https://agsi.gie.eu/api-docs/

No API key required for basic access. GIE does rate-limit heavy usage.

Why this matters for AN forecasting:
- AN is ~70-80% gas cost (natural gas -> ammonia -> ammonium nitrate).
- EU gas storage levels are the primary forward indicator of European
  gas price direction, which feeds directly into AN production economics.
- Low storage = higher winter gas prices = higher AN costs = higher prices.
- This series is daily, making it the highest-frequency energy signal
  available for free (FRED gas series are only monthly).
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from utils.config import GIE_BASE_URL, GIE_COUNTRY_CODE, START_DATE, END_DATE
from utils.db import write_dataframe
from utils.logger import get_logger

logger = get_logger(__name__)

# GIE paginates results — 30 days per page is a safe batch size
BATCH_DAYS = 30
REQUEST_DELAY_SECONDS = 1  # Be polite to the API


def _fetch_gie_page(date_from: str, date_to: str) -> list[dict]:
    """
    Fetches one page of GIE storage data for the EU aggregate.

    GIE AGSI+ now requires a free API key. Register at:
    https://agsi.gie.eu/ -> click 'API Key' in the top menu.
    Add to your .env file as: GIE_API_KEY=your_key_here

    Args:
        date_from: ISO date string (YYYY-MM-DD)
        date_to:   ISO date string (YYYY-MM-DD)

    Returns:
        List of raw data dicts from the API response, or empty list on failure.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GIE_API_KEY")

    url = f"{GIE_BASE_URL}/data/{GIE_COUNTRY_CODE}"
    params = {
        "date_from": date_from,
        "date_to":   date_to,
        "size":      300,
    }
    headers = {}
    if api_key:
        headers["x-key"] = api_key
    else:
        logger.warning(
            "GIE_API_KEY not set in .env — requests may be rejected. "
            "Register free at https://agsi.gie.eu/"
        )

    try:
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    except requests.exceptions.HTTPError as e:
        logger.error(f"GIE HTTP error for {date_from} to {date_to}: {e}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"GIE request failed for {date_from} to {date_to}: {e}")
        return []
    except ValueError as e:
        logger.error(f"GIE JSON parse error: {e}")
        return []


def run():
    """
    Pull EU gas storage data in monthly batches and write to gie_raw table.
    Batches to avoid overloading the API and hitting rate limits.
    """
    retrieved_at = datetime.utcnow().isoformat()

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    all_rows = []
    current = start

    logger.info(f"Pulling GIE AGSI+ EU gas storage from {START_DATE} to {END_DATE}")

    from dotenv import load_dotenv
    import os
    load_dotenv()
    logger.debug(f"GIE API key present: {bool(os.getenv('GIE_API_KEY'))}")

    while current <= end:
        batch_end = min(current + timedelta(days=BATCH_DAYS), end)
        date_from_str = current.strftime("%Y-%m-%d")
        date_to_str = batch_end.strftime("%Y-%m-%d")

        logger.debug(f"  Fetching batch: {date_from_str} to {date_to_str}")
        records = _fetch_gie_page(date_from_str, date_to_str)

        for record in records:
            # GIE response fields (field names may vary slightly by API version)
            # Log raw record on first iteration to catch any field name changes
            all_rows.append({
                "data_date":       record.get("gasDayStart", record.get("date", "")),
                "country_code":    GIE_COUNTRY_CODE.upper(),
                "gas_in_storage":  _safe_float(record.get("gasInStorage")),
                "full_pct":        _safe_float(record.get("full")),
                "injection":       _safe_float(record.get("injection")),
                "withdrawal":      _safe_float(record.get("withdrawal")),
                "retrieved_at":    retrieved_at,
            })

        current = batch_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY_SECONDS)

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Normalise date format
        df["data_date"] = pd.to_datetime(df["data_date"]).dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["gas_in_storage"])

        write_dataframe(df, "gie_raw")
        logger.info(f"GIE ingestion complete — {len(df)} rows written.")
    else:
        logger.warning(
            "GIE ingestion returned no data. "
            "Check API availability at https://agsi.gie.eu and verify the endpoint. "
            "The GIE API structure can change — inspect _fetch_gie_page() if needed."
        )


def _safe_float(value) -> float | None:
    """Converts GIE string values to float, returns None on failure."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    from utils.db import initialise_tables
    initialise_tables()
    run()