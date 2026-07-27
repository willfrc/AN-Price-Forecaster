"""
World Bank Pink Sheet ingestion module.

Pulls: urea, DAP, and ammonia monthly prices (USD/tonne).
Source: World Bank Commodity Price Data (Pink Sheet).
Direct Excel download — no API key required.

Download URL: https://www.worldbank.org/en/research/commodity-markets
(Links to the "Download data" Excel file)

Why these commodities matter for AN forecasting:
- Urea:    direct substitute for AN in many markets; global price anchor.
- Ammonia: upstream feedstock for AN production; cost signal.
- DAP:     phosphate fertilizer; cross-commodity demand signal.
"""

import io
import os
from datetime import datetime

import pandas as pd
import requests

from utils.config import WORLD_BANK_PINK_SHEET_URL, START_DATE
from utils.db import write_dataframe
from utils.logger import get_logger

logger = get_logger(__name__)

# Map from Pink Sheet column names to our standard commodity labels
# These column names are as they appear in the CMO Historical Data workbook.
# If the World Bank updates the file format, check the 'Monthly Prices' sheet.
COMMODITY_COLUMN_MAP = {
    "Urea, E. Europe, bulk, spot, fob, $/mt":          "urea_eeurope",
    "Urea, US Gulf, bulk, spot, fob, $/mt":            "urea_us_gulf",
    "DAP, spot, f.o.b. US Gulf, $/mt":                 "dap_us_gulf",
    "Ammonia, spot, fob Black Sea, $/mt":              "ammonia_black_sea",
}


def run():
    """
    Downloads World Bank Pink Sheet Excel, extracts fertilizer price columns,
    and writes to worldbank_raw table.
    """
    retrieved_at = datetime.utcnow().isoformat()

    local_path = "data/raw/worldbank_pink_sheet.xlsx"

    if not os.path.exists(local_path):
        logger.error(
            f"World Bank Pink Sheet not found at {local_path}.\n"
            "Download manually from:\n"
            "  https://www.worldbank.org/en/research/commodity-markets\n"
            "Click 'Download data' and save the Excel file as:\n"
            "  data/raw/worldbank_pink_sheet.xlsx"
        )
        return
 
    logger.info(f"Loading World Bank Pink Sheet from {local_path}...")
 
    try:
        with open(local_path, "rb") as f:
            file_bytes = f.read()
 
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        logger.debug(f"Available sheets: {xl.sheet_names}")
 
        # Find the monthly prices sheet (name can vary slightly across vintages)
        sheet_name = next(
            (s for s in xl.sheet_names if "monthly" in s.lower()),
            xl.sheet_names[0]
        )
        logger.info(f"Reading sheet: '{sheet_name}'")
 
        # The Pink Sheet has metadata rows at the top — skip to find the header
        raw = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name,
            header=None,
        )
 
    except Exception as e:
        logger.error(f"Failed to parse Pink Sheet Excel: {e}")
        return
 
    # ---------------------------------------------------------------------------
    # Parse the Pink Sheet structure
    # The file has commodity names in row ~4, units in row ~5, then monthly data.
    # We find the date column and the commodity columns we want dynamically.
    # ---------------------------------------------------------------------------
 
    try:
        df = _parse_pink_sheet(raw)
    except Exception as e:
        logger.error(
            f"Pink Sheet parsing failed: {e}\n"
            "The World Bank occasionally reformats this file. "
            "Inspect data/raw/worldbank_pink_sheet_debug.xlsx if saved."
        )
        return
 
    if df.empty:
        logger.warning("No matching commodity columns found in Pink Sheet.")
        return
 
    # Filter to START_DATE onwards
    df = df[df["data_date"] >= START_DATE]
    df["retrieved_at"] = retrieved_at
 
    write_dataframe(df, "worldbank_raw")
    logger.info(f"World Bank ingestion complete — {len(df)} rows written.")
 
 
def _parse_pink_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the raw Pink Sheet DataFrame into a normalised long-format table.

    Structure confirmed:
    - Row 0-3: metadata (skip)
    - Row 4:   commodity names (header)
    - Row 5:   units (skip)
    - Row 6+:  data, date in column 0 as '1960M01' format
    """
    # Row 4 is the commodity header row
    header = raw.iloc[4].tolist()

    # Row 6 onwards is data
    data = raw.iloc[6:].copy()
    data.columns = [str(c).strip() if pd.notna(c) else c for c in header]
    data = data.reset_index(drop=True)

    # Date column is the first column (NaN label in header)
    date_col = data.columns[0]

    # Target commodities — short names matching actual Pink Sheet headers
    target_commodities = {
        "Urea":                "urea",
        "DAP":                 "dap",
        "Phosphate rock":      "phosphate_rock",
        "Natural gas, Europe": "gas_europe",
    }

    rows = []
    for col_name, commodity_label in target_commodities.items():
        if col_name not in data.columns:
            logger.warning(f"Column '{col_name}' not found in Pink Sheet — skipping {commodity_label}")
            continue

        series = data[[date_col, col_name]].copy()
        series.columns = ["raw_date", "value"]

        # Replace '…' (World Bank missing value marker) with NaN
        series["value"] = series["value"].replace("…", pd.NA)
        series["value"] = pd.to_numeric(series["value"], errors="coerce")
        series = series.dropna(subset=["value"])

        # Parse date format '1960M01' -> first day of that month
        series["data_date"] = pd.to_datetime(
            series["raw_date"].astype(str).str.replace("M", "-"),
            format="%Y-%m",
            errors="coerce"
        )
        series = series.dropna(subset=["data_date"])
        series["data_date"] = series["data_date"].dt.strftime("%Y-%m-%d")

        series["commodity"] = commodity_label
        series["unit"] = "USD/tonne"

        rows.append(series[["commodity", "data_date", "value", "unit"]])
        logger.info(f"  Parsed {len(series)} rows for {commodity_label}")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
 
 
# def _find_column(columns, target: str) -> str | None:
#     """
#     Fuzzy column finder — handles minor formatting differences between
#     Pink Sheet vintages. Tries exact match first, then keyword match.
#     """
#     # Exact match
#     if target in columns:
#         return target
 
#     # Keyword match on first meaningful word (e.g. "Urea", "DAP", "Ammonia")
#     keyword = target.split(",")[0].strip().lower()
#     matches = [c for c in columns if keyword in str(c).lower()]
#     if len(matches) == 1:
#         return matches[0]
#     elif len(matches) > 1:
#         # Return closest match by string length similarity
#         return min(matches, key=lambda c: abs(len(str(c)) - len(target)))
 
#     return None
 
 
if __name__ == "__main__":
    from utils.db import initialise_tables
    initialise_tables()
    run()   