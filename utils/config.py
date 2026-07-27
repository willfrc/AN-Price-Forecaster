"""
Central configuration for the AN price forecasting pipeline.
Adjust START_DATE to control how much history to pull on first run.
"""

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

# Pull 5 years of history on first run — sufficient for seasonal patterns
START_DATE = "2019-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# FRED series IDs
# Reference: https://fred.stlouisfed.org/
# ---------------------------------------------------------------------------

FRED_SERIES = {
    # Energy — primary AN cost driver
    "hh_gas_monthly":   "MHHNGSP",    # UHenry Hub Natural Gas Spot Price, USD/MMBtu, monthly
    "ttf_gas_monthly":   "PNGASEUUSDM",    # EU TTF natural gas price, USD/MMBtu, monthly
    "brent_crude_daily": "DCOILBRENTEU",   # Brent crude oil, USD/barrel, daily

    # FX — fertilizer trades globally in USD, forecasting in GBP
    "usd_gbp_daily":     "DEXUSUK",        # USD/GBP exchange rate, daily

    # Global commodity benchmarks
    # "urea_monthly":      "PUREA_MONTHLY",  # Urea price, USD/tonne, monthly (World Bank via FRED)
}

# ---------------------------------------------------------------------------
# yfinance tickers
# Reference: https://finance.yahoo.com/
# ---------------------------------------------------------------------------

YFINANCE_TICKERS = {
    # Fertilizer sector equity proxies (sentiment / production economics signal)
    "cf_industries":  "CF",        # CF Industries — major nitrogen fertilizer producer
    "yara":           "YAR.OL",    # Yara International — largest AN producer globally
    "soil_etf":       "SOIL",      # Global X Fertilizers/Potash ETF

    # CME urea futures front month — directional proxy only, treat with caution
    "cme_urea_fut":   "UBU=F",
}

# ---------------------------------------------------------------------------
# GIE AGSI+ — EU gas storage
# API docs: https://agsi.gie.eu/api-docs/
# ---------------------------------------------------------------------------

GIE_BASE_URL = "https://agsi.gie.eu/api"

# Europe aggregate storage (most relevant for NBP/TTF correlation)
GIE_COUNTRY_CODE = "eu"

# ---------------------------------------------------------------------------
# World Bank Pink Sheet
# API docs: https://datacatalog.worldbank.org/dataset/commodity-price-data
# ---------------------------------------------------------------------------

WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2/en/indicator"

WORLD_BANK_INDICATORS = {
    "urea":    "PNGASUKUSDM",   # Placeholder — WB Pink Sheet via direct download
    "dap":     "PDAP_USD",      # DAP price, USD/tonne
    "ammonia": "PNRG_USD",      # Ammonia price, USD/tonne (upstream of AN)
}

# World Bank Pink Sheet direct download URL (Excel)
WORLD_BANK_PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59255601b2"
    "-0050012023/original/CMO-Historical-Data-Annual.xlsx"
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = "data/an_price_data.sqlite"

# Table names — one per source, raw data stored as-is
DB_TABLES = {
    "fred":       "fred_raw",
    "yfinance":   "yfinance_raw",
    "gie":        "gie_raw",
    "worldbank":  "worldbank_raw",
    "ahdb":       "ahdb_raw",
    "ember":      "ember_raw",
}

# ---------------------------------------------------------------------------
# AHDB manual file path
# Download from: https://ahdb.org.uk/dairy/uk-fertiliser-price-tracker
# ---------------------------------------------------------------------------

AHDB_FILE_PATH = "data/raw/ahdb_an_prices.csv"

# ---------------------------------------------------------------------------
# Ember ETS manual file path
# Download from: https://ember-climate.org/data/carbon-price-viewer/
# ---------------------------------------------------------------------------

# EMBER_FILE_PATH = "data/raw/ember_ets_carbon_price.csv"
