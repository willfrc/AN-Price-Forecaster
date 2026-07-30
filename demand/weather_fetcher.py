"""
Weather data fetcher using Open-Meteo API.

Retrieves historical and forecast rainfall (and temperature) for a given
lat/lon. No API key required — Open-Meteo is free for non-commercial use.

API docs: https://open-meteo.com/en/docs

We pull two datasets:
    1. Historical daily rainfall (past 90 days) — establishes soil moisture context
    2. 16-day forecast daily rainfall — drives application timing adjustments

For the Phase 2 weather adjuster we need:
    - Daily precipitation (mm) for the forecast window
    - Whether any forecast application window has heavy rain (>10mm/day)
    - Soil temperature proxy (air temp as indicator of ground conditions)

Open-Meteo returns data in JSON. We parse into a clean DataFrame.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import pandas as pd
import requests

from utils.logger import get_logger

logger = get_logger(__name__)

OPEN_METEO_BASE = "https://api.open-meteo.com/v1"

# Thresholds for application suitability (mm/day)
RAIN_HEAVY_THRESHOLD   = 10.0   # >10mm/day = unsuitable for application
RAIN_MODERATE_THRESHOLD = 5.0   # 5-10mm/day = marginal, flag as caution
RAIN_LIGHT_THRESHOLD    = 2.0   # <2mm/day = suitable


def fetch_forecast_weather(lat: float, lon: float,
                           days_ahead: int = 16) -> pd.DataFrame:
    """
    Fetches 16-day daily weather forecast from Open-Meteo.

    Args:
        lat:        Latitude (WGS84)
        lon:        Longitude (WGS84)
        days_ahead: Number of forecast days (max 16 on free tier)

    Returns:
        DataFrame with columns: date, precipitation_mm, temp_max_c,
        temp_min_c, application_suitable (bool)
    """
    days_ahead = min(days_ahead, 16)

    params = {
        "latitude":             lat,
        "longitude":            lon,
        "daily":                "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "forecast_days":        days_ahead,
        "timezone":             "Europe/London",
    }

    try:
        response = requests.get(f"{OPEN_METEO_BASE}/forecast", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        df = _parse_daily_response(data)
        logger.info(f"Forecast weather fetched: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo forecast request failed: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        logger.error(f"Open-Meteo forecast parse error: {e}")
        return pd.DataFrame()


def fetch_historical_weather(lat: float, lon: float,
                             days_back: int = 90) -> pd.DataFrame:
    """
    Fetches historical daily weather from Open-Meteo (ERA5 reanalysis).
    Used to establish antecedent soil moisture context.

    Args:
        lat:      Latitude
        lon:      Longitude
        days_back: How many days of history to fetch (default 90)

    Returns:
        DataFrame with daily weather data
    """
    end_date   = date.today() - timedelta(days=7)  # ERA5 archive lags ~5-7 days    
    start_date = end_date - timedelta(days=days_back)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily":      "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone":   "Europe/London",
    }

    try:
        response = requests.get(f"{OPEN_METEO_BASE}/archive", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        df = _parse_daily_response(data)
        logger.info(f"Historical weather fetched: {len(df)} days ending {end_date}")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo historical request failed: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        logger.error(f"Open-Meteo historical parse error: {e}")
        return pd.DataFrame()


def fetch_monthly_climatology(lat: float, lon: float,
                              year: int = None) -> pd.DataFrame:
    """
    Fetches monthly aggregated weather for the past 3 years to build
    a seasonal rainfall profile. Used by the weather adjuster to
    assess whether a given month is typically wet or dry.

    Returns:
        DataFrame with monthly mean precipitation and temperature,
        indexed by month (1-12)
    """
    if year is None:
        year = date.today().year

    start_date = date(year - 3, 1, 1)
    end_date   = date(year - 2, 12, 31)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily":      "precipitation_sum,temperature_2m_max",
        "timezone":   "Europe/London",
    }

    try:
        response = requests.get(f"{OPEN_METEO_BASE}/archive", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = _parse_daily_response(data)

        # Aggregate to monthly mean
        df["month"] = pd.to_datetime(df["date"]).dt.month
        monthly = df.groupby("month").agg(
            mean_precip_mm = ("precipitation_mm", "mean"),
            total_precip_mm = ("precipitation_mm", "sum"),
            mean_temp_max_c = ("temp_max_c", "mean"),
        ).reset_index()

        # Normalise total to 30-day month equivalent
        monthly["mean_precip_mm"] = monthly["mean_precip_mm"].round(2)

        logger.info(f"Monthly climatology built from {start_date} to {end_date}")
        return monthly

    except Exception as e:
        logger.error(f"Monthly climatology fetch failed: {e}")
        return pd.DataFrame()


def _parse_daily_response(data: dict) -> pd.DataFrame:
    """
    Parses Open-Meteo daily JSON response into a clean DataFrame.
    Adds application_suitable boolean based on precipitation threshold.
    """
    daily = data.get("daily", {})

    df = pd.DataFrame({
        "date":            daily.get("time", []),
        "precipitation_mm": daily.get("precipitation_sum", []),
        "temp_max_c":      daily.get("temperature_2m_max", []),
        "temp_min_c":      daily.get("temperature_2m_min", []),
    })

    df["date"] = pd.to_datetime(df["date"])

    # Fill None values (Open-Meteo uses null for future dates sometimes)
    df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0)
    df["temp_max_c"]       = pd.to_numeric(df["temp_max_c"], errors="coerce")
    df["temp_min_c"]       = pd.to_numeric(df["temp_min_c"], errors="coerce")

    # Application suitability flags
    df["application_suitable"] = (
        (df["precipitation_mm"] < RAIN_HEAVY_THRESHOLD) &
        (df["temp_max_c"] > 0)   # No application on frozen ground
    )
    df["rain_caution"] = (
        (df["precipitation_mm"] >= RAIN_MODERATE_THRESHOLD) &
        (df["precipitation_mm"] < RAIN_HEAVY_THRESHOLD)
    )

    return df


def get_monthly_rainfall_forecast(lat: float, lon: float) -> dict[int, float]:
    """
    Builds a month -> expected_rainfall_mm lookup for the next 12 months.

    Combines:
    - 16-day actual forecast (high confidence)
    - Historical monthly climatology (for months beyond the 16-day window)

    Returns:
        Dict mapping calendar month (1-12) to expected total monthly rainfall (mm)
    """
    forecast = fetch_forecast_weather(lat, lon, days_ahead=16)
    climatology = fetch_monthly_climatology(lat, lon)

    monthly_rainfall = {}

    # Use climatology as the base for all months
    if not climatology.empty:
        for _, row in climatology.iterrows():
            monthly_rainfall[int(row["month"])] = float(row["mean_precip_mm"]) * 30

    # Override near-term months with actual forecast aggregation
    if not forecast.empty:
        forecast["month"] = forecast["date"].dt.month
        fc_monthly = forecast.groupby("month")["precipitation_mm"].sum()
        for month, total in fc_monthly.items():
            monthly_rainfall[int(month)] = float(total)

    return monthly_rainfall
