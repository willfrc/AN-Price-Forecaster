"""
Weather adjustment layer — Phase 2 of the farm demand engine.

Takes the deterministic N schedule from Phase 1 and adjusts application
timing based on weather forecast and seasonal rainfall patterns.

Adjustment rules (derived from standard agronomic practice):

1. HEAVY RAIN RULE: If forecast rainfall in the application week exceeds
   10mm/day average, shift application forward by 1-2 weeks. Reason:
   heavy rain after AN application causes N leaching (nutrient loss and
   environmental risk).

2. WATERLOGGING RISK: If the preceding 30 days had >200mm total rainfall
   (wet antecedent conditions), flag as high leaching risk and recommend
   splitting the application or reducing rate.

3. FROST RULE: If forecast min temperature < 0°C in the application week,
   delay application. AN applied to frozen ground cannot be incorporated
   and causes surface runoff risk.

4. DROUGHT ADJUSTMENT: If the month is historically dry (<30mm typical),
   flag that N uptake may be limited by moisture stress. No timing change
   but adds a yield risk note.

5. NVZ INTERACTION: If an adjusted date falls inside an NVZ closed period,
   flag explicitly rather than silently moving it — farmer must make
   the compliance decision.

Output is an AdjustedSchedule — the Phase 1 schedule with timing
adjustments, confidence flags, and agronomic notes per application.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import math

from demand.n_calculator import NRequirementResult, MonthlyApplication
from demand.weather_fetcher import (
    fetch_forecast_weather,
    fetch_historical_weather,
    get_monthly_rainfall_forecast,
    RAIN_HEAVY_THRESHOLD,
)
from demand.rb209_tables import NVZ_CLOSED_PERIODS, SUPPORTED_CROPS, _in_closed_period
from utils.logger import get_logger

logger = get_logger(__name__)

# Adjustment thresholds
ANTECEDENT_WET_THRESHOLD_MM   = 200   # mm over 30 days = waterlogged risk
FROST_TEMP_THRESHOLD_C        = 0.0   # Min temp below this = no application
DRY_MONTH_THRESHOLD_MM        = 30    # Monthly total below this = drought flag
MAX_WEEKS_SHIFT               = 3     # Maximum weeks we'll shift an application


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdjustedApplication:
    """A single application event after weather adjustment."""
    # Original schedule from Phase 1
    original:           MonthlyApplication

    # Adjusted timing
    adjusted_month:     int
    adjusted_month_name: str
    weeks_shifted:      int          # +ve = delayed, -ve = brought forward

    # Weather context
    forecast_precip_mm: Optional[float]   # Forecast rainfall for that period
    antecedent_wet:     bool              # Was the preceding 30 days wet?
    frost_risk:         bool              # Frost forecast in application window?

    # Flags
    timing_changed:     bool
    nvz_conflict:       bool              # Adjusted date is in NVZ closed period
    confidence:         str              # "high" / "medium" / "low"

    # Agronomic notes
    notes:              list[str] = field(default_factory=list)


@dataclass
class AdjustedSchedule:
    """Full adjusted schedule with weather context."""
    farm_result:        NRequirementResult
    applications:       list[AdjustedApplication]
    weather_available:  bool
    assessment_date:    date
    notes:              list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "WEATHER-ADJUSTED APPLICATION SCHEDULE",
            f"Assessment date: {self.assessment_date.strftime('%d %B %Y')}",
            f"Weather data:    {'Available' if self.weather_available else 'Unavailable — using unadjusted schedule'}",
            "=" * 65,
        ]

        any_change = any(a.timing_changed for a in self.applications)
        if any_change:
            lines.append("NOTE: One or more applications have been timing-adjusted based on weather.")
        else:
            lines.append("NOTE: All applications are on schedule — no weather adjustments required.")

        lines += [
            "",
            f"{'Month':<14} {'Original':<12} {'Application':<28} {'AN (t)':>7} {'Conf':>6} {'Notes'}",
            "-" * 80,
        ]

        for app in self.applications:
            changed_marker = f"→ {app.adjusted_month_name[:3]}" if app.timing_changed else "  (on schedule)"
            conf_symbol    = {"high": "HIGH", "medium": "MED ", "low": "LOW "}.get(app.confidence, "?   ")
            note_str       = " | ".join(app.notes[:2]) if app.notes else ""  # First 2 notes only

            lines.append(
                f"{app.original.month_name[:3]:<14} {changed_marker:<12} "
                f"{app.original.label:<28} {app.original.an_tonnes:>7.1f} "
                f"{conf_symbol:>6}  {note_str}"
            )

        lines += ["", "=" * 65]

        if self.notes:
            lines.append("WEATHER SUMMARY:")
            for note in self.notes:
                lines.append(f"  - {note}")
            lines.append("=" * 65)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main adjuster
# ---------------------------------------------------------------------------

def adjust_for_weather(farm_result: NRequirementResult,
                       weather_override: Optional[dict] = None) -> AdjustedSchedule:
    """
    Adjusts the Phase 1 application schedule for weather conditions.

    Args:
        farm_result:      Output from n_calculator.calculate_n_requirement()
        weather_override: Optional dict to override weather fetching for testing.
                          Format: {"forecast": pd.DataFrame, "historical": pd.DataFrame,
                                   "monthly_rainfall": dict[int, float]}

    Returns:
        AdjustedSchedule with adjusted timing and agronomic notes
    """
    assessment_date    = date.today()
    schedule_notes     = []
    weather_available  = False

    # --- Fetch weather data ---
    if weather_override:
        forecast_df       = weather_override.get("forecast", pd.DataFrame())
        historical_df     = weather_override.get("historical", pd.DataFrame())
        monthly_rainfall  = weather_override.get("monthly_rainfall", {})
        weather_available = True
        logger.info("Using weather override data (test mode)")

    elif farm_result.latitude and farm_result.longitude:
        logger.info(f"Fetching weather for ({farm_result.latitude:.4f}, {farm_result.longitude:.4f})")
        forecast_df      = fetch_forecast_weather(farm_result.latitude, farm_result.longitude)
        historical_df    = fetch_historical_weather(farm_result.latitude, farm_result.longitude, days_back=30)
        monthly_rainfall = get_monthly_rainfall_forecast(farm_result.latitude, farm_result.longitude)
        weather_available = not forecast_df.empty

        if not weather_available:
            schedule_notes.append(
                "Weather data unavailable — using unadjusted Phase 1 schedule. "
                "Check internet connection or Open-Meteo API status."
            )
    else:
        schedule_notes.append(
            "No coordinates available (NVZ lookup may have failed). "
            "Using unadjusted Phase 1 schedule."
        )
        forecast_df     = pd.DataFrame()
        historical_df   = pd.DataFrame()
        monthly_rainfall = {}

    # --- Antecedent conditions: was the last 30 days wet? ---
    antecedent_wet = False
    antecedent_mm  = 0.0
    if not historical_df.empty:
        antecedent_mm  = float(historical_df["precipitation_mm"].sum())
        antecedent_wet = antecedent_mm > ANTECEDENT_WET_THRESHOLD_MM
        if antecedent_wet:
            schedule_notes.append(
                f"Antecedent conditions: {antecedent_mm:.0f}mm in last 30 days "
                f"(>{ANTECEDENT_WET_THRESHOLD_MM}mm threshold). "
                "Soil likely saturated — increased leaching risk on early applications."
            )

    # --- Add weather summary notes ---
    if monthly_rainfall:
        wet_months  = [m for m, r in monthly_rainfall.items() if r > 80]
        dry_months  = [m for m, r in monthly_rainfall.items() if r < DRY_MONTH_THRESHOLD_MM]
        if wet_months:
            schedule_notes.append(
                f"Wet months forecast: {', '.join(_month_name(m) for m in sorted(wet_months)[:3])}. "
                "Plan applications to avoid heavy rain windows."
            )
        if dry_months:
            schedule_notes.append(
                f"Dry months forecast: {', '.join(_month_name(m) for m in sorted(dry_months)[:3])}. "
                "N uptake may be moisture-limited in these periods."
            )

    # --- Adjust each application ---
    adjusted_apps = []
    for app in farm_result.applications:
        adjusted = _adjust_single_application(
            app=app,
            crop=farm_result.crop,
            in_nvz=farm_result.in_nvz,
            forecast_df=forecast_df,
            monthly_rainfall=monthly_rainfall,
            antecedent_wet=antecedent_wet,
            weather_available=weather_available,
            assessment_date=assessment_date,
        )
        adjusted_apps.append(adjusted)

    return AdjustedSchedule(
        farm_result=farm_result,
        applications=adjusted_apps,
        weather_available=weather_available,
        assessment_date=assessment_date,
        notes=schedule_notes,
    )


# ---------------------------------------------------------------------------
# Single application adjuster
# ---------------------------------------------------------------------------

def _adjust_single_application(
    app:              MonthlyApplication,
    crop:             str,
    in_nvz:           bool,
    forecast_df:      pd.DataFrame,
    monthly_rainfall: dict,
    antecedent_wet:   bool,
    weather_available: bool,
    assessment_date:  date,
) -> AdjustedApplication:
    """Applies weather adjustment rules to a single application event."""

    notes           = []
    weeks_shifted   = 0
    frost_risk      = False
    forecast_precip = None
    confidence      = "high"

    # Get expected rainfall for this application month
    month_rainfall = monthly_rainfall.get(app.month, None)
    if month_rainfall is not None:
        forecast_precip = month_rainfall

    # --- Rule 1: Heavy rain ---
    if month_rainfall is not None and month_rainfall > RAIN_HEAVY_THRESHOLD * 30:
        weeks_shifted  = max(weeks_shifted, 2)
        confidence     = "medium"
        notes.append(
            f"High rainfall forecast in {_month_name(app.month)} "
            f"({month_rainfall:.0f}mm expected). "
            f"Delayed by ~{weeks_shifted} weeks to reduce leaching risk."
        )

    # --- Rule 2: Antecedent wet conditions ---
    if antecedent_wet and app.month == assessment_date.month:
        # Only flag for near-term applications
        confidence = "medium"
        notes.append(
            "Wet antecedent conditions. Consider split application "
            "or reduced rate to minimise leaching risk."
        )

    # --- Rule 3: Frost risk (16-day forecast only) ---
    if not forecast_df.empty:
        # Check if any days in this month fall within our 16-day window
        app_month_dates = forecast_df[forecast_df["date"].dt.month == app.month]
        if not app_month_dates.empty:
            min_temp = app_month_dates["temp_min_c"].min()
            if pd.notna(min_temp) and min_temp < FROST_TEMP_THRESHOLD_C:
                frost_risk  = True
                weeks_shifted = max(weeks_shifted, 1)
                confidence  = "medium"
                notes.append(
                    f"Frost forecast in {_month_name(app.month)} "
                    f"(min temp {min_temp:.1f}°C). "
                    "Delay application until ground thaws."
                )
            forecast_precip = float(app_month_dates["precipitation_mm"].sum())

    # --- Rule 4: Dry month flag ---
    if month_rainfall is not None and month_rainfall < DRY_MONTH_THRESHOLD_MM:
        notes.append(
            f"Dry conditions forecast in {_month_name(app.month)} "
            f"({month_rainfall:.0f}mm). "
            "N uptake may be moisture-limited — consider irrigation if available."
        )

    # --- Apply shift ---
    if weeks_shifted > MAX_WEEKS_SHIFT:
        weeks_shifted = MAX_WEEKS_SHIFT
        notes.append(f"Shift capped at {MAX_WEEKS_SHIFT} weeks maximum.")

    # Convert weeks to months (approximate)
    months_shifted  = math.ceil(weeks_shifted / 4)
    adjusted_month  = _offset_month(app.month, months_shifted)
    timing_changed  = months_shifted != 0

    # --- NVZ conflict check on adjusted date ---
    nvz_conflict = False
    if in_nvz and timing_changed:
        closed = NVZ_CLOSED_PERIODS.get(crop, (99, 99))
        if _in_closed_period(adjusted_month, closed[0], closed[1]):
            nvz_conflict = True
            confidence   = "low"
            notes.append(
                f"WARNING: Adjusted date ({_month_name(adjusted_month)}) "
                "falls within NVZ closed period. "
                "You must not apply N during this period. "
                "Review original timing or consult your agronomist."
            )

    # --- Confidence: no weather data ---
    if not weather_available:
        confidence = "medium"
        notes.append("No weather data available — timing based on standard RB209 schedule only.")

    return AdjustedApplication(
        original=app,
        adjusted_month=adjusted_month,
        adjusted_month_name=_month_name(adjusted_month),
        weeks_shifted=weeks_shifted,
        forecast_precip_mm=forecast_precip,
        antecedent_wet=antecedent_wet,
        frost_risk=frost_risk,
        timing_changed=timing_changed,
        nvz_conflict=nvz_conflict,
        confidence=confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _offset_month(base_month: int, offset: int) -> int:
    return ((base_month - 1 + offset) % 12) + 1


MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

def _month_name(month: int) -> str:
    return MONTH_NAMES.get(month, f"Month {month}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from demand.n_calculator import calculate_n_requirement

    print("Running Phase 2 example: 120ha winter wheat, PE1 1AB\n")

    farm = calculate_n_requirement(
        crop="winter_wheat",
        farm_size_ha=120.0,
        soil_quality=3,
        planting_month=9,
        postcode="PE1 1AB",
    )

    print(farm.summary())
    print("\n--- Applying weather adjustment ---\n")

    adjusted = adjust_for_weather(farm)
    print(adjusted.summary())
