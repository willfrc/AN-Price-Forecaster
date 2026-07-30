"""
Procurement planner — Phase 3 of the farm demand engine.

Combines the weather-adjusted demand schedule (Phase 2) with the AN
price forecast (existing model) to produce a buy timing recommendation.

Core logic:
    For each application in the adjusted schedule, compare:
    - Price in the month of application (buy spot)
    - Price in earlier months (buy forward)

    If the price forecast shows prices rising into the application window,
    recommend buying earlier. If prices are forecast to fall, recommend
    waiting and buying spot.

    Output: monthly procurement schedule with:
        - How many tonnes to buy and when
        - Forecast price at purchase time (point + CI)
        - Expected cost range per application
        - Total seasonal procurement cost range
        - Buy/wait recommendation with rationale

Confidence note:
    The 1-month price model has MAPE ~7%. The 12-month model has MAPE ~11.6%.
    Cost estimates beyond 3 months carry significant uncertainty and should
    be presented as ranges, not precise figures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from demand.weather_adjuster import AdjustedSchedule, AdjustedApplication
from utils.logger import get_logger

logger = get_logger(__name__)

FORECAST_PATH  = "data/processed/forecast_output.csv"
SAVED_DIR      = "model/saved"

# How much price rise justifies buying forward (% threshold)
# If price is forecast to rise >5% between now and application, recommend buying forward
FORWARD_BUY_THRESHOLD_PCT = 5.0

# How much price fall justifies waiting (%)
# If price is forecast to fall >3%, recommend waiting
WAIT_THRESHOLD_PCT = 3.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProcurementRecommendation:
    """Buy recommendation for a single application event."""
    application:        AdjustedApplication

    # Timing
    recommended_buy_month:      int
    recommended_buy_month_name: str
    months_before_application:  int    # 0 = buy spot, >0 = buy forward

    # Price at recommended buy time
    forecast_price_point:  float   # £/tonne point forecast
    forecast_price_lower:  float   # £/tonne lower bound
    forecast_price_upper:  float   # £/tonne upper bound
    price_horizon_label:   str     # "1-month model" or "12-month model"

    # Cost estimates
    an_tonnes:             float
    cost_point:            float   # £ point estimate
    cost_lower:            float   # £ lower bound
    cost_upper:            float   # £ upper bound

    # Recommendation
    action:                str     # "BUY NOW" / "BUY FORWARD" / "WAIT — BUY SPOT"
    rationale:             str
    confidence:            str     # "high" / "medium" / "low"

    # Price context
    current_price:         Optional[float]   # Latest actual AN price
    price_change_pct:      Optional[float]   # Forecast % change from now to buy month


@dataclass
class ProcurementPlan:
    """Full procurement plan for the season."""
    farm_postcode:         str
    crop_name:             str
    farm_size_ha:          float
    assessment_date:       date

    recommendations:       list[ProcurementRecommendation]

    # Season totals
    total_an_tonnes:       float
    total_cost_point:      float
    total_cost_lower:      float
    total_cost_upper:      float

    # Data quality flags
    forecast_available:    bool
    forecast_generated:    Optional[str]   # Date forecast was generated
    notes:                 list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "SEASONAL PROCUREMENT PLAN",
            "=" * 70,
            f"Farm:              {self.farm_postcode}",
            f"Crop:              {self.crop_name}",
            f"Farm size:         {self.farm_size_ha:.0f} ha",
            f"Assessment date:   {self.assessment_date.strftime('%d %B %Y')}",
            f"Price forecast:    {'Available' if self.forecast_available else 'Not available — using current price'}",
            "",
            f"TOTAL AN REQUIRED: {self.total_an_tonnes:.1f} tonnes",
            f"ESTIMATED COST:    £{self.total_cost_lower:,.0f} – £{self.total_cost_upper:,.0f}",
            f"                   (point estimate: £{self.total_cost_point:,.0f})",
            "",
            "PROCUREMENT SCHEDULE:",
            f"{'Buy Month':<12} {'Application':<22} {'Tonnes':>7} {'£/t (point)':>12} "
            f"{'£/t range':>18} {'Est. Cost':>12} {'Action'}",
            "-" * 100,
        ]

        for rec in self.recommendations:
            app_label = rec.application.original.label[:20]
            price_range = f"£{rec.forecast_price_lower:.0f}–£{rec.forecast_price_upper:.0f}"
            lines.append(
                f"{rec.recommended_buy_month_name[:3]+' '+str(_buy_year(rec)):<12} "
                f"{app_label:<22} "
                f"{rec.an_tonnes:>7.1f} "
                f"£{rec.forecast_price_point:>10.0f} "
                f"{price_range:>18} "
                f"£{rec.cost_point:>10,.0f} "
                f"  {rec.action}"
            )

        lines += [
            "-" * 100,
            f"{'TOTAL':<35} {self.total_an_tonnes:>7.1f} {'':>12} {'':>18} "
            f"£{self.total_cost_point:>10,.0f}",
            "",
            "RATIONALE:",
        ]

        for rec in self.recommendations:
            lines.append(
                f"  {rec.application.original.label[:25]:<25} "
                f"[{rec.action}] {rec.rationale}"
            )

        if self.notes:
            lines += ["", "NOTES:"]
            for note in self.notes:
                lines.append(f"  - {note}")

        lines += [
            "",
            "=" * 70,
            "DISCLAIMER",
            "-" * 70,
            "Price forecasts are model outputs, not guaranteed prices.",
            f"1-month model MAPE: ~7% | 12-month model MAPE: ~11.6%.",
            "Cost estimates beyond 3 months carry significant uncertainty.",
            "This tool does not constitute financial or agronomic advice.",
            "=" * 70,
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def build_procurement_plan(
    adjusted_schedule: AdjustedSchedule,
    forecast_override: Optional[pd.DataFrame] = None,
) -> ProcurementPlan:
    """
    Builds the full procurement plan from an adjusted demand schedule
    and the AN price forecast.

    Args:
        adjusted_schedule: Output from weather_adjuster.adjust_for_weather()
        forecast_override: Optional price forecast DataFrame for testing.
                           If None, loads from data/processed/forecast_output.csv

    Returns:
        ProcurementPlan with buy timing recommendations and cost estimates
    """
    farm     = adjusted_schedule.farm_result
    notes    = []

    # --- Load price forecast ---
    forecast_df, forecast_available, forecast_generated = _load_forecast(
        forecast_override
    )

    # Get current (latest actual) AN price
    current_price = _get_current_price()
    if current_price:
        logger.info(f"Current AN price: £{current_price:.0f}/tonne")
    else:
        notes.append(
            "Could not retrieve current AN price from processed data. "
            "Cost estimates use forecast prices only."
        )

    # Build price lookup: month -> (point, lower, upper, model_label)
    price_lookup = _build_price_lookup(forecast_df, current_price)

    # --- Build recommendation for each application ---
    recommendations = []
    for adj_app in adjusted_schedule.applications:
        rec = _build_recommendation(
            adj_app=adj_app,
            price_lookup=price_lookup,
            current_price=current_price,
            assessment_date=adjusted_schedule.assessment_date,
        )
        recommendations.append(rec)

    # Sort by recommended buy month
    recommendations.sort(key=lambda r: r.recommended_buy_month)

    # --- Season totals ---
    total_tonnes    = sum(r.an_tonnes     for r in recommendations)
    total_cost_pt   = sum(r.cost_point    for r in recommendations)
    total_cost_lo   = sum(r.cost_lower    for r in recommendations)
    total_cost_hi   = sum(r.cost_upper    for r in recommendations)

    # --- Plan-level notes ---
    if not forecast_available:
        notes.append(
            "Price forecast not available — cost estimates based on current price. "
            "Run py model/forecast.py to generate a price forecast."
        )

    rising_count = sum(
        1 for r in recommendations
        if r.price_change_pct and r.price_change_pct > FORWARD_BUY_THRESHOLD_PCT
    )
    if rising_count > 0:
        notes.append(
            f"{rising_count} application(s) have rising price forecasts. "
            "Consider forward purchasing to lock in lower prices."
        )

    logger.info(
        f"Procurement plan complete: {total_tonnes:.1f}t AN, "
        f"£{total_cost_lo:,.0f}–£{total_cost_hi:,.0f} estimated cost"
    )

    return ProcurementPlan(
        farm_postcode=farm.postcode,
        crop_name=farm.crop_name,
        farm_size_ha=farm.farm_size_ha,
        assessment_date=adjusted_schedule.assessment_date,
        recommendations=recommendations,
        total_an_tonnes=round(total_tonnes, 1),
        total_cost_point=round(total_cost_pt, 0),
        total_cost_lower=round(total_cost_lo, 0),
        total_cost_upper=round(total_cost_hi, 0),
        forecast_available=forecast_available,
        forecast_generated=forecast_generated,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def _build_recommendation(
    adj_app:         AdjustedApplication,
    price_lookup:    dict,
    current_price:   Optional[float],
    assessment_date: date,
) -> ProcurementRecommendation:
    """
    Builds a buy recommendation for a single application event.

    Decision logic:
    1. Get forecast price at application month
    2. Compare to current price and near-term prices
    3. If rising: recommend buying forward (1-2 months ahead)
    4. If falling: recommend waiting (buy spot, 0 months ahead)
    5. If flat: recommend buying 4-6 weeks ahead (logistics buffer)
    """
    app_month = adj_app.adjusted_month
    app_price = price_lookup.get(app_month)

    if app_price is None:
        # No forecast for this month — use current price as fallback
        app_price = {
            "point": current_price or 400.0,
            "lower": (current_price or 400.0) * 0.90,
            "upper": (current_price or 400.0) * 1.10,
            "label": "current price (no forecast)",
        }

    # --- Determine optimal buy month ---
    # Look back up to 3 months before application to find lowest forecast price
    candidate_months = [
        max(1, app_month - offset) for offset in range(0, 4)
    ]
    candidate_months = list(dict.fromkeys(candidate_months))  # Deduplicate

    # Find month with lowest point forecast price
    best_month     = app_month
    best_price_pt  = app_price["point"]

    for month in candidate_months:
        candidate = price_lookup.get(month)
        if candidate and candidate["point"] < best_price_pt:
            best_month    = month
            best_price_pt = candidate["point"]

    buy_price = price_lookup.get(best_month, app_price)

    # --- Calculate price change % ---
    price_change_pct = None
    if current_price and current_price > 0:
        price_change_pct = (app_price["point"] - current_price) / current_price * 100

    # --- Determine action ---
    months_before = (app_month - best_month) % 12

    if price_change_pct is not None:
        if price_change_pct > FORWARD_BUY_THRESHOLD_PCT and months_before >= 1:
            action    = "BUY FORWARD"
            rationale = (
                f"Price forecast to rise {price_change_pct:.1f}% by application month. "
                f"Buy {months_before} month(s) early to save "
                f"~£{(app_price['point'] - buy_price['point']) * adj_app.original.an_tonnes:,.0f}."
            )
        elif price_change_pct < -WAIT_THRESHOLD_PCT:
            action    = "WAIT — BUY SPOT"
            best_month = app_month
            buy_price  = app_price
            months_before = 0
            rationale = (
                f"Price forecast to fall {abs(price_change_pct):.1f}% by application month. "
                "Buy spot at application time."
            )
        else:
            action    = "BUY 4-6 WEEKS AHEAD"
            best_month = max(1, app_month - 1)
            buy_price  = price_lookup.get(best_month, app_price)
            months_before = 1
            rationale = (
                "Price broadly flat. Buy 4–6 weeks ahead to allow for "
                "delivery logistics and avoid last-minute supply constraints."
            )
    else:
        # No current price available — default to 4-6 weeks ahead
        action        = "BUY 4-6 WEEKS AHEAD"
        best_month    = max(1, app_month - 1)
        buy_price     = price_lookup.get(best_month, app_price)
        months_before = 1
        rationale     = "No current price available — standard 4–6 week forward purchase recommended."

    # --- Cost estimates ---
    tonnes    = adj_app.original.an_tonnes
    cost_pt   = tonnes * buy_price["point"]
    cost_lo   = tonnes * buy_price["lower"]
    cost_hi   = tonnes * buy_price["upper"]

    # Confidence: inherit from weather adjustment, downgrade for long horizons
    confidence = adj_app.confidence
    if months_before > 6:
        confidence = "low"
    elif months_before > 3:
        confidence = min_confidence(confidence, "medium")

    return ProcurementRecommendation(
        application=adj_app,
        recommended_buy_month=best_month,
        recommended_buy_month_name=_month_name(best_month),
        months_before_application=months_before,
        forecast_price_point=round(buy_price["point"], 1),
        forecast_price_lower=round(buy_price["lower"], 1),
        forecast_price_upper=round(buy_price["upper"], 1),
        price_horizon_label=buy_price.get("label", "forecast"),
        an_tonnes=tonnes,
        cost_point=round(cost_pt, 0),
        cost_lower=round(cost_lo, 0),
        cost_upper=round(cost_hi, 0),
        action=action,
        rationale=rationale,
        confidence=confidence,
        current_price=current_price,
        price_change_pct=round(price_change_pct, 1) if price_change_pct is not None else None,
    )


# ---------------------------------------------------------------------------
# Price data loaders
# ---------------------------------------------------------------------------

def _load_forecast(
    override: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, bool, Optional[str]]:
    """Loads the price forecast, returns (df, available, generated_date)."""
    if override is not None:
        return override, True, date.today().isoformat()

    if not os.path.exists(FORECAST_PATH):
        logger.warning(f"Forecast file not found: {FORECAST_PATH}")
        return pd.DataFrame(), False, None

    df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    # Get generation date from file modification time
    mtime = os.path.getmtime(FORECAST_PATH)
    from datetime import datetime
    generated = datetime.fromtimestamp(mtime).strftime("%d %B %Y")

    logger.info(f"Loaded price forecast: {len(df)} rows (generated {generated})")
    return df, True, generated


def _get_current_price() -> Optional[float]:
    """Gets the most recent actual AN price from processed features."""
    features_path = "data/processed/model_features.csv"
    if not os.path.exists(features_path):
        return None
    try:
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        if "an_price_gbp_t" in df.columns:
            latest = df["an_price_gbp_t"].dropna().iloc[-1]
            return float(latest)
    except Exception as e:
        logger.warning(f"Could not read current price: {e}")
    return None


def _build_price_lookup(
    forecast_df: pd.DataFrame,
    current_price: Optional[float],
) -> dict[int, dict]:
    """
    Builds a month -> price dict from the forecast DataFrame.
    Falls back to current price for months with no forecast.
    """
    lookup = {}

    if not forecast_df.empty:
        # Use monthly frequency rows (one point per month)
        monthly = forecast_df[forecast_df["frequency"] == "monthly"].copy()

        # Also use the first weekly entry per month as the short-term price
        weekly = forecast_df[forecast_df["frequency"] == "weekly"].copy()
        if not weekly.empty:
            weekly["month"] = weekly["date"].dt.month
            weekly_first = weekly.groupby("month").first().reset_index()
            monthly = pd.concat([monthly, weekly_first], ignore_index=True)

        for _, row in monthly.iterrows():
            try:
                month = int(pd.to_datetime(row["date"]).month)
                if month not in lookup:
                    lookup[month] = {
                        "point": float(row["point_forecast"]),
                        "lower": float(row["lower_bound"]),
                        "upper": float(row["upper_bound"]),
                        "label": f"{row.get('model_used', '?')} model",
                    }
            except (ValueError, KeyError):
                continue

    # Fill any missing months with current price (flat assumption)
    if current_price:
        for month in range(1, 13):
            if month not in lookup:
                lookup[month] = {
                    "point": current_price,
                    "lower": current_price * 0.88,
                    "upper": current_price * 1.12,
                    "label": "current price (no forecast)",
                }

    return lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_confidence(a: str, b: str) -> str:
    """Returns the lower of two confidence levels."""
    order = {"high": 2, "medium": 1, "low": 0}
    return a if order.get(a, 0) <= order.get(b, 0) else b


def _buy_year(rec: ProcurementRecommendation) -> int:
    """Infers the calendar year for the buy month (handles year boundary)."""
    current_month = date.today().month
    buy_month     = rec.recommended_buy_month
    return date.today().year if buy_month >= current_month else date.today().year + 1


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
    from demand.weather_adjuster import adjust_for_weather

    print("Running Phase 3 example: 120ha winter wheat, PE1 1AB\n")

    farm = calculate_n_requirement(
        crop="winter_wheat",
        farm_size_ha=120.0,
        soil_quality=3,
        planting_month=9,
        postcode="PE1 1AB",
        nvz_override=False,
    )

    adjusted = adjust_for_weather(farm)
    plan     = build_procurement_plan(adjusted)

    print(plan.summary())
