"""
Phase 3 test scenarios.

Tests the procurement planner with:
    1. Rising price forecast — should recommend buying forward
    2. Falling price forecast — should recommend waiting (buy spot)
    3. Flat price forecast — should recommend buying 4-6 weeks ahead
    4. No forecast available — should fall back gracefully
    5. Full pipeline test (Phases 1+2+3 end-to-end)

Run: py demand/test_phase3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import pandas as pd

from demand.n_calculator import calculate_n_requirement
from demand.weather_adjuster import adjust_for_weather
from demand.procurement_planner import build_procurement_plan


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_forecast(monthly_prices: dict[int, tuple[float, float, float]]) -> pd.DataFrame:
    """
    Builds a synthetic forecast DataFrame.
    monthly_prices: {month: (point, lower, upper)}
    """
    rows = []
    base_year = date.today().year
    for month, (point, lower, upper) in monthly_prices.items():
        forecast_date = date(base_year if month >= date.today().month else base_year + 1,
                             month, 1)
        rows.append({
            "date":           pd.Timestamp(forecast_date),
            "horizon_months": month,
            "frequency":      "monthly",
            "point_forecast": point,
            "lower_bound":    lower,
            "upper_bound":    upper,
            "ci_width":       upper - lower,
            "model_used":     "test",
        })
    return pd.DataFrame(rows)


def _make_farm(crop="winter_wheat", size_ha=100.0,
               soil=3, plant_month=9, postcode="PE1 1AB"):
    return calculate_n_requirement(
        crop=crop, farm_size_ha=size_ha,
        soil_quality=soil, planting_month=plant_month,
        postcode=postcode, nvz_override=False,
    )


def _make_adjusted(farm):
    return adjust_for_weather(
        farm,
        weather_override={
            "forecast":         _wf_df(),
            "historical":       _wf_df(daily_rain=3.0),
            "monthly_rainfall": {m: 50.0 for m in range(1, 13)},
        }
    )


def _wf_df(days=16, daily_rain=2.0, min_temp=5.0, max_temp=12.0):
    """Minimal synthetic weather DataFrame."""
    dates = [date.today() + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({
        "date":                pd.to_datetime(dates),
        "precipitation_mm":    [daily_rain] * days,
        "temp_max_c":          [max_temp] * days,
        "temp_min_c":          [min_temp] * days,
        "application_suitable": [True] * days,
        "rain_caution":         [False] * days,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rising_prices():
    """Rising prices — all applications should recommend BUY FORWARD."""
    print("\n[TEST 1] Rising price forecast — expect BUY FORWARD recommendations")
    print("-" * 65)

    farm     = _make_farm()
    adjusted = _make_adjusted(farm)

    # Prices rising sharply from a low base of £300 now to £480-520 at application
    # Set forecast prices well above the "current" price we'll inject
    forecast = _make_forecast({
        1: (480, 420, 540),
        2: (500, 440, 560),
        4: (520, 460, 580),
        7: (300, 270, 330),   # "Now" — low price, good time to buy forward
        8: (310, 280, 340),
        9: (320, 290, 350),
        10: (340, 305, 375),
        11: (360, 325, 395),
        12: (400, 360, 440),
    })

    # Patch current price to £300 so forecast prices look like a rise
    import unittest.mock as mock
    with mock.patch(
        "demand.procurement_planner._get_current_price",
        return_value=300.0
    ):
        plan = build_procurement_plan(adjusted, forecast_override=forecast)

    print(plan.summary())

    forward_buys = [r for r in plan.recommendations if "FORWARD" in r.action]
    assert len(forward_buys) > 0, "Expected at least one BUY FORWARD recommendation"
    assert plan.total_an_tonnes > 0
    assert plan.total_cost_point > 0
    print(f"[PASS] {len(forward_buys)} forward buy recommendation(s)")


def test_falling_prices():
    """Falling prices — applications should recommend WAIT / BUY SPOT."""
    print("\n[TEST 2] Falling price forecast — expect WAIT — BUY SPOT recommendations")
    print("-" * 65)

    farm     = _make_farm()
    adjusted = _make_adjusted(farm)

    # Prices falling from £500 now to £350 by spring
    forecast = _make_forecast({
        1: (350, 310, 390),
        2: (340, 300, 380),
        4: (330, 290, 370),
        10: (490, 440, 540),
        11: (470, 420, 520),
        12: (450, 400, 500),
    })

    plan = build_procurement_plan(adjusted, forecast_override=forecast)
    print(plan.summary())

    wait_recs = [r for r in plan.recommendations if "WAIT" in r.action]
    assert len(wait_recs) > 0, "Expected at least one WAIT recommendation"
    print(f"[PASS] {len(wait_recs)} wait/buy-spot recommendation(s)")


def test_flat_prices():
    """Flat prices — should recommend buying 4-6 weeks ahead."""
    print("\n[TEST 3] Flat price forecast — expect BUY 4-6 WEEKS AHEAD")
    print("-" * 65)

    farm     = _make_farm()
    adjusted = _make_adjusted(farm)

    # Prices broadly flat at £400
    flat_price = {m: (400, 360, 440) for m in range(1, 13)}
    forecast   = _make_forecast(flat_price)

    plan = build_procurement_plan(adjusted, forecast_override=forecast)
    print(plan.summary())

    assert plan.total_an_tonnes > 0
    assert plan.total_cost_lower < plan.total_cost_point < plan.total_cost_upper
    print("[PASS] Flat price plan produced with valid cost range")


def test_no_forecast():
    """No forecast available — should fall back gracefully to current price."""
    print("\n[TEST 4] No forecast available — graceful fallback expected")
    print("-" * 65)

    farm     = _make_farm()
    adjusted = _make_adjusted(farm)

    # Pass empty DataFrame as forecast
    plan = build_procurement_plan(adjusted, forecast_override=pd.DataFrame())
    print(plan.summary())

    assert plan.total_an_tonnes > 0, "Should still calculate tonnage without forecast"
    assert not plan.forecast_available or plan.total_cost_point > 0
    print("[PASS] Graceful fallback without forecast data")


def test_full_pipeline():
    """End-to-end: Phase 1 + Phase 2 + Phase 3 for four different farm types."""
    print("\n[TEST 5] Full pipeline — four farm scenarios")
    print("-" * 65)

    scenarios = [
        ("winter_wheat",  200, 3, 9,  "PE1 1AB",  "Fenland wheat"),
        ("spring_barley",  50, 2, 4,  "AB51 0AB", "Scottish barley"),
        ("oilseed_rape",   80, 4, 8,  "YO42 1AB", "Yorkshire OSR"),
        ("maize",          60, 5, 5,  "LN1 1AB",  "Lincolnshire maize"),
    ]

    flat_forecast = _make_forecast({m: (420, 375, 465) for m in range(1, 13)})

    for crop, size, soil, month, postcode, label in scenarios:
        print(f"\n  -> {label}")
        farm     = calculate_n_requirement(
            crop=crop, farm_size_ha=size, soil_quality=soil,
            planting_month=month, postcode=postcode, nvz_override=False,
        )
        adjusted = _make_adjusted(farm)
        plan     = build_procurement_plan(adjusted, forecast_override=flat_forecast)

        assert plan.total_an_tonnes > 0,     f"{label}: zero AN tonnes"
        assert plan.total_cost_point > 0,    f"{label}: zero cost"
        assert len(plan.recommendations) > 0, f"{label}: no recommendations"

        print(f"     Total AN: {plan.total_an_tonnes:.1f}t | "
              f"Est. cost: £{plan.total_cost_lower:,.0f}–£{plan.total_cost_upper:,.0f} | "
              f"Recommendations: {len(plan.recommendations)}")

    print("\n[PASS] All four scenarios completed successfully")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "=" * 70)
    print("PHASE 3 TEST RUN — PROCUREMENT PLANNER")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_fn in [
        test_rising_prices,
        test_falling_prices,
        test_flat_prices,
        test_no_forecast,
        test_full_pipeline,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] Assertion: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nPhase 3 complete. Ready to build Phase 4 (farm dashboard).")


if __name__ == "__main__":
    run_tests()
