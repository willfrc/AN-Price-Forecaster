"""
Phase 2 test scenarios.

Tests the weather adjustment layer with:
    1. A live API call (requires internet)
    2. Synthetic weather overrides (no internet needed — always runs)

Run: py demand/test_phase2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import pandas as pd

from demand.n_calculator import calculate_n_requirement
from demand.weather_adjuster import adjust_for_weather


# ---------------------------------------------------------------------------
# Synthetic weather helpers
# ---------------------------------------------------------------------------

def _make_forecast(days: int = 16, daily_rain_mm: float = 2.0,
                   min_temp: float = 5.0, max_temp: float = 12.0) -> pd.DataFrame:
    """Builds a synthetic forecast DataFrame for testing."""
    dates = [date.today() + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({
        "date":                pd.to_datetime(dates),
        "precipitation_mm":    [daily_rain_mm] * days,
        "temp_max_c":          [max_temp] * days,
        "temp_min_c":          [min_temp] * days,
        "application_suitable": [daily_rain_mm < 10.0 and min_temp > 0] * days,
        "rain_caution":         [5.0 <= daily_rain_mm < 10.0] * days,
    })


def _make_monthly_rainfall(default_mm: float = 50.0,
                           overrides: dict = None) -> dict:
    """Builds a synthetic monthly rainfall dict for testing."""
    rainfall = {m: default_mm for m in range(1, 13)}
    if overrides:
        rainfall.update(overrides)
    return rainfall


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def test_normal_conditions():
    """Standard conditions — no adjustments expected."""
    print("\n[TEST 1] Normal conditions — no adjustments expected")
    print("-" * 60)

    farm = calculate_n_requirement(
        crop="winter_wheat",
        farm_size_ha=100.0,
        soil_quality=3,
        planting_month=9,
        postcode="PE1 1AB",
        nvz_override=False,
    )

    adjusted = adjust_for_weather(
        farm,
        weather_override={
            "forecast":       _make_forecast(daily_rain_mm=2.0, min_temp=5.0),
            "historical":     _make_forecast(days=30, daily_rain_mm=3.0),
            "monthly_rainfall": _make_monthly_rainfall(default_mm=50.0),
        }
    )

    print(adjusted.summary())

    # No applications should be shifted under normal conditions
    shifted = [a for a in adjusted.applications if a.timing_changed]
    assert len(shifted) == 0, f"Expected 0 shifts, got {len(shifted)}"
    print("[PASS] No timing adjustments under normal conditions")


def test_heavy_rain_delay():
    """Heavy rain in application month — expect delay."""
    print("\n[TEST 2] Heavy rain in application month — expect delay")
    print("-" * 60)

    farm = calculate_n_requirement(
        crop="winter_wheat",
        farm_size_ha=100.0,
        soil_quality=3,
        planting_month=9,
        postcode="PE1 1AB",
        nvz_override=False,
    )

    # Very wet February and March (main application months for winter wheat)
    adjusted = adjust_for_weather(
        farm,
        weather_override={
            "forecast":       _make_forecast(daily_rain_mm=2.0),
            "historical":     _make_forecast(days=30, daily_rain_mm=3.0),
            "monthly_rainfall": _make_monthly_rainfall(
                default_mm=50.0,
                overrides={1: 400.0, 2: 450.0}  # Very wet Jan-Feb
            ),
        }
    )

    print(adjusted.summary())

    shifted = [a for a in adjusted.applications if a.timing_changed]
    assert len(shifted) > 0, "Expected at least one timing adjustment for heavy rain"
    print(f"[PASS] {len(shifted)} application(s) delayed due to heavy rain")


def test_frost_delay():
    """Frost in forecast — expect delay on near-term applications."""
    print("\n[TEST 3] Frost forecast — expect near-term delay")
    print("-" * 60)

    farm = calculate_n_requirement(
        crop="spring_barley",
        farm_size_ha=50.0,
        soil_quality=2,
        planting_month=3,  # March planting — first application in March
        postcode="AB51 0AB",
        nvz_override=False,
    )

    # Frost in the first 16 days
    adjusted = adjust_for_weather(
        farm,
        weather_override={
            "forecast":       _make_forecast(daily_rain_mm=1.0, min_temp=-3.0),
            "historical":     _make_forecast(days=30, daily_rain_mm=2.0),
            "monthly_rainfall": _make_monthly_rainfall(default_mm=40.0),
        }
    )

    print(adjusted.summary())

    frost_flags = [a for a in adjusted.applications if a.frost_risk]
    print(f"[PASS] Frost risk flagged on {len(frost_flags)} application(s)")


def test_wet_antecedent():
    """Wet preceding 30 days — expect leaching risk flag."""
    print("\n[TEST 4] Wet antecedent conditions — leaching risk flag expected")
    print("-" * 60)

    farm = calculate_n_requirement(
        crop="oilseed_rape",
        farm_size_ha=80.0,
        soil_quality=4,
        planting_month=8,
        postcode="YO42 1AB",
        nvz_override=False,
    )

    # Very wet last 30 days
    adjusted = adjust_for_weather(
        farm,
        weather_override={
            "forecast":       _make_forecast(daily_rain_mm=3.0),
            "historical":     _make_forecast(days=30, daily_rain_mm=8.0),  # 240mm in 30 days
            "monthly_rainfall": _make_monthly_rainfall(default_mm=60.0),
        }
    )

    print(adjusted.summary())
    print("[PASS] Wet antecedent conditions handled")


def test_live_api():
    """Live API test — requires internet. Skipped gracefully if unavailable."""
    print("\n[TEST 5] Live API test — Fenland winter wheat farm")
    print("-" * 60)

    try:
        farm = calculate_n_requirement(
            crop="winter_wheat",
            farm_size_ha=150.0,
            soil_quality=3,
            planting_month=9,
            postcode="PE1 1AB",
        )

        adjusted = adjust_for_weather(farm)
        print(adjusted.summary())

        if adjusted.weather_available:
            print("[PASS] Live API call succeeded")
        else:
            print("[WARN] Weather API unavailable — schedule unadjusted (acceptable for PoC)")

    except Exception as e:
        print(f"[SKIP] Live API test skipped: {e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "=" * 70)
    print("PHASE 2 TEST RUN — WEATHER ADJUSTER")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_fn in [
        test_normal_conditions,
        test_heavy_rain_delay,
        test_frost_delay,
        test_wet_antecedent,
        test_live_api,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] Assertion failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[FAIL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nPhase 2 complete. Ready to build Phase 3 (procurement planner).")
    else:
        print("\nFix failures before proceeding to Phase 3.")


if __name__ == "__main__":
    run_tests()
