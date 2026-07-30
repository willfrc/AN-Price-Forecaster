"""
Phase 1 test scenarios.

Runs four representative UK farm scenarios to verify the N calculator
is producing sensible outputs before we build Phase 2.

Run: py demand/test_phase1.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demand.n_calculator import calculate_n_requirement

SCENARIOS = [
    {
        "name":           "Fenland winter wheat — large arable, NVZ likely",
        "crop":           "winter_wheat",
        "farm_size_ha":   200.0,
        "soil_quality":   3,
        "planting_month": 9,
        "postcode":       "PE1 1AB",   # Peterborough
    },
    {
        "name":           "Scottish spring barley — small farm, non-NVZ",
        "crop":           "spring_barley",
        "farm_size_ha":   45.0,
        "soil_quality":   2,
        "planting_month": 4,
        "postcode":       "AB51 0AB",  # Inverurie, Aberdeenshire
    },
    {
        "name":           "Yorkshire oilseed rape — medium farm",
        "crop":           "oilseed_rape",
        "farm_size_ha":   80.0,
        "soil_quality":   4,
        "planting_month": 8,
        "postcode":       "YO42 1AB",  # Pocklington, East Yorkshire
    },
    {
        "name":           "Lincolnshire maize — good soil",
        "crop":           "maize",
        "farm_size_ha":   60.0,
        "soil_quality":   5,
        "planting_month": 5,
        "postcode":       "LN1 1AB",   # Lincoln
    },
]


def run_tests():
    print("\n" + "=" * 70)
    print("PHASE 1 TEST RUN — N CALCULATOR")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] {scenario['name']}")
        print("-" * 70)

        try:
            result = calculate_n_requirement(
                crop           = scenario["crop"],
                farm_size_ha   = scenario["farm_size_ha"],
                soil_quality   = scenario["soil_quality"],
                planting_month = scenario["planting_month"],
                postcode       = scenario["postcode"],
            )

            print(result.summary())

            # Basic sanity checks
            assert result.an_total_tonnes > 0,         "AN total should be positive"
            assert len(result.applications) > 0,       "Should have at least one application"
            assert result.n_rate_kg_ha > 0,            "N rate should be positive"
            assert result.an_total_tonnes < 2000,      "AN total implausibly large"

            total_from_apps = sum(a.an_tonnes for a in result.applications)
            assert abs(total_from_apps - result.an_total_tonnes) < 0.5, \
                f"Application sum {total_from_apps:.1f}t != total {result.an_total_tonnes:.1f}t"

            print(f"\n[PASS] All checks passed")
            passed += 1

        except Exception as e:
            print(f"\n[FAIL] {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nPhase 1 complete. Ready to build Phase 2 (weather adjustment).")
    else:
        print("\nFix failures before proceeding to Phase 2.")


if __name__ == "__main__":
    run_tests()
