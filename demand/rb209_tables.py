"""
RB209 Nitrogen Requirement Lookup Tables

Source: AHDB RB209 Field Crop Recommendations (Section 2: Arable Crops)
URL: https://ahdb.org.uk/knowledge-library/rb209-section-2-cereals-and-oilseeds

These tables encode the standard N recommendations for the four dominant
UK arable crops. Values are in kg N/ha total seasonal requirement.

Soil type definitions (SNS = Soil Nitrogen Supply):
    1 = Low SNS   (light sandy soils, low organic matter)
    2 = Medium SNS (typical arable soils)
    3 = High SNS  (heavy clay, high organic matter, recently manured)

These map to our 1-5 soil quality scale as follows:
    1-2 -> Low SNS
    3   -> Medium SNS
    4-5 -> High SNS

Application timing splits are derived from standard RB209 split application
guidance. These represent typical practice — actual timing depends on
crop growth stage and weather, which is handled in Phase 2.

All values represent kg N/ha. Divide by 0.345 to convert to kg AN/ha
(AN is 34.5% nitrogen). Multiply by hectares for total tonnes.

IMPORTANT: These are indicative values for PoC purposes. Production use
should reference the current RB209 edition directly, as recommendations
are updated periodically by AHDB.
"""

from typing import TypedDict


# ---------------------------------------------------------------------------
# Crop definitions
# ---------------------------------------------------------------------------

SUPPORTED_CROPS = {
    "winter_wheat":   "Winter Wheat",
    "spring_barley":  "Spring Barley",
    "oilseed_rape":   "Oilseed Rape (Winter)",
    "maize":          "Forage/Grain Maize",
}

# ---------------------------------------------------------------------------
# Total N requirement (kg N/ha) by crop and SNS class
# Source: RB209 Section 2, Table 2 (Recommended N rates for arable crops)
# ---------------------------------------------------------------------------

N_REQUIREMENT_KG_HA: dict[str, dict[str, int]] = {
    "winter_wheat": {
        "low":    220,   # Light soils, low SNS
        "medium": 190,   # Typical arable
        "high":   160,   # Heavy/high OM soils
    },
    "spring_barley": {
        "low":    150,
        "medium": 130,
        "high":   110,
    },
    "oilseed_rape": {
        "low":    250,
        "medium": 220,
        "high":   190,
    },
    "maize": {
        "low":    180,
        "medium": 150,
        "high":   120,
    },
}

# ---------------------------------------------------------------------------
# NVZ adjustments
# Nitrate Vulnerable Zone rules cap total N and restrict timing.
# Source: The Nitrate Pollution Prevention Regulations 2015 (England)
# NVZ rules reduce the effective N allowance by restricting autumn/winter
# applications. We model this as a reduction in total N applied.
# ---------------------------------------------------------------------------

NVZ_N_CAP_KG_HA: dict[str, int] = {
    "winter_wheat":  220,   # NVZ N limit for cereals: 220 kg N/ha total
    "spring_barley": 150,   # Spring cereals have lower N anyway
    "oilseed_rape":  250,   # OSR limit: 250 kg N/ha
    "maize":         150,   # Maize: 150 kg N/ha in NVZ
}

# Closed periods for NVZ farms — no N fertiliser applications allowed
# Format: (start_month, end_month) inclusive
NVZ_CLOSED_PERIODS: dict[str, tuple[int, int]] = {
    "winter_wheat":   (9, 1),    # Sep to Jan closed for cereals
    "spring_barley":  (9, 1),
    "oilseed_rape":   (11, 1),   # Nov to Jan closed for OSR
    "maize":          (10, 1),   # Oct to Jan closed for maize
}

# ---------------------------------------------------------------------------
# Application timing splits (% of total N per month)
# Based on standard RB209 split application recommendations.
# Month numbers relative to planting month.
#
# Structure: {crop: {months_after_planting: fraction_of_total_N}}
#
# Winter wheat (autumn planted):
#   - Tillering application: ~Feb/Mar (+4-5 months after Sep planting)
#   - Stem extension:        ~Apr     (+7 months)
#   - Flag leaf:             ~May     (+8 months) — only for high-yield targets
#
# Spring barley (spring planted):
#   - Single application at drilling or shortly after, then top-dress
#
# Oilseed rape (autumn planted):
#   - Pre-winter small dose, then main application in late Feb/Mar
#
# Maize (spring planted):
#   - Pre-drilling incorporated, sometimes split with top-dress
# ---------------------------------------------------------------------------

APPLICATION_SPLITS: dict[str, list[dict]] = {
    "winter_wheat": [
        {"months_after_planting": 4,  "fraction": 0.20, "label": "Tillering (early)"},
        {"months_after_planting": 5,  "fraction": 0.45, "label": "Tillering (main)"},
        {"months_after_planting": 7,  "fraction": 0.35, "label": "Stem extension"},
    ],
    "spring_barley": [
        {"months_after_planting": 0,  "fraction": 0.40, "label": "Pre-drilling/drilling"},
        {"months_after_planting": 1,  "fraction": 0.60, "label": "Top-dress (3-leaf stage)"},
    ],
    "oilseed_rape": [
        {"months_after_planting": 4,  "fraction": 0.15, "label": "Pre-winter (small dose)"},
        {"months_after_planting": 5,  "fraction": 0.50, "label": "Stem extension (early)"},
        {"months_after_planting": 6,  "fraction": 0.35, "label": "Stem extension (main)"},
    ],
    "maize": [
        {"months_after_planting": -1, "fraction": 0.50, "label": "Pre-drilling (incorporated)"},
        {"months_after_planting": 1,  "fraction": 0.50, "label": "Top-dress (6-leaf stage)"},
    ],
}

# ---------------------------------------------------------------------------
# Soil quality scale to SNS class mapping
# User provides 1-5; we convert to RB209 SNS class
# ---------------------------------------------------------------------------

def soil_quality_to_sns(soil_quality: int) -> str:
    """
    Maps our 1-5 soil quality scale to RB209 SNS class.

    1-2: Poor soils (sandy, low OM) -> Low SNS (highest N requirement)
    3:   Average soils              -> Medium SNS
    4-5: Good soils (clay, high OM) -> High SNS (lowest N requirement)
    """
    if soil_quality <= 2:
        return "low"
    elif soil_quality == 3:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Typical planting month by crop (for validation and default guidance)
# ---------------------------------------------------------------------------

TYPICAL_PLANTING_MONTHS: dict[str, list[int]] = {
    "winter_wheat":   [9, 10, 11],       # Sep-Nov
    "spring_barley":  [3, 4],            # Mar-Apr
    "oilseed_rape":   [8, 9],            # Aug-Sep
    "maize":          [4, 5],            # Apr-May
}

# Harvest month (approximate, for full season context)
TYPICAL_HARVEST_MONTHS: dict[str, list[int]] = {
    "winter_wheat":   [7, 8],
    "spring_barley":  [7, 8],
    "oilseed_rape":   [7],
    "maize":          [9, 10],
}


# ---------------------------------------------------------------------------
# Shared utility functions
# Defined here so they can be imported by both n_calculator and weather_adjuster
# ---------------------------------------------------------------------------

def _in_closed_period(month: int, start: int, end: int) -> bool:
    """
    Checks if a month falls within a closed period.
    Handles periods that span the year boundary (e.g. Sep=9 to Jan=1).
    """
    if start > end:  # Wraps year boundary (e.g. Sep-Jan)
        return month >= start or month <= end
    return start <= month <= end


MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

def _month_name(month: int) -> str:
    return MONTH_NAMES.get(month, f"Month {month}")
