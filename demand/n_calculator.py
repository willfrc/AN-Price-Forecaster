"""
Nitrogen requirement calculator — Phase 1 of the farm demand engine.

Takes farm inputs and returns:
    1. Total N requirement (kg N/ha and total kg N for the farm)
    2. Total AN requirement (tonnes, accounting for 34.5% N content)
    3. Monthly application schedule (how many tonnes of AN per month)
    4. NVZ adjustments if applicable

This is a rules-based calculation derived from AHDB RB209 recommendations.
No ML involved — agronomic lookup tables are more reliable than any model
we could train without farm-level historical data.

Usage:
    from demand.n_calculator import calculate_n_requirement
    result = calculate_n_requirement(
        crop="winter_wheat",
        farm_size_ha=120.0,
        soil_quality=3,
        planting_month=9,
        postcode="PE1 1AB",
    )
    print(result.summary())
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from demand.rb209_tables import (
    _in_closed_period, _month_name, MONTH_NAMES,
    SUPPORTED_CROPS, N_REQUIREMENT_KG_HA, NVZ_N_CAP_KG_HA,
    APPLICATION_SPLITS, NVZ_CLOSED_PERIODS,
    soil_quality_to_sns, TYPICAL_PLANTING_MONTHS, TYPICAL_HARVEST_MONTHS,
)
from demand.nvz_lookup import get_nvz_status_from_postcode
from utils.logger import get_logger

logger = get_logger(__name__)

# AN nitrogen content (%)
AN_N_CONTENT = 0.345   # 34.5% N — standard agricultural grade AN


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MonthlyApplication:
    """A single N application event."""
    month:          int       # Calendar month (1-12)
    month_name:     str
    label:          str       # Application type (e.g. "Tillering (main)")
    n_kg_ha:        float     # N applied this event (kg/ha)
    n_kg_total:     float     # N applied this event (total kg for farm)
    an_tonnes:      float     # AN required for this application (tonnes)
    nvz_restricted: bool = False  # True if this application is in a closed period


@dataclass
class NRequirementResult:
    """Full output from the N requirement calculator."""

    # Farm inputs
    crop:             str
    crop_name:        str
    farm_size_ha:     float
    soil_quality:     int
    sns_class:        str
    planting_month:   int
    postcode:         str

    # NVZ
    in_nvz:           bool
    nvz_lookup_ok:    bool
    latitude:         Optional[float]
    longitude:        Optional[float]

    # N requirements
    n_rate_kg_ha:     float    # Total N kg/ha (pre-NVZ)
    n_rate_kg_ha_nvz: float    # Total N kg/ha (post-NVZ cap if applicable)
    n_total_kg:       float    # Total N for whole farm
    an_total_tonnes:  float    # Total AN for whole farm

    # Schedule
    applications:     list[MonthlyApplication] = field(default_factory=list)

    # Warnings
    warnings:         list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Returns a human-readable summary of the N requirement."""
        lines = [
            "=" * 60,
            "FARM N REQUIREMENT SUMMARY",
            "=" * 60,
            f"Crop:              {self.crop_name}",
            f"Farm size:         {self.farm_size_ha:.1f} ha",
            f"Soil quality:      {self.soil_quality}/5 ({self.sns_class.upper()} SNS)",
            f"Planting month:    {_month_name(self.planting_month)}",
            f"Location:          {self.postcode}",
            f"NVZ status:        {'IN NVZ' if self.in_nvz else 'Not in NVZ'}",
            "",
            f"N rate:            {self.n_rate_kg_ha:.0f} kg N/ha",
        ]

        if self.in_nvz and self.n_rate_kg_ha_nvz < self.n_rate_kg_ha:
            lines.append(f"N rate (NVZ cap):   {self.n_rate_kg_ha_nvz:.0f} kg N/ha (reduced by NVZ rules)")

        lines += [
            f"Total N required:  {self.n_total_kg:,.0f} kg N",
            f"Total AN required: {self.an_total_tonnes:.1f} tonnes",
            "",
            "APPLICATION SCHEDULE:",
            f"{'Month':<12} {'Application':<30} {'AN (t)':>8} {'N (kg/ha)':>10}",
            "-" * 65,
        ]

        for app in self.applications:
            nvz_flag = " [NVZ CLOSED PERIOD]" if app.nvz_restricted else ""
            lines.append(
                f"{app.month_name:<12} {app.label:<30} "
                f"{app.an_tonnes:>8.1f} {app.n_kg_ha:>10.0f}"
                f"{nvz_flag}"
            )

        lines += ["", "=" * 60]

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("=" * 60)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

def calculate_n_requirement(
    crop:           str,
    farm_size_ha:   float,
    soil_quality:   int,
    planting_month: int,
    postcode:       str,
    nvz_override:   Optional[bool] = None,
) -> NRequirementResult:
    """
    Calculates the full N and AN requirement for a farm, including a
    monthly application schedule.

    Args:
        crop:           Crop key (see SUPPORTED_CROPS)
        farm_size_ha:   Total arable area in hectares
        soil_quality:   Soil quality score 1-5 (1=poor, 5=excellent)
        planting_month: Month of planting (1-12)
        postcode:       UK postcode for NVZ lookup
        nvz_override:   Optional manual NVZ override (True/False).
                        Use if API lookup fails or for testing.

    Returns:
        NRequirementResult with full schedule and summary
    """
    warnings = []

    # --- Input validation ---
    crop = crop.lower().strip()
    if crop not in SUPPORTED_CROPS:
        raise ValueError(
            f"Unsupported crop: '{crop}'. "
            f"Supported crops: {list(SUPPORTED_CROPS.keys())}"
        )

    if not 1 <= soil_quality <= 5:
        raise ValueError(f"Soil quality must be 1-5, got {soil_quality}")

    if not 1 <= planting_month <= 12:
        raise ValueError(f"Planting month must be 1-12, got {planting_month}")

    if farm_size_ha <= 0:
        raise ValueError(f"Farm size must be positive, got {farm_size_ha}")

    # Warn if planting month is unusual for crop
    typical = TYPICAL_PLANTING_MONTHS[crop]
    if planting_month not in typical:
        warnings.append(
            f"Planting month {_month_name(planting_month)} is unusual for "
            f"{SUPPORTED_CROPS[crop]}. Typical planting: "
            f"{', '.join(_month_name(m) for m in typical)}. "
            "Application timing may be suboptimal."
        )

    # --- NVZ lookup ---
    if nvz_override is not None:
        in_nvz     = nvz_override
        nvz_ok     = True
        latitude   = None
        longitude  = None
        logger.info(f"NVZ status manually overridden: {in_nvz}")
    else:
        logger.info(f"Looking up NVZ status for postcode: {postcode}")
        nvz_data   = get_nvz_status_from_postcode(postcode)
        in_nvz     = nvz_data["in_nvz"]
        nvz_ok     = nvz_data["lookup_ok"]
        latitude   = nvz_data["latitude"]
        longitude  = nvz_data["longitude"]
        if not nvz_ok:
            warnings.append(
                "NVZ lookup failed — check postcode format or internet connection. "
                "NVZ status defaulted to False (non-NVZ). "
                "If farm is in an NVZ, use nvz_override=True."
            )

    # --- N requirement ---
    sns_class    = soil_quality_to_sns(soil_quality)
    n_rate_kg_ha = N_REQUIREMENT_KG_HA[crop][sns_class]

    # Apply NVZ cap if applicable
    nvz_cap       = NVZ_N_CAP_KG_HA[crop]
    n_rate_nvz    = min(n_rate_kg_ha, nvz_cap) if in_nvz else n_rate_kg_ha

    if in_nvz and n_rate_nvz < n_rate_kg_ha:
        warnings.append(
            f"NVZ N cap of {nvz_cap} kg N/ha applied "
            f"(standard rate was {n_rate_kg_ha} kg N/ha). "
            "Ensure compliance with NVZ closed period rules."
        )

    n_total_kg    = n_rate_nvz * farm_size_ha
    an_total_t    = n_total_kg / 1000 / AN_N_CONTENT  # kg -> tonnes, adjust for AN N content

    # --- Build application schedule ---
    applications = _build_schedule(
        crop=crop,
        n_rate_kg_ha=n_rate_nvz,
        farm_size_ha=farm_size_ha,
        planting_month=planting_month,
        in_nvz=in_nvz,
        warnings=warnings,
    )

    result = NRequirementResult(
        crop=crop,
        crop_name=SUPPORTED_CROPS[crop],
        farm_size_ha=farm_size_ha,
        soil_quality=soil_quality,
        sns_class=sns_class,
        planting_month=planting_month,
        postcode=postcode.upper(),
        in_nvz=in_nvz,
        nvz_lookup_ok=nvz_ok,
        latitude=latitude,
        longitude=longitude,
        n_rate_kg_ha=n_rate_kg_ha,
        n_rate_kg_ha_nvz=n_rate_nvz,
        n_total_kg=n_total_kg,
        an_total_tonnes=round(an_total_t, 2),
        applications=applications,
        warnings=warnings,
    )

    logger.info(
        f"N calculation complete: {SUPPORTED_CROPS[crop]}, "
        f"{farm_size_ha:.0f}ha, {n_rate_nvz:.0f} kg N/ha, "
        f"{an_total_t:.1f}t AN total"
    )

    return result


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def _build_schedule(
    crop:           str,
    n_rate_kg_ha:   float,
    farm_size_ha:   float,
    planting_month: int,
    in_nvz:         bool,
    warnings:       list,
) -> list[MonthlyApplication]:
    """
    Builds the monthly application schedule from RB209 split fractions.
    Checks each application against NVZ closed periods.
    """
    splits      = APPLICATION_SPLITS[crop]
    closed      = NVZ_CLOSED_PERIODS.get(crop, (99, 99))  # (start, end) months
    applications = []

    for split in splits:
        # Calculate calendar month for this application
        months_offset = split["months_after_planting"]
        app_month     = _offset_month(planting_month, months_offset)

        n_this_kg_ha  = n_rate_kg_ha * split["fraction"]
        n_this_kg     = n_this_kg_ha * farm_size_ha
        an_this_t     = n_this_kg / 1000 / AN_N_CONTENT

        # NVZ closed period check
        nvz_restricted = False
        if in_nvz and _in_closed_period(app_month, closed[0], closed[1]):
            nvz_restricted = True
            warnings.append(
                f"Application '{split['label']}' falls in "
                f"{_month_name(app_month)}, which is an NVZ closed period "
                f"for {SUPPORTED_CROPS[crop]}. "
                "This application should be rescheduled to after the closed period."
            )

        applications.append(MonthlyApplication(
            month=app_month,
            month_name=_month_name(app_month),
            label=split["label"],
            n_kg_ha=round(n_this_kg_ha, 1),
            n_kg_total=round(n_this_kg, 1),
            an_tonnes=round(an_this_t, 2),
            nvz_restricted=nvz_restricted,
        ))

    return sorted(applications, key=lambda x: x.month)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _offset_month(base_month: int, offset: int) -> int:
    """Adds months to a base month, wrapping around year boundary."""
    return ((base_month - 1 + offset) % 12) + 1


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


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running example: 120ha winter wheat farm, PE1 1AB, soil quality 3\n")

    result = calculate_n_requirement(
        crop="winter_wheat",
        farm_size_ha=120.0,
        soil_quality=3,
        planting_month=9,       # September planting
        postcode="PE1 1AB",     # Peterborough — typical arable Fenland
    )

    print(result.summary())

    print("\n--- Raw output ---")
    print(f"Total AN required: {result.an_total_tonnes} tonnes")
    print(f"Applications: {len(result.applications)}")
    for app in result.applications:
        print(f"  {app.month_name}: {app.an_tonnes}t AN ({app.label})")
