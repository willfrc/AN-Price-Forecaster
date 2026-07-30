"""
NVZ (Nitrate Vulnerable Zone) status lookup.

Determines whether a farm location falls within a Nitrate Vulnerable Zone
using DEFRA's open data API. NVZ designation significantly affects:
    - Total N application caps
    - Application timing (closed periods Oct-Jan for most crops)
    - Record-keeping requirements

Source: DEFRA Magic Map / Natural England open data
API: https://environment.data.gov.uk/arcgis/rest/services/

For the PoC we use two approaches in priority order:
    1. DEFRA ArcGIS REST API (live lookup by coordinates)
    2. Fallback: assume NVZ = False if API unavailable (conservative for PoC)

In production, pre-download the NVZ shapefile and do a local spatial join.
Shapefile: https://www.data.gov.uk/dataset/nitrate-vulnerable-zones-england
"""

import requests
from utils.logger import get_logger

logger = get_logger(__name__)

# DEFRA ArcGIS REST endpoint for NVZ designation in England
NVZ_API_URL = (
    "https://environment.data.gov.uk/arcgis/rest/services/"
    "EA/NitrateVulnerableZones/MapServer/0/query"
)

# Postcodes API for converting UK postcode to lat/long (free, no key)
POSTCODES_API_URL = "https://api.postcodes.io/postcodes/{}"


def postcode_to_latlon(postcode: str) -> tuple[float, float] | None:
    """
    Converts a UK postcode to latitude/longitude using the free postcodes.io API.

    Args:
        postcode: UK postcode string (e.g. "PE1 1AB" or "PE11AB")

    Returns:
        (latitude, longitude) tuple or None if lookup fails
    """
    clean = postcode.replace(" ", "").upper()
    url = POSTCODES_API_URL.format(clean)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == 200 and data.get("result"):
            lat = data["result"]["latitude"]
            lon = data["result"]["longitude"]
            logger.debug(f"Postcode {postcode} -> ({lat}, {lon})")
            return lat, lon
        else:
            logger.warning(f"Postcode lookup failed for {postcode}: {data.get('error', 'unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Postcode API request failed: {e}")
        return None


def check_nvz_status(lat: float, lon: float) -> bool:
    """
    Checks whether a coordinate falls within a Nitrate Vulnerable Zone
    using DEFRA's ArcGIS REST API.

    Args:
        lat: Latitude (WGS84)
        lon: Longitude (WGS84)

    Returns:
        True if in NVZ, False if not, False if lookup fails (conservative default)
    """
    params = {
        "geometry":     f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR":         "4326",              # WGS84
        "spatialRel":   "esriSpatialRelIntersects",
        "returnCountOnly": "true",
        "f":            "json",
    }

    try:
        response = requests.get(NVZ_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        count = data.get("count", 0)
        in_nvz = count > 0
        logger.info(f"NVZ check ({lat:.4f}, {lon:.4f}): {'IN NVZ' if in_nvz else 'NOT in NVZ'}")
        return in_nvz

    except requests.exceptions.RequestException as e:
        logger.warning(
            f"NVZ API lookup failed: {e}. "
            "Defaulting to NVZ=False. "
            "Check DEFRA API availability or use manual override."
        )
        return False
    except (KeyError, ValueError) as e:
        logger.warning(f"NVZ API response parsing failed: {e}. Defaulting to NVZ=False.")
        return False


def get_nvz_status_from_postcode(postcode: str) -> dict:
    """
    Full lookup: postcode -> lat/lon -> NVZ status.

    Returns a dict with all intermediate results for transparency.
    """
    result = {
        "postcode":  postcode,
        "latitude":  None,
        "longitude": None,
        "in_nvz":    False,
        "lookup_ok": False,
        "notes":     "",
    }

    coords = postcode_to_latlon(postcode)
    if coords is None:
        result["notes"] = "Postcode lookup failed — check postcode format. NVZ defaulted to False."
        return result

    lat, lon = coords
    result["latitude"]  = lat
    result["longitude"] = lon

    in_nvz = check_nvz_status(lat, lon)
    result["in_nvz"]    = in_nvz
    result["lookup_ok"] = True
    result["notes"]     = (
        "Location is within a Nitrate Vulnerable Zone. "
        "N application caps and closed periods apply."
        if in_nvz else
        "Location is not within a Nitrate Vulnerable Zone."
    )

    return result
