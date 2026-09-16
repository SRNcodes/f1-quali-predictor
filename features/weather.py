"""
Weather feature (optional, most useful for wet-prone circuits like Spa or
Silverstone — Baku is usually dry so this matters less there).

Uses Open-Meteo's free historical weather API (no key required) to fetch
precipitation on a given date/location, and can be merged into features as
a simple "was it wet" signal.
"""

import requests

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_precipitation_mm(lat, lon, date):
    """Returns total precipitation (mm) at a location on a given date (YYYY-MM-DD).
    Returns None if unavailable (e.g. date is in the future, before data exists)."""
    try:
        resp = requests.get(OPEN_METEO_URL, params={
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "daily": "precipitation_sum",
            "timezone": "auto",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        values = data.get("daily", {}).get("precipitation_sum", [])
        return values[0] if values else None
    except requests.RequestException:
        return None


def is_wet_session(lat, lon, date, threshold_mm=1.0):
    """Simple boolean: was there meaningful rain that day."""
    precip = get_precipitation_mm(lat, lon, date)
    if precip is None:
        return None
    return precip >= threshold_mm


# NOTE: for an UPCOMING race (predicting before it happens), historical
# weather at that date obviously doesn't exist yet. For prediction use cases,
# swap this for a weather FORECAST API instead (Open-Meteo also has a free
# forecast endpoint: https://api.open-meteo.com/v1/forecast). This module as
# written is for building TRAINING features (was last year's / this year's
# past races wet or dry), not for the upcoming race itself.
