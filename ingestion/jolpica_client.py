"""
Thin client for the Jolpica-F1 API (free, no key required; successor to Ergast).
Used by fetch_data.py to pull qualifying results.
"""

import time
import requests
import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def fetch_json(url, params=None):
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    time.sleep(0.3)  # be polite to the free API
    return resp.json()


def time_to_seconds(t):
    """Convert 'M:SS.sss' qualifying time string to seconds. Returns None if missing."""
    if not t:
        return None
    if ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return float(t)


def get_qualifying_results(season, round_=None, circuit_id=None):
    """Pull qualifying results for a season (optionally a specific round or circuit)."""
    url = f"{BASE_URL}/{season}"
    if round_:
        url += f"/{round_}"
    url += "/qualifying.json"

    rows = []
    offset, limit = 0, 100
    while True:
        data = fetch_json(url, params={"limit": limit, "offset": offset})
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            break
        for race in races:
            if circuit_id and race["Circuit"]["circuitId"] != circuit_id:
                continue
            round_num = int(race["round"])
            for q in race["QualifyingResults"]:
                times = [time_to_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
                times = [t for t in times if t is not None]
                if not times:
                    continue
                rows.append({
                    "season": season,
                    "round": round_num,
                    "circuit_id": race["Circuit"]["circuitId"],
                    "driver_id": q["Driver"]["driverId"],
                    "constructor_id": q["Constructor"]["constructorId"],
                    "best_time": min(times),
                })
        total = int(data["MRData"]["total"])
        offset += limit
        if offset >= total:
            break
    return pd.DataFrame(rows)


def get_next_race(season):
    """Return the next unrun race (round, circuit_id, race_name, date) in a season,
    based on today's date. Falls back to the last race in the season if none remain."""
    import datetime
    data = fetch_json(f"{BASE_URL}/{season}.json", params={"limit": 100})
    races = data["MRData"]["RaceTable"]["Races"]
    today = datetime.date.today().isoformat()
    upcoming = [r for r in races if r["date"] >= today]
    target = upcoming[0] if upcoming else races[-1]
    return {
        "round": int(target["round"]),
        "circuit_id": target["Circuit"]["circuitId"],
        "race_name": target["raceName"],
        "date": target["date"],
    }


def get_race_results(season):
    """Pull all race results for a season (driver, constructor, grid, finish position)."""
    rows = []
    offset, limit = 0, 100
    while True:
        data = fetch_json(f"{BASE_URL}/{season}/results.json",
                           params={"limit": limit, "offset": offset})
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            break
        for race in races:
            round_num = int(race["round"])
            for r in race["Results"]:
                rows.append({
                    "season": season,
                    "round": round_num,
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": r["Driver"]["driverId"],
                    "constructor_id": r["Constructor"]["constructorId"],
                    "grid": int(r["grid"]) if r["grid"] else None,
                    "finish_position": int(r["position"]) if r["position"].isdigit() else None,
                    "status": r["status"],
                    "points": float(r["points"]),
                })
        total = int(data["MRData"]["total"])
        offset += limit
        if offset >= total:
            break
    return pd.DataFrame(rows)
