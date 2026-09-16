"""
Inference stage: loads the trained model + scaler from ../models/, fetches
fresh data for the requested (or auto-detected) circuit, builds the same
prediction features used at training time, and serves the predicted
qualifying order over HTTP.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET /health
    GET /predict/next              -- auto-detects the next race on the calendar
    GET /predict/{circuit_id}      -- predicts a specific circuit (e.g. /predict/monza)
"""

import json
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from jolpica_client import get_qualifying_results, get_next_race  # noqa: E402
from build_features import add_pace_delta, build_prediction_features  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DEFAULT_LAST_YEAR = 2025
DEFAULT_THIS_YEAR = 2026

app = FastAPI(title="F1 Qualifying Predictor")

# Model, scaler, and feature order are loaded once at process startup, not
# per-request -- deserializing a joblib file on every call would add latency
# and is unnecessary since none of these artifacts change until the next
# training run (which redeploys the service).
model = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    metadata = json.load(f)
FEATURE_COLS = metadata["features"]


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["best_model"]}


def _predict_for_circuit(circuit_id, last_year=DEFAULT_LAST_YEAR, this_year=DEFAULT_THIS_YEAR):
    circuit_last_year = get_qualifying_results(last_year, circuit_id=circuit_id)
    if circuit_last_year.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {last_year} qualifying data found for circuit '{circuit_id}'.",
        )
    season_so_far = get_qualifying_results(this_year)
    if season_so_far.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {this_year} qualifying results yet -- season hasn't started.",
        )

    circuit_last_year = add_pace_delta(circuit_last_year)
    season_so_far = add_pace_delta(season_so_far)
    circuit_pole_time = float(circuit_last_year["pole_time"].iloc[0])

    features = build_prediction_features(circuit_last_year, season_so_far)

    X = features[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    predicted_delta = model.predict(X_scaled)

    features = features.copy()
    features["predicted_pace_delta"] = predicted_delta
    features["predicted_lap_time_sec"] = circuit_pole_time + predicted_delta
    features = features.sort_values("predicted_pace_delta")

    predicted_order = [
        {
            "position": i + 1,
            "driver_id": driver_id,
            "constructor_id": row["constructor_id"],
            "predicted_pace_delta": round(float(row["predicted_pace_delta"]), 3),
            "predicted_lap_time_sec": round(float(row["predicted_lap_time_sec"]), 3),
        }
        for i, (driver_id, row) in enumerate(features.iterrows())
    ]

    return {
        "circuit_id": circuit_id,
        "model": metadata["best_model"],
        "predicted_order": predicted_order,
    }


@app.get("/predict/next")
def predict_next(this_year: int = DEFAULT_THIS_YEAR, last_year: int = DEFAULT_LAST_YEAR):
    next_race = get_next_race(this_year)
    result = _predict_for_circuit(next_race["circuit_id"], last_year, this_year)
    result["race_name"] = next_race["race_name"]
    result["date"] = next_race["date"]
    return result


@app.get("/predict/{circuit_id}")
def predict_circuit(circuit_id: str, this_year: int = DEFAULT_THIS_YEAR, last_year: int = DEFAULT_LAST_YEAR):
    return _predict_for_circuit(circuit_id, last_year, this_year)
