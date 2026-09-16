"""
Inference stage (placeholder — build this out after training is validated).
Will load models/model.joblib + scaler.joblib and serve predictions over HTTP.

Usage (once implemented):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI

app = FastAPI(title="F1 Qualifying Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: load model + scaler from ../models/, add a /predict/{circuit} endpoint
