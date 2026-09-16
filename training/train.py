"""
Training stage: reads data/processed/training_set.csv, compares Ridge /
RandomForest / XGBoost / GaussianProcess via 5-fold CV, picks the best model,
retrains on all data, and saves it + the scaler to models/.

Usage:
    python train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

X_COLS = [
    "driver_season_avg_delta",
    "driver_recent_avg_delta",
    "constructor_season_avg_delta",
    "driver_baku_last_year_delta",
]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "training_set.csv"))
    X, y = train_df[X_COLS], train_df["target_pace_delta"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05),
        "GaussianProcess": GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), normalize_y=True),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("--- Cross-validated MAE (seconds off pole) per model ---")
    results = {}
    for name, model in models.items():
        scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring="neg_mean_absolute_error")
        mean_mae = float(scores.mean())
        sem = float(scores.std(ddof=1) / np.sqrt(len(scores)))  # standard error of the mean
        results[name] = {"mean_mae": mean_mae, "sem": sem, "fold_scores": scores.tolist()}
        print(f"{name:15s}: MAE = {mean_mae:.3f}s  |  SEM = {sem:.3f}s  (95% CI: +/- {1.96 * sem:.3f}s)")

    best_name = min(results, key=lambda k: results[k]["mean_mae"])
    print(f"\nBest model: {best_name}")

    best_model = models[best_name]
    best_model.fit(X_scaled, y)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump({"best_model": best_name, "cv_results": results, "features": X_COLS}, f, indent=2)

    print(f"Saved model, scaler, and metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
