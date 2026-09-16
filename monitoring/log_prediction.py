"""
Prediction tracking: logs what the model predicted before each race weekend,
then (once qualifying has actually happened) fetches the real result and
scores the prediction. This builds up a real track record over time, which
is a much stronger "does this work" signal than cross-validation alone.

Usage:
    # Right after running training/train.py and getting predictions for the
    # next race, log them:
    python monitoring/log_prediction.py log --circuit baku --season 2026 \
        --predictions-csv data/processed/prediction_features.csv

    # After that race's qualifying has actually happened, score it:
    python monitoring/log_prediction.py score --circuit baku --season 2026
"""

import argparse
import os
import sys
import json
import datetime
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from jolpica_client import get_qualifying_results  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")


def log_prediction(season, circuit, predicted_order):
    """predicted_order: list of driver_ids in predicted qualifying order (fastest first)."""
    row = {
        "logged_at": datetime.datetime.utcnow().isoformat(),
        "season": season,
        "circuit_id": circuit,
        "predicted_order": json.dumps(predicted_order),
        "actual_order": None,
        "top1_correct": None,
        "mean_position_error": None,
    }
    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)
    print(f"Logged prediction for {circuit} {season}: {predicted_order}")


def score_prediction(season, circuit):
    """Fetch actual qualifying results and score the most recent unscored
    prediction for this circuit/season."""
    if not os.path.exists(LOG_PATH):
        print("No prediction log found yet.")
        return

    log_df = pd.read_csv(LOG_PATH)
    mask = (log_df["season"] == season) & (log_df["circuit_id"] == circuit) & (
        log_df["actual_order"].isna()
    )
    if not mask.any():
        print(f"No unscored prediction found for {circuit} {season}.")
        return

    from jolpica_client import get_qualifying_results as _get_q

    actual_df = _get_q(season, circuit_id=circuit)
    if actual_df.empty:
        print("Actual qualifying results not available yet (race hasn't happened).")
        return
    actual_order = actual_df.sort_values("best_time")["driver_id"].tolist()

    idx = log_df[mask].index[-1]
    predicted_order = json.loads(log_df.loc[idx, "predicted_order"])

    top1_correct = int(predicted_order[0] == actual_order[0]) if actual_order else None

    pos_errors = []
    for i, driver in enumerate(predicted_order):
        if driver in actual_order:
            pos_errors.append(abs(i - actual_order.index(driver)))
    mean_position_error = sum(pos_errors) / len(pos_errors) if pos_errors else None

    log_df.loc[idx, "actual_order"] = json.dumps(actual_order)
    log_df.loc[idx, "top1_correct"] = top1_correct
    log_df.loc[idx, "mean_position_error"] = mean_position_error
    log_df.to_csv(LOG_PATH, index=False)

    print(f"Scored {circuit} {season}: pole prediction correct = {bool(top1_correct)}, "
          f"mean position error = {mean_position_error:.2f} places")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["log", "score"])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--predictions-csv", type=str,
                         help="Path to prediction_features.csv with predicted_lap_time_sec column, "
                              "required for 'log' mode")
    args = parser.parse_args()

    if args.mode == "log":
        if not args.predictions_csv:
            raise SystemExit("--predictions-csv is required for 'log' mode")
        pred_df = pd.read_csv(args.predictions_csv, index_col=0)
        sort_col = "predicted_lap_time_sec" if "predicted_lap_time_sec" in pred_df.columns else pred_df.columns[0]
        predicted_order = pred_df.sort_values(sort_col).index.tolist()
        log_prediction(args.season, args.circuit, predicted_order)
    else:
        score_prediction(args.season, args.circuit)


if __name__ == "__main__":
    main()
