"""
Feature stage: reads raw qualifying CSVs from data/raw/, normalizes lap times
into "pace delta to pole" (comparable across circuits/years), and writes
model-ready features to data/processed/.

Usage:
    python build_features.py --last-year 2025 --this-year 2026 --circuit baku
"""

import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from jolpica_client import get_next_race  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def add_pace_delta(df):
    """Add pace delta to pole (fastest time) within each (season, round)."""
    df["pole_time"] = df.groupby(["season", "round"])["best_time"].transform("min")
    df["pace_delta"] = df["best_time"] - df["pole_time"]
    return df


def build_training_set(season_so_far):
    """For each past race this season, use only PRIOR rounds' form to predict
    that race's actual pace delta. Avoids leaking the target into features."""
    rows = []
    rounds = sorted(season_so_far["round"].unique())
    for rnd in rounds:
        prior = season_so_far[season_so_far["round"] < rnd]
        actual = season_so_far[season_so_far["round"] == rnd]
        if prior.empty:
            continue
        driver_form = prior.groupby("driver_id")["pace_delta"].mean()
        recent_mask = prior["round"] >= max(1, rnd - 3)
        driver_recent = prior[recent_mask].groupby("driver_id")["pace_delta"].mean()
        constructor_form = prior.groupby("constructor_id")["pace_delta"].mean()

        for _, r in actual.iterrows():
            rows.append({
                "driver_season_avg_delta": driver_form.get(r["driver_id"], driver_form.mean()),
                "driver_recent_avg_delta": driver_recent.get(
                    r["driver_id"], driver_form.get(r["driver_id"], driver_form.mean())
                ),
                "constructor_season_avg_delta": constructor_form.get(
                    r["constructor_id"], constructor_form.mean()
                ),
                "driver_circuit_last_year_delta": driver_form.get(r["driver_id"], driver_form.mean()),
                "target_pace_delta": r["pace_delta"],
            })
    return pd.DataFrame(rows).dropna()


def build_prediction_features(circuit_last_year, season_so_far):
    """Build the feature row for each current driver, to predict the upcoming race."""
    last_round = season_so_far["round"].max()
    recent_mask = season_so_far["round"] >= max(1, last_round - 2)

    driver_form = season_so_far.groupby("driver_id")["pace_delta"].mean().rename("driver_season_avg_delta")
    driver_recent = season_so_far[recent_mask].groupby("driver_id")["pace_delta"].mean().rename("driver_recent_avg_delta")
    constructor_form = season_so_far.groupby("constructor_id")["pace_delta"].mean().rename("constructor_season_avg_delta")
    circuit_last_year_delta = circuit_last_year.set_index("driver_id")["pace_delta"].rename("driver_circuit_last_year_delta")

    latest_round = season_so_far[season_so_far["round"] == last_round]
    driver_to_constructor = latest_round.set_index("driver_id")["constructor_id"]

    feat = pd.DataFrame(index=driver_to_constructor.index)
    feat["constructor_id"] = driver_to_constructor
    feat = feat.join(driver_form).join(driver_recent).join(circuit_last_year_delta)
    feat = feat.join(constructor_form, on="constructor_id")

    feat["driver_season_avg_delta"] = feat["driver_season_avg_delta"].fillna(feat["driver_season_avg_delta"].median())
    feat["driver_recent_avg_delta"] = feat["driver_recent_avg_delta"].fillna(feat["driver_season_avg_delta"])
    feat["driver_circuit_last_year_delta"] = feat["driver_circuit_last_year_delta"].fillna(feat["driver_season_avg_delta"])
    feat["constructor_season_avg_delta"] = feat["constructor_season_avg_delta"].fillna(
        feat["constructor_season_avg_delta"].median()
    )
    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--this-year", type=int, default=2026)
    parser.add_argument("--circuit", type=str, default=None,
                         help="Circuit id (e.g. 'baku'). If omitted, auto-detects the next "
                              "upcoming race on the calendar for --this-year, same as fetch_data.py.")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    circuit = args.circuit
    if circuit is None:
        next_race = get_next_race(args.this_year)
        circuit = next_race["circuit_id"]
        print(f"No --circuit given. Auto-detected next race: {next_race['race_name']} "
              f"({circuit}) on {next_race['date']}.")

    circuit_last_year = pd.read_csv(
        os.path.join(RAW_DIR, f"qualifying_{args.last_year}_{circuit}.csv")
    )
    season_so_far = pd.read_csv(
        os.path.join(RAW_DIR, f"qualifying_{args.this_year}_season.csv")
    )

    circuit_last_year = add_pace_delta(circuit_last_year)
    season_so_far = add_pace_delta(season_so_far)
    circuit_pole_time = float(circuit_last_year["pole_time"].iloc[0])

    training_set = build_training_set(season_so_far)
    training_set.to_csv(os.path.join(PROCESSED_DIR, "training_set.csv"), index=False)

    prediction_features = build_prediction_features(circuit_last_year, season_so_far)
    prediction_features.to_csv(os.path.join(PROCESSED_DIR, "prediction_features.csv"))

    with open(os.path.join(PROCESSED_DIR, "circuit_pole_time.txt"), "w") as f:
        f.write(str(circuit_pole_time))

    print(f"Wrote training_set.csv ({len(training_set)} rows) and prediction_features.csv "
          f"({len(prediction_features)} drivers) to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
