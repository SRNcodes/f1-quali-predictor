"""
Ingestion stage: pulls raw qualifying data from the Jolpica-F1 API and
saves it to data/raw/ as CSVs (later: this writes to S3 instead).

Usage:
    python fetch_data.py --last-year 2025 --this-year 2026 --circuit baku
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from jolpica_client import get_qualifying_results, get_next_race  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--this-year", type=int, default=2026)
    parser.add_argument("--circuit", type=str, default=None,
                         help="Circuit id (e.g. 'baku'). If omitted, auto-detects the next "
                              "upcoming race on the calendar for --this-year.")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    circuit = args.circuit
    if circuit is None:
        next_race = get_next_race(args.this_year)
        circuit = next_race["circuit_id"]
        print(f"No --circuit given. Auto-detected next race: {next_race['race_name']} "
              f"({circuit}) on {next_race['date']}.")

    print(f"Fetching {args.last_year} {circuit} qualifying...")
    args.circuit = circuit
    last_year_circuit = get_qualifying_results(args.last_year, circuit_id=args.circuit)
    last_year_circuit.to_csv(
        os.path.join(RAW_DIR, f"qualifying_{args.last_year}_{args.circuit}.csv"), index=False
    )

    print(f"Fetching {args.this_year} season-to-date qualifying...")
    season_so_far = get_qualifying_results(args.this_year)
    season_so_far.to_csv(
        os.path.join(RAW_DIR, f"qualifying_{args.this_year}_season.csv"), index=False
    )

    print(f"Saved raw data to {RAW_DIR}")


if __name__ == "__main__":
    main()
