"""Compare supplied and recomputed speed and acceleration values."""

from pathlib import Path
import argparse
import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent


def numbers(values):
    return np.array([np.nan if value is None else value for value in values], dtype=float)


def read_list(text):
    """The CSV contains both JSON lists and older Python-style lists."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="plot one penalty instead of the complete CSV")
    args = parser.parse_args()

    data = pd.read_csv(HERE / "data" / "penalty.csv", sep=";", dtype=str)
    if args.id:
        data = data[data["id"] == args.id]
    if data.empty:
        raise SystemExit(f"Penalty {args.id} was not found")

    for _, row in data.iterrows():
        plot_penalty(row)


def plot_penalty(row):
    trajectory = json.loads(row["trajectory_json"])
    time = np.arange(len(trajectory)) * 0.025  # fixture data is sampled at 40 Hz
    try:
        release_index = int(float(row["release_idx"]))
    except (TypeError, ValueError):
        release_index = None

    series = [
        ("velocity", "m/s", numbers([p.get("v") for p in trajectory]),
         numbers(read_list(row["velocity_per_point"]))),
        ("acceleration", "m/s²", numbers([p.get("a") for p in trajectory]),
         numbers(read_list(row["acceleration_per_point"]))),
    ]
    for name, unit, fixture, recomputed in series:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(time, fixture, label="fixture", linewidth=2)
        axis.plot(time, recomputed, label="recomputed", linewidth=2)
        if release_index is not None and release_index < len(time):
            axis.axvline(time[release_index], color="black", linestyle="--", label="release")
        axis.set(xlabel="time since first sample (s)", ylabel=f"{name} ({unit})",
                 title=f"Penalty {row['id']}: fixture vs recomputed {name}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        output_dir = HERE / "timeseries" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{row['id']}.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
