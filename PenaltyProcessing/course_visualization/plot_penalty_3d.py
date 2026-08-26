"""Plot one tracked penalty throw in 3D."""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent


def draw_goal(axis, goal_x):
    for y in (-1.5, 1.5):
        axis.plot([goal_x, goal_x], [y, y], [0, 2], color="black", linewidth=2)
    axis.plot([goal_x, goal_x], [-1.5, 1.5], [2, 2], color="black", linewidth=2)


def add_markers(axis, points, xyz):
    """Mark the main events along a penalty trajectory."""
    axis.scatter(*xyz[0], color="#2ca02c", marker="o", s=70, label="start")

    velocity = np.array([point.get("v", np.nan) for point in points], dtype=float)
    acceleration = np.array([point.get("a", np.nan) for point in points], dtype=float)
    if np.isfinite(velocity).any():
        index = np.nanargmax(velocity)
        axis.scatter(*xyz[index], color="#ff9900", marker="D", s=75, label="max velocity")
    if np.isfinite(acceleration).any():
        index = np.nanargmax(acceleration)
        axis.scatter(*xyz[index], color="#8e44ad", marker="s", s=75, label="max acceleration")

    crossing = np.flatnonzero(np.abs(xyz[:, 0]) >= 20)
    if len(crossing):
        axis.scatter(*xyz[crossing[0]], color="black", marker="X", s=90, label="goal crossing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="plot one penalty instead of the complete CSV")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    data = pd.read_csv(HERE / "data" / "penalty.csv", sep=";", dtype=str)
    if args.id:
        data = data[data["id"] == args.id]
    if data.empty:
        raise SystemExit(f"Penalty {args.id} was not found")

    output_dir = HERE / "plots_3d"
    output_dir.mkdir(exist_ok=True)
    for _, row in data.iterrows():
        try:
            points = json.loads(row["trajectory_json"])
            xyz = np.array([[point[axis] for axis in "xyz"] for point in points], dtype=float)
            release = json.loads(row["release_point_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            print(f"Skipped {row['id']}: incomplete trajectory")
            continue

        figure = plt.figure(figsize=(9, 6))
        axis = figure.add_subplot(projection="3d")
        draw_goal(axis, 20 if xyz[:, 0].mean() > 0 else -20)
        axis.plot(*xyz.T, color="#2878b5", linewidth=2)
        axis.scatter(*xyz.T, c=np.arange(len(xyz)), cmap="viridis", s=18)
        add_markers(axis, points, xyz)
        axis.scatter(release["x"], release["y"], release["z"], color="#e45756",
                     marker="*", s=180, label="release")
        axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                 title=f"Penalty {row['id']}: {row['home_team']} vs {row['away_team']}")
        axis.legend()
        figure.tight_layout()

        output = output_dir / f"{row['id']}.png"
        figure.savefig(output, dpi=150)
        print(f"Saved {output}")
        if args.show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
