#!/usr/bin/env python3
"""Plot one League throw and its synthetic ranking queries in 3D."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_COLUMNS = ("throw_id", "league_throw_id")
QUERY_FILES = ("train_queries.csv", "validation_queries.csv", "test_queries.csv")


def trajectory(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the common PoR-relative displacement samples from a feature row."""
    times = sorted(
        int(match.group(1))
        for column in row.index
        if (match := re.fullmatch(r"disp_x_t(\d+)ms", column))
        and f"disp_y_t{match.group(1)}ms" in row.index
        and f"disp_z_t{match.group(1)}ms" in row.index
    )
    coordinates = np.asarray([
        [row[f"disp_{axis}_t{time_ms}ms"] for time_ms in times]
        for axis in "xyz"
    ], dtype=float)
    return coordinates[0], coordinates[1], coordinates[2]


def load_queries(directory: Path) -> pd.DataFrame:
    files = [directory / name for name in QUERY_FILES if (directory / name).exists()]
    if not files:
        raise FileNotFoundError(f"No *_queries.csv files found in {directory}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def plot_line(ax, points: tuple[np.ndarray, np.ndarray, np.ndarray], **kwargs) -> None:
    x, y, z = points
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    ax.plot(x[valid], y[valid], z[valid], **kwargs)
    ax.scatter(x[valid], y[valid], z[valid], s=9, color=kwargs.get("color"), alpha=.7)


def set_common_axes(axes, trajectories) -> None:
    if not any(np.isfinite(values).any() for points in trajectories for values in points):
        return
    ranges = []
    for coordinate in range(3):
        values = np.concatenate([
            points[coordinate][np.isfinite(points[coordinate])] for points in trajectories
        ])
        low, high = float(values.min()), float(values.max())
        margin = max((high - low) * .08, .01)
        ranges.append((low - margin, high + margin))

    aspect = np.asarray([high - low for low, high in ranges])
    aspect = np.maximum(aspect / aspect.max(), .2)
    for ax in axes:
        ax.set_xlim(*ranges[0])
        ax.set_ylim(*ranges[1])
        ax.set_zlim(*ranges[2])
        ax.set_box_aspect(aspect)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.grid(True, alpha=.25)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    default_features = project / "out" / "throw_features"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", type=Path, default=default_features / "features_league.csv")
    parser.add_argument("--queries-dir", type=Path, default=default_features / "ranking_dataset")
    parser.add_argument("--throw-id", help="League throw ID; defaults to the first source in train_queries.csv")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it")
    args = parser.parse_args()

    league = pd.read_csv(args.league, sep=None, engine="python")
    queries = load_queries(args.queries_dir)
    if "source_league_throw_id" not in queries:
        raise ValueError("Query CSVs need a source_league_throw_id column")

    id_column = next((name for name in ID_COLUMNS if name in league), None)
    if id_column is None:
        raise ValueError(f"League CSV needs one of these ID columns: {ID_COLUMNS}")

    throw_id = args.throw_id or str(queries.iloc[0]["source_league_throw_id"])
    ground_truth_rows = league[league[id_column].astype(str) == throw_id]
    synthetic_rows = queries[queries["source_league_throw_id"].astype(str) == throw_id]
    if ground_truth_rows.empty:
        raise ValueError(f"League throw {throw_id!r} was not found in {args.league}")
    if synthetic_rows.empty:
        raise ValueError(f"No synthetic queries found for League throw {throw_id!r}")

    ground_truth = trajectory(ground_truth_rows.iloc[0])
    synthetics = [trajectory(row) for _, row in synthetic_rows.iterrows()]
    colors = plt.cm.tab10(np.linspace(0, 1, len(synthetics)))

    panel_count = 2 + len(synthetics)
    columns = min(4, panel_count)
    rows = math.ceil(panel_count / columns)
    figure = plt.figure(figsize=(4.8 * columns, 4.2 * rows), constrained_layout=True)
    axes = [figure.add_subplot(rows, columns, index + 1, projection="3d")
            for index in range(panel_count)]

    plot_line(axes[0], ground_truth, color="black", linewidth=3, label="ground truth")
    for index, (points, color) in enumerate(zip(synthetics, colors), start=1):
        plot_line(axes[0], points, color=color, linewidth=1.5, alpha=.8,
                  label=f"synthetic {index}")
    axes[0].set_title("Overlay")
    axes[0].legend(fontsize=8)

    plot_line(axes[1], ground_truth, color="black", linewidth=3)
    axes[1].set_title("Ground truth")
    for index, (points, color, (_, query)) in enumerate(
        zip(synthetics, colors, synthetic_rows.iterrows()), start=2
    ):
        plot_line(axes[index], ground_truth, color="0.65", linewidth=1.5,
                  linestyle="--", label="ground truth")
        plot_line(axes[index], points, color=color, linewidth=2, label="synthetic")
        axes[index].set_title(str(query.get("throw_id", f"Synthetic {index - 1}")))
        axes[index].legend(fontsize=8)

    set_common_axes(axes, [ground_truth, *synthetics])
    figure.suptitle(
        f"League throw {throw_id}: PoR-relative ground truth and "
        f"{len(synthetics)} synthetic queries",
        fontsize=15,
    )

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", throw_id)
    output = args.output or args.queries_dir / f"synthetic_trajectories_{safe_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Wrote {output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
