#!/usr/bin/env python3
"""Quick 3D viewer for QTM-like mocap TSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd


def load_mocap_tsv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Frame\tTime"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find data header line starting with 'Frame\\tTime'.")

    df = pd.read_csv(path, sep="\t", skiprows=header_idx, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    xyz_cols = [c for c in df.columns if c.endswith(" X") or c.endswith(" Y") or c.endswith(" Z")]
    if not xyz_cols:
        raise ValueError("No marker coordinate columns ending in ' X', ' Y', or ' Z' found.")

    df[xyz_cols] = df[xyz_cols].apply(pd.to_numeric, errors="coerce")
    markers = sorted({c[:-2] for c in xyz_cols})
    return df, markers


def build_frame_points(df: pd.DataFrame, frame_index: int, markers: list[str]) -> pd.DataFrame:
    row = df.iloc[frame_index]
    points = []

    for marker in markers:
        x_col, y_col, z_col = f"{marker} X", f"{marker} Y", f"{marker} Z"
        if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
            continue

        x, y, z = row[x_col], row[y_col], row[z_col]
        if pd.notna(x) and pd.notna(y) and pd.notna(z):
            points.append((marker, float(x), float(y), float(z)))

    return pd.DataFrame(points, columns=["marker", "x", "y", "z"])


def compute_axes_bounds(df: pd.DataFrame, markers: list[str]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_cols = [f"{m} X" for m in markers if f"{m} X" in df.columns]
    y_cols = [f"{m} Y" for m in markers if f"{m} Y" in df.columns]
    z_cols = [f"{m} Z" for m in markers if f"{m} Z" in df.columns]

    x_vals = df[x_cols].to_numpy(dtype=float).ravel()
    y_vals = df[y_cols].to_numpy(dtype=float).ravel()
    z_vals = df[z_cols].to_numpy(dtype=float).ravel()

    x_vals = x_vals[~pd.isna(x_vals)]
    y_vals = y_vals[~pd.isna(y_vals)]
    z_vals = z_vals[~pd.isna(z_vals)]

    if len(x_vals) == 0 or len(y_vals) == 0 or len(z_vals) == 0:
        raise ValueError("Could not determine axis bounds from marker coordinates.")

    mins = [x_vals.min(), y_vals.min(), z_vals.min()]
    maxs = [x_vals.max(), y_vals.max(), z_vals.max()]
    center = [(lo + hi) / 2 for lo, hi in zip(mins, maxs)]
    radius = max((hi - lo) for lo, hi in zip(mins, maxs)) / 2
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def plot_frame(points: pd.DataFrame, frame_no: int, time_s: float | None, annotate: bool) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points["x"], points["y"], points["z"], s=24, alpha=0.9)

    if annotate:
        for _, p in points.iterrows():
            ax.text(p["x"], p["y"], p["z"], p["marker"], fontsize=7)

    title = f"Mocap 3D markers - Frame {frame_no}"
    if time_s is not None and pd.notna(time_s):
        title += f" (t={time_s:.3f}s)"
    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Keep equal scale on all axes so body proportions look correct.
    mins = points[["x", "y", "z"]].min().to_numpy()
    maxs = points[["x", "y", "z"]].max().to_numpy()
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    plt.show()


def animate_motion(df: pd.DataFrame, markers: list[str], annotate: bool, fps: float) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xlim, ylim, zlim = compute_axes_bounds(df, markers)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scat = ax.scatter([], [], [], s=24, alpha=0.9)
    label_artists = []

    def update(i: int):
        nonlocal label_artists

        for txt in label_artists:
            txt.remove()
        label_artists = []

        points = build_frame_points(df, i, markers)
        if points.empty:
            scat._offsets3d = ([], [], [])
        else:
            scat._offsets3d = (
                points["x"].to_numpy(),
                points["y"].to_numpy(),
                points["z"].to_numpy(),
            )

            if annotate:
                for _, p in points.iterrows():
                    label_artists.append(ax.text(p["x"], p["y"], p["z"], p["marker"], fontsize=7))

        frame_no = int(df.iloc[i]["Frame"]) if "Frame" in df.columns else i
        time_s = pd.to_numeric(df.iloc[i].get("Time"), errors="coerce")
        title = f"Mocap 3D markers - Frame {frame_no}"
        if pd.notna(time_s):
            title += f" (t={time_s:.3f}s)"
        ax.set_title(title)

        return [scat, *label_artists]

    interval_ms = max(1, int(1000 / max(1e-6, fps)))
    ani = FuncAnimation(fig, update, frames=len(df), interval=interval_ms, blit=False, repeat=True)

    # Keep a live reference so the animation is not garbage-collected.
    fig._ani = ani
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a mocap TSV in 3D (single frame or full motion)")
    parser.add_argument("file", nargs="?", default="first_throw(test)0221.tsv", help="Path to TSV file")
    parser.add_argument("--frame", type=int, default=0, help="0-based frame index to show (only with --single-frame)")
    parser.add_argument("--single-frame", action="store_true", help="Show only one frame instead of animation")
    parser.add_argument("--labels", action="store_true", help="Show marker labels")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS for animation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file)

    df, markers = load_mocap_tsv(path)
    if df.empty:
        raise ValueError("No data rows found in TSV.")

    if args.single_frame:
        frame_index = max(0, min(args.frame, len(df) - 1))
        points = build_frame_points(df, frame_index, markers)
        if points.empty:
            raise ValueError("Selected frame has no valid XYZ marker points.")

        frame_no = int(df.iloc[frame_index]["Frame"]) if "Frame" in df.columns else frame_index
        time_s = pd.to_numeric(df.iloc[frame_index].get("Time"), errors="coerce")
        plot_frame(points, frame_no, time_s, args.labels)
    else:
        animate_motion(df, markers, args.labels, args.fps)


if __name__ == "__main__":
    main()
