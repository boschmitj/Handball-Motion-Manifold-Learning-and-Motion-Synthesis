#!/usr/bin/env python3
"""Plot penalty trajectories in 3D from penalty_trajectories.csv.

Usage examples:
  python3 plot_penalty_3d.py --shot-id 8008206
  python3 plot_penalty_3d.py --home-team HC_Erlangen --away-team TSV_Hannover_Burgdorf
  python3 plot_penalty_3d.py --random 10  # plot 10 random penalties
    python3 plot_penalty_3d.py --shot-id 8008206 --only-interactive
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

# Backend will be set dynamically based on --show flag


GOAL_X = 20.0
POST_Y = 1.5
GOAL_HEIGHT = 2.0


def load_trajectories(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", dtype=str)
    return df


def find_latest_run_dir(csv_path: Path) -> Optional[Path]:
    penalty_traj_dir = csv_path.parent.parent
    if not penalty_traj_dir.exists():
        return None

    latest_id = None
    latest_dir = None
    for path in penalty_traj_dir.glob("run_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"run_(\d+)", path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        if latest_id is None or run_id > latest_id:
            latest_id = run_id
            latest_dir = path

    if latest_dir is not None:
        return latest_dir

    if re.fullmatch(r"run_[A-Za-z0-9_\-]+", penalty_traj_dir.name):
        return penalty_traj_dir

    return None


def parse_trajectory_json(s: str) -> List[Dict[str, Any]]:
    if not s or pd.isna(s):
        return []
    try:
        return json.loads(s)
    except Exception:
        # sometimes the JSON is quoted twice; try to unquote
        try:
            return json.loads(s.strip('"'))
        except Exception:
            return []


def parse_release_point_json(s: str) -> Optional[Dict[str, Any]]:
    if not s or pd.isna(s):
        return None
    try:
        value = json.loads(s)
        return value if isinstance(value, dict) else None
    except Exception:
        try:
            value = json.loads(str(s).strip('"'))
            return value if isinstance(value, dict) else None
        except Exception:
            return None


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")




def build_shot_list(df: pd.DataFrame, shot_id: Optional[str], home: Optional[str], away: Optional[str]) -> pd.DataFrame:
    if shot_id:
        sel = df[df["id"] == str(shot_id)]
    elif home and away:
        sel = df[(df["home_team"] == home) & (df["away_team"] == away)]
    else:
        sel = df
    return sel


def detect_shot_direction(traj_points: List[Dict[str, Any]]) -> str:
    """Determine shot direction (positive or negative x) from trajectory."""
    xs = [float(p.get("x", 0)) for p in traj_points if p.get("x") is not None]
    if not xs:
        return "positive"
    avg_x = sum(xs) / len(xs)
    return "negative" if avg_x < 0 else "positive"


def plot_field_section(ax, direction: str):
    """Plot only the relevant goal side and penalty area.
    
    direction: "positive" (x=12..20) or "negative" (x=-20..-12)
    """
    if direction == "negative":
        goal_x = -20.0
        x_min, x_max = -20.4, -12
    else:
        goal_x = 20.0
        x_min, x_max = 12, 20.4
    
    # Draw faint grid lines on floor (penalty area only)
    for x in np.linspace(x_min, x_max, 5):
        ax.plot([x, x], [-2.5, 2.5], [0, 0], color="#ddd", linewidth=0.5)
    for y in np.linspace(-2.5, 2.5, 6):
        ax.plot([x_min, x_max], [y, y], [0, 0], color="#ddd", linewidth=0.5)

    # Goal frame
    # left post to right post (vertical lines)
    ax.plot([goal_x, goal_x], [-POST_Y, -POST_Y], [0, GOAL_HEIGHT], color="k", linewidth=2)
    ax.plot([goal_x, goal_x], [POST_Y, POST_Y], [0, GOAL_HEIGHT], color="k", linewidth=2)
    # crossbar
    ax.plot([goal_x, goal_x], [-POST_Y, POST_Y], [GOAL_HEIGHT, GOAL_HEIGHT], color="k", linewidth=2)

    # 7m line for this direction
    x_7m = goal_x + (7.0 if direction == "positive" else -7.0)
    y0, y1 = -3.0, 3.0
    z0 = 0.02
    corners = [
        (x_7m - 0.25, y0, z0),
        (x_7m + 0.25, y0, z0),
        (x_7m + 0.25, y1, z0),
        (x_7m - 0.25, y1, z0),
    ]
    verts = [corners]
    poly = Poly3DCollection(verts, facecolors=(0.2, 0.6, 1.0, 0.6), linewidths=0.5)
    ax.add_collection3d(poly)


def plot_shot(
    ax,
    traj_points: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    decimate: int,
    cmap: mcolors.Colormap,
    point_size: float,
    smooth_window: int,
    goalkeeper_points: List[Dict[str, Any]],
):
    pts = traj_points[::decimate]
    if not pts:
        return
    xs = np.array([float(p.get("x", 0)) for p in pts])
    ys = np.array([float(p.get("y", 0)) for p in pts])
    zs = np.array([float(p.get("z", 0)) for p in pts])
    vs = np.array([float(p.get("v", float("nan"))) if p.get("v") is not None else float("nan") for p in pts])
    accs = np.array([float(p.get("a", float("nan"))) if p.get("a") is not None else float("nan") for p in pts])

    n = len(xs)
    colors = cmap(np.linspace(0, 1, n))

    ax.scatter(xs, ys, zs, c=colors, s=point_size, depthshade=True)

    if smooth_window > 1 and len(xs) > 1:
        xs_s = smooth_series(xs, smooth_window)
        ys_s = smooth_series(ys, smooth_window)
        zs_s = smooth_series(zs, smooth_window)
        ax.plot(xs_s, ys_s, zs_s, color="#1f4e79", linewidth=2.2, alpha=0.9, label=f"ball_smoothed_{smooth_window}")

    # special points
    start_idx = 0
    max_v_idx = int(np.nanargmax(vs)) if np.any(~np.isnan(vs)) else 0
    max_a_idx = int(np.nanargmax(accs)) if np.any(~np.isnan(accs)) else 0

    # plot markers
    ax.scatter([xs[start_idx]], [ys[start_idx]], [zs[start_idx]], color="lime", s=80, marker="*", label="start")
    ax.scatter([xs[max_v_idx]], [ys[max_v_idx]], [zs[max_v_idx]], color="orange", s=80, marker="D", label="max_v")
    ax.scatter([xs[max_a_idx]], [ys[max_a_idx]], [zs[max_a_idx]], color="purple", s=80, marker="s", label="max_a")

    # release point: parse directly from release_point_json
    release_point = parse_release_point_json(metadata.get("release_point_json", ""))
    if release_point:
        rx = release_point.get("x")
        ry = release_point.get("y")
        rz = release_point.get("z")
        if all(v is not None for v in (rx, ry, rz)):
            ax.scatter([rx], [ry], [rz], color="yellow", s=80, marker="^", label="release")

    if goalkeeper_points:
        gxs = np.array([float(p.get("x", 0)) for p in goalkeeper_points])
        gys = np.array([float(p.get("y", 0)) for p in goalkeeper_points])
        gzs = np.array([float(p.get("z", 0)) for p in goalkeeper_points])
        ax.plot(gxs, gys, gzs, color="black", linewidth=1.8, alpha=0.95, label="goalkeeper")

    # hit position: last point
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="black", s=90, marker="X", label="hit")


def format_release_point_label(metadata: Dict[str, Any]) -> str:
    shot_id = metadata.get("id", "unknown")
    release_point = parse_release_point_json(metadata.get("release_point_json", ""))
    if release_point:
        x = release_point.get("x")
        y = release_point.get("y")
        z = release_point.get("z")
        release_time = release_point.get("t_local") or metadata.get("release_time_local", "")
        coord_text = f"({x:.2f}, {y:.2f}, {z:.2f})" if all(v is not None for v in (x, y, z)) else "unknown"
        return f"Shot id: {shot_id}\nRelease point: {coord_text}\nRelease time: {release_time}"

    release_time = metadata.get("release_time_local", "")
    return f"Shot id: {shot_id}\nRelease point: unavailable\nRelease time: {release_time}"


def main():
    parser = argparse.ArgumentParser(description="3D plotter for penalty trajectories")
    parser.add_argument("--input", default="penalty_trajectories.csv")
    parser.add_argument("--shot-id", default=None)
    parser.add_argument("--home-team", default=None)
    parser.add_argument("--away-team", default=None)
    parser.add_argument("--output-dir", default="out/penalty_plots", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively (opens matplotlib windows)")
    parser.add_argument("--only-interactive", action="store_true", help="Show the plot interactively without saving an image")
    parser.add_argument("--decimate", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--azim", type=float, default=45.0, help="azimuth for view_init (45° default for x-y plane diagonal view)")
    parser.add_argument("--elev", type=float, default=30.0, help="elevation for view_init")
    parser.add_argument("--random", type=int, default=None, help="Plot N random penalties")
    parser.add_argument("--smooth-window", type=int, default=1, help="Apply moving-average smoothing to the ball trajectory line")
    args = parser.parse_args()

    # Set matplotlib backend based on whether we're showing interactively
    if args.show or args.only_interactive:
        # Try to use an interactive backend for 3D visualization
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                # Fallback to a basic interactive backend
                matplotlib.use("TkAgg")
    else:
        # Use non-interactive backend for headless servers
        matplotlib.use("Agg")

    # Create output directory only when plots will be saved.
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if output_dir == Path("penalty_plots"):
        latest_run_dir = find_latest_run_dir(input_path)
        if latest_run_dir is not None:
            output_dir = latest_run_dir / "penalty_plots"
    if not args.only_interactive:
        output_dir.mkdir(exist_ok=True)

    df = load_trajectories(Path(args.input))
    sel = build_shot_list(df, args.shot_id, args.home_team, args.away_team)

    # Apply random sampling if requested
    if args.random is not None:
        if len(sel) > args.random:
            sel = sel.sample(n=args.random, random_state=None)
            print(f"Randomly selected {len(sel)} penalties from {len(build_shot_list(df, args.shot_id, args.home_team, args.away_team))} matches")

    if sel.empty:
        print("No matching shots found. Exiting.")
        return

    cmap = mcolors.LinearSegmentedColormap.from_list("gr2rd", ["#00FF00", "#FF0000"])

    for i, row in sel.reset_index(drop=True).iterrows():
        traj = parse_trajectory_json(row.get("trajectory_json", ""))
        if not traj:
            continue

        meta = row.to_dict()
        shot_id = row.get("id", f"shot_{i}")
        successful = str(row.get("success", "")).strip() == "1"

        # Determine shot direction
        direction = detect_shot_direction(traj)
        goalkeeper_points = parse_trajectory_json(row.get("goalkeeper_trajectory_json", ""))
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        plot_field_section(ax, direction)
        plot_shot(ax, traj, meta, max(1, args.decimate), cmap, args.point_size, max(1, args.smooth_window), goalkeeper_points)

        fig.text(
            0.985,
            0.965,
            format_release_point_label(meta),
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.5", "boxstyle": "round,pad=0.35"},
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        # Set axis limits based on direction
        if direction == "negative":
            ax.set_xlim(-20.4, -12)
        else:
            ax.set_xlim(12, 20.4)
        
        ax.set_ylim(-2.5, 2.5)
        ax.set_zlim(0, 4)

        ax.view_init(elev=args.elev, azim=args.azim)

        plt.legend(loc="upper left")
        title_prefix = "Penalty Shot"
        filename_suffix = ""
        if not successful:
            title_prefix = "Unsuccessful Penalty Shot"
            filename_suffix = "_unsuccessful"
        fig.suptitle(f"{title_prefix} ID: {shot_id}", fontsize=14, fontweight="bold")

        # Save plot with shot id as filename unless running interactive-only
        if not args.only_interactive:
            output_file = output_dir / f"{shot_id}{filename_suffix}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {output_file}")
        
        # Show interactively if requested
        if args.show or args.only_interactive:
            print(f"Displaying shot {shot_id} interactively. Use mouse to rotate, zoom with scroll wheel.")
            plt.show()
        
        plt.close(fig)


if __name__ == "__main__":
    main()
