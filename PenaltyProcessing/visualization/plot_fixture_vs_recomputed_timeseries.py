#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def _parse_json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        try:
            parsed = json.loads(text.strip('"'))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            # Fallback for Python-literal lists like [None, 1.2, ...]
            # written by csv from a Python list repr.
            try:
                parsed = ast.literal_eval(text)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _trajectory_time_axis_seconds(trajectory: List[Dict[str, Any]]) -> Tuple[List[float], str]:
    if not trajectory:
        return [], "sample index"

    parsed_times: List[Optional[datetime]] = []
    for point in trajectory:
        t_local = point.get("t_local")
        if not t_local:
            parsed_times.append(None)
            continue
        try:
            parsed_times.append(datetime.fromisoformat(str(t_local)))
        except Exception:
            parsed_times.append(None)

    if not parsed_times or parsed_times[0] is None or any(t is None for t in parsed_times):
        return [float(i) for i in range(len(trajectory))], "sample index"

    t0 = parsed_times[0]
    assert t0 is not None
    return [float((t - t0).total_seconds()) for t in parsed_times if t is not None], "time since first point (s)"


def _extract_fixture_series(trajectory: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    return [_to_optional_float(point.get(key)) for point in trajectory]


def _normalize_length(values: List[Optional[float]], n: int) -> List[Optional[float]]:
    if len(values) == n:
        return values
    if len(values) > n:
        return values[:n]
    return values + [None] * (n - len(values))


def _valid_xy(x: List[float], y: List[Optional[float]]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for xv, yv in zip(x, y):
        if yv is None:
            continue
        if not math.isfinite(yv):
            continue
        xs.append(xv)
        ys.append(float(yv))
    return xs, ys


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _por_x_position(row: Dict[str, Any], trajectory: List[Dict[str, Any]], x: List[float]) -> Optional[float]:
    if not trajectory or not x:
        return None

    release_idx = _safe_int(row.get("release_idx"))
    if release_idx is not None and 0 <= release_idx < len(x):
        return x[release_idx]

    release_time = str(row.get("release_time_local", "")).strip()
    if release_time:
        release_time_iso = release_time.replace(" ", "T")
        for idx, point in enumerate(trajectory):
            t_local = str(point.get("t_local", "")).strip()
            if t_local == release_time_iso and idx < len(x):
                return x[idx]

    release_point_raw = row.get("release_point_json")
    if release_point_raw:
        try:
            release_point = json.loads(str(release_point_raw))
            release_point_t = str(release_point.get("t_local", "")).strip()
            if release_point_t:
                for idx, point in enumerate(trajectory):
                    t_local = str(point.get("t_local", "")).strip()
                    if t_local == release_point_t and idx < len(x):
                        return x[idx]
        except Exception:
            pass
    return None


def _goal_cross_x_position(trajectory: List[Dict[str, Any]], x: List[float]) -> Optional[float]:
    if not trajectory or not x:
        return None
    for idx, point in enumerate(trajectory):
        px = _to_optional_float(point.get("x"))
        if px is None:
            continue
        if abs(px) > 20.0 and idx < len(x):
            return x[idx]
    return None


def _plot_pair(
    x: List[float],
    x_label: str,
    fixture_values: List[Optional[float]],
    recomputed_values: List[Optional[float]],
    por_x: Optional[float],
    goal_cross_x: Optional[float],
    metric_label: str,
    title: str,
    output_file: Path,
) -> None:
    plt.figure(figsize=(11, 5.5))

    fx, fy = _valid_xy(x, fixture_values)
    cx, cy = _valid_xy(x, recomputed_values)
    if fx:
        plt.plot(fx, fy, label=f"{metric_label} fixture", linewidth=2.0)
    if cx:
        plt.plot(cx, cy, label=f"{metric_label} recomputed", linewidth=2.0)

    if por_x is not None:
        plt.axvline(por_x, color="black", linestyle="--", linewidth=1.3, alpha=0.8, label="PoR")
    if goal_cross_x is not None:
        plt.axvline(
            goal_cross_x,
            color="#c0392b",
            linestyle=":",
            linewidth=1.6,
            alpha=0.85,
            label="first |x| > 20",
        )

    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=160)
    plt.close()


def _row_plot_id(row: Dict[str, Any], fallback_idx: int) -> str:
    raw_id = str(row.get("id", "")).strip()
    return raw_id if raw_id else f"row_{fallback_idx}"


def plot_row(
    row: Dict[str, Any],
    row_idx: int,
    output_parent: Path,
    por_window_frames: Optional[int] = None,
) -> Tuple[Path, Path]:
    trajectory = _parse_json_list(row.get("trajectory_json"))
    if not trajectory:
        raise ValueError("trajectory_json is empty or invalid")

    x, x_label = _trajectory_time_axis_seconds(trajectory)
    n = len(trajectory)
    x = _normalize_length([float(v) for v in x], n)
    
    # Extract metrics before windowing
    fixture_v_full = _extract_fixture_series(trajectory, "v")
    fixture_a_full = _extract_fixture_series(trajectory, "a")
    
    recomputed_v_full = [_to_optional_float(v) for v in _parse_json_list(row.get("velocity_per_point"))]
    
    accel_source = row.get("tangential_acceleration_per_point")
    if accel_source in {None, "", "nan", "null"}:
        accel_source = row.get("acceleration_per_point")
    recomputed_a_full = [_to_optional_float(a) for a in _parse_json_list(accel_source)]

    # Window data around PoR if requested
    release_idx = _safe_int(row.get("release_idx"))
    start_idx = 0
    if por_window_frames is not None and release_idx is not None and release_idx >= 0 and release_idx < len(trajectory):
        start_idx = max(0, release_idx - por_window_frames)
        end_idx = min(len(trajectory), release_idx + por_window_frames + 1)
        trajectory = trajectory[start_idx:end_idx]
        x = x[start_idx:end_idx]
        fixture_v_full = fixture_v_full[start_idx:end_idx]
        fixture_a_full = fixture_a_full[start_idx:end_idx]
        recomputed_v_full = recomputed_v_full[start_idx:end_idx]
        recomputed_a_full = recomputed_a_full[start_idx:end_idx]
        release_idx = release_idx - start_idx
    
    n = len(trajectory)
    fixture_v = _normalize_length(fixture_v_full, n)
    fixture_a = _normalize_length(fixture_a_full, n)
    recomputed_v = _normalize_length(recomputed_v_full, n)
    recomputed_a = _normalize_length(recomputed_a_full, n)
    
    goal_cross_x = _goal_cross_x_position(trajectory, x)

    shot_id = _row_plot_id(row, row_idx)
    v_dir = output_parent / "v_fixture_vs_v_recomputed"
    a_dir = output_parent / "a_fixture_vs_a_recomputed"
    v_file = v_dir / f"{shot_id}.png"
    a_file = a_dir / f"{shot_id}.png"

    matchup = f"{row.get('home_team', '')} vs {row.get('away_team', '')}".strip()
    title_base = f"id={shot_id} | {matchup}" if matchup else f"id={shot_id}"

    # Adjust PoR x-position if windowing was applied
    if release_idx >= 0 and release_idx < len(x):
        por_x = x[release_idx]
    else:
        por_x = None
    
    # Recompute goal_cross_x for windowed trajectory
    goal_cross_x = _goal_cross_x_position(trajectory, x)

    _plot_pair(
        x=x,
        x_label=x_label,
        fixture_values=fixture_v,
        recomputed_values=recomputed_v,
        por_x=por_x,
        goal_cross_x=goal_cross_x,
        metric_label="velocity (m/s)",
        title=f"Fixture vs recomputed velocity | {title_base}",
        output_file=v_file,
    )
    _plot_pair(
        x=x,
        x_label=x_label,
        fixture_values=fixture_a,
        recomputed_values=recomputed_a,
        por_x=por_x,
        goal_cross_x=goal_cross_x,
        metric_label="signed tangential acceleration (m/s^2)",
        title=f"Fixture vs recomputed tangential acceleration | {title_base}",
        output_file=a_file,
    )
    return v_file, a_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fixture-reported vs recomputed velocity and acceleration "
            "time series from simple_penalty_trajectories.csv"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to simple_penalty_trajectories.csv",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Optional single penalty id to plot",
    )
    parser.add_argument(
        "--por-window-frames",
        type=int,
        default=None,
        help="Optional window size (frames on each side of PoR) to zoom into the release region. "
             "If not provided, plots the entire trajectory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = args.input.resolve()
    df = pd.read_csv(input_path, sep=";", dtype=str)

    if args.id is not None:
        rows = df[df["id"].astype(str).str.strip() == str(args.id).strip()]
    else:
        rows = df

    if rows.empty:
        print("No matching rows found.")
        return 1

    out_parent = input_path.parent
    generated = 0
    for i, row in enumerate(rows.to_dict(orient="records"), start=1):
        try:
            v_file, a_file = plot_row(
                row=row,
                row_idx=i,
                output_parent=out_parent,
                por_window_frames=args.por_window_frames,
            )
            generated += 1
            print(f"Saved: {v_file}")
            print(f"Saved: {a_file}")
        except Exception as exc:
            row_id = str(row.get("id", "")).strip() or f"row_{i}"
            print(f"Skipped id={row_id}: {exc}")

    if generated == 0:
        print("No plots were generated.")
        return 2

    print(f"Generated plots for {generated} row(s).")
    return 0


if __name__ == "__main__":
    matplotlib.use("Agg")
    raise SystemExit(main())
