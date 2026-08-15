from __future__ import annotations

import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gc
import os
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

# Module-level logger. Configured in main(); defaults to WARNING so library
# use does not spam stdout unless the caller opts in.
logger = logging.getLogger("release_detector_trajectory_based")

from ball_trajectory import BallPoint, serialize_point, serialize_trajectory
from fixture_resolution import build_edge_case_mappings, build_fixture_index, resolve_fixture_file
from penalty_time_utils import canonical_team_name, parse_penalty_local_time, parse_position_local_time, try_float, try_int


@dataclass
class ReleaseCandidate:
    idx: int
    score: float
    x: float
    z: float
    speed: float
    accel: float
    direction: Optional[float]


class FixtureCache:
    def __init__(self) -> None:
        # Simple single-entry cache: fixtures are processed in contiguous blocks
        # so we only keep the currently-loaded fixture in memory. This avoids
        # accumulating many large files.
        self._current_path: Optional[Path] = None
        self._current_points: Optional[List[BallPoint]] = None

    def get_points(self, path: Path, start_dt: Optional[datetime] = None, time_window_seconds: int = 5) -> List[BallPoint]:
        # Resolve path to a stable absolute path for comparisons and caching.
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        # If requested path is already loaded, return cached points.
        if self._current_path == resolved and self._current_points is not None and start_dt is None:
            return self._current_points

        # Load via pandas-backed loader (memory efficient for column selection)
        pts = _build_ball_points_from_file(path, start_dt=start_dt, time_window_seconds=time_window_seconds)
        # Replace current cache entry
        # Store resolved path in cache
        self._current_path = resolved
        self._current_points = pts
        # Trigger GC to free previous large DataFrames if any
        gc.collect()
        return pts

    def prefetch(self, path: Path, start_dt: Optional[datetime] = None, time_window_seconds: int = 5) -> None:
        try:
            _ = self.get_points(path, start_dt=start_dt, time_window_seconds=time_window_seconds)
        except Exception:
            pass


def _coerce_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_id(value: Any) -> str:
    text = _coerce_text(value)
    if not text:
        return ""
    return text.replace(".0", "")


def _build_ball_points_from_rows(rows: Sequence[Dict[str, str]], start_dt: Optional[datetime] = None) -> List[BallPoint]:
    points: List[BallPoint] = []
    for row in rows:
        if (row.get("group name") or "").strip() != "Ball":
            continue
        local_dt = parse_position_local_time(_coerce_text(row.get("formatted local time")))
        x = try_float(row.get("x in m", ""))
        y = try_float(row.get("y in m", ""))
        z = try_float(row.get("z in m", ""))
        if local_dt is None or x is None or y is None or z is None:
            continue
        if start_dt is not None and local_dt < start_dt - timedelta(seconds=5):
            continue
        if start_dt is not None and local_dt > start_dt + timedelta(seconds=5):
            continue
        points.append(
            BallPoint(
                local_dt=local_dt,
                ts_ms=try_int(row.get("ts in ms", "")),
                x=x,
                y=y,
                z=z,
                speed=try_float(row.get("speed in m/s", "")) or float("nan"),
                accel=try_float(row.get("acceleration in m/s2", "")) or float("nan"),
                direction=try_float(row.get("direction of movement in deg", "")),
            )
        )
    points.sort(key=lambda p: (p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


def _load_fixture_rows(positions_file: Path) -> List[Dict[str, str]]:
    with positions_file.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _build_ball_points_from_file(positions_file: Path, start_dt: Optional[datetime] = None, time_window_seconds: int = 5) -> List[BallPoint]:
    # Try a pandas-backed load for performance and column selection. If pandas
    # fails for any reason, fall back to the csv-based streaming loader.
    usecols = [
        "group name",
        "formatted local time",
        "x in m",
        "y in m",
        "z in m",
        "ts in ms",
        "speed in m/s",
        "acceleration in m/s2",
        "direction of movement in deg",
    ]
    try:
        df = pd.read_csv(positions_file, delimiter=";", usecols=usecols, dtype=str, low_memory=True)
    except Exception:
        return _build_ball_points_from_rows(_load_fixture_rows(positions_file), start_dt=start_dt)

    # Keep only ball rows
    if "group name" in df.columns:
        df = df[df["group name"].str.strip() == "Ball"]
    if df.empty:
        return []

    # Parse datetime strings into python datetimes using existing parser
    df["local_dt"] = df["formatted local time"].map(lambda s: parse_position_local_time(_coerce_text(s)))

    # Convert numeric columns
    for col in ["x in m", "y in m", "z in m", "ts in ms", "speed in m/s", "acceleration in m/s2", "direction of movement in deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    if "local_dt" not in df.columns:
        return []
    df = df[df["local_dt"].notna() & df["x in m"].notna() & df["y in m"].notna() & df["z in m"].notna()]
    if df.empty:
        return []

    if start_dt is not None:
        before = start_dt - timedelta(seconds=time_window_seconds)
        after = start_dt + timedelta(seconds=time_window_seconds)
        df = df[(df["local_dt"] >= before) & (df["local_dt"] <= after)]
        if df.empty:
            return []

    df = df.sort_values(["local_dt", "ts in ms"], ascending=[True, True])

    points: List[BallPoint] = []
    for _, row in df.iterrows():
        ts_val = row.get("ts in ms")
        ts_ms = int(ts_val) if not (pd.isna(ts_val)) else None
        speed = float(row["speed in m/s"]) if not pd.isna(row.get("speed in m/s")) else float("nan")
        accel = float(row["acceleration in m/s2"]) if not pd.isna(row.get("acceleration in m/s2")) else float("nan")
        direction = float(row["direction of movement in deg"]) if not pd.isna(row.get("direction of movement in deg")) else None
        points.append(
            BallPoint(
                local_dt=row["local_dt"],
                ts_ms=ts_ms,
                x=float(row["x in m"]),
                y=float(row["y in m"]),
                z=float(row["z in m"]),
                speed=speed,
                accel=accel,
                direction=direction,
            )
        )

    return points


def _filter_points_for_penalty(rows: Sequence[Dict[str, str]], start_dt: Optional[datetime] = None) -> List[BallPoint]:
    # Backwards-compatible wrapper: if the caller passes a Path-like object,
    # allow streaming directly from file. Otherwise assume pre-read rows.
    if isinstance(rows, (str, Path)):
        return _build_ball_points_from_file(Path(rows), start_dt=start_dt)
    return _build_ball_points_from_rows(rows, start_dt=start_dt)


def _find_window(points: Sequence[BallPoint], start_dt: datetime) -> Tuple[int, int]:
    start_idx = None
    for idx, point in enumerate(points):
        if start_idx is None and point.local_dt >= start_dt - timedelta(seconds=1):
            start_idx = idx
            break
    if start_idx is None:
        start_idx = 0

    end_idx = None
    for idx in range(start_idx, len(points)):
        if abs(points[idx].x) > 20.0:
            end_idx = min(len(points) - 1, idx + 5)
            break
    if end_idx is None:
        end_idx = len(points) - 1
    return start_idx, end_idx


def _estimate_motion_energy(points: Sequence[BallPoint], start_idx: int, end_idx: int) -> Tuple[float, float]:
    if start_idx >= end_idx:
        return 0.0, 0.0
    xs = [p.x for p in points[start_idx : end_idx + 1]]
    zs = [p.z for p in points[start_idx : end_idx + 1]]
    dx = xs[-1] - xs[0]
    dz = zs[-1] - zs[0]
    return abs(dx), abs(dz)


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Optional[List[float]]:
    size = len(matrix)
    aug = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda r: abs(aug[r][pivot]))
        if abs(aug[pivot_row][pivot]) < 1e-9:
            return None
        aug[pivot], aug[pivot_row] = aug[pivot_row], aug[pivot]
        for row in range(pivot + 1, size):
            factor = aug[row][pivot] / aug[pivot][pivot]
            for col in range(pivot, size + 1):
                aug[row][col] -= factor * aug[pivot][col]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = aug[row][size]
        for col in range(row + 1, size):
            total -= aug[row][col] * solution[col]
        if abs(aug[row][row]) < 1e-9:
            return None
        solution[row] = total / aug[row][row]
    return solution


def _fit_poly(values: Sequence[float], degree: int) -> Tuple[Optional[List[float]], Optional[float]]:
    values_list = [float(v) for v in values]
    n = len(values_list)
    if n < degree + 1:
        return None, None
    t = [i / 20.0 for i in range(n)]
    if degree == 1:
        design = [[1.0, t_i] for t_i in t]
    elif degree == 2:
        design = [[1.0, t_i, t_i * t_i] for t_i in t]
    else:
        return None, None
    rhs = values_list
    coeffs = _solve_linear_system(
        [[sum(design[i][j] * design[i][k] for i in range(n)) for k in range(degree + 1)] for j in range(degree + 1)],
        [sum(design[i][j] * rhs[i] for i in range(n)) for j in range(degree + 1)],
    )
    if coeffs is None:
        return None, None
    preds = [sum(coeffs[j] * design[i][j] for j in range(degree + 1)) for i in range(n)]
    rmse = math.sqrt(sum((values_list[i] - preds[i]) ** 2 for i in range(n)) / max(n, 1))
    return coeffs, rmse


def _normalize_rmse(rmse: float, values: Sequence[float]) -> float:
    if not values:
        return 1.0
    span = max(abs(max(values, default=0.0)), abs(min(values, default=0.0)), 1.0)
    return rmse / max(span, 1e-3)


def _fit_reference_line(points: Sequence[BallPoint]) -> Optional[Tuple[float, float, float]]:
    reference_points = [p for p in points if 14.0 < abs(p.x) < 16.0]
    if len(reference_points) < 3:
        return None

    reference_points = reference_points[:5]
    xs = [p.x for p in reference_points]
    ys = [p.y for p in reference_points]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in residuals) / max(n, 1))
    tolerance = max(0.07, 2.5 * rmse + 0.03)
    return slope, intercept, tolerance


def _has_strong_direction_change(points: Sequence[BallPoint], idx: int, candidate_idx: int, reference_line: Optional[Tuple[float, float, float]] = None) -> bool:
    if idx <= candidate_idx + 1:
        return False

    point = points[idx]
    if abs(point.x) >= 16.0:
        return False

    prev_prev = points[idx - 2]
    prev = points[idx - 1]
    curr = points[idx]

    prev_vel = prev.x - prev_prev.x
    curr_vel = curr.x - prev.x
    if prev_vel * curr_vel < 0.0 and abs(prev_vel) > 0.35 and abs(curr_vel) > 0.35:
        if idx + 1 < len(points):
            next_vel = points[idx + 1].x - curr.x
            if abs(next_vel) > 0.35 and next_vel * curr_vel > 0.0:
                return True
        return True

    if reference_line is not None:
        slope, intercept, tolerance = reference_line
        residual = point.y - (slope * point.x + intercept)
        if abs(residual) > tolerance:
            return True

    return False


def _rotate_180_z(points: Sequence[BallPoint]) -> List[BallPoint]:
    """Rotate points by 180 degrees around the Z axis.

    Applies ``newX = -oldX``, ``newY = -oldY``, ``newZ = oldZ`` to every point.
    Also rotates the ``direction`` heading by 180° so the direction stays
    consistent with the rotated coordinate frame (e.g. a throw toward -x
    becomes a throw toward +x, so 180° becomes 0°).
    This maps a throw performed on the left side of the field (-x) onto the
    right side (+x) while preserving right-handedness.
    """
    rotated: List[BallPoint] = []
    for point in points:
        rotated.append(
            BallPoint(
                local_dt=point.local_dt,
                ts_ms=point.ts_ms,
                x=-point.x,
                y=-point.y,
                z=point.z,
                speed=point.speed,
                accel=point.accel,
                direction=(point.direction + 180.0) % 360.0 if point.direction is not None else None,
            )
        )
    return rotated


def _is_left_side(points: Sequence[BallPoint]) -> bool:
    """Return True if the throw is performed on the left side of the field (-x).

    Determines the side from the median x-coordinate of the points near the
    release area (|x| < 14 m, i.e. before the goal line).
    """
    near_points = [p.x for p in points if abs(p.x) > 12.0]
    if not near_points:
        return False
    median_x = float(np.median(near_points))
    return median_x < 0.0


def _has_deflection(points: Sequence[BallPoint], start_idx: int, end_idx: int) -> bool:
    """Detect whether the trajectory contains a deflection (e.g. by the goalkeeper).

    Uses the same strong-direction-change heuristic as the release detector:
    a deflection is present if any point after the release area shows a strong
    direction change or deviates from the fitted reference line.
    """
    if len(points) < 5:
        return False

    # Find a candidate release index near the 13 m gate.
    release_idx = select_release_index(points, start_idx, end_idx)
    if release_idx is None:
        return False

    # Fit a reference line from the post-release points and scan for strong
    # direction changes or reference-line deviations. Ground hits and goal-line
    # crossings are NOT deflections.
    reference_line = _fit_reference_line(points[release_idx + 1 : end_idx + 1])
    for idx in range(release_idx + 2, end_idx + 1):
        if abs(points[idx].x) > 20.0:
            break
        if _is_ground_contact(points, idx):
            break
        if _has_strong_direction_change(points, idx, release_idx, reference_line=reference_line):
            return True

    return False


def _build_valid_post_segment(points: Sequence[BallPoint], candidate_idx: int, end_idx: int) -> Optional[Tuple[int, int]]:
    if candidate_idx >= end_idx:
        return None

    reference_line = _fit_reference_line(points[candidate_idx + 1 : end_idx + 1])
    segment_end = end_idx
    for idx in range(candidate_idx + 1, end_idx + 1):
        if abs(points[idx].x) > 20.0:
            segment_end = idx - 1
            break
        # TODO: I doubt this works, because points are scarse, 2 points do not necessarily are both below 0.18
        if _is_ground_contact(points, idx):
            segment_end = idx - 1
            break
        if _has_strong_direction_change(points, idx, candidate_idx, reference_line=reference_line):
            segment_end = idx - 1
            break
    if segment_end < candidate_idx + 4:
        return None
    return candidate_idx, segment_end

def _is_ground_contact(
    points: Sequence[BallPoint],
    idx: int,
    ground_z: float = 0.0,
    tolerance: float = 0.2,
    window: int = 3,
) -> bool:
    if idx + window >= len(points):
        return False

    window_points = points[idx : idx + window + 1]
    zs = [p.z for p in window_points]

    # Ball is close to the ground.
    near_ground = min(zs) <= ground_z + tolerance

    # It is approaching the ground rather than moving away from it.
    descending = zs[0] > zs[-1]

    return near_ground and descending


def _projectile_fit_score(points: Sequence[BallPoint], candidate_idx: int, start_idx: int, end_idx: int) -> Tuple[float, Dict[str, Any]]:
    segment = _build_valid_post_segment(points, candidate_idx, end_idx)
    if segment is None:
        return -1e9, {"postfit": 0.0, "forward": 0.0, "gravity": 0.0, "segment_length": 0}

    candidate_idx, segment_end = segment
    post_points = points[candidate_idx : segment_end + 1]
    if len(post_points) < 5:
        return -1e9, {"postfit": 0.0, "forward": 0.0, "gravity": 0.0, "segment_length": len(post_points)}

    release_point = points[candidate_idx]
    goal_sign = 1.0 if (post_points[-1].x - release_point.x) >= 0.0 else -1.0
    post_long = [(p.x - release_point.x) * goal_sign for p in post_points]
    post_lat = [p.y - release_point.y for p in post_points]
    post_vert = [p.z - release_point.z for p in post_points]

    fit_values: List[float] = []
    for values, degree in ((post_long, 1), (post_lat, 1), (post_vert, 2)):
        _, rmse = _fit_poly(values, degree)
        if rmse is None:
            continue
        fit_values.append(_normalize_rmse(rmse, values))

    if not fit_values:
        return -1e9, {"postfit": 0.0, "forward": 0.0, "gravity": 0.0, "segment_length": len(post_points)}

    postfit = float(sum(fit_values) / len(fit_values))
    postfit_component = max(0.0, 1.0 - postfit)

    forward = 0.0
    if len(post_points) >= 2:
        forward_gain = (post_points[-1].x - release_point.x) * goal_sign
        forward = max(0.0, min(1.0, forward_gain / max(abs(forward_gain) + 1.0, 1.0)))

    coeffs, _ = _fit_poly(post_vert, 2)
    gravity_score = 0.0
    if coeffs is not None:
        accel_fit = 2.0 * coeffs[2]
        gravity_score = max(0.0, 1.0 - min(1.0, abs(accel_fit - 9.81) / (2.0 * 9.81)))

    segment_persistence = 1.0 if len(post_points) >= 7 else 0.55 + 0.075 * (len(post_points) - 5)
    score = 0.7 * postfit_component + 0.2 * forward + 0.1 * gravity_score * segment_persistence
    details = {
        "postfit": float(postfit_component),
        "forward": float(forward),
        "gravity": float(gravity_score),
        "segment_length": len(post_points),
        "persistence": float(segment_persistence),
    }
    return score, details


def _candidate_score(points: Sequence[BallPoint], idx: int, start_idx: int, end_idx: int) -> ReleaseCandidate:
    if idx >= end_idx:
        return ReleaseCandidate(idx=idx, score=-1e9, x=points[idx].x, z=points[idx].z, speed=points[idx].speed, accel=points[idx].accel, direction=points[idx].direction)

    motion_score, details = _projectile_fit_score(points, idx, start_idx, end_idx)
    score = motion_score
    return ReleaseCandidate(idx=idx, score=score, x=points[idx].x, z=points[idx].z, speed=points[idx].speed, accel=points[idx].accel, direction=points[idx].direction)


def select_release_index(points: Sequence[BallPoint], start_idx: int, end_idx: int) -> Optional[int]:
    """Return a simple release index by scoring candidate points with a projectile fit.

    The detector examines points between start_idx and end_idx and prefers a
    candidate that is near the 13 m x-gate, has a strong acceleration increase,
    and is followed by a projectile-like segment.
    """
    if not points:
        return None
    start_idx = max(0, min(start_idx, len(points) - 1))
    end_idx = max(start_idx, min(end_idx, len(points) - 1))

    candidates: List[ReleaseCandidate] = []
    for idx in range(start_idx, end_idx + 1):
        if abs(points[idx].x) > 14:
            continue
        if idx < start_idx + 2:
            continue
        cand = _candidate_score(points, idx, start_idx, end_idx)
        candidates.append(cand)

    if not candidates:
        return None

    threshold = 0.9
    strong_candidates = [cand for cand in candidates if cand.score >= threshold]
    if strong_candidates:
        return min(strong_candidates, key=lambda c: c.idx).idx

    best = max(candidates, key=lambda c: c.score)
    return best.idx


def compute_velocity_acceleration_from_points(points: Sequence[BallPoint]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Compute per-point speed (m/s) and acceleration magnitude (m/s^2) purely
    from the trajectory's positions using finite differences over time.

    Unlike the fixture CSV's ``speed in m/s`` / ``acceleration in m/s2``
    columns, these values are derived only from the sampled positions (x, y, z)
    and local timestamps of the ball points.

    Speed is the magnitude of the 3D velocity vector: central differences over
    the two-neighbour span for interior points, forward/backward differences at
    the boundaries. Acceleration is the magnitude of the central difference of
    consecutive velocity vectors; it is None for the first and last point where
    no two-sided neighbour support exists.
    """
    n = len(points)
    speeds: List[Optional[float]] = [None] * n
    accels: List[Optional[float]] = [None] * n
    if n < 2:
        return speeds, accels

    # 3D velocity vectors aligned to each point.
    velocities: List[Optional[Tuple[float, float, float]]] = [None] * n

    for i in range(n):
        if i == 0:
            a_idx, b_idx = 0, 1
        elif i == n - 1:
            a_idx, b_idx = n - 2, n - 1
        else:
            a_idx, b_idx = i - 1, i + 1

        dt = (points[b_idx].local_dt - points[a_idx].local_dt).total_seconds()
        if dt <= 0:
            continue
        vx = (points[b_idx].x - points[a_idx].x) / dt
        vy = (points[b_idx].y - points[a_idx].y) / dt
        vz = (points[b_idx].z - points[a_idx].z) / dt
        velocities[i] = (vx, vy, vz)
        speeds[i] = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Acceleration: central difference of the velocity vectors.
    for i in range(1, n - 1):
        v_prev = velocities[i - 1]
        v_next = velocities[i + 1]
        if v_prev is None or v_next is None:
            continue
        dt = (points[i + 1].local_dt - points[i - 1].local_dt).total_seconds()
        if dt <= 0:
            continue
        ax = (v_next[0] - v_prev[0]) / dt
        ay = (v_next[1] - v_prev[1]) / dt
        az = (v_next[2] - v_prev[2]) / dt
        accels[i] = math.sqrt(ax * ax + ay * ay + az * az)

    return speeds, accels


def detect_simple_release_point(
    penalty_row: Dict[str, str],
    positions_dir: Path,
    fixture_cache: Optional[FixtureCache] = None,
    fixture_path: Optional[Path] = None,
    points: Optional[List[BallPoint]] = None,
    normalize_side: bool = False,
    include_deflections: bool = True,
) -> Dict[str, Any]:
    player_id = _coerce_text(penalty_row.get("player_id"))
    goalkeeper_id = _coerce_text(penalty_row.get("goalkeeper_id"))

    # Resolve fixture once if not provided
    if fixture_path is None:
        fixture_index, _ = build_fixture_index(positions_dir)
        fixture_path, _ = resolve_fixture_file(penalty_row, fixture_index, build_edge_case_mappings())
    if fixture_path is None:
        raise FileNotFoundError(f"Could not resolve fixture for {penalty_row.get('id', 'unknown')}")

    raw_start = _coerce_text(penalty_row.get("timestamp_local_timezone"))
    start_dt = parse_penalty_local_time(raw_start)

    # If caller already supplied points, use them directly (avoids reloading).
    if points is None:
        # Prefer cached points when available; otherwise stream from file and only
        # keep points in the time window around the reported start time.
        if fixture_cache is not None:
            try:
                pts = fixture_cache.get_points(fixture_path, start_dt=start_dt, time_window_seconds=5)
            except Exception:
                pts = _build_ball_points_from_file(fixture_path, start_dt=start_dt)
            points = pts
        else:
            points = _build_ball_points_from_file(fixture_path, start_dt=start_dt)
    if start_dt is None:
        start_dt = points[0].local_dt if points else datetime.now()
    if not points:
        raise ValueError(f"No ball points found for penalty {penalty_row.get('id', 'unknown')}")

    # Optionally transform throws performed on the left side of the field (-x)
    # onto the right side (+x) via a 180-degree rotation around the Z axis.
    was_left_side = False
    if normalize_side and _is_left_side(points):
        points = _rotate_180_z(points)
        was_left_side = True

    # Use the reported start time as the anchor and extend the window until the
    # ball clears the goal line, then include a few extra points for context.
    start_idx, end_idx = _find_window(points, start_dt)
    if start_idx >= end_idx:
        start_idx = max(0, end_idx - 2)

    # Optionally skip penalties with a deflection (e.g. by the goalkeeper).
    if not include_deflections and _has_deflection(points, start_idx, end_idx):
        raise ValueError(f"Penalty {penalty_row.get('id', 'unknown')} has a deflection and is excluded")

    release_idx = select_release_index(points, start_idx, end_idx)
    if release_idx is None:
        release_idx = max(start_idx, min(end_idx, start_idx + 2))

    release_point = points[release_idx]
    window_points = points[start_idx:end_idx + 1]
    # Keep both index spaces explicit:
    # - release_idx_global: index in the full loaded points list
    # - release_idx: index relative to trajectory/window_points
    release_idx_global = release_idx
    release_idx_local = max(0, release_idx_global - start_idx)
    trajectory = [
        {
            "t_local": p.local_dt.isoformat(timespec="milliseconds"),
            "ts_ms": p.ts_ms,
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "v": None if math.isnan(p.speed) else p.speed,
            "a": None if math.isnan(p.accel) else p.accel,
            "dir": p.direction,
        }
        for p in window_points
    ]
    # Max velocity/acceleration over the throw trajectory (w.r.t. the ball points).
    finite_vs = [p["v"] for p in trajectory if p["v"] is not None]
    finite_as = [p["a"] for p in trajectory if p["a"] is not None]
    max_v = max(finite_vs) if finite_vs else None
    max_a = max(finite_as) if finite_as else None
    # Velocity/acceleration derived from the positions themselves (finite
    # differences over time), independent of the fixture CSV's speed/accel columns.
    computed_speeds, computed_accels = compute_velocity_acceleration_from_points(window_points)
    return {
        "fixture_file": str(fixture_path),
        "trajectory": trajectory,
        "max_v": max_v,
        "max_a": max_a,
        "velocity_per_point": computed_speeds,
        "acceleration_per_point": computed_accels,
        "release_idx": release_idx_local,
        "release_idx_global": release_idx_global,
        "release_point": {
            "t_local": release_point.local_dt.isoformat(timespec="milliseconds"),
            "ts_ms": release_point.ts_ms,
            "x": release_point.x,
            "y": release_point.y,
            "z": release_point.z,
            "v": None if math.isnan(release_point.speed) else release_point.speed,
            "a": None if math.isnan(release_point.accel) else release_point.accel,
            "dir": release_point.direction,
        },
        "release_speed": None if math.isnan(release_point.speed) else release_point.speed,
        "release_accel": None if math.isnan(release_point.accel) else release_point.accel,
        "release_direction": release_point.direction,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "trajectory_points": len(points[start_idx:end_idx + 1]),
        "normalized_side": was_left_side,
    }


def process_penalties_csv(
    penalties_csv: Path,
    positions_dir: Path,
    output_csv: Path,
    errors_csv: Optional[Path] = None,
    include_unsuccessful: bool = False,
    penalty_id: Optional[str] = None,
    normalize_side: bool = False,
    include_deflections: bool = True,
) -> Path:
    with penalties_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    if penalty_id is not None:
        rows = [row for row in rows if _coerce_text(row.get("id")) == penalty_id]

    if not include_unsuccessful:
        rows = [row for row in rows if _coerce_text(row.get("success")) == "1"]

    fixture_cache = FixtureCache()
    records: List[Dict[str, str]] = []
    # Build fixture index once to avoid repeated filesystem scans
    fixture_index, _ = build_fixture_index(positions_dir)
    logger.info("Processing %d penalty rows from %s", len(rows), penalties_csv)
    for row_idx, row in enumerate(rows, start=1):
        try:
            fixture_path, _ = resolve_fixture_file(row, fixture_index, build_edge_case_mappings())
            if fixture_path is None:
                logger.warning(
                    "Row %d (id=%s): could not resolve fixture for %s vs %s",
                    row_idx,
                    _coerce_text(row.get("id")),
                    _coerce_text(row.get("home_team")),
                    _coerce_text(row.get("away_team")),
                )
                result = {"error": f"Could not resolve fixture for {_coerce_text(row.get('id'))}"}
            else:
                logger.info(
                    "Row %d/%d (id=%s): processing fixture %s",
                    row_idx,
                    len(rows),
                    _coerce_text(row.get("id")),
                    fixture_path.name,
                )
                # NOTE: Do NOT preload points once per fixture and reuse them
                # across rows. Each penalty has its own start time, so the ball
                # points must be loaded/filtered for that penalty's own time
                # window. detect_simple_release_point handles this correctly
                # when points=None (it loads via the cache with the row's
                # start_dt). Reusing a single fixture-level window caused the
                # same (first penalty's) points to be applied to every later
                # penalty in the same fixture.
                result = detect_simple_release_point(
                    penalty_row=row,
                    positions_dir=positions_dir,
                    fixture_cache=fixture_cache,
                    fixture_path=fixture_path,
                    normalize_side=normalize_side,
                    include_deflections=include_deflections,
                )
                if result.get("release_idx") is not None:
                    logger.info(
                        "Row %d (id=%s): release detected at idx=%s time=%s",
                        row_idx,
                        _coerce_text(row.get("id")),
                        result.get("release_idx"),
                        result.get("release_point", {}).get("t_local", ""),
                    )
                else:
                    logger.info(
                        "Row %d (id=%s): release detection returned no result",
                        row_idx,
                        _coerce_text(row.get("id")),
                    )
        except Exception as exc:
            logger.error(
                "Row %d (id=%s): error during processing: %s",
                row_idx,
                _coerce_text(row.get("id")),
                exc,
            )
            result = {"error": str(exc)}
        record = {
            "id": _coerce_text(row.get("id")),
            "home_team": _coerce_text(row.get("home_team")),
            "away_team": _coerce_text(row.get("away_team")),
            "player_id": _coerce_text(row.get("player_id")),
            "goalkeeper_id": _coerce_text(row.get("goalkeeper_id")),
            "success": _coerce_text(row.get("success")),
            "distance": _coerce_text(row.get("distance")),
            "timestamp_local_timezone": _coerce_text(row.get("timestamp_local_timezone")),
            "fixture_file": result.get("fixture_file", ""),
            "release_point_json": json.dumps(result.get("release_point", {}), ensure_ascii=False, separators=(",", ":")),
            "release_time_local": result.get("release_point", {}).get("t_local", ""),
            "release_speed": result.get("release_speed"),
            "release_accel": result.get("release_accel"),
            "release_direction": result.get("release_direction"),
            "max_v": result.get("max_v"),
            "max_a": result.get("max_a"),
            "velocity_per_point": result.get("velocity_per_point", []),
            "acceleration_per_point": result.get("acceleration_per_point", []),
            "trajectory_json": json.dumps(result.get("trajectory", []), ensure_ascii=False, separators=(",", ":")),
            "trajectory_point_count": result.get("trajectory_points", 0),
            "release_idx": result.get("release_idx"),
            "release_idx_global": result.get("release_idx_global"),
            "normalized_side": result.get("normalized_side", False),
            "error": result.get("error", ""),
        }
        records.append(record)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "home_team",
        "away_team",
        "player_id",
        "goalkeeper_id",
        "success",
        "distance",
        "timestamp_local_timezone",
        "fixture_file",
        "release_point_json",
        "release_time_local",
        "release_speed",
        "release_accel",
        "release_direction",
        "max_v",
        "max_a",
        "velocity_per_point",
        "acceleration_per_point",
        "trajectory_json",
        "trajectory_point_count",
        "release_idx",
        "release_idx_global",
        "normalized_side",
        "error",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(records)

    # Separate kinematics CSV in the same output directory: per-throw lists of
    # velocity (m/s) and acceleration (m/s^2) computed from the trajectory's
    # positions (finite differences), one row per penalty id.
    kinematics_csv = output_csv.parent / (output_csv.stem + "_kinematics" + output_csv.suffix)
    with kinematics_csv.open("w", encoding="utf-8", newline="") as handle:
        kin_writer = csv.DictWriter(
            handle,
            fieldnames=["id", "velocity_per_point", "acceleration_per_point"],
            delimiter=";",
        )
        kin_writer.writeheader()
        for record in records:
            kin_writer.writerow(
                {
                    "id": record["id"],
                    "velocity_per_point": json.dumps(
                        record.get("velocity_per_point", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "acceleration_per_point": json.dumps(
                        record.get("acceleration_per_point", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    logger.info("Kinematics CSV written to %s", kinematics_csv)

    errors = [r for r in records if r.get("error")]
    if errors and errors_csv is not None:
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        # write only the error rows for quick inspection
        with errors_csv.open("w", encoding="utf-8", newline="") as eh:
            err_writer = csv.DictWriter(eh, fieldnames=fieldnames, delimiter=";")
            err_writer.writeheader()
            err_writer.writerows(errors)

    logger.info("Processed %d rows -> %d records written to %s (%d with errors)",
                len(rows), len(records), output_csv, len(errors))
    return output_csv


def main() -> None:
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Detect simple point-of-release trajectories for penalties")
    parser.add_argument("--penalties", default=str(project_root / "penalties.csv"))
    parser.add_argument("--positions-dir", default=str(project_root / "games_position_files"))
    parser.add_argument("--output", default=str(project_root / "out" / "simple_penalty_trajectories.csv"))
    parser.add_argument("--include-unsuccessful", action="store_true")
    parser.add_argument("--penalty-id", default=None)
    parser.add_argument(
        "--normalize-side",
        action="store_true",
        help="Transform throws on the left side of the field (-x) onto the right side (+x) via a 180-degree Z rotation",
        default=True
    )
    parser.add_argument(
        "--exclude-deflections",
        action="store_true",
        help="Exclude penalties with a deflection (e.g. by the goalkeeper)",
    )
    args = parser.parse_args()

    # Configure logging so the user sees which fixture is currently being
    # processed. Intended for CLI use; library callers can configure the
    # "simple_release_detector" logger themselves.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create timestamped run directory under out/, matching other pipeline outputs
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "out" / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_csv = run_dir / Path(args.output).name
    errors_csv = run_dir / "simple_penalty_errors.csv"

    process_penalties_csv(
        Path(args.penalties),
        Path(args.positions_dir),
        output_csv,
        errors_csv=errors_csv,
        include_unsuccessful=args.include_unsuccessful,
        penalty_id=args.penalty_id,
        normalize_side=args.normalize_side,
        include_deflections=not args.exclude_deflections,
    )


if __name__ == "__main__":
    main()
