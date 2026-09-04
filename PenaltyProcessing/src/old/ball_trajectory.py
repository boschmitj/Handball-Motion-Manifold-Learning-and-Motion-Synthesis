from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from penalty_time_utils import parse_position_local_time, try_float, try_int


@dataclass
class BallPoint:
    local_dt: datetime
    ts_ms: Optional[int]
    x: float
    y: float
    z: float
    speed: float
    accel: float
    direction: Optional[float]


@dataclass
class GoalkeeperPoint:
    local_dt: datetime
    ts_ms: Optional[int]
    x: float
    y: float
    z: float
    sensor_id: str


def load_ball_points(positions_file: Path) -> List[BallPoint]:
    """Load only Ball rows with required columns from one positions file."""
    points: List[BallPoint] = []

    with positions_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        required = {
            "formatted local time",
            "group name",
            "x in m",
            "y in m",
            "z in m",
            "speed in m/s",
            "acceleration in m/s2",
            "direction of movement in deg",
            "ts in ms",
        }
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {positions_file.name}: {missing}")

        for row in reader:
            if (row.get("group name") or "").strip() != "Ball":
                continue

            local_dt = parse_position_local_time(row.get("formatted local time", ""))
            x = try_float(row.get("x in m", ""))
            y = try_float(row.get("y in m", ""))
            z = try_float(row.get("z in m", ""))
            speed = try_float(row.get("speed in m/s", ""))
            accel = try_float(row.get("acceleration in m/s2", ""))
            direction = try_float(row.get("direction of movement in deg", ""))
            ts_ms = try_int(row.get("ts in ms", ""))

            if local_dt is None or x is None or y is None or z is None:
                continue

            points.append(
                BallPoint(
                    local_dt=local_dt,
                    ts_ms=ts_ms,
                    x=x,
                    y=y,
                    z=z,
                    speed=speed if speed is not None else float("nan"),
                    accel=accel if accel is not None else float("nan"),
                    direction=direction,
                )
            )

    points.sort(key=lambda p: (p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


def load_goalkeeper_candidates(positions_file: Path) -> List[GoalkeeperPoint]:
    """Load potential goalkeeper points from one positions file.

    We keep only points close to either goal area so downstream matching remains fast.
    """
    points: List[GoalkeeperPoint] = []

    with positions_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        required = {
            "formatted local time",
            "group name",
            "x in m",
            "y in m",
            "z in m",
            "ts in ms",
        }
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {positions_file.name}: {missing}")

        for row in reader:
            if (row.get("group name") or "").strip() == "Ball":
                continue

            local_dt = parse_position_local_time(row.get("formatted local time", ""))
            x = try_float(row.get("x in m", ""))
            y = try_float(row.get("y in m", ""))
            z = try_float(row.get("z in m", ""))
            ts_ms = try_int(row.get("ts in ms", ""))

            if local_dt is None or x is None or y is None or z is None:
                continue

            # Restrict to likely goalkeeper area near either goal.
            if not (16.0 <= abs(x) <= 20.0 and abs(y) <= 1.5):
                continue

            sensor_id = (
                (row.get("sensor id") or "").strip()
                or (row.get("mapped id") or "").strip()
                or (row.get("full name") or "").strip()
                or "unknown"
            )

            points.append(
                GoalkeeperPoint(
                    local_dt=local_dt,
                    ts_ms=ts_ms,
                    x=x,
                    y=y,
                    z=z,
                    sensor_id=sensor_id,
                )
            )

    points.sort(key=lambda p: (p.sensor_id, p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


def _detect_shot_direction(points: List[BallPoint]) -> int:
    if not points:
        return 1
    avg_x = sum(p.x for p in points) / len(points)
    return -1 if avg_x < 0 else 1


def _align_goalkeeper_to_ball_timestamps(
    sensor_points: List[GoalkeeperPoint],
    ball_traj: List[BallPoint],
    sensor_id: str,
) -> List[GoalkeeperPoint]:
    """Build goalkeeper samples exactly at ball timestamps (same window, same count)."""
    if not sensor_points or not ball_traj:
        return []

    aligned: List[GoalkeeperPoint] = []
    idx = 0
    
    # TODO: See below
    # This can be heavily simplified: Instead of checking for every point, check if the points for the goalkeeper are continuous, so full 20 Hz is matched
    # Then align first and last point of ball points and goalkeeper points. Fill the rest with the points between these two times of the appropriate sensor
    # Keep below implementation as fallback function in case the ball points or goalkeeper points are not continuous (50ms increments) -> log a warning in this case
    for ball_point in ball_traj:
        while idx + 1 < len(sensor_points) and sensor_points[idx + 1].local_dt <= ball_point.local_dt:
            idx += 1

        best = sensor_points[idx]
        if idx + 1 < len(sensor_points):
            left_delta = abs((ball_point.local_dt - sensor_points[idx].local_dt).total_seconds())
            right_delta = abs((sensor_points[idx + 1].local_dt - ball_point.local_dt).total_seconds())
            if right_delta < left_delta:
                best = sensor_points[idx + 1]

        aligned.append(
            GoalkeeperPoint(
                local_dt=ball_point.local_dt,
                ts_ms=ball_point.ts_ms,
                x=best.x,
                y=best.y,
                z=best.z,
                sensor_id=sensor_id,
            )
        )

    return aligned


def extract_goalkeeper_trajectory(
    candidates: List[GoalkeeperPoint], # already filtered for positions close to either goal
    ball_traj: List[BallPoint],
    prepend_frames: int = 5, # not used
) -> Tuple[List[GoalkeeperPoint], str]:
    """Select one goalkeeper trajectory aligned to ball timestamps.

    Result uses the exact same timestamp grid as ball_traj.
    """
    if not candidates or not ball_traj: 
        return [], ""

    start_dt = ball_traj[0].local_dt
    end_dt = ball_traj[-1].local_dt
    same_side_sign = _detect_shot_direction(ball_traj)

    # Create dict from sensor id to list of points of the sensor in the candidates time range
    by_sensor: Dict[str, List[GoalkeeperPoint]] = {}
    for p in candidates:
        by_sensor.setdefault(p.sensor_id, []).append(p) # Create empty list if not existing yet, append the point

    # Now gradually filter out to get for best matching sensor, go on to next sensor_id if sensor_points is empty
    best_sensor = ""
    best_segment: List[GoalkeeperPoint] = []
    best_score: Optional[Tuple[int, float, float]] = None

    for sensor_id, points in by_sensor.items():
        # Filter out sensors on wrong side of the field
        sensor_points = [
            p
            for p in points
            if int(math.copysign(1, p.x)) == same_side_sign
        ]
        if not sensor_points:
            continue
        
        in_range_indices = [
            i for i, p in enumerate(sensor_points) if start_dt <= p.local_dt <= end_dt
        ]
        if not in_range_indices:
            continue

        # Create time range for goalkeeper candidate
        first_in_range = in_range_indices[0]
        last_in_range = in_range_indices[-1]
        segment = sensor_points[first_in_range : last_in_range + 1]
        if not segment:
            continue

        mean_abs_x = sum(abs(p.x) for p in segment) / len(segment)
        mean_abs_y = sum(abs(p.y) for p in segment) / len(segment)
        score = (len(segment), mean_abs_x, -mean_abs_y) # Decides by length of segment, then how far away from midline, how close to center of goalline (y, less is better).

        if best_score is None or score > best_score:
            best_score = score
            best_sensor = sensor_id
            best_segment = _align_goalkeeper_to_ball_timestamps(segment, ball_traj, sensor_id)

    return best_segment, best_sensor


def first_idx_at_or_after(points: List[BallPoint], dt: datetime) -> Optional[int]:
    for i, p in enumerate(points):
        if p.local_dt >= dt:
            return i
    return None


def first_idx_at_or_before(points: List[BallPoint], dt: datetime) -> Optional[int]:
    for i in range(len(points) - 1, -1, -1):
        if points[i].local_dt <= dt:
            return i
    return None


def serialize_point(point: BallPoint) -> str:
    return json.dumps(
        {
            "t_local": point.local_dt.isoformat(timespec="milliseconds"),
            "ts_ms": point.ts_ms,
            "x": point.x,
            "y": point.y,
            "z": point.z,
            "v": None if math.isnan(point.speed) else point.speed,
            "a": None if math.isnan(point.accel) else point.accel,
            "dir": point.direction,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

# Finds first index of given points, where x ordinate is behind either goal line
def find_goalline_crossing_idx(points: List[BallPoint], start_idx: int) -> Optional[int]:
    for i in range(start_idx, len(points)):
        p = points[i]
        if p.x > 20.0 or p.x < -20.0:
            return i
    return None

# Finds the release point earlier
def detect_release_for_short_distance(
    points: List[BallPoint], start_idx: int, back_window_ms: int = 3000
) -> Optional[int]:
    # Find original start point
    start_dt = points[start_idx].local_dt
    # Go back back_window_ms ms (per default 3s)
    min_dt = start_dt - timedelta(milliseconds=back_window_ms)
    
    # Find the new points for the new time range
    candidates = [i for i in range(len(points)) if min_dt <= points[i].local_dt <= start_dt]
    if not candidates:
        return None

    # Filter out those points with no acceleration
    candidates_with_accel = [i for i in candidates if not math.isnan(points[i].accel)]
    if not candidates_with_accel:
        return None

    # Find out the max accel now
    max_accel = max(points[i].accel for i in candidates_with_accel)
    eps = 1e-6
    # Find out corresponding indices
    max_idxs = [i for i in candidates_with_accel if abs(points[i].accel - max_accel) <= eps]
    if not max_idxs:
        return None

    # Tie break for the last one (makes sense, afterwards ball should become slower)
    peak_idx = max(max_idxs)

    # TODO: Check if this makes sense. A better implementation would do a quick sanity check if the previous point is even inside a sensible area around the 7m point (defined like earlier, independent of field side)
    # Now going back as long as the previous acceleration was higher as to indicate that PoR was earlier
    # Goes before the selected prepend time window (default 3s)
    release_idx = peak_idx
    while release_idx > 0:
        prev_i = release_idx - 1
        prev_a = points[prev_i].accel
        cur_a = points[release_idx].accel
        if math.isnan(prev_a) or math.isnan(cur_a):
            break
        if prev_a >= cur_a:
            release_idx = prev_i
            continue
        break

    return release_idx


def has_x_direction_reversal(points: List[BallPoint], start_idx: int) -> bool:
    if start_idx >= len(points) - 1:
        return False

    start_x = points[start_idx].x
    direction = None

    for i in range(start_idx + 1, len(points)):
        x = points[i].x
        if x == start_x:
            continue

        current_direction = 1 if x > start_x else -1

        if direction is None:
            direction = current_direction
        elif direction != current_direction:
            return True

    return False

# TODO: This function is EXTREMELY naive. It doesnt check for the highest acceleration in the near proximity, which is a liability. 
# TODO: It should also check current z and y coords. It should also check for the highest accel. in a plausible positional window, namely look at 200ms before and after, find out if there are higher accelerations, go back (or forth) until (while checking if position makes sense) decrease in accel is found, select point before
def is_plausible_release_point(
    points: List[BallPoint],
    start_idx: int,
    distance: Optional[float],
) -> bool:
    if start_idx >= len(points):
        return False

    if distance is None:
        return False

    if not (6.5 <= distance <= 7.5):
        return False

    if has_x_direction_reversal(points, start_idx):
        return False

    return True


def adjust_start_idx_by_distance(
    points: List[BallPoint],
    start_idx: int,
    distance: Optional[float],
) -> Tuple[int, dict[str, str], Optional[int]]:
    flags: dict[str, str] = {}
    release_idx: Optional[int] = None

    if distance is None:
        flags["distance_check"] = "missing_distance"
        return start_idx, flags, release_idx

    if distance < 6.95: # TODO: is this threshold too low, e.g. should the distance be even closer for this adaption?
        # Finds highest acceleration before (extends time range up to 3s before, maybe even more if throw started at this edge)
        rel_idx = detect_release_for_short_distance(points, start_idx)
        if rel_idx is None:
            flags["short_distance_correction"] = "no_release_detected"
            return start_idx, flags, release_idx
        
        release_idx = rel_idx
        # Set start for visualisation to 1s before release
        corrected_start_dt = points[release_idx].local_dt - timedelta(seconds=1)
        corrected_start_idx = first_idx_at_or_after(points, corrected_start_dt)
        if corrected_start_idx is not None:
            # TODO: Show such flags in the visualisation plots
            flags["short_distance_correction"] = "applied"
            return corrected_start_idx, flags, release_idx
        
        flags["short_distance_correction"] = "failed_index_lookup"
        return start_idx, flags, release_idx

    # TODO: What is this? This is no correction at all.
    # Just returns the first index which is in this range
    # No check for release_index, returns None?
    # TODO: Maybe its just better to append a second to the range before the point where it has distance < 7.5 or smth like that
    if distance > 8:
        for i in range(start_idx, len(points)):
            x = points[i].x
            if (12.0 <= x <= 14.0) or (-14.0 <= x <= -12.0):
                flags["long_distance_correction"] = "applied"
                return i, flags, release_idx
        flags["long_distance_correction"] = "no_x_gate_match"
        return start_idx, flags, release_idx

    # Also returns None for release_index
    flags["distance_check"] = "in_range"
    return start_idx, flags, release_idx


def serialize_trajectory(points: List[BallPoint]) -> str:
    payload = []
    for p in points:
        payload.append(
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
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def serialize_goalkeeper_trajectory(points: List[GoalkeeperPoint]) -> str:
    payload = []
    for p in points:
        payload.append(
            {
                "t_local": p.local_dt.isoformat(timespec="milliseconds"),
                "ts_ms": p.ts_ms,
                "x": p.x,
                "y": p.y,
                "z": p.z,
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def select_release_point(points: List[BallPoint]) -> Tuple[BallPoint, str]:
    if not points:
        return points[0], "release_point:empty_trajectory"

    start_x = points[0].x if points else 0.0
    goal_x = 20.0 if start_x > 0 else -20.0

    release_zone_indices = []
    for i, p in enumerate(points):
        dx = p.x - goal_x
        dy = p.y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        if 6.5 <= distance_to_goal <= 7.5:
            release_zone_indices.append(i)

    finite_accel_indices = [i for i in release_zone_indices if not math.isnan(points[i].accel)]
    if finite_accel_indices:
        rel_local_i = max(finite_accel_indices, key=lambda i: points[i].accel)
        return points[rel_local_i], "release_point:max_a"

    # TODO: This is a stupid fallback -> needs to go over release_zone_indies too or at least check the points position as a release point outside [6.5, 7.5] really does not make sense
    finite_accel_indices = [i for i, p in enumerate(points) if not math.isnan(p.accel)]
    if finite_accel_indices:
        rel_local_i = max(finite_accel_indices, key=lambda i: points[i].accel)
        return points[rel_local_i], "release_point:max_a_fallback_global"

    # TODO: Same as comment above
    finite_speed_indices = [i for i, p in enumerate(points) if not math.isnan(p.speed)]
    if finite_speed_indices:
        rel_local_i = max(finite_speed_indices, key=lambda i: points[i].speed)
        return points[rel_local_i], "release_point:max_v_fallback"

    return points[0], "release_point:first_point_fallback"