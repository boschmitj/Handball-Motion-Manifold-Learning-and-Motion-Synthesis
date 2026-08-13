#!/usr/bin/env python3
"""
Create unified throw representation CSVs for Mocap and League data.

This script processes Mocap throws from a given throw_type directory and
produces three output CSVs:
  1. raw_mocap.csv   - Mocap throws with PoR-relative trajectories
  2. raw_league.csv  - League throws (loaded from penalties + position files)
  3. throw_index.csv - Throw metadata: id, whole-throw trajectory, PoR index,
                       global segment/PoR indices

Coordinate transformation (per mocap_league_throw_matching.md):
  1. Swap X/Y (already done by swap_xy_tsv.py in the pipeline)
  2. Translate: Mocap (0,0,0) → League (12.6, 0, 0) i.e. +12.6m in x
     after the swap, so the 7m line (League x=13) aligns with Mocap origin.
  3. Convert mm → m (Mocap data is in millimeters)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ball_trajectory import (
    BallPoint,
    load_ball_points,
    load_goalkeeper_candidates,
    load_possession_context_candidates,
    first_idx_at_or_after,
    first_idx_at_or_before,
    find_goalline_crossing_idx,
    detect_physics_based_release_point,
    is_plausible_release_point,
    adjust_start_idx_by_distance,
    serialize_trajectory,
    serialize_point,
)
from fixture_resolution import build_fixture_index, resolve_fixture_file, build_edge_case_mappings
from penalty_time_utils import parse_penalty_local_time, parse_position_local_time, try_float, try_int
from release_detector_trajectory_based import (
    compute_velocity_acceleration_from_points,
    select_release_index,
    _rotate_180_z,
    _is_left_side,
)


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

# League canonical court: origin at court centre, +x toward goal
# Goal line at x = +20 m, 7 m line at x = +13 m
# Mocap origin (0,0,0) ≈ (13 m - 0.40 m, 0, 0) in League coords per goal doc
# → after swap_xy, translate Mocap x by +12.6 m

MOCAP_TO_LEAGUE_TRANSLATION_M = 12.6  # mm→m handled below; pure x-translation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThrowRepresentation:
    """Unified representation of a throw (Mocap or League)."""
    throw_id: str
    source: str  # "mocap" or "league"
    # Canonical (League) coordinates in metres
    por_position: Tuple[float, float, float]  # (x, y, z) in m
    release_speed_m_s: float
    release_direction_deg: float
    release_height_m: float
    # Trajectory: PoR-relative, starting at (0,0,0)
    trajectory_json: str  # JSON of BallPoint list (PoR-relative, in m)
    trajectory_point_count: int
    trajectory_duration_ms: float
    # Full-throw info (for throw_index.csv)
    full_trajectory_json: Optional[str] = None
    full_trajectory_point_count: Optional[int] = None
    por_index_in_full: Optional[int] = None
    segment_start_global_idx: Optional[int] = None
    por_global_idx: Optional[int] = None
    # Error/validity
    valid: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def mocap_to_league_coords(
    x_mm: float, y_mm: float, z_mm: float
) -> Tuple[float, float, float]:
    """
    Convert Mocap coordinates (mm) to League canonical coordinates (m).

    Step 1: Swap X/Y (already done by swap_xy_tsv.py, but we apply it here
             for completeness if the file wasn't pre-swapped).
    Step 2: Translate x by +12.6 m so Mocap (0,0,0) → League (12.6, 0, 0)
             (at the 7m line position per goal doc).
    Step 3: Convert mm → m.

    Returns (x_League_m, y_League_m, z_League_m).
    """
    # Step 1: swap (newX = oldY, newY = -oldX)
    x_swapped = y_mm
    y_swapped = -x_mm

    # Step 2: translate x by +12.6 m (Mocap origin → 7m line in League)
    x_translated = x_swapped + MOCAP_TO_LEAGUE_TRANSLATION_M

    # Step 3: convert mm → m
    x_m = x_translated / 1000.0
    y_m = y_swapped / 1000.0
    z_m = z_mm / 1000.0

    return x_m, y_m, z_m


def league_to_mocap_coords(
    x_m: float, y_m: float, z_m: float
) -> Tuple[float, float, float]:
    """
    Reverse transformation: League canonical → Mocap original (mm).
    Useful for debugging / validation.
    """
    # Reverse: convert m → mm, translate -12.6m, then un-swap
    x_mm = (x_m - MOCAP_TO_LEAGUE_TRANSLATION_M) * 1000.0
    y_mm = y_m * 1000.0
    # un-swap: originalX = -newY, originalY = newX
    orig_x = -y_mm
    orig_y = x_mm
    orig_z = z_mm
    return orig_x, orig_y, orig_z


# ---------------------------------------------------------------------------
# Trajectory utilities
# ---------------------------------------------------------------------------

def compute_kinematics_from_points(
    points: List[BallPoint],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute per-point velocity (m/s), acceleration (m/s²), and direction (deg)
    from BallPoint positions using the SAME methodology as compute_velocity_acceleration_from_points
    in release_detector_trajectory_based.py, but returning all three quantities.

    Returns (speeds, accels, directions) aligned to input points.
    Speeds and accels may be None where computation is not possible.
    """
    n = len(points)
    if n < 2:
        return [None] * n, [None] * n, [None] * n

    # 3D velocity vectors via central differences
    velocities: List[Optional[Tuple[float, float, float]]] = [None] * n
    speeds: List[Optional[float]] = [None] * n
    directions: List[Optional[float]] = [None] * n
    acceleration_magnitudes: List[Optional[float]] = [None] * n

    # Forward/backward differences at boundaries, central otherwise
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
        directions[i] = math.degrees(math.atan2(vy, vx))

    # Acceleration: central difference of velocity vectors
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
        acceleration_magnitudes[i] = math.sqrt(ax * ax + ay * ay + az * az)

    # Boundary acceleration: forward/backward difference
    if n >= 2:
        # First point: difference between point 1 and point 0 velocities
        if velocities[0] is not None and velocities[1] is not None:
            dt = (points[1].local_dt - points[0].local_dt).total_seconds()
            if dt > 0:
                ax = (velocities[1][0] - velocities[0][0]) / dt
                ay = (velocities[1][1] - velocities[0][1]) / dt
                az = (velocities[1][2] - velocities[0][2]) / dt
                acceleration_magnitudes[0] = math.sqrt(ax * ax + ay * ay + az * az)

        # Last point: difference between point n-1 and point n-2 velocities
        if velocities[n - 2] is not None and velocities[n - 1] is not None:
            dt = (points[n - 1].local_dt - points[n - 2].local_dt).total_seconds()
            if dt > 0:
                ax = (velocities[n - 1][0] - velocities[n - 2][0]) / dt
                ay = (velocities[n - 1][1] - velocities[n - 2][1]) / dt
                az = (velocities[n - 1][2] - velocities[n - 2][2]) / dt
                acceleration_magnitudes[n - 1] = math.sqrt(ax * ax + ay * ay + az * az)

    return speeds, acceleration_magnitudes, directions


def po_relative_trajectory(
    points: List[BallPoint],
    por_idx: int,
) -> Tuple[List[BallPoint], List[float], List[float], List[float]]:
    """
    Return PoR-relative trajectory starting from (0,0,0).

    For each point at index i: p_rel = points[i] - points[por_idx]
    Also returns t_since_release, speed, acceleration, direction for each point.

    The returned points have x_rel, y_rel, z_rel in metres, and we also
    compute velocity/accel/direction at each point using the same finite-difference
    methodology as the parent functions.
    """
    if por_idx is None or por_idx < 0 or por_idx >= len(points):
        return [], [], [], []

    por_point = points[por_idx]
    origin = datetime(1970, 1, 1)

    rel_points: List[BallPoint] = []
    t_since: List[float] = []
    speeds: List[Optional[float]] = []
    accels: List[Optional[float]] = []
    dirs: List[Optional[float]] = []

    for i, p in enumerate(points):
        # PoR-relative position
        x_rel = (p.x - por_point.x) / 1000.0  # mm → m, but points already in m?
        y_rel = (p.y - por_point.y) / 1000.0
        z_rel = (p.z - por_point.z) / 1000.0

        # Time since PoR
        dt = (p.local_dt - por_point.local_dt).total_seconds()
        t_s = dt * 1000.0  # ms

        rel_points.append(
            BallPoint(
                local_dt=p.local_dt,
                ts_ms=p.ts_ms,
                x=x_rel,
                y=y_rel,
                z=z_rel,
                speed=p.speed,
                accel=p.accel,
                direction=p.direction,
            )
        )
        t_since.append(t_s)
        speeds.append(p.speed)
        accels.append(p.accel)
        dirs.append(p.direction)

    return rel_points, t_since, speeds, accels, dirs


def find_goal_line_crossing_idx(points: List[BallPoint]) -> Optional[int]:
    """
    Return the first index where |x| > 20 m (goal line crossed).
    Goal lines are at x = +20 and x = -20 in League canonical coords.
    """
    for i, p in enumerate(points):
        if abs(p.x) > 20.0:
            return i
    return None


# ---------------------------------------------------------------------------
# League throw loading
# ---------------------------------------------------------------------------

def load_league_throws(
    penalties_file: Path,
    positions_dir: Path,
    limit: Optional[int] = None,
    include_unsuccessful: bool = False,
    penalty_id: Optional[str] = None,
) -> List[ThrowRepresentation]:
    """
    Load League throws from penalties.csv + position files,
    applying the same pipeline as simple_release_detector.py but
    returning ThrowRepresentation objects with PoR-relative trajectories.
    """
    from penalty_processing import process_penalties

    # Run the pipeline to get the basic output
    run_dir = process_penalties(
        penalties_file=penalties_file,
        positions_dir=positions_dir,
        output_dir=Path("./out_test"),
        tol_y=0.35,
        tol_z=0.35,
        limit=limit,
        include_unsuccessful=include_unsuccessful,
        penalty_id=penalty_id,
    )

    # Read the output CSV
    output_file = run_dir / "penalty_trajectories.csv"
    if not output_file.exists():
        return []

    df = pd.read_csv(output_file, sep=";", dtype=str)
    representations: List[ThrowRepresentation] = []

    for _, row in df.iterrows():
        throw_id = row.get("id", "")
        if not throw_id:
            continue

        try:
            # Parse release point
            release_point_json = row.get("release_point_json", "{}")
            release_point = json.loads(release_point_json)

            por_x = try_float(release_point.get("x", "0"))
            por_y = try_float(release_point.get("y", "0"))
            por_z = try_float(release_point.get("z", "0"))

            # Parse trajectory
            traj_json = row.get("trajectory_json", "[]")
            trajectory = json.loads(traj_json)

            # Convert trajectory points to BallPoint-like structure
            # and apply coordinate transformation + PoR-relative
            ball_points: List[BallPoint] = []
            for pt in trajectory:
                x = try_float(pt.get("x", "0"))
                y = try_float(pt.get("y", "0"))
                z = try_float(pt.get("z", "0"))
                # These are already in League canonical coords from the pipeline
                ball_points.append(
                    BallPoint(
                        local_dt=parse_position_local_time(pt.get("t_local", "")),
                        ts_ms=try_int(pt.get("ts_ms", "")),
                        x=x,
                        y=y,
                        z=z,
                        speed=try_float(pt.get("v", "")),
                        accel=try_float(pt.get("a", "")),
                        direction=try_float(pt.get("dir", "")),
                    )
                )

            if not ball_points:
                continue

            # Find PoR index in the trajectory
            # The release_point_json should have the PoR position
            # We need to find the index where the point matches the release point
            por_idx = None
            for i, bp in enumerate(ball_points):
                if (abs(bp.x - por_x) < 0.01 and abs(bp.y - por_y) < 0.01
                        and abs(bp.z - por_z) < 0.01):
                    por_idx = i
                    break

            if por_idx is None:
                # Use the release_idx from the row
                release_idx = try_int(row.get("release_idx", ""))
                if release_idx is not None and 0 <= release_idx < len(ball_points):
                    por_idx = release_idx

            # Compute PoR-relative trajectory
            if por_idx is not None and 0 <= por_idx < len(ball_points):
                rel_points, t_since, rel_speeds, rel_accels, rel_dirs = po_relative_trajectory(
                    ball_points, por_idx
                )
            else:
                # Fallback: just use first point as PoR
                rel_points, t_since, rel_speeds, rel_accels, rel_dirs = po_relative_trajectory(
                    ball_points, 0
                )

            # trajectory duration (from first to last point)
            if len(t_since) >= 2:
                duration_ms = t_since[-1] - t_since[0]
            else:
                duration_ms = 0.0

            # Release info
            release_speed = rel_speeds[por_idx] if por_idx is not None and por_idx < len(rel_speeds) else (
                ball_points[por_idx].speed if por_idx is not None and por_idx < len(ball_points) else None
            ) or 0.0

            release_dir = rel_dirs[por_idx] if por_idx is not None and por_idx < len(rel_dirs) else (
                ball_points[por_idx].direction if por_idx is not None and por_idx < len(ball_points) else None
            ) or 0.0

            release_height = por_point.z if por_point is not None else 0.0

            # Serialize PoR-relative trajectory
            traj_ser = serialize_trajectory(rel_points) if rel_points else "[]"

            rep = ThrowRepresentation(
                throw_id=throw_id,
                source="league",
                por_position=(por_x, por_y, por_z),
                release_speed_m_s=float(release_speed) if release_speed is not None else 0.0,
                release_direction_deg=float(release_dir) if release_dir is not None else 0.0,
                release_height_m=float(release_height) if release_height is not None else 0.0,
                trajectory_json=traj_ser,
                trajectory_point_count=len(rel_points),
                trajectory_duration_ms=float(duration_ms),
                full_trajectory_json=traj_json,
                full_trajectory_point_count=len(ball_points),
                por_index_in_full=por_idx,
                segment_start_global_idx=None,  # League has no "segment start" in same sense
                por_global_idx=por_idx,
                valid=True,
            )
            representations.append(rep)

        except Exception as exc:
            warnings.warn(f"Error processing league throw {throw_id}: {exc}")
            continue

    return representations


# ---------------------------------------------------------------------------
# Mocap throw processing
# ---------------------------------------------------------------------------

def process_mocap_throw(
    throw_dir: Path,
    throw_id: str,
) -> Optional[ThrowRepresentation]:
    """
    Process a single Mocap throw directory and return a ThrowRepresentation.

    The throw_dir should contain:
      - skeleton_*.tsv   (may be swapped already)
      - ball_3d_*.tsv  or ball_6d_*.tsv
      - body_*.tsv     (optional)

    Returns ThrowRepresentation or None if processing fails.
    """
    # Find the relevant files
    skeleton_files = list(throw_dir.glob("skeleton*.tsv"))
    ball_files = list(throw_dir.glob("ball_3d*.tsv")) + list(throw_dir.glob("ball_6d*.tsv"))
    body_files = list(throw_dir.glob("body*.tsv"))

    if not skeleton_files or not ball_files:
        print(f"  Warning: missing files in {throw_dir}, skipping")
        return None

    skeleton_path = skeleton_files[0]
    ball_path = ball_files[0]
    body_path = body_files[0] if body_files else None

    # Index the files (this will also swap XY if swap_xy is enabled in the pipeline)
    from mocap_por_detection_pipeline import (
        index_skeleton_file, index_ball_3d_file, index_ball_6d_file,
    )

    if ball_path.suffix == ".6d" or "6D" in str(ball_path):
        ball_index = index_ball_6d_file(ball_path)
    else:
        ball_index = index_ball_3d_file(ball_path)
    skeleton_index = index_skeleton_file(skeleton_path)

    # Find common frames
    shared_frames = sorted(set(skeleton_index.frames) & set(ball_index.frames))
    if not shared_frames:
        print(f"  Warning: no shared frames in {throw_dir}, skipping")
        return None

    # Load ball samples per frame
    from mocap_por_detection_pipeline import (
        _load_ball_sample_from_fit, _load_ball_sample_from_gt,
    )

    ball_by_frame: Dict[int, Any] = {}
    # We'll use fit-based loading (6DOF sphere fit) as default
    for frame in shared_frames:
        ball_by_frame[frame] = _load_ball_sample_from_fit(frame, 0.0, ball_index)

    # Detect throw segments
    from mocap_por_detection_pipeline import detect_throw_segments

    # Dummy parameters for segment detection
    segments = detect_throw_segments(
        frames=shared_frames,
        positions_by_frame={},
        ball_by_frame=ball_by_frame,
        reacquire_distance_mm=120.0,
        sigma_multiplier=6.0,
        baseline_min_samples=100,
        baseline_std_floor_mm=1.0,
    )

    if not segments:
        print(f"  Warning: no throw segments detected in {throw_dir}, skipping")
        return None

    segment = segments[0]  # Use first detected segment

    # Now collect the full window samples from segment start to end
    # We need to re-load all samples in the window
    from mocap_por_detection_pipeline import _collect_window_samples

    valid_window_samples, _ = _collect_window_samples(
        ball_by_frame=ball_by_frame,
        all_frames=shared_frames,
        start_frame=segment.start_frame,
        end_frame=segment.end_frame,
    )

    if not valid_window_samples:
        print(f"  Warning: no valid window samples in {throw_dir}, skipping")
        return None

    # Detect PoR using method 1 (the 6-sigma hand-ball distance spike)
    # We need kinematics for this
    from mocap_por_detection_pipeline import _compute_kinematics

    full_kinematics = _compute_kinematics(valid_window_samples, fs_hz=300.0)
    method1 = _method_result_at_frame(full_kinematics, segment.method1_por_frame)

    por_frame = method1.por_frame if method1.por_frame is not None else segment.method1_por_frame

    # Find the index of the PoR in our samples
    por_idx_in_samples = None
    for i, sample in enumerate(valid_window_samples):
        if sample.frame == por_frame:
            por_idx_in_samples = i
            break

    if por_idx_in_samples is None:
        # Fallback: use the index from the segment
        por_idx_in_samples = 0

    # Get the PoR point
    por_sample = valid_window_samples[por_idx_in_samples] if por_idx_in_samples < len(valid_window_samples) else valid_window_samples[0]
    por_x_m, por_y_m, por_z_m = mocap_to_league_coords(
        por_sample.center_mm[0], por_sample.center_mm[1], por_sample.center_mm[2]
    )

    # Build the full trajectory (PoR to end)
    # End: when ball becomes unidentified or hits wall
    # We'll use the pipeline's free flight window detection concept
    # For now, use all samples from PoR to end of segment

    # Build BallPoint list from samples
    all_ball_points: List[BallPoint] = []
    for i, sample in enumerate(valid_window_samples):
        if sample.center_mm is None:
            continue
        center_m = np.asarray(sample.center_mm, dtype=np.float64) / 1000.0  # mm → m
        origin = datetime(1970, 1, 1)
        all_ball_points.append(
            BallPoint(
                local_dt=origin + timedelta(seconds=sample.time_s),
                ts_ms=sample.frame,  # approximate
                x=float(center_m[0]),
                y=float(center_m[1]),
                z=float(center_m[2]),
                speed=float("nan"),
                accel=float("nan"),
                direction=None,
            )
        )

    if not all_ball_points:
        print(f"  Warning: no valid ball points in {throw_dir}, skipping")
        return None

    # PoR-relative trajectory
    if por_idx_in_samples is not None and 0 <= por_idx_in_samples < len(all_ball_points):
        rel_points, t_since, rel_speeds, rel_accels, rel_dirs = po_relative_trajectory(
            all_ball_points, por_idx_in_samples
        )
    else:
        rel_points, t_since, rel_speeds, rel_accels, rel_dirs = po_relative_trajectory(
            all_ball_points, 0
        )

    # Trajectory duration
    if len(t_since) >= 2:
        duration_ms = t_since[-1] - t_since[0]
    else:
        duration_ms = 0.0

    # Compute kinematics from the PoR-relative trajectory using the unified method
    computed_speeds, computed_accels, computed_dirs = compute_kinematics_from_points(rel_points)

    # Release speed at PoR
    release_speed = computed_speeds[por_idx_in_samples] if por_idx_in_samples is not None and por_idx_in_samples < len(computed_speeds) else (rel_speeds[por_idx_in_samples] if rel_speeds else None) or 0.0

    # Release direction at PoR
    release_dir = computed_dirs[por_idx_in_samples] if por_idx_in_samples is not None and por_idx_in_samples < len(computed_dirs) else (rel_dirs[por_idx_in_samples] if rel_dirs else None) or 0.0

    # Serialize PoR-relative trajectory with kinematics
    # We need to add velocity, acceleration, direction to the BallPoints
    enhanced_rel_points: List[BallPoint] = []
    for i, p in enumerate(rel_points):
        enhanced_rel_points.append(
            BallPoint(
                local_dt=p.local_dt,
                ts_ms=p.ts_ms,
                x=p.x,
                y=p.y,
                z=p.z,
                speed=computed_speeds[i] if computed_speeds and i < len(computed_speeds) else None,
                accel=computed_accels[i] if computed_accels and i < len(computed_accels) else None,
                direction=computed_dirs[i] if computed_dirs and i < len(computed_dirs) else None,
            )
        )

    traj_ser = serialize_trajectory(enhanced_rel_points) if enhanced_rel_points else "[]"

    # Full trajectory (whole throw, not PoR-relative)
    # Serialize all original ball points
    full_traj_ser = serialize_trajectory(all_ball_points) if all_ball_points else "[]"

    # Determine global indices
    # segment_start_global_idx: index in the full shared_frames list
    segment_start_global_idx = None
    for i, frame in enumerate(shared_frames):
        if frame == segment.start_frame:
            segment_start_global_idx = i
            break

    # por_global_idx: global index of PoR
    por_global_idx = None
    for i, frame in enumerate(shared_frames):
        if frame == por_frame:
            por_global_idx = i
            break

    # Release height
    release_height_m = por_z_m

    rep = ThrowRepresentation(
        throw_id=throw_id,
        source="mocap",
        por_position=(por_x_m, por_y_m, por_z_m),
        release_speed_m_s=float(release_speed) if release_speed is not None else 0.0,
        release_direction_deg=float(release_dir) if release_dir is not None else 0.0,
        release_height_m=float(release_height_m) if release_height_m is not None else 0.0,
        trajectory_json=traj_ser,
        trajectory_point_count=len(rel_points),
        trajectory_duration_ms=float(duration_ms),
        full_trajectory_json=full_traj_ser,
        full_trajectory_point_count=len(all_ball_points),
        por_index_in_full=por_idx_in_samples,
        segment_start_global_idx=segment_start_global_idx,
        por_global_idx=por_global_idx,
        valid=True,
    )

    return rep


def _method_result_at_frame(kinematics, frame):
    """Helper: get MethodResult at a given frame (copied from mocap_por_detection_pipeline)."""
    if frame is None:
        return None
    try:
        idx = kinematics.frames.index(frame)
    except ValueError:
        return None
    return type('MethodResult', (), {
        'por_frame': kinematics.frames[idx],
        'velocity_m_s': kinematics.speed_mm_s[idx] / 1000.0 if kinematics.speed_mm_s[idx] is not None else None,
        'acceleration_m_s2': kinematics.acceleration_mm_s2[idx] / 1000.0 if kinematics.acceleration_mm_s2[idx] is not None else None,
        'direction_deg': kinematics.direction_deg[idx],
    })()


# ---------------------------------------------------------------------------
# Throw directory organization
# ---------------------------------------------------------------------------

def find_throw_directories(root_dir: Path, throw_type: str) -> List[Tuple[str, Path]]:
    """
    Find all Mocap throw directories under root_dir/throw_type/.

    Returns list of (throw_id, directory_path) tuples.
    """
    throw_type_dir = root_dir / throw_type
    if not throw_type_dir.exists():
        print(f"  Warning: throw type directory {throw_type_dir} does not exist")
        return []

    results: List[Tuple[str, Path]] = []
    # Each subdirectory should be a throw
    for throw_dir in sorted(throw_type_dir.iterdir()):
        if not throw_dir.is_dir():
            continue
        # Throw ID could be derived from directory name or from files inside
        throw_id = throw_dir.name
        results.append((throw_id, throw_dir))

    return results


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_raw_mocap_csv(representations: List[ThrowRepresentation], output_path: Path) -> None:
    """Write raw_mocap.csv with PoR-relative trajectory data."""
    fieldnames = [
        "throw_id",
        "source",
        "por_x_m", "por_y_m", "por_z_m",
        "release_speed_m_s", "release_direction_deg", "release_height_m",
        "trajectory_json",
        "trajectory_point_count", "trajectory_duration_ms",
        "valid", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rep in representations:
            if rep.source != "mocap":
                continue
            row = {
                "throw_id": rep.throw_id,
                "source": rep.source,
                "por_x_m": rep.por_position[0],
                "por_y_m": rep.por_position[1],
                "por_z_m": rep.por_position[2],
                "release_speed_m_s": rep.release_speed_m_s,
                "release_direction_deg": rep.release_direction_deg,
                "release_height_m": rep.release_height_m,
                "trajectory_json": rep.trajectory_json,
                "trajectory_point_count": rep.trajectory_point_count,
                "trajectory_duration_ms": rep.trajectory_duration_ms,
                "valid": rep.valid,
                "error": rep.error,
            }
            writer.writerow(row)


def write_raw_league_csv(representations: List[ThrowRepresentation], output_path: Path) -> None:
    """Write raw_league.csv with PoR-relative trajectory data."""
    fieldnames = [
        "throw_id",
        "source",
        "por_x_m", "por_y_m", "por_z_m",
        "release_speed_m_s", "release_direction_deg", "release_height_m",
        "trajectory_json",
        "trajectory_point_count", "trajectory_duration_ms",
        "valid", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rep in representations:
            if rep.source != "league":
                continue
            row = {
                "throw_id": rep.throw_id,
                "source": rep.source,
                "por_x_m": rep.por_position[0],
                "por_y_m": rep.por_position[1],
                "por_z_m": rep.por_position[2],
                "release_speed_m_s": rep.release_speed_m_s,
                "release_direction_deg": rep.release_direction_deg,
                "release_height_m": rep.release_height_m,
                "trajectory_json": rep.trajectory_json,
                "trajectory_point_count": rep.trajectory_point_count,
                "trajectory_duration_ms": rep.trajectory_duration_ms,
                "valid": rep.valid,
                "error": rep.error,
            }
            writer.writerow(row)


def write_throw_index_csv(representations: List[ThrowRepresentation], output_path: Path) -> None:
    """
    Write throw_index.csv with throw metadata:
    - throw_id
    - trajectory_json_list of the WHOLE mocap throw (from segment start to end)
    - index relative to that list where the PoR is
    - global indices for segment start, PoR
    - Also includes league data with appropriate fields
    """
    fieldnames = [
        "throw_id",
        "source",
        "trajectory_json_list",  # whole-throw trajectory JSON
        "por_index_relative",    # index within trajectory_json_list where PoR is
        "segment_start_global_idx",
        "por_global_idx",
        "por_x_m", "por_y_m", "por_z_m",
        "release_speed_m_s",
        "release_direction_deg",
        "release_height_m",
        "trajectory_point_count_full",
        "trajectory_duration_ms_full",
        "valid", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rep in representations:
            row = {
                "throw_id": rep.throw_id,
                "source": rep.source,
                "trajectory_json_list": rep.full_trajectory_json or "[]",
                "por_index_relative": rep.por_index_in_full,
                "segment_start_global_idx": rep.segment_start_global_idx,
                "por_global_idx": rep.por_global_idx,
                "por_x_m": rep.por_position[0],
                "por_y_m": rep.por_position[1],
                "por_z_m": rep.por_position[2],
                "release_speed_m_s": rep.release_speed_m_s,
                "release_direction_deg": rep.release_direction_deg,
                "release_height_m": rep.release_height_m,
                "trajectory_point_count_full": rep.full_trajectory_point_count or 0,
                "trajectory_duration_ms_full": rep.trajectory_duration_ms,
                "valid": rep.valid,
                "error": rep.error,
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create unified throw representation CSVs for Mocap and League data. "
            "Processes Mocap throws from a throw_type directory and produces: "
            "raw_mocap.csv, raw_league.csv, throw_index.csv"
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory of the PenaltyProcessing project",
    )
    parser.add_argument(
        "--throw-type",
        type=str,
        default="mocap_files",
        help="Subdirectory under root-dir containing Mocap throw data "
             "(e.g., 'mocap_files', 'mocap_por')",
    )
    parser.add_argument(
        "--league-penalties",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "penalties.csv",
        help="Path to penalties.csv for League data",
    )
    parser.add_argument(
        "--league-positions-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "games_position_files",
        help="Directory containing *_2_phases_positions.csv for League data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of penalties to process",
    )
    parser.add_argument(
        "--include-unsuccessful",
        action="store_true",
        help="Include unsuccessful throws in League processing",
    )
    parser.add_argument(
        "--penalty-id",
        type=str,
        default=None,
        help="If set, only process the penalty row with this id",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out",
        help="Base output directory for the CSV files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information",
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    throw_type = args.throw_type
    output_dir = args.output_dir

    print(f"=== Create Throw Representation ===")
    print(f"Root dir: {root_dir}")
    print(f"Throw type: {throw_type}")
    print(f"Output dir: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Process Mocap throws ----
    print("--- Processing Mocap throws ---")
    mocap_throws = find_throw_directories(root_dir, throw_type)
    print(f"Found {len(mocap_throws)} Mocap throw directories")

    mocap_reps: List[ThrowRepresentation] = []
    for throw_id, throw_dir in mocap_throws:
        if args.verbose:
            print(f"  Processing throw {throw_id}...")
        rep = process_mocap_throw(throw_dir, throw_id)
        if rep is not None:
            mocap_reps.append(rep)
            if args.verbose:
                print(f"    -> Success: por at ({rep.por_position[0]:.3f}, {rep.por_position[1]:.3f}, {rep.por_position[2]:.3f}) m")
        else:
            if args.verbose:
                print(f"    -> Failed (skipped)")

    print(f"Successfully processed {len(mocap_reps)} Mocap throws")
    print()

    # ---- Phase 2: Load League throws ----
    print("--- Loading League throws ---")
    league_reps = load_league_throws(
        penalties_file=args.league_penalties,
        positions_dir=args.league_positions_dir,
        limit=args.limit,
        include_unsuccessful=args.include_unsuccessful,
        penalty_id=args.penalty_id,
    )
    print(f"Successfully loaded {len(league_reps)} League throws")
    print()

    # ---- Phase 3: Write output CSVs ----
    print("--- Writing output CSVs ---")

    raw_mocap_path = output_dir / "raw_mocap.csv"
    write_raw_mocap_csv(mocap_reps, raw_mocap_path)
    print(f"Wrote {len(mocap_reps)} Mocap rows to {raw_mocap_path}")

    raw_league_path = output_dir / "raw_league.csv"
    write_raw_league_csv(league_reps, raw_league_path)
    print(f"Wrote {len(league_reps)} League rows to {raw_league_path}")

    throw_index_path = output_dir / "throw_index.csv"
    write_throw_index_csv(mocap_reps, throw_index_path)
    print(f"Wrote {len(mocap_reps)} rows to {throw_index_path}")
    print()

    # ---- Summary ----
    print("=== Summary ===")
    print(f"Mocap throws processed: {len(mocap_reps)}")
    print(f"League throws loaded: {len(league_reps)}")
    print()
    print(f"Output files:")
    print(f"  {raw_mocap_path}")
    print(f"  {raw_league_path}")
    print(f"  {throw_index_path}")
    print()
    print("=== Done ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())