"""
Main penalty processing pipeline.

This module ties together the whole penalty trajectory extraction pipeline:

1. Load the penalties CSV and apply filters (successful only, skip known-bad
   clock times, optional penalty-id / limit / random-test).
2. Resolve each penalty row to its position-tracking fixture file.
3. For each fixture, load the ball, goalkeeper, and possession-context points.
4. For each penalty row, build a trajectory window around the shot, correct
   the start time based on the reported distance, detect the release point
   (both a legacy heuristic and a physics-based method), and extract the
   goalkeeper trajectory.
5. Write one output row per penalty to a CSV, plus an issues CSV for rows
   that could not be processed.

The output CSV contains serialized JSON trajectories and release-point
metadata that downstream visualization code can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ball_trajectory import (
    adjust_start_idx_by_distance,
    extract_goalkeeper_trajectory,
    find_goalline_crossing_idx,
    first_idx_at_or_after,
    first_idx_at_or_before,
    detect_physics_based_release_point,
    is_plausible_release_point,
    load_ball_points,
    load_goalkeeper_candidates,
    load_possession_context_candidates,
    select_release_point,
    serialize_goalkeeper_trajectory,
    serialize_point,
    serialize_trajectory,
)
from fixture_resolution import (
    build_edge_case_mappings,
    build_fixture_index,
    create_run_folder,
    is_successful_penalty,
    load_penalties,
    resolve_fixture_file,
    should_skip_penalty_row,
)
from penalty_time_utils import parse_penalty_local_time, try_float


# A trajectory longer than this many points (at 20 Hz ~ 3 s) is considered
# too long and is skipped (likely a tracking error or a non-penalty sequence).
MAX_TRAJECTORY_POINT_COUNT = 60
# Number of extra frames prepended to the trajectory for visualization context.
PREPEND_FRAME_COUNT = 5


def _build_trajectory_window(
    points: List,
    start_idx: int,
    end_idx: int,
    extend_start_ms: int,
) -> Tuple[List, int, int]:
    """Build the visualization trajectory window around the release search span."""
    extended_start_dt = points[start_idx].local_dt - timedelta(milliseconds=extend_start_ms)
    extended_end_dt = points[end_idx].local_dt + timedelta(milliseconds=300)

    extended_start_idx = first_idx_at_or_after(points, extended_start_dt)
    if extended_start_idx is None:
        extended_start_idx = 0

    extended_end_idx = first_idx_at_or_before(points, extended_end_dt)
    if extended_end_idx is None:
        extended_end_idx = len(points) - 1

    traj = points[extended_start_idx : extended_end_idx + 1]
    return traj, extended_start_idx, extended_end_idx


def _build_release_detection_json(
    traj: List,
    physics_release,
    legacy_release_flag: str,
    release_point,
) -> str:
    """Serialize the physics-based diagnostics payload."""
    return json.dumps(
        {
            "release_timestamp": release_point.local_dt.isoformat(timespec="milliseconds"),
            "release_position": None
            if physics_release.release_point is None
            else {
                "x": physics_release.release_point.x,
                "y": physics_release.release_point.y,
                "z": physics_release.release_point.z,
            },
            "release_velocity": physics_release.release_velocity,
            "release_distance_to_goal": physics_release.release_distance_to_goal,
            "release_score": physics_release.release_score,
            "projectile_score": physics_release.projectile_score,
            "kinematic_score": physics_release.kinematic_score,
            "core_score": physics_release.core_score,
            "distance_score": physics_release.distance_score,
            "confidence": physics_release.confidence,
            "candidate_window": list(physics_release.candidate_window),
            "goal_cutoff_idx": physics_release.goal_cutoff_idx,
            "ground_contact_idx": physics_release.ground_contact_idx,
            "ground_contact_time_local": None
            if physics_release.ground_contact_idx is None
            else traj[min(physics_release.ground_contact_idx, len(traj) - 1)].local_dt.isoformat(timespec="milliseconds"),
            "legacy_release_flag": legacy_release_flag,
            "diagnostics": physics_release.diagnostics,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

# TODO: Modularize this function heavily -> break it up into smaller functions, reusable; For example for appending unresolved (its always nearly the same)
def process_penalties(
    penalties_file: Path,
    positions_dir: Path,
    output_dir: Path,
    tol_y: float, # Unused?
    tol_z: float, # Unused?
    limit: Optional[int],
    extend_start_ms: int = 0,
    include_unsuccessful: bool = False,
    penalty_id: Optional[str] = None,
    random_test: Optional[int] = None,
) -> Path:
    """Run the full penalty trajectory extraction pipeline.

    Args:
        penalties_file: Path to the penalties.csv file.
        positions_dir: Directory containing ``*_2_phases_positions.csv`` files.
        output_dir: Base output directory; a numbered run folder is created
            inside it.
        tol_y: Unused parameter (kept for API compatibility).
        tol_z: Unused parameter (kept for API compatibility).
        limit: Optional cap on the number of penalty rows to process.
        extend_start_ms: Extra milliseconds to prepend to the trajectory start
            window before the shot timestamp.
        include_unsuccessful: If True, also process unsuccessful throws.
        penalty_id: If set, only process the penalty row with this id.
        random_test: If set, randomly select this many rows for testing.

    Returns:
        The path to the created run folder.
    """
    rows = load_penalties(penalties_file)
    total_rows = len(rows)

    # Filter first by penalty_id, if provided -> limits rows to only 1 row
    if penalty_id is not None:
        rows = [row for row in rows if row.get("id", "").strip() == str(penalty_id)]
        print(f"Filtered to penalty id {penalty_id}: {len(rows)}/{total_rows}", flush=True)
        if not rows:
            raise SystemExit(f"No penalty with id {penalty_id} found in {penalties_file}")

    # Filter further by potentially filtering for successful throws only
    if not include_unsuccessful:
        rows = [r for r in rows if is_successful_penalty(r)]
    # Filter out throws which started at 00:00, 30:00, 60:00 -> TODO: Need to check if there is some better way to skip errornous throws, as 30:00 and 60:00 is possible
    rows = [r for r in rows if not should_skip_penalty_row(r)]
    print(
        f"Filtered to {'all shots' if include_unsuccessful else 'successful shots'}: {len(rows)}/{total_rows}",
        flush=True,
    )

    # Run directory is named after running count and time appended.
    run_dir = create_run_folder(output_dir)
    output_file = run_dir / "penalty_trajectories.csv"
    issues_file = run_dir / "penalty_trajectories_issues.csv"
    print(f"Output directory: {run_dir}", flush=True)

    # Edge case for test run (with some random rows or the first k rows)
    if random_test is not None:
        if random_test < len(rows):
            rows = random.sample(rows, random_test)
            print(f"Random test mode: selected {len(rows)} penalties", flush=True)
    elif limit is not None:
        rows = rows[:limit]

    # Build an index of all available fixture files, keyed by canonical team pair.
    fixture_index, fixture_index_issues = build_fixture_index(positions_dir) # Consists of the canonical team_names as a tuple (this is the key) and a list (len=1) of the Paths leading to those files
    edge_cases = build_edge_case_mappings()

    # Used to build an index of fixtures and their corresponding rows
    # Group penalty rows by the fixture file they belong to, so we only load
    # each position file once and process all its penalties together.
    grouped_rows: Dict[Path, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    # Unresolved rows with concrete issues
    unresolved: List[Dict[str, str]] = []

    for idx, row in enumerate(rows):
        fixture_file, errs = resolve_fixture_file(row, fixture_index, edge_cases)
        if fixture_file is None:
            unresolved.append(
                {
                    "row_idx": str(idx),
                    "penalty_id": row.get("id", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "issue": ";".join(errs),
                }
            )
            continue
        grouped_rows[fixture_file].append((idx, row))

    results: List[Dict[str, str]] = []
    # For debugging
    skipped_by_point_count = 0

    for fixture_file, fixture_rows in grouped_rows.items():
        # Load ball points and potential goalkeeper points (will get reevaluated later)
        # TODO: Optimize here -> Preload fixture files to reduce main overhead of loading a big csv file for ball processing (collecting ball points and goalkeeper points)
        try:
            points = load_ball_points(fixture_file)
            goalkeeper_candidates = load_goalkeeper_candidates(fixture_file)
            possession_candidates = load_possession_context_candidates(fixture_file)
        except Exception as exc:
            for idx, row in fixture_rows:
                unresolved.append(
                    {
                        "row_idx": str(idx),
                        "penalty_id": row.get("id", ""),
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "issue": f"positions_load_error:{exc}",
                    }
                )
            continue

        if not points:
            for idx, row in fixture_rows:
                unresolved.append(
                    {
                        "row_idx": str(idx),
                        "penalty_id": row.get("id", ""),
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "issue": "no_ball_points",
                    }
                )
            continue

        # Process a penalty row
        for idx, row in fixture_rows:
            print(f"Processing penalty row {idx+1}/{len(rows)} id:{row.get('id','')} fixture:{fixture_file.name}", flush=True)
            flags: List[str] = []
            # Load time (seperate function here)
            # Parse the shot's local timestamp from the penalty row.
            shot_dt = parse_penalty_local_time(row.get("timestamp_local_timezone", ""))
            if shot_dt is None:
                unresolved.append(
                    {
                        "row_idx": str(idx),
                        "penalty_id": row.get("id", ""),
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "issue": "invalid_shot_local_time",
                    }
                )
                continue

            # Find first point after penalty start time (seperate function here)
            # Locate the first ball point at or after the shot timestamp.
            start_idx = first_idx_at_or_after(points, shot_dt)
            if start_idx is None:
                unresolved.append(
                    {
                        "row_idx": str(idx),
                        "penalty_id": row.get("id", ""),
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "issue": "shot_start_after_last_ball_point",
                    }
                )
                continue

            distance = try_float(row.get("distance", ""))
            # Finds out better start index if necessary, for early releases also finds release_idx
            # Correct the start index based on the reported shot distance
            # (e.g. move it earlier for short shots, or to the run-up zone for long shots).
            start_idx, corr_flags, release_idx = adjust_start_idx_by_distance(points, start_idx, distance)
            for k, v in corr_flags.items():
                flags.append(f"{k}:{v}")

            is_success = is_successful_penalty(row)

            # Find where the ball crosses the goal line to bound the trajectory end.
            end_idx = find_goalline_crossing_idx(points, start_idx)
            if end_idx is None:
                if include_unsuccessful and not is_success:
                    # TODO: Find out what the "last" point refers to -> global last point of ball in fixture match or what?
                    end_idx = len(points) - 1
                    flags.append("unsuccessful:using_last_point")
                else:
                    unresolved.append(
                        {
                            "row_idx": str(idx),
                            "penalty_id": row.get("id", ""),
                            "home_team": row.get("home_team", ""),
                            "away_team": row.get("away_team", ""),
                            "issue": "no_goalline_crossing_found",
                        }
                    )
                    continue

            if end_idx < start_idx:
                unresolved.append(
                    {
                        "row_idx": str(idx),
                        "penalty_id": row.get("id", ""),
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "issue": "end_before_start",
                    }
                )
                continue

            # Keep the old 7 m plausibility check as a diagnostic baseline, but
            # the new detector no longer depends on it to choose the release.
            use_start_as_release = is_plausible_release_point(points, start_idx, distance)
            if use_start_as_release:
                flags.append("start_timestamp_plausible:1")
            else:
                flags.append("start_timestamp_plausible:0")

            window_traj, extended_start_idx, extended_end_idx = _build_trajectory_window(
                points,
                start_idx,
                end_idx,
                extend_start_ms,
            )
            if not window_traj:
                unresolved.append(
                    {
                        "row_idx": str(idx),
                        "penalty_id": row.get("id", ""),
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "issue": "empty_trajectory",
                    }
                )
                continue

            legacy_release_point, legacy_release_flag = select_release_point(window_traj)
            flags.append(f"legacy_{legacy_release_flag}")

            physics_release = detect_physics_based_release_point(
                window_traj,
                0,
                possession_candidates=possession_candidates,
            )

            if physics_release.release_point is not None:
                release_point = physics_release.release_point
                release_flag = "release_point:physics_based"
                flags.append("physics_release:applied")
            else:
                release_point = legacy_release_point
                release_flag = legacy_release_flag
                flags.append("physics_release:failed")

            # Prepend a fixed number of frames before the (possibly extended)
            # start so the visualization shows some context before the release.
            prepended_ball_count = min(PREPEND_FRAME_COUNT, max(0, extended_start_idx)) # So only if our extended start index is not "extended" enough for our PREPEND_FRAME_COUNT
            ball_start_idx_with_prepend = max(0, extended_start_idx - PREPEND_FRAME_COUNT) # Ah, i get why. But this is very confusing to read. TODO: Clarify why this is needed
            traj = points[ball_start_idx_with_prepend : extended_end_idx + 1]
            if prepended_ball_count > 0:
                flags.append(f"ball_prepend_frames:{prepended_ball_count}")

            # TODO: Seperate function
            # Extract the goalkeeper trajectory aligned to the ball trajectory.
            goalkeeper_traj, goalkeeper_sensor_id = extract_goalkeeper_trajectory(
                goalkeeper_candidates,
                traj,
            )
            if goalkeeper_traj:
                flags.append(f"goalkeeper_points:{len(goalkeeper_traj)}")
                if len(goalkeeper_traj) == len(traj):
                    flags.append("goalkeeper_aligned_with_ball:1")
                else:
                    flags.append("goalkeeper_aligned_with_ball:0")
            else:
                flags.append("goalkeeper_points:0")

            # Compute max speed/accel over the trajectory for the output row.
            finite_speeds = [p.speed for p in traj if not math.isnan(p.speed)]
            finite_accels = [p.accel for p in traj if not math.isnan(p.accel)]
            max_v = max(finite_speeds) if finite_speeds else float("nan")
            max_a = max(finite_accels) if finite_accels else float("nan")

            player_name = row.get("player_name", "")
            if not player_name:
                player_name = row.get("name", "")

            # Skip throws longer than 3s
            trajectory_point_count = len(traj)
            if trajectory_point_count >= MAX_TRAJECTORY_POINT_COUNT:
                skipped_by_point_count += 1
                continue

            # Assemble the output row with all serialized data.
            results.append(
                {
                    "id": row.get("id", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "fixture_file": fixture_file.name,
                    "player_id": row.get("player_id", ""),
                    "player_name": player_name,
                    "success": row.get("success", ""),
                    "distance": row.get("distance", ""),
                    "timestamp_local_timezone": row.get("timestamp_local_timezone", ""),
                    "start_time_local": traj[0].local_dt.isoformat(timespec="milliseconds"),
                    "end_time_local": traj[-1].local_dt.isoformat(timespec="milliseconds"),
                    "release_point_json": serialize_point(release_point), # This includes everything
                    "release_time_local": release_point.local_dt.isoformat(timespec="milliseconds"),
                    "legacy_release_point_json": serialize_point(legacy_release_point),
                    "legacy_release_time_local": legacy_release_point.local_dt.isoformat(timespec="milliseconds"),
                    "release_detection_json": _build_release_detection_json(window_traj, physics_release, legacy_release_flag, release_point),
                    "release_confidence": "" if physics_release.release_point is None else f"{physics_release.confidence:.6f}",
                    "release_score": "" if physics_release.release_point is None else f"{physics_release.release_score:.6f}",
                    "projectile_score": "" if physics_release.release_point is None else f"{physics_release.projectile_score:.6f}",
                    "kinematic_score": "" if physics_release.release_point is None else f"{physics_release.kinematic_score:.6f}",
                    "core_score": "" if physics_release.release_point is None else f"{physics_release.core_score:.6f}",
                    "distance_score": "" if physics_release.release_point is None else f"{physics_release.distance_score:.6f}",
                    "release_distance_to_goal": "" if physics_release.release_distance_to_goal is None else f"{physics_release.release_distance_to_goal:.6f}",
                    "max_v": "" if math.isnan(max_v) else f"{max_v:.6f}",
                    "max_a": "" if math.isnan(max_a) else f"{max_a:.6f}",
                    "release_angle": "" if release_point.direction is None else f"{release_point.direction:.6f}",
                    "trajectory_point_count": str(trajectory_point_count),
                    "trajectory_json": serialize_trajectory(traj),
                    "goalkeeper_sensor_id": goalkeeper_sensor_id,
                    "goalkeeper_trajectory_json": serialize_goalkeeper_trajectory(goalkeeper_traj),
                    "flags": ";".join(flags),
                }
            )

    # Column order for the output CSV.
    output_fields = [
        "id",
        "home_team",
        "away_team",
        "fixture_file",
        "player_id",
        "player_name",
        "success",
        "distance",
        "timestamp_local_timezone",
        "start_time_local",
        "end_time_local",
        "release_point_json",
        "release_time_local",
        "legacy_release_point_json",
        "legacy_release_time_local",
        "release_detection_json",
        "release_confidence",
        "release_score",
        "projectile_score",
        "kinematic_score",
        "core_score",
        "distance_score",
        "release_distance_to_goal",
        "max_v",
        "max_a",
        "release_angle",
        "trajectory_point_count",
        "trajectory_json",
        "goalkeeper_sensor_id",
        "goalkeeper_trajectory_json",
        "flags",
    ]

    # TODO: Seperate functions for below stuff
    # Write the main results CSV.
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter=";")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Write the issues CSV (unresolved rows + fixture index issues).
    issue_fields = ["row_idx", "penalty_id", "home_team", "away_team", "issue"]
    with issues_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=issue_fields, delimiter=";")
        writer.writeheader()
        for issue in unresolved:
            writer.writerow(issue)

        for issue in fixture_index_issues:
            writer.writerow(
                {
                    "row_idx": "",
                    "penalty_id": "",
                    "home_team": "",
                    "away_team": "",
                    "issue": f"fixture_index:{issue}",
                }
            )

    # Print a summary of the run.
    print("=== shot_matcher summary ===")
    print(f"Penalties input rows: {len(rows)}")
    print(f"Resolved trajectories: {len(results)}")
    print(
        f"Skipped by trajectory point count (>={MAX_TRAJECTORY_POINT_COUNT}): {skipped_by_point_count}"
    )
    print(f"Unresolved/issues: {len(unresolved)}")
    print(f"Output file: {output_file}")
    print(f"Issues file: {issues_file}")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the pipeline."""
    parser = argparse.ArgumentParser(description="Extract trajectories for penalty shots.")
    parser.add_argument("--penalties", default="../penalties.csv", help="Path to penalties.csv")
    parser.add_argument(
        "--positions-dir",
        default="../games_position_files",
        help="Directory containing *_2_phases_positions.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="../out",
        help="Base output directory (penalty_trajectories folder will be created here)",
    )
    parser.add_argument("--tol-y", type=float, default=0.35, help="Y-axis hit-position tolerance in meters")
    parser.add_argument("--tol-z", type=float, default=0.35, help="Z-axis hit-position tolerance in meters")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of penalties rows to process (ignored if --random-test is set)",
    )
    parser.add_argument(
        "--random-test",
        type=int,
        default=None,
        help="Optional: randomly select N successful penalties for testing instead of processing all",
    )
    parser.add_argument(
        "--include-unsuccessful",
        action="store_true",
        help="Include unsuccessful penalty throws in the output CSV",
    )
    parser.add_argument("--penalty-id", default=None, help="Optional: only process the penalty row with this id")
    parser.add_argument(
        "--extend-start-ms",
        type=int,
        default=0,
        help="Extend trajectory start window by N milliseconds before the shot timestamp (default 0 = no extension)",
    )
    return parser


def main() -> None:
    """CLI entry point: parse args and run the pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    run_dir = process_penalties(
        penalties_file=Path(args.penalties),
        positions_dir=Path(args.positions_dir),
        output_dir=Path(args.output_dir),
        tol_y=args.tol_y,
        tol_z=args.tol_z,
        limit=args.limit,
        extend_start_ms=args.extend_start_ms,
        include_unsuccessful=args.include_unsuccessful,
        penalty_id=args.penalty_id,
        random_test=args.random_test,
    )

    print(f"\nRun folder: {run_dir}")