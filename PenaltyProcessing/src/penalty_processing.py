from __future__ import annotations

import argparse
import csv
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
    is_plausible_release_point,
    load_ball_points,
    load_goalkeeper_candidates,
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


MAX_TRAJECTORY_POINT_COUNT = 60
PREPEND_FRAME_COUNT = 5


def process_penalties(
    penalties_file: Path,
    positions_dir: Path,
    output_dir: Path,
    tol_y: float,
    tol_z: float,
    limit: Optional[int],
    extend_start_ms: int = 0,
    include_unsuccessful: bool = False,
    penalty_id: Optional[str] = None,
    random_test: Optional[int] = None,
) -> Path:
    rows = load_penalties(penalties_file)
    total_rows = len(rows)

    if penalty_id is not None:
        rows = [row for row in rows if row.get("id", "").strip() == str(penalty_id)]
        print(f"Filtered to penalty id {penalty_id}: {len(rows)}/{total_rows}", flush=True)
        if not rows:
            raise SystemExit(f"No penalty with id {penalty_id} found in {penalties_file}")

    if not include_unsuccessful:
        rows = [r for r in rows if is_successful_penalty(r)]
    rows = [r for r in rows if not should_skip_penalty_row(r)]
    print(
        f"Filtered to {'all shots' if include_unsuccessful else 'successful shots'}: {len(rows)}/{total_rows}",
        flush=True,
    )

    run_dir = create_run_folder(output_dir)
    output_file = run_dir / "penalty_trajectories.csv"
    issues_file = run_dir / "penalty_trajectories_issues.csv"
    print(f"Output directory: {run_dir}", flush=True)

    if random_test is not None:
        if random_test < len(rows):
            rows = random.sample(rows, random_test)
            print(f"Random test mode: selected {len(rows)} penalties", flush=True)
    elif limit is not None:
        rows = rows[:limit]

    fixture_index, fixture_index_issues = build_fixture_index(positions_dir)
    edge_cases = build_edge_case_mappings()

    grouped_rows: Dict[Path, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
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
    skipped_by_point_count = 0

    for fixture_file, fixture_rows in grouped_rows.items():
        try:
            points = load_ball_points(fixture_file)
            goalkeeper_candidates = load_goalkeeper_candidates(fixture_file)
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

        for idx, row in fixture_rows:
            print(f"Processing penalty row {idx+1}/{len(rows)} id:{row.get('id','')} fixture:{fixture_file.name}", flush=True)
            flags: List[str] = []
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
            start_idx, corr_flags, release_idx = adjust_start_idx_by_distance(points, start_idx, distance)
            for k, v in corr_flags.items():
                flags.append(f"{k}:{v}")

            is_success = is_successful_penalty(row)

            end_idx = find_goalline_crossing_idx(points, start_idx)
            if end_idx is None:
                if include_unsuccessful and not is_success:
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

            use_start_as_release = is_plausible_release_point(points, start_idx, distance)

            if use_start_as_release:
                extended_start_dt = points[start_idx].local_dt - timedelta(milliseconds=extend_start_ms)
                extended_end_dt = points[end_idx].local_dt + timedelta(milliseconds=300)

                extended_start_idx = first_idx_at_or_after(points, extended_start_dt)
                if extended_start_idx is None:
                    extended_start_idx = 0

                extended_end_idx = first_idx_at_or_before(points, extended_end_dt)
                if extended_end_idx is None:
                    extended_end_idx = len(points) - 1

                traj = points[extended_start_idx : extended_end_idx + 1]
                if not traj:
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

                release_point = points[start_idx]
                release_flag = "release_point:start_timestamp_plausible"
                flags.append(release_flag)
            else:
                extended_start_dt = points[start_idx].local_dt - timedelta(milliseconds=extend_start_ms)
                extended_end_dt = points[end_idx].local_dt + timedelta(milliseconds=300)

                extended_start_idx = first_idx_at_or_after(points, extended_start_dt)
                if extended_start_idx is None:
                    extended_start_idx = 0

                extended_end_idx = first_idx_at_or_before(points, extended_end_dt)
                if extended_end_idx is None:
                    extended_end_idx = len(points) - 1

                traj = points[extended_start_idx : extended_end_idx + 1]
                if not traj:
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

                release_point, release_flag = select_release_point(traj)
                flags.append(release_flag)

                goal_x = 20.0 if release_point.x > 0 else -20.0
                dx = release_point.x - goal_x
                d_por = math.sqrt(dx * dx + (release_point.y) * (release_point.y))

                if d_por < 6.5 and not math.isnan(release_point.speed) and release_point.speed > 0:
                    dt_needed = (7.5 - d_por) / release_point.speed
                    if dt_needed > 0 and dt_needed < 5.0:
                        margin_s = 0.1
                        target_dt = release_point.local_dt - timedelta(seconds=(dt_needed + margin_s))
                        if target_dt < points[extended_start_idx].local_dt:
                            new_start_idx = first_idx_at_or_after(points, target_dt)
                            if new_start_idx is None:
                                new_start_idx = 0
                            if new_start_idx < extended_start_idx:
                                extended_start_idx = new_start_idx
                                traj = points[extended_start_idx : extended_end_idx + 1]
                                if traj:
                                    release_point, release_flag = select_release_point(traj)
                                    flags.append("release_point:dynamic_extended")

            # Keep existing trajectory logic and prepend a fixed number of frames.
            prepended_ball_count = min(PREPEND_FRAME_COUNT, max(0, extended_start_idx))
            ball_start_idx_with_prepend = max(0, extended_start_idx - PREPEND_FRAME_COUNT)
            traj = points[ball_start_idx_with_prepend : extended_end_idx + 1]
            if prepended_ball_count > 0:
                flags.append(f"ball_prepend_frames:{prepended_ball_count}")

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

            finite_speeds = [p.speed for p in traj if not math.isnan(p.speed)]
            finite_accels = [p.accel for p in traj if not math.isnan(p.accel)]
            max_v = max(finite_speeds) if finite_speeds else float("nan")
            max_a = max(finite_accels) if finite_accels else float("nan")

            player_name = row.get("player_name", "")
            if not player_name:
                player_name = row.get("name", "")

            trajectory_point_count = len(traj)
            if trajectory_point_count >= MAX_TRAJECTORY_POINT_COUNT:
                skipped_by_point_count += 1
                continue

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
                    "release_point_json": serialize_point(release_point),
                    "release_time_local": release_point.local_dt.isoformat(timespec="milliseconds"),
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
        "max_v",
        "max_a",
        "release_angle",
        "trajectory_point_count",
        "trajectory_json",
        "goalkeeper_sensor_id",
        "goalkeeper_trajectory_json",
        "flags",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter=";")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

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