from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_reconstruction import (  # noqa: E402
    BounceWindow, align_bounce_trajectory_to_por, align_trajectory_to_por,
    detect_bounce_window, estimate_bounce_time,
    insert_bounce_event, load_raw_throw, parse_trajectory_json, reconstruct_league_continuation,
    reconstruct_matches, resolve_random_weight_run, output_path_for_run,
    select_random_weight_run, upsample_at_times, upsample_trajectory,
)


def _point(t: float, x: float, y: float, z: float) -> dict:
    return {"frame": t / 50.0, "t_since_ms": t, "x": x, "y": y, "z": z}


def test_constant_translation_preserves_all_measured_geometry() -> None:
    original = parse_trajectory_json(json.dumps([
        _point(0, 0, 0, 0), _point(25, 0.5, 0.1, 0.2), _point(75, 1.4, 0.3, 0.1)
    ]))
    aligned = align_trajectory_to_por(original, (12.7, -0.2, 1.8))
    before = np.asarray([[p[a] for a in ("x", "y", "z")] for p in original])
    after = np.asarray([[p[a] for a in ("x", "y", "z")] for p in aligned])
    np.testing.assert_allclose(after - before, [[12.7, -0.2, 1.8]] * 3, atol=1e-12)
    np.testing.assert_allclose(after[:, None] - after[None, :], before[:, None] - before[None, :])
    assert [p["source_frame"] for p in aligned] == [0.0, 0.5, 1.5]
    assert all(p["is_original_sample"] for p in aligned)


def test_actual_timestamps_are_knots_and_duration_is_unchanged() -> None:
    points = parse_trajectory_json(json.dumps([
        _point(0, 0, 0, 0), _point(25, 0.5, 0.1, 0.2),
        _point(75, 1.5, 0.2, 0.3), _point(125, 2.5, 0.3, 0.25),
    ]))
    evaluated = upsample_at_times(points, [0, 25, 75, 125])
    np.testing.assert_allclose([[p[a] for a in ("x", "y", "z")] for p in evaluated],
                               [[p[a] for a in ("x", "y", "z")] for p in points])
    dense = upsample_trajectory(points, 300.0)
    assert dense[0]["t_since_ms"] == 0
    assert dense[-1]["t_since_ms"] == 125
    assert len(dense) > len(points)


def test_bounce_is_guarded_inserted_and_piecewise_preserved() -> None:
    def ballistic_z(t_ms: float, tb_ms: float, vz: float) -> float:
        dt = (t_ms - tb_ms) / 1000.0
        return 0.095 + vz * dt - 0.5 * 9.81 * dt * dt
    points = parse_trajectory_json(json.dumps([
        _point(0, 0, 0, ballistic_z(0, 160, -4)),
        _point(50, 1, 0, ballistic_z(50, 160, -4)),
        _point(100, 2, 0, ballistic_z(100, 160, -4)),
        _point(150, 3, 0, 0.20),
        _point(200, 4, 0, ballistic_z(200, 160, 4)),
        _point(250, 5, 0, ballistic_z(250, 160, 4)),
        _point(300, 6, 0, ballistic_z(300, 160, 4)),
    ]))
    window = detect_bounce_window(points)
    assert window == BounceWindow(3, 1, 5)
    event = estimate_bounce_time(points, window, ground_z=0.095)
    controls = insert_bounce_event(points, event)
    contact = next(point for point in controls if point.get("event") == "bounce")
    assert 100 < contact["t_since_ms"] < 200
    assert contact["z"] == 0.095
    assert contact["bounce_vz_in_m_s"] < 0 < contact["bounce_vz_out_m_s"]
    dense = upsample_trajectory(controls, 300)
    bounce = next(p for p in dense if p.get("event") == "bounce")
    assert bounce["z"] == 0.095
    exact_points = [points[i] for i in (0, 1, 5, 6)]
    measured = upsample_at_times(controls, [p["t_since_ms"] for p in exact_points])
    np.testing.assert_allclose([[p[a] for a in ("x", "y", "z")] for p in measured],
                               [[p[a] for a in ("x", "y", "z")] for p in exact_points])
    assert contact["bounce_reconstruction_rmse_m"] >= 0
    assert contact["bounce_reconstruction_max_error_m"] >= 0


def test_bounce_alignment_fades_z_correction_and_keeps_floor_after_contact() -> None:
    points = [
        {**_point(0, 10, 1, 1.5), "is_original_sample": True},
        {**_point(50, 11, 1, 0.7), "is_original_sample": False},
        {**_point(100, 12, 1, 0.095), "is_original_sample": False, "event": "bounce"},
        {**_point(150, 13, 1, 0.5), "is_original_sample": False},
    ]
    aligned = align_bounce_trajectory_to_por(points, (20, -2, 2.0))
    np.testing.assert_allclose([aligned[0][a] for a in ("x", "y", "z")], [20, -2, 2.0])
    assert math.isclose(aligned[1]["z"], 0.95)  # smootherstep(0.5) == 0.5
    assert math.isclose(aligned[2]["z"], 0.095)
    assert math.isclose(aligned[3]["z"], 0.5)
    np.testing.assert_allclose(
        [[q["x"] - p["x"], q["y"] - p["y"]] for p, q in zip(points, aligned)],
        [[10, -3]] * 4,
    )


def test_bounce_at_goal_line_with_only_two_outgoing_samples_is_detected() -> None:
    """A late bounce must not require samples beyond the goal-line crossing."""
    points = parse_trajectory_json(json.dumps([
        _point(0, 0.0, 0, 1.720),
        _point(25, 0.380, 0, 1.604),
        _point(75, 1.643, 0, 1.121),
        _point(125, 3.141, 0, 0.602),
        _point(175, 4.224, 0, 0.165),
        _point(225, 5.080, 0, 0.079),
        _point(275, 5.810, 0, 0.169),
        _point(325, 6.540, 0, 0.328),
    ]))
    window = detect_bounce_window(points)
    assert window == BounceWindow(minimum_idx=5, start_idx=3, end_idx=7)
    event = estimate_bounce_time(points, window)
    assert 225 < event.time_ms < 275
    assert event.vz_in_m_s < 0 < event.vz_out_m_s


def test_bounce_uses_window_change_when_adjacent_incoming_sample_is_near_ground() -> None:
    """Regression for League throw 10169014: 0.115 -> 0.099 is only 16 mm."""
    points = parse_trajectory_json(json.dumps([
        _point(0, 0.0, 0, 1.446),
        _point(25, 0.249, 0, 1.236),
        _point(75, 1.313, 0, 0.818),
        _point(125, 2.827, 0, 0.326),
        _point(175, 4.053, 0, 0.115),
        _point(225, 4.858, 0, 0.099),
        _point(275, 5.494, 0, 0.154),
        _point(325, 6.057, 0, 0.470),
        _point(375, 6.513, 0, 1.017),
        _point(425, 6.933, 0, 1.528),
    ]))
    window = detect_bounce_window(points)
    assert window == BounceWindow(minimum_idx=5, start_idx=3, end_idx=7)


def test_high_noisy_minimum_is_not_a_bounce() -> None:
    points = parse_trajectory_json(json.dumps([
        _point(0, 0, 0, 1.2), _point(50, 1, 0, 1.19), _point(100, 2, 0, 1.21)
    ]))
    assert detect_bounce_window(points) is None


def test_high_level_reconstruction_aligns_por_and_derives_kinematics() -> None:
    league = {"por_x_m": "13.0", "por_y_m": "0.1", "por_z_m": "1.5",
              "trajectory_json": json.dumps([
        _point(0, 0, 0, 0), _point(25, 0.5, 0.1, 0.1), _point(75, 1.5, 0.2, 0.15)
    ])}
    mocap = {"por_x_m": "12.6", "por_y_m": "-0.3", "por_z_m": "1.7"}
    result = reconstruct_league_continuation(league, mocap)
    np.testing.assert_allclose([result[0][a] for a in ("x", "y", "z")], [12.6, -0.3, 1.7])
    assert result[-1]["t_since_ms"] == 75
    assert all(key in result[0] for key in ("v", "vx", "vy", "vz", "a", "dir", "vert_angle"))


def test_csv_export_uses_selected_rank_and_reports_translation(tmp_path: Path) -> None:
    trajectory = json.dumps([_point(0, 0, 0, 0), _point(50, 1, 0, 0.1), _point(100, 2, 0, 0.15)])
    def write(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
            writer.writeheader(); writer.writerows(rows)
    league_path, mocap_path, matches_path, output = (tmp_path / name for name in ("league.csv", "mocap.csv", "matches.csv", "out.csv"))
    raw_fields = ["throw_id", "por_x_m", "por_y_m", "por_z_m", "trajectory_json"]
    write(league_path, raw_fields, [{"throw_id": "L01", "por_x_m": 1, "por_y_m": 2, "por_z_m": 3, "trajectory_json": trajectory}])
    write(mocap_path, raw_fields, [{"throw_id": "M01", "por_x_m": 12.6, "por_y_m": -0.2, "por_z_m": 1.8, "trajectory_json": trajectory}])
    write(matches_path, ["mocap_throw_id", "rank", "league_throw_id"], [
        {"mocap_throw_id": "M01", "rank": 1, "league_throw_id": "L01"},
        {"mocap_throw_id": "M01", "rank": 2, "league_throw_id": "L01"},
    ])
    assert reconstruct_matches(matches_path, league_path, mocap_path, output, rank=1) == 1
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["mocap_throw_id"] == "M01" and row["league_throw_id"] == "L01"
    assert row["match_rank"] == "1"
    assert math.isclose(float(row["translation_x_m"]), 11.6)
    assert "bounce_reconstruction_rmse_m" in row
    assert "bounce_reconstruction_max_error_m" in row
    assert json.loads(row["trajectory_json"])[-1]["t_since_ms"] == 100


def test_csv_export_reconstructs_all_match_ranks_by_default(tmp_path: Path) -> None:
    trajectory = json.dumps([
        _point(0, 0, 0, 0), _point(50, 1, 0, 0.1), _point(100, 2, 0, 0.15),
    ])

    def write(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    league_path, mocap_path, matches_path, output = (
        tmp_path / name for name in ("league.csv", "mocap.csv", "matches.csv", "out.csv")
    )
    raw_fields = ["throw_id", "por_x_m", "por_y_m", "por_z_m", "trajectory_json"]
    write(league_path, raw_fields, [
        {"throw_id": f"L{rank}", "por_x_m": 1, "por_y_m": 2, "por_z_m": 3,
         "trajectory_json": trajectory}
        for rank in range(1, 6)
    ])
    write(mocap_path, raw_fields, [
        {"throw_id": "M01", "por_x_m": 12.6, "por_y_m": -0.2, "por_z_m": 1.8,
         "trajectory_json": trajectory},
    ])
    write(matches_path, ["mocap_throw_id", "rank", "league_throw_id"], [
        {"mocap_throw_id": "M01", "rank": rank, "league_throw_id": f"L{rank}"}
        for rank in range(1, 6)
    ])

    assert reconstruct_matches(matches_path, league_path, mocap_path, output) == 5
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["match_rank"]) for row in rows] == [1, 2, 3, 4, 5]
    assert [row["league_throw_id"] for row in rows] == ["L1", "L2", "L3", "L4", "L5"]


def test_csv_export_can_skip_invalid_matches_and_report_them(tmp_path: Path) -> None:
    valid = json.dumps([
        _point(0, 0, 0, 0), _point(50, 1, 0, 0.1), _point(100, 2, 0, 0.15),
    ])
    invalid = json.dumps([
        _point(0, 0, 0, 0), _point(0, 1, 0, 0.1),
    ])

    def write(path: Path, fields: list[str], rows: list[dict], delimiter: str = ",") -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    league_path, mocap_path, matches_path, output = (
        tmp_path / name for name in ("league.csv", "mocap.csv", "matches.csv", "out.csv")
    )
    raw_fields = ["throw_id", "por_x_m", "por_y_m", "por_z_m", "trajectory_json"]
    write(league_path, raw_fields, [
        {"throw_id": "L1", "por_x_m": 1, "por_y_m": 2, "por_z_m": 3,
         "trajectory_json": valid},
        {"throw_id": "L2", "por_x_m": 1, "por_y_m": 2, "por_z_m": 3,
         "trajectory_json": invalid},
    ], delimiter=";")
    write(mocap_path, raw_fields, [
        {"throw_id": "M1", "por_x_m": 12.6, "por_y_m": -.2, "por_z_m": 1.8,
         "trajectory_json": valid},
    ], delimiter=";")
    write(matches_path, ["mocap_throw_id", "rank", "league_throw_id"], [
        {"mocap_throw_id": "M1", "rank": 1, "league_throw_id": "L1"},
        {"mocap_throw_id": "M1", "rank": 2, "league_throw_id": "L2"},
    ])

    assert reconstruct_matches(
        matches_path, league_path, mocap_path, output, skip_invalid=True,
    ) == 1
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["match_rank"], row["league_throw_id"]) for row in rows] == [("1", "L1")]
    errors = output.with_name("out_errors.csv")
    with errors.open(newline="", encoding="utf-8") as handle:
        error_rows = list(csv.DictReader(handle))
    assert len(error_rows) == 1
    assert error_rows[0]["league_throw_id"] == "L2"
    assert error_rows[0]["error_type"] == "ValueError"


def test_semicolon_csv_with_large_json_field_is_read_completely(tmp_path: Path) -> None:
    path = tmp_path / "raw_league.csv"
    long_trajectory = json.dumps([_point(i * 50, i, i / 10, i / 20) for i in range(2500)])
    assert len(long_trajectory) > 131_072
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["throw_id", "trajectory_json"], delimiter=";")
        writer.writeheader()
        writer.writerow({"throw_id": "first", "trajectory_json": long_trajectory})
        writer.writerow({"throw_id": "9082364", "trajectory_json": long_trajectory})
    row = load_raw_throw(path, "9082364")
    assert row["throw_id"] == "9082364"
    assert len(json.loads(row["trajectory_json"])) == 2500


def test_random_weight_run_resolution_and_output_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0007"
    run_dir.mkdir()
    matches = run_dir / "weighted_knn_matches.csv"
    matches.write_text("mocap_throw_id,rank,league_throw_id\nM1,1,L1\n", encoding="utf-8")
    weights = {"release_speed": 2.5, "release_angles": 1.5}
    (run_dir / "weights.json").write_text(json.dumps(weights), encoding="utf-8")
    resolved_matches, resolved_weights = resolve_random_weight_run(tmp_path, 7)
    assert resolved_matches == matches
    assert resolved_weights == weights
    assert output_path_for_run(Path("out/reconstructed.csv"), 7) == Path(
        "out/reconstructed_run_0007.csv"
    )
    assert output_path_for_run(Path("out/reconstructed_run_0007.csv"), 7) == Path(
        "out/reconstructed_run_0007.csv"
    )


def test_selects_joint_weight_importance_only_within_top_distance_runs(tmp_path: Path) -> None:
    rows = [
        {"run": 1, "balanced_mean_top1_distance": 1.0, "release_speed": 5.0, "release_angles": 1.0},
        {"run": 2, "balanced_mean_top1_distance": 1.1, "release_speed": 3.0, "release_angles": 3.0},
        {"run": 3, "balanced_mean_top1_distance": 1.2, "release_speed": 1.0, "release_angles": 5.0},
        {"run": 4, "balanced_mean_top1_distance": 99.0, "release_speed": 20.0, "release_angles": 20.0},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "random_search_summary.csv", index=False)
    run, importance, distance = select_random_weight_run(
        tmp_path, ["release_speed"], top_runs=3,
    )
    assert (run, importance, distance) == (1, 5.0, 1.0)
    run, importance, distance = select_random_weight_run(
        tmp_path, ["release_speed", "release_angles"], top_runs=3,
    )
    assert run == 2
    assert math.isclose(importance, 3.0)
    assert distance == 1.1
