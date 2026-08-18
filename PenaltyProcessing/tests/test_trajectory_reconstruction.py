from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_reconstruction import (  # noqa: E402
    BounceWindow, align_trajectory_to_por, detect_bounce_window, estimate_bounce_time,
    insert_bounce_event, load_raw_throw, parse_trajectory_json, reconstruct_league_continuation,
    reconstruct_matches, upsample_at_times, upsample_trajectory,
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
    assert reconstruct_matches(matches_path, league_path, mocap_path, output) == 1
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["mocap_throw_id"] == "M01" and row["league_throw_id"] == "L01"
    assert math.isclose(float(row["translation_x_m"]), 11.6)
    assert "bounce_reconstruction_rmse_m" in row
    assert "bounce_reconstruction_max_error_m" in row
    assert json.loads(row["trajectory_json"])[-1]["t_since_ms"] == 100


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
