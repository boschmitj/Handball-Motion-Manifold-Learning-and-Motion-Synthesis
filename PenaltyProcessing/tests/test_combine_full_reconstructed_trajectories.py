from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from combine_full_reconstructed_trajectories import combine_file  # noqa: E402


def test_combines_every_match_and_converts_pre_por_to_absolute(tmp_path: Path) -> None:
    raw = tmp_path / "raw_mocap.tsv"
    reconstructed = tmp_path / "reconstructed.csv"
    output = tmp_path / "full.csv"
    mocap_points = [
        {"frame": -2, "t_since_ms": -20, "x": -1, "y": 0.5, "z": 0.2},
        {"frame": -1, "t_since_ms": -10, "x": -0.5, "y": 0.2, "z": 0.1},
        {"frame": 0, "t_since_ms": 0, "x": 0, "y": 0, "z": 0},
        {"frame": 1, "t_since_ms": 10, "x": 0.5, "y": 0, "z": -0.1},
    ]
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t",
            fieldnames=["throw_id", "por_x_m", "por_y_m", "por_z_m", "trajectory_json"],
        )
        writer.writeheader()
        writer.writerow({"throw_id": "M1", "por_x_m": 10, "por_y_m": 2, "por_z_m": 1.5,
                         "trajectory_json": json.dumps(mocap_points)})
    reconstructed_points = [
        {"frame": 0, "t_since_ms": 0, "x": 10, "y": 2, "z": 1.5},
        {"frame": 1, "t_since_ms": 10, "x": 11, "y": 2.1, "z": 1.4},
    ]
    with reconstructed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["mocap_throw_id", "league_throw_id", "trajectory_json"]
        )
        writer.writeheader()
        for league_id in ("L1", "L2"):
            writer.writerow({"mocap_throw_id": "M1", "league_throw_id": league_id,
                             "trajectory_json": json.dumps(reconstructed_points)})

    assert combine_file(raw, reconstructed, output) == 2
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["mocap_throw_id"], row["league_throw_id"]) for row in rows] == [
        ("M1", "L1"), ("M1", "L2")
    ]
    points = json.loads(rows[0]["trajectory_json"])
    assert [point["frame"] for point in points] == [-2, -1, 0, 1]
    assert [point["t_since_ms"] for point in points] == [-20.0, -10.0, 0.0, 10.0]
    assert points[0]["x"] == 9.0 and points[0]["y"] == 2.5 and points[0]["z"] == 1.7
    assert points[2]["x"] == 10.0 and rows[0]["trajectory_point_count"] == "4"
