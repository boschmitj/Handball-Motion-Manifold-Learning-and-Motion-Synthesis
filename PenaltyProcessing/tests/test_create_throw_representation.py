from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from create_throw_representation import load_league_throws
from feature_extraction import COMMON_TIME_STEPS_MS, extract_features_from_row


def test_precomputed_league_csv_inserts_synthetic_por_with_half_frame_timing() -> None:
    trajectory = [
        {"t_local": "2024-01-01T12:00:00.000", "ts_ms": 0, "x": 12.0, "y": 0.0, "z": 1.4, "v": 10.0, "a": 0.0, "dir": 0.0},
        {"t_local": "2024-01-01T12:00:00.050", "ts_ms": 50, "x": 13.0, "y": 0.1, "z": 1.6, "v": 20.0, "a": 0.0, "dir": 0.0},
        {"t_local": "2024-01-01T12:00:00.100", "ts_ms": 100, "x": 14.0, "y": 0.2, "z": 1.7, "v": 20.0, "a": 0.0, "dir": 0.0},
        {"t_local": "2024-01-01T12:00:00.150", "ts_ms": 150, "x": 15.0, "y": 0.3, "z": 1.75, "v": 20.0, "a": 0.0, "dir": 0.0},
    ]
    release = {"t_local": "2024-01-01T12:00:00.025", "ts_ms": 25, "x": 12.5, "y": 0.05, "z": 1.5}
    row = {
        "id": "synthetic-test",
        "release_point_json": json.dumps(release),
        "trajectory_json": json.dumps(trajectory),
        "release_idx": "1",
        "first_projectile_idx": "1",
        "projectile_end_idx": "3",
        "release_speed_solved": "20.1",
        "release_direction_solved": "5.0",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "league.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), delimiter=";")
            writer.writeheader()
            writer.writerow(row)
        representations, _ = load_league_throws(Path("unused"), Path("unused"), precomputed_csv=csv_path)

    assert len(representations) == 1
    representation = representations[0]
    assert representation.release_timing_type == "half_frame_adjacent"
    assert representation.first_projectile_offset_ms == 25.0
    points = json.loads(representation.trajectory_json)
    assert [point["t_since_ms"] for point in points] == [0.0, 25.0, 75.0, 125.0]
    assert [point["frame"] for point in points] == [0.0, 0.5, 1.5, 2.5]
    assert (points[0]["x"], points[0]["y"], points[0]["z"]) == (0.0, 0.0, 0.0)
    assert all(math.isclose(actual, expected, abs_tol=1e-9) for actual, expected in zip(
        (points[1]["x"], points[1]["y"], points[1]["z"]), (0.5, 0.05, 0.1)
    ))

    features = extract_features_from_row(pd.Series({
        "throw_id": representation.throw_id,
        "source": representation.source,
        "trajectory_json": representation.trajectory_json,
        "sampling_rate_hz": representation.sampling_rate_hz,
    }))
    assert COMMON_TIME_STEPS_MS[:4] == [0, 25, 50, 75]
    assert features["disp_x_t0ms"] == 0.0
    assert features["disp_x_t25ms"] == 0.5
    assert features["disp_x_t50ms"] is None
    assert features["disp_x_t75ms"] == 1.5


def test_sparse_league_phase_and_dense_mocap_phase_align_through_none_columns() -> None:
    def point(t_ms: float, x: float) -> dict:
        return {"frame": t_ms / 50.0, "t_since_ms": t_ms, "x": x, "y": 0.0, "z": 0.0,
                "v": x, "a": 0.0, "dir": 0.0, "vx": x, "vy": 0.0, "vz": 0.0, "vert_angle": 0.0}

    shifted_league = pd.Series({
        "throw_id": "shifted", "source": "league", "sampling_rate_hz": 20.0,
        "release_timing_type": "half_frame_shifted", "first_projectile_offset_ms": 75.0,
        "trajectory_json": json.dumps([point(0.0, 0.0), point(75.0, 1.5), point(125.0, 2.5)]),
    })
    dense_mocap = pd.Series({
        "throw_id": "mocap", "source": "mocap", "sampling_rate_hz": 300.0,
        "trajectory_json": json.dumps([point(float(t), t / 50.0) for t in range(0, 151, 5)]),
    })
    league_features = extract_features_from_row(shifted_league)
    mocap_features = extract_features_from_row(dense_mocap)

    assert league_features["disp_x_t25ms"] is None
    assert league_features["disp_x_t50ms"] is None
    assert league_features["disp_x_t75ms"] == 1.5
    assert mocap_features["disp_x_t25ms"] == 0.5
    assert mocap_features["disp_x_t50ms"] == 1.0
    assert mocap_features["disp_x_t75ms"] == 1.5
