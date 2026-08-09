from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ball_trajectory import BallPoint
from release_detector_trajectory_based import _has_strong_direction_change, select_release_index


def make_point(ts: datetime, x: float, z: float, speed: float, accel: float) -> BallPoint:
    return BallPoint(
        local_dt=ts,
        ts_ms=None,
        x=x,
        y=0.0,
        z=z,
        speed=speed,
        accel=accel,
        direction=0.0,
    )


def test_select_release_index_prefers_projectile_segment() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points: list[BallPoint] = []

    # Early throw-like motion: increasing speed/acceleration
    for i in range(3):
        points.append(
            make_point(
                base + timedelta(milliseconds=100 * i),
                8.0 + i * 1.6,
                1.15 + 0.02 * i,
                3.0 + i * 0.8,
                2.0 + i * 0.3,
            )
        )

    # Projectile-like motion after release; x increases roughly linearly and z follows a parabola.
    for i in range(8):
        t = 0.1 * i
        z = 1.25 + 0.9 * t - 4.9 * t * t
        x = 13.0 + 5.0 * t
        points.append(
            make_point(
                base + timedelta(milliseconds=300 + 100 * i),
                x,
                z,
                6.0,
                0.2,
            )
        )

    idx = select_release_index(points, 0, len(points) - 1)
    assert idx is not None
    assert idx == 3, f"expected the release index near 3, got {idx}"


def test_select_release_index_uses_earliest_stable_projectile_segment() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points: list[BallPoint] = []

    # Pre-release motion is smooth enough to look projectile-like, but the true release is the first
    # point that is followed by a sufficiently long, stable projectile segment.
    for i in range(4):
        points.append(
            make_point(
                base + timedelta(milliseconds=100 * i),
                8.0 + i * 0.7,
                1.15 + 0.01 * i,
                3.0 + i * 0.2,
                0.5 + i * 0.1,
            )
        )

    for i in range(6):
        t = 0.1 * i
        z = 1.2 + 0.8 * t - 4.8 * t * t
        x = 13.0 + 5.0 * t
        points.append(
            make_point(
                base + timedelta(milliseconds=400 + 100 * i),
                x,
                z,
                6.0,
                0.2,
            )
        )

    idx = select_release_index(points, 0, len(points) - 1)
    assert idx is not None
    assert idx == 3, f"expected the earliest stable candidate, got {idx}"


def test_has_strong_direction_change_detects_reference_line_deflection() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = [
        BallPoint(local_dt=base + timedelta(milliseconds=100 * i), ts_ms=None, x=14.0 + 0.25 * i, y=0.0, z=1.0, speed=0.0, accel=0.0, direction=0.0)
        for i in range(4)
    ]
    points.append(BallPoint(local_dt=base + timedelta(milliseconds=500), ts_ms=None, x=15.2, y=0.8, z=1.0, speed=0.0, accel=0.0, direction=0.0))

    line = _has_strong_direction_change(points, 4, 0, reference_line=(0.0, 0.0, 0.05))
    assert line is True
