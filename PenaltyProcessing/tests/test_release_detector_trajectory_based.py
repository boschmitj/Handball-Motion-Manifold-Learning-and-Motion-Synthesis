from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ball_trajectory import BallPoint
from release_detector_trajectory_based import (
    ProjectileModel,
    _has_strong_direction_change,
    _has_deflection,
    detect_backward_projectile_segment,
    detect_simple_release_point,
    estimate_release_midpoint,
    estimate_release_solved,
    find_bounce_idx,
    select_release_index,
)


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
    assert idx == 4, f"expected the first projectile sample, got {idx}"


def test_has_strong_direction_change_detects_reference_line_deflection() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = [
        BallPoint(local_dt=base + timedelta(milliseconds=100 * i), ts_ms=None, x=14.0 + 0.25 * i, y=0.0, z=1.0, speed=0.0, accel=0.0, direction=0.0)
        for i in range(4)
    ]
    points.append(BallPoint(local_dt=base + timedelta(milliseconds=500), ts_ms=None, x=17.2, y=0.8, z=1.0, speed=0.0, accel=0.0, direction=0.0))

    line = _has_strong_direction_change(points, 4, reference_line=(0.0, 0.0, 0.05))
    assert line is True


def test_direction_change_before_goalkeeper_zone_is_not_a_deflection() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = [
        BallPoint(base + timedelta(milliseconds=50*i), None, 13.0 + 0.5*i,
                  0.0 if i < 4 else 1.0, 1.2, 0.0, 0.0, 0.0)
        for i in range(6)
    ]
    assert all(abs(point.x) <= 15.5 for point in points)
    assert _has_deflection(points, 0, len(points) - 1) is False


def test_backward_segment_uses_real_timestamps_and_stops_at_transition() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    offsets = [0, 47, 101]
    points = [make_point(base + timedelta(milliseconds=ms), 10.0 + i, 1.1, 0.0, 0.0) for i, ms in enumerate(offsets)]
    flight_offsets = [151, 203, 254, 309, 361, 416, 470, 523]
    for ms in flight_offsets:
        t = (ms - flight_offsets[0]) / 1000.0
        points.append(make_point(base + timedelta(milliseconds=ms), 13.0 + 18.0*t, 1.7 + 3.0*t - 4.905*t*t, 0.0, 0.0))

    result = detect_backward_projectile_segment(points, 0, len(points) - 1)
    assert result is not None
    assert result["last_non_projectile_idx"] == 2
    assert result["first_projectile_idx"] == 3
    assert abs(2.0 * result["projectile_model"].z[2] + 9.81) < 0.05

    midpoint = estimate_release_midpoint(points[2], points[3])
    solved = estimate_release_solved(points[2], points[3], result["anchor_model"])
    assert midpoint["alpha"] == 0.5
    assert solved["alpha"] == 0.5
    assert solved["solve_mode"] == "anchor_bisector"
    solved_point = np.asarray(solved["point"])
    before = np.asarray((points[2].x, points[2].y, points[2].z))
    after = np.asarray((points[3].x, points[3].y, points[3].z))
    assert abs(np.linalg.norm(solved_point - before) - np.linalg.norm(solved_point - after)) < 1e-9


def test_find_bounce_idx_searches_backward_in_goal_zone() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    zs = [0.8, 0.35, 0.08, 0.32, 0.55]
    points = [make_point(base + timedelta(milliseconds=50*i), 15.0 + i, z, 0.0, 0.0) for i, z in enumerate(zs)]
    assert find_bounce_idx(points, 0, len(points) - 1) == 2


def test_anchor_bisector_release_is_on_anchor_and_equidistant() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    before = BallPoint(base, None, 12.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    after = BallPoint(base + timedelta(milliseconds=50), None, 13.0, 2.0, 2.0, 0.0, 0.0, 0.0)
    anchor = ProjectileModel(
        origin_dt=base,
        x=np.asarray([14.0, 10.0]),
        y=np.asarray([1.0, 2.0]),
        z=np.asarray([1.0, 1.0, -4.905]),
    )
    solved = estimate_release_solved(before, after, anchor)
    x, y, z = solved["point"]
    assert solved["alpha"] == 0.5
    assert solved["solve_mode"] == "anchor_bisector"
    assert abs(y - (0.2*x - 1.8)) < 1e-9
    assert z == 1.5
    p = np.asarray(solved["point"])
    assert abs(np.linalg.norm(p - np.asarray((12.0, 0.0, 1.0))) - np.linalg.norm(p - np.asarray((13.0, 2.0, 2.0)))) < 1e-9


def test_four_point_seed_does_not_pull_in_pre_release_sample() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = [make_point(base + timedelta(milliseconds=50*i), 11.0 + i, 1.0, 0.0, 0.0) for i in range(3)]
    # Deliberately incompatible sample immediately before exactly four clean
    # post-gate projectile samples.
    points.append(make_point(base + timedelta(milliseconds=150), 13.8, 3.5, 0.0, 0.0))
    for i in range(4):
        t = 0.05 * i
        points.append(make_point(base + timedelta(milliseconds=200 + 50*i), 14.2 + 16*t, 1.8 + 2*t - 4.905*t*t, 0.0, 0.0))

    result = detect_backward_projectile_segment(points, 0, len(points) - 1)
    assert result is not None
    assert result["first_projectile_idx"] == 4
    assert result["last_non_projectile_idx"] == 3


def test_three_point_anchor_is_used_and_marked_low_confidence() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = [make_point(base + timedelta(milliseconds=50*i), 10.0 + 0.4*i, 0.8, 0.0, 0.0) for i in range(4)]
    for i in range(3):
        t = 0.05 * i
        points.append(make_point(base + timedelta(milliseconds=200 + 50*i), 14.2 + 18*t, 1.7 + 2*t - 4.905*t*t, 0.0, 0.0))

    result = detect_backward_projectile_segment(points, 0, len(points) - 1)
    assert result is not None
    assert result["first_projectile_idx"] == 4
    assert result["por_anchor_point_count"] == 3
    assert result["por_anchor_low_confidence"] is True


def test_spatial_prior_moves_too_late_release_interval_one_sample_earlier() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = [
        BallPoint(base + timedelta(milliseconds=50*i), None, x, 0.0,
                  1.1 + 0.025*i, 0.0, 0.0, 0.0)
        for i, x in enumerate([11.0, 11.7, 12.4, 13.1, 13.8])
    ]
    for i in range(9):
        t = 0.05*i
        points.append(BallPoint(base + timedelta(milliseconds=250 + 50*i), None,
                                14.5 + 16*t, 0.0, 1.7 + 2*t - 4.905*t*t,
                                0.0, 0.0, 0.0))

    result = detect_simple_release_point(
        {"id": "spatial-prior", "timestamp_local_timezone": "2024-01-01T12:00:00"},
        Path("."), fixture_path=Path("unused"), points=points,
    )
    assert result["last_non_projectile_idx"] == 4
    assert result["first_projectile_idx"] == 5
    assert result["release_interval_start_idx"] == 3
    assert result["release_interval_end_idx"] == 4
    assert result["release_boundary_shift"] == -1
    assert result["release_distance_before_prior"] < 6.0
    assert result["release_distance_after_prior"] >= 6.4


def test_deflection_filter_does_not_require_projectile_release_fit() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    points = []
    for i in range(9):
        points.append(BallPoint(
            local_dt=base + timedelta(milliseconds=50*i), ts_ms=None,
            x=11.0 + i, y=0.0 if i < 7 else 1.0, z=1.4,
            speed=0.0, accel=0.0, direction=0.0,
        ))
    assert _has_deflection(points, 0, len(points) - 1) is True
