from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse indexing and row extraction logic from the interactive TSV viewer.
VIS_DIR = Path(__file__).resolve().parents[1] / "visualization"
if str(VIS_DIR) not in sys.path:
    sys.path.append(str(VIS_DIR))

from tsv_frame_viewer import (  # type: ignore
    fit_sphere,
    index_ball_3d_file,
    index_ball_6d_file,
    index_skeleton_file,
    parse_ball_markers_cached,
    parse_skeleton_row_cached,
)

from ball_trajectory import BallPoint
from release_detector_trajectory_based import select_release_index

FS_HZ = 300.0
DT_S = 1.0 / FS_HZ

RIGHT_HAND_FINGER_PREFIXES = (
    "RightHandThumb",
    "RightHandIndex",
    "RightHandMiddle",
    "RightHandRing",
    "RightHandPinky",
)
RIGHT_HAND_INNER_PREFIX = "RightInHand"


@dataclass
class BallSample:
    frame: int
    time_s: float
    center_mm: Optional[np.ndarray]
    valid: bool
    reason: str
    n_markers: Optional[int] = None


@dataclass
class ThrowSegment:
    start_frame: int
    end_frame: int
    method1_por_frame: int
    baseline_mean_mm: Optional[float]
    baseline_std_mm: Optional[float]
    distance_at_release_mm: Optional[float]


@dataclass
class MethodResult:
    por_frame: Optional[int]
    velocity_mm_s: Optional[float]
    acceleration_mm_s2: Optional[float]
    direction_deg: Optional[float]
    coordinate_mm: Optional[np.ndarray]


@dataclass
class Kinematics:
    frames: List[int]
    positions_mm: List[np.ndarray]
    velocity_vec_mm_s: List[Optional[np.ndarray]]
    speed_mm_s: List[Optional[float]]
    acceleration_vec_mm_s2: List[Optional[np.ndarray]]
    acceleration_mm_s2: List[Optional[float]]
    tangential_accel_mm_s2: List[Optional[float]]
    direction_deg: List[Optional[float]]


@lru_cache(maxsize=2048)
def _read_row_at_offset(path_str: str, offset: int) -> List[str]:
    path = Path(path_str)
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline().decode("utf-8", errors="replace").rstrip("\r\n")
    return line.split("\t")


def _parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_gt_6d_invalid_row(row: Sequence[str], min_numeric_payload: int = 6) -> bool:
    if len(row) < 9:
        return True

    numeric_payload: List[float] = []
    for raw in row[2:]:
        value = _parse_float(raw)
        if value is not None and np.isfinite(value):
            numeric_payload.append(float(value))

    if len(numeric_payload) < min_numeric_payload:
        return True

    return bool(np.all(np.isclose(np.asarray(numeric_payload, dtype=np.float64), 0.0)))


def _extract_right_hand_points(positions: Dict[str, np.ndarray]) -> List[np.ndarray]:
    selected: List[np.ndarray] = []
    for name, point in positions.items():
        if name.startswith(RIGHT_HAND_INNER_PREFIX):
            selected.append(point)
            continue
        if name.startswith(RIGHT_HAND_FINGER_PREFIXES):
            selected.append(point)
    return selected


def _mean_hand_ball_distance_mm(positions: Dict[str, np.ndarray], ball_center_mm: np.ndarray) -> Optional[float]:
    points = _extract_right_hand_points(positions)
    if not points:
        return None
    dists = [float(np.linalg.norm(point - ball_center_mm)) for point in points]
    return float(np.mean(dists)) if dists else None


def _load_ball_sample_from_fit(frame: int, time_s: float, ball_3d_index) -> BallSample:
    offset = ball_3d_index.frame_to_offset.get(frame)
    if offset is None:
        return BallSample(frame=frame, time_s=time_s, center_mm=None, valid=False, reason="missing 3D marker row")

    _, _, markers = parse_ball_markers_cached(str(ball_3d_index.path), offset)
    n_markers = int(markers.shape[0])
    if n_markers < 4:
        return BallSample(
            frame=frame,
            time_s=time_s,
            center_mm=None,
            valid=False,
            reason="<4 non-zero marker points",
            n_markers=n_markers,
        )

    fit = fit_sphere(markers)
    if fit is None:
        return BallSample(
            frame=frame,
            time_s=time_s,
            center_mm=None,
            valid=False,
            reason="sphere fit failed despite >=4 markers",
            n_markers=n_markers,
        )

    center, _, _ = fit
    if not np.isfinite(center).all():
        return BallSample(
            frame=frame,
            time_s=time_s,
            center_mm=None,
            valid=False,
            reason="fitted centre not finite",
            n_markers=n_markers,
        )

    return BallSample(
        frame=frame,
        time_s=time_s,
        center_mm=np.asarray(center, dtype=np.float64),
        valid=True,
        reason=f"fit from {n_markers} markers",
        n_markers=n_markers,
    )


def _load_ball_sample_from_gt(frame: int, time_s: float, ball_6d_index) -> BallSample:
    offset = ball_6d_index.frame_to_offset.get(frame)
    if offset is None:
        return BallSample(frame=frame, time_s=time_s, center_mm=None, valid=False, reason="missing 6D row")

    row = _read_row_at_offset(str(ball_6d_index.path), offset)
    if len(row) < 5:
        return BallSample(frame=frame, time_s=time_s, center_mm=None, valid=False, reason="short 6D row")

    if _is_gt_6d_invalid_row(row):
        return BallSample(frame=frame, time_s=time_s, center_mm=None, valid=False, reason="6D payload all zero")

    cx = _parse_float(row[2])
    cy = _parse_float(row[3])
    cz = _parse_float(row[4])
    if cx is None or cy is None or cz is None:
        return BallSample(frame=frame, time_s=time_s, center_mm=None, valid=False, reason="invalid GT centre columns")

    center = np.array([cx, cy, cz], dtype=np.float64)
    if not np.isfinite(center).all():
        return BallSample(frame=frame, time_s=time_s, center_mm=None, valid=False, reason="GT centre not finite")

    return BallSample(frame=frame, time_s=time_s, center_mm=center, valid=True, reason="GT 6D centre")


def _compute_kinematics(samples: Sequence[BallSample], fs_hz: float) -> Kinematics:
    frames = [sample.frame for sample in samples]
    positions_mm = [np.asarray(sample.center_mm, dtype=np.float64) for sample in samples if sample.center_mm is not None]

    n = len(frames)
    velocity_vec: List[Optional[np.ndarray]] = [None] * n
    speed: List[Optional[float]] = [None] * n
    acceleration_vec: List[Optional[np.ndarray]] = [None] * n
    acceleration_mag: List[Optional[float]] = [None] * n
    tangential_accel: List[Optional[float]] = [None] * n
    direction_deg: List[Optional[float]] = [None] * n

    if n < 2:
        return Kinematics(
            frames=frames,
            positions_mm=positions_mm,
            velocity_vec_mm_s=velocity_vec,
            speed_mm_s=speed,
            acceleration_vec_mm_s2=acceleration_vec,
            acceleration_mm_s2=acceleration_mag,
            tangential_accel_mm_s2=tangential_accel,
            direction_deg=direction_deg,
        )

    for i in range(n):
        if i == 0:
            a_idx, b_idx = 0, 1
        elif i == n - 1:
            a_idx, b_idx = n - 2, n - 1
        else:
            a_idx, b_idx = i - 1, i + 1

        frame_delta = frames[b_idx] - frames[a_idx]
        if frame_delta <= 0:
            continue
        dt = frame_delta / fs_hz
        vel = (positions_mm[b_idx] - positions_mm[a_idx]) / dt
        velocity_vec[i] = vel
        speed[i] = float(np.linalg.norm(vel))
        direction_deg[i] = float(math.degrees(math.atan2(vel[1], vel[0])))

    for i in range(n):
        if i == 0:
            a_idx, b_idx = 0, 1
        elif i == n - 1:
            a_idx, b_idx = n - 2, n - 1
        else:
            a_idx, b_idx = i - 1, i + 1

        va = velocity_vec[a_idx]
        vb = velocity_vec[b_idx]
        if va is None or vb is None:
            continue

        frame_delta = frames[b_idx] - frames[a_idx]
        if frame_delta <= 0:
            continue
        dt = frame_delta / fs_hz
        acc = (vb - va) / dt
        acceleration_vec[i] = acc
        acceleration_mag[i] = float(np.linalg.norm(acc))

        sa = speed[a_idx]
        sb = speed[b_idx]
        if sa is not None and sb is not None:
            tangential_accel[i] = float((sb - sa) / dt)

    return Kinematics(
        frames=frames,
        positions_mm=positions_mm,
        velocity_vec_mm_s=velocity_vec,
        speed_mm_s=speed,
        acceleration_vec_mm_s2=acceleration_vec,
        acceleration_mm_s2=acceleration_mag,
        tangential_accel_mm_s2=tangential_accel,
        direction_deg=direction_deg,
    )


def _method_result_at_frame(kinematics: Kinematics, frame: Optional[int]) -> MethodResult:
    if frame is None:
        return MethodResult(None, None, None, None, None)

    try:
        idx = kinematics.frames.index(frame)
    except ValueError:
        return MethodResult(None, None, None, None, None)

    return MethodResult(
        por_frame=frame,
        velocity_mm_s=kinematics.speed_mm_s[idx],
        acceleration_mm_s2=kinematics.acceleration_mm_s2[idx],
        direction_deg=kinematics.direction_deg[idx],
        coordinate_mm=np.asarray(kinematics.positions_mm[idx], dtype=np.float64),
    )


def _detect_method2_por_frame(kinematics: Kinematics) -> Optional[int]:
    if not kinematics.frames:
        return None

    valid_speed_indices = [idx for idx, value in enumerate(kinematics.speed_mm_s) if value is not None]
    if not valid_speed_indices:
        return None

    zero_crossing_candidates: List[int] = []
    for idx in range(1, len(kinematics.frames)):
        prev_a = kinematics.tangential_accel_mm_s2[idx - 1]
        curr_a = kinematics.tangential_accel_mm_s2[idx]
        if prev_a is None or curr_a is None:
            continue
        if prev_a > 0.0 and curr_a <= 0.0:
            zero_crossing_candidates.append(idx)

    if zero_crossing_candidates:
        best = max(
            zero_crossing_candidates,
            key=lambda idx: (kinematics.speed_mm_s[idx] if kinematics.speed_mm_s[idx] is not None else -1e12),
        )
        return kinematics.frames[best]

    best = max(
        valid_speed_indices,
        key=lambda idx: (kinematics.speed_mm_s[idx] if kinematics.speed_mm_s[idx] is not None else -1e12),
    )
    return kinematics.frames[best]


def _detect_method3_por_frame(valid_window_samples: Sequence[BallSample]) -> Optional[int]:
    if len(valid_window_samples) < 5:
        return None

    origin = datetime(1970, 1, 1)
    points_m: List[BallPoint] = []
    for sample in valid_window_samples:
        if sample.center_mm is None:
            continue
        center_m = np.asarray(sample.center_mm, dtype=np.float64) / 1000.0
        points_m.append(
            BallPoint(
                local_dt=origin + timedelta(seconds=sample.time_s),
                ts_ms=int(round(sample.time_s * 1000.0)),
                x=float(center_m[0]),
                y=float(center_m[1]),
                z=float(center_m[2]),
                speed=float("nan"),
                accel=float("nan"),
                direction=None,
            )
        )

    if len(points_m) < 5:
        return None

    idx = select_release_index(points_m, 0, len(points_m) - 1)
    if idx is None or idx < 0 or idx >= len(valid_window_samples):
        return None
    return valid_window_samples[idx].frame


def _detect_pre_impact_frame(
    valid_samples: Sequence[BallSample],
    ground_hit_z_mm: float,
    y_reversal_epsilon_mm: float = 0.0,
) -> int:
    if not valid_samples:
        raise ValueError("Cannot detect pre-impact frame without valid ball samples")
    if len(valid_samples) == 1:
        return valid_samples[0].frame

    for idx in range(1, len(valid_samples) - 1):
        prev_sample = valid_samples[idx - 1]
        curr_sample = valid_samples[idx]
        next_sample = valid_samples[idx + 1]

        prev_center = np.asarray(prev_sample.center_mm, dtype=np.float64)
        curr_center = np.asarray(curr_sample.center_mm, dtype=np.float64)
        next_center = np.asarray(next_sample.center_mm, dtype=np.float64)

        dy_prev = curr_center[1] - prev_center[1]
        dy_next = next_center[1] - curr_center[1]
        y_reversal = (dy_prev > y_reversal_epsilon_mm) and (dy_next < -y_reversal_epsilon_mm)
        if y_reversal:
            return prev_sample.frame

        ground_hit = (curr_center[2] <= ground_hit_z_mm) and (prev_center[2] <= ground_hit_z_mm)
        if ground_hit:
            return prev_sample.frame

    return valid_samples[-1].frame


def _collect_window_samples(
    ball_by_frame: Dict[int, BallSample],
    all_frames: Sequence[int],
    start_frame: int,
    end_frame: int,
) -> Tuple[List[BallSample], int]:
    samples: List[BallSample] = []
    invalid_count = 0
    for frame in all_frames:
        if frame < start_frame or frame > end_frame:
            continue
        sample = ball_by_frame.get(frame)
        if sample is None:
            continue
        if sample.valid:
            samples.append(sample)
        else:
            invalid_count += 1
    return samples, invalid_count


def detect_throw_segments(
    frames: Sequence[int],
    positions_by_frame: Dict[int, Dict[str, np.ndarray]],
    ball_by_frame: Dict[int, BallSample],
    reacquire_distance_mm: float,
    sigma_multiplier: float,
    baseline_min_samples: int,
    baseline_std_floor_mm: float,
) -> List[ThrowSegment]:
    segments: List[ThrowSegment] = []

    holding_ball = False
    start_frame: Optional[int] = None
    baseline_values: List[float] = []

    for frame in frames:
        positions = positions_by_frame.get(frame, {})
        ball = ball_by_frame.get(frame)
        if ball is None or not ball.valid or ball.center_mm is None:
            # Invalid ball samples preserve holding state.
            continue

        distance = _mean_hand_ball_distance_mm(positions, ball.center_mm)
        if distance is None:
            continue

        if not holding_ball:
            if distance <= reacquire_distance_mm:
                holding_ball = True
                start_frame = frame
                baseline_values = [distance]
            continue

        baseline_values.append(distance)
        baseline_mean = float(np.mean(baseline_values)) if baseline_values else None
        baseline_std = float(np.std(baseline_values)) if baseline_values else None
        if baseline_std is not None:
            baseline_std = max(baseline_std, baseline_std_floor_mm)

        is_release = False
        if (
            baseline_mean is not None
            and baseline_std is not None
            and len(baseline_values) >= baseline_min_samples
            and distance > baseline_mean + sigma_multiplier * baseline_std
        ):
            is_release = True

        if not is_release:
            continue

        holding_ball = False
        if start_frame is not None and frame > start_frame:
            segments.append(
                ThrowSegment(
                    start_frame=start_frame,
                    end_frame=frame,
                    method1_por_frame=frame,
                    baseline_mean_mm=baseline_mean,
                    baseline_std_mm=baseline_std,
                    distance_at_release_mm=distance,
                )
            )
        start_frame = None
        baseline_values = []

    return segments


def _np_to_json_array(value: Optional[np.ndarray], digits: int = 3) -> str:
    if value is None:
        return ""
    arr = np.asarray(value, dtype=np.float64)
    rounded = [round(float(v), digits) for v in arr.tolist()]
    return json.dumps(rounded, separators=(",", ":"))


def run_pipeline(
    skeleton_path: Path,
    ball_3d_path: Path,
    ball_6d_path: Path,
    ball_source: str,
    output_csv: Path,
    reacquire_distance_mm: float,
    sigma_multiplier: float,
    baseline_min_samples: int,
    baseline_std_floor_mm: float,
    ground_hit_z_mm: float,
) -> Path:
    skeleton_index = index_skeleton_file(skeleton_path)
    ball_3d_index = index_ball_3d_file(ball_3d_path)
    ball_6d_index = index_ball_6d_file(ball_6d_path)

    if ball_source == "gt":
        shared_frames = sorted(set(skeleton_index.frames) & set(ball_6d_index.frames))
    else:
        shared_frames = sorted(set(skeleton_index.frames) & set(ball_3d_index.frames))

    if not shared_frames:
        raise ValueError("No common frames between skeleton and selected ball source")

    positions_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    ball_by_frame: Dict[int, BallSample] = {}

    segment_names = tuple(skeleton_index.segment_names or ())
    for frame in shared_frames:
        skeleton_offset = skeleton_index.frame_to_offset.get(frame)
        if skeleton_offset is None:
            continue
        _, time_s, positions = parse_skeleton_row_cached(str(skeleton_index.path), skeleton_offset, segment_names)
        positions_by_frame[frame] = positions

        if ball_source == "gt":
            sample = _load_ball_sample_from_gt(frame, time_s, ball_6d_index)
        else:
            sample = _load_ball_sample_from_fit(frame, time_s, ball_3d_index)
        ball_by_frame[frame] = sample

    segments = detect_throw_segments(
        frames=shared_frames,
        positions_by_frame=positions_by_frame,
        ball_by_frame=ball_by_frame,
        reacquire_distance_mm=reacquire_distance_mm,
        sigma_multiplier=sigma_multiplier,
        baseline_min_samples=baseline_min_samples,
        baseline_std_floor_mm=baseline_std_floor_mm,
    )

    records: List[Dict[str, object]] = []
    for segment_idx, segment in enumerate(segments):
        next_start = segments[segment_idx + 1].start_frame if segment_idx + 1 < len(segments) else shared_frames[-1] + 1
        segment_search_end = max(segment.end_frame, next_start - 1)

        valid_window_samples, invalid_count = _collect_window_samples(
            ball_by_frame=ball_by_frame,
            all_frames=shared_frames,
            start_frame=segment.start_frame,
            end_frame=segment_search_end,
        )

        if not valid_window_samples:
            pre_impact_frame = None
            method1 = MethodResult(None, None, None, None, None)
            method2 = MethodResult(None, None, None, None, None)
            method3 = MethodResult(None, None, None, None, None)
            valid_count = 0
        else:
            pre_impact_frame = _detect_pre_impact_frame(valid_window_samples, ground_hit_z_mm=ground_hit_z_mm)
            pre_impact_samples = [sample for sample in valid_window_samples if sample.frame <= pre_impact_frame]
            valid_count = len(pre_impact_samples)

            # Method 1 is defined by the holding-state transition and therefore
            # uses kinematics from the full valid segment window.
            full_kinematics = _compute_kinematics(valid_window_samples, fs_hz=FS_HZ)
            method1 = _method_result_at_frame(full_kinematics, segment.method1_por_frame)

            # Methods 2 and 3 search only up to the pre-impact boundary.
            kinematics = _compute_kinematics(pre_impact_samples, fs_hz=FS_HZ)

            method2_frame = _detect_method2_por_frame(kinematics)
            method2 = _method_result_at_frame(kinematics, method2_frame)

            method3_frame = _detect_method3_por_frame(pre_impact_samples)
            method3 = _method_result_at_frame(kinematics, method3_frame)

        records.append(
            {
                "PoR_method1": method1.por_frame,
                "acceleration_method1": method1.acceleration_mm_s2,
                "velocity_method1": method1.velocity_mm_s,
                "direction_method1": method1.direction_deg,
                "coordinate_method1": _np_to_json_array(method1.coordinate_mm),
                "PoR_method2": method2.por_frame,
                "acceleration_method2": method2.acceleration_mm_s2,
                "velocity_method2": method2.velocity_mm_s,
                "direction_method2": method2.direction_deg,
                "coordinate_method2": _np_to_json_array(method2.coordinate_mm),
                "PoR_method3": method3.por_frame,
                "acceleration_method3": method3.acceleration_mm_s2,
                "velocity_method3": method3.velocity_mm_s,
                "direction_method3": method3.direction_deg,
                "coordinate_method3": _np_to_json_array(method3.coordinate_mm),
                "throw_segment": f"{segment.start_frame}-{segment.end_frame}",
                "segment_start": segment.start_frame,
                "holding_por_frame": segment.end_frame,
                "pre_impact_frame": pre_impact_frame,
                "ball_valid_samples_preimpact": valid_count,
                "ball_invalid_samples_in_segment": invalid_count,
                "method1_baseline_mean_mm": segment.baseline_mean_mm,
                "method1_baseline_std_mm": segment.baseline_std_mm,
                "method1_release_distance_mm": segment.distance_at_release_mm,
                "ball_source": ball_source,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PoR_method1",
        "acceleration_method1",
        "velocity_method1",
        "direction_method1",
        "coordinate_method1",
        "PoR_method2",
        "acceleration_method2",
        "velocity_method2",
        "direction_method2",
        "coordinate_method2",
        "PoR_method3",
        "acceleration_method3",
        "velocity_method3",
        "direction_method3",
        "coordinate_method3",
        "throw_segment",
        "segment_start",
        "holding_por_frame",
        "pre_impact_frame",
        "ball_valid_samples_preimpact",
        "ball_invalid_samples_in_segment",
        "method1_baseline_mean_mm",
        "method1_baseline_std_mm",
        "method1_release_distance_mm",
        "ball_source",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(records)

    print(f"Shared frames: {len(shared_frames)}")
    print(f"Detected throws: {len(records)}")
    for idx, row in enumerate(records, start=1):
        print(
            f"Throw {idx}: start={row['segment_start']} "
            f"holding_por={row['holding_por_frame']} "
            f"pre_impact={row['pre_impact_frame']} "
            f"m1={row['PoR_method1']} m2={row['PoR_method2']} m3={row['PoR_method3']} "
            f"valid={row['ball_valid_samples_preimpact']} invalid={row['ball_invalid_samples_in_segment']}"
        )
    print(f"Comparison CSV written to {output_csv}")
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Detect point-of-release (PoR) for mocap throws with three methods: "
            "(1) 6-sigma hand-ball distance spike, (2) max-velocity/acceleration-zero-crossing, "
            "and (3) trajectory-based detector adapted from release_detector_trajectory_based."
        )
    )
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=root / "mocap_files" / "skeleton" / "Mo_Wurfvariation4_ul0251_reprocessing_labeling_done_s_Josh.tsv",
        help="Skeleton TSV with segment points",
    )
    parser.add_argument(
        "--ball-3d",
        type=Path,
        default=root / "mocap_files" / "6DOF" / "Mo_Wurfvariation4_ul0251_reprocessing_6DOF_3D.tsv",
        help="Ball 3D marker TSV (for computed sphere centre)",
    )
    parser.add_argument(
        "--ball-6d",
        type=Path,
        default=root / "mocap_files" / "6DOF" / "Mo_Wurfvariation4_ul0251_reprocessing_labeling_done_6D.tsv",
        help="Ball 6D TSV (for GT centre)",
    )
    parser.add_argument(
        "--ball-source",
        choices=("gt", "fit"),
        default="gt",
        help="Ball centre source: gt=6D, fit=6DOF sphere fit",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "out" / "por_method_comparison.csv",
        help="Output comparison CSV",
    )
    parser.add_argument(
        "--reacquire-distance-mm",
        type=float,
        default=250.0,
        help="Distance threshold for False->True holding transition",
    )
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=6.0,
        help="Sigma multiplier for Method-1 release detection",
    )
    parser.add_argument(
        "--baseline-min-samples",
        type=int,
        default=20,
        help="Minimum holding samples required before Method-1 release can trigger",
    )
    parser.add_argument(
        "--baseline-std-floor-mm",
        type=float,
        default=1.0,
        help="Lower bound for baseline std in Method-1 to avoid divide-by-noise",
    )
    parser.add_argument(
        "--ground-hit-z-mm",
        type=float,
        default=180.0,
        help="Ground-hit z threshold in mm (existing detector uses 0.18 m)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(
        skeleton_path=args.skeleton,
        ball_3d_path=args.ball_3d,
        ball_6d_path=args.ball_6d,
        ball_source=args.ball_source,
        output_csv=args.output_csv,
        reacquire_distance_mm=args.reacquire_distance_mm,
        sigma_multiplier=args.sigma_multiplier,
        baseline_min_samples=args.baseline_min_samples,
        baseline_std_floor_mm=args.baseline_std_floor_mm,
        ground_hit_z_mm=args.ground_hit_z_mm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
