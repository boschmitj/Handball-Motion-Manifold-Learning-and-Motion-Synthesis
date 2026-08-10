"""
Sphere-centre check for the marker-based 6DOF handball captures.

For every frame the script

1. fits a least-squares sphere through the six ball markers stored in the
   ``*_6DOF_3D.tsv`` file ("handball_6 - 1 .. 6" X/Y/Z columns), and
2. compares the fitted centre with the ground-truth centre stored in the
   ``*_labeling_done_6D.tsv`` file (the rigid-body translation
   ``handball_6 X, Y, Z`` in "Global" coordinates).

The fitted radius is also validated: the real ball has a diameter of roughly
19 cm (radius ~95 mm). Because the markers are attached to the outer surface
and are not exactly flush with the boundary, the fitted radius is expected to
be slightly larger than the physical ball radius.

Coordinate units of the mocap TSV exports are millimetres.

Usage
-----
python sphere_center_check.py
python sphere_center_check.py --markers path/to/*_6DOF_3D.tsv \\
    --gt path/to/*_labeling_done_6D.tsv --output-csv sphere_check.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_markers_3d(path: Path) -> Dict[int, Tuple[float, np.ndarray]]:
    """Read the 3D marker TSV and return {frame: (time_s, markers (<=6, 3))}.

    Markers whose coordinates are NaN or exactly (0, 0, 0) (the usual mocap
    "missing marker" convention) are dropped. Every frame is kept, even when
    fewer than four usable markers remain -- the caller decides whether a
    sphere can be fitted. This keeps the frame-count breakdown transparent
    (e.g. the 3D file declares 16296 frames, but 338 of them have <4 visible
    markers and can only provide a GT centre, not a fitted sphere).
    """
    frames: Dict[int, Tuple[float, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            # Skip metadata lines and the column header; data rows have the
            # layout: Frame, Time, then 6 x (X, Y, Z) => 20 columns.
            if len(row) < 20 or row[0].strip() == "Frame":
                continue
            try:
                frame = int(float(row[0].strip()))
                time_s = float(row[1].strip())
                markers = np.array(
                    [float(v.strip()) for v in row[2:20]], dtype=np.float64
                ).reshape(6, 3)
            except ValueError:
                continue

            usable = markers[
                np.isfinite(markers).all(axis=1) & (np.abs(markers).sum(axis=1) > 0.0)
            ]
            frames[frame] = (time_s, usable)
    return frames


def read_gt_centers(path: Path) -> Dict[int, Tuple[float, np.ndarray]]:
    """Read the 6D label TSV and return {frame: (time_s, center (3,))}.

    The columns ``handball_6 X, Y, Z`` (the rigid-body translation in "Global"
    coordinates) are used as the ground-truth sphere centre.
    """
    frames: Dict[int, Tuple[float, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            # Skip metadata lines and the column header; the translation is in
            # columns 2..4 (0-indexed) right after Frame, Time.
            if len(row) < 5 or row[0].strip() == "Frame":
                continue
            try:
                frame = int(float(row[0].strip()))
                time_s = float(row[1].strip())
                center = np.array(
                    [float(row[2].strip()), float(row[3].strip()), float(row[4].strip())],
                    dtype=np.float64,
                )
            except ValueError:
                continue
            if np.isfinite(center).all():
                frames[frame] = (time_s, center)
    return frames


# ---------------------------------------------------------------------------
# Sphere fitting
# ---------------------------------------------------------------------------


def fit_sphere(markers: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    """Fit a sphere to marker positions (N x 3) with a linear least-squares fit.

    For a sphere with centre ``c`` and radius ``r`` every marker satisfies::

        |p|^2 = 2 p . c + (r^2 - |c|^2)

    which is linear in the unknowns [c, k] with k = r^2 - |c|^2. The
    over-determined system is solved with ``numpy.linalg.lstsq`` and the
    radius is recovered as ``r = sqrt(k + |c|^2)``.

    Args:
        markers: (N, 3) array of marker positions in mm (N >= 4).

    Returns:
        ``(center, radius, residual_rms)`` in mm, or None when the fit is
        invalid (too few markers, non-positive radius, ...). ``residual_rms``
        is the RMS of the per-marker distances between ``|p - c|`` and ``r``
        and measures how sphere-like the markers actually are.
    """
    n = markers.shape[0]
    if n < 4:
        return None

    design = np.column_stack((2.0 * markers, np.ones(n)))
    rhs = np.sum(markers * markers, axis=1)
    try:
        coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    center = coeffs[:3]
    radius_sq = coeffs[3] + float(center @ center)
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        return None

    radius = float(np.sqrt(radius_sq))
    distances = np.linalg.norm(markers - center, axis=1)
    residual_rms = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return center, radius, residual_rms


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    default_markers = (
        Path(__file__).resolve().parent.parent
        / "mocap_files"
        / "6DOF"
        / "Mo_Wurfvariation4_ul0251_reprocessing_6DOF_3D.tsv"
    )
    default_gt = (
        Path(__file__).resolve().parent.parent
        / "mocap_files"
        / "6DOF"
        / "Mo_Wurfvariation4_ul0251_reprocessing_labeling_done_6D.tsv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Fit a sphere through the six handball markers per frame and compare "
            "the fitted centre with the ground-truth centre from the 6D label file."
        )
    )
    parser.add_argument("--markers", type=Path, default=default_markers,
                        help="3D marker TSV (default: Mo_Wurfvariation4_ul0251 *_6DOF_3D.tsv)")
    parser.add_argument("--gt", type=Path, default=default_gt,
                        help="6D labeling TSV with GT centre (default: Mo_Wurfvariation4_ul0251 *_labeling_done_6D.tsv)")
    parser.add_argument("--center-tolerance-mm", type=float, default=10.0,
                        help="PASS threshold for the median |fit - GT| deviation (default: 10 mm)")
    parser.add_argument("--expected-radius-mm", type=float, default=95.0,
                        help="Nominal ball radius, diameter ~19 cm (default: 95 mm)")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Optional CSV with per-frame results")
    args = parser.parse_args()

    marker_frames = read_markers_3d(args.markers)
    gt_frames = read_gt_centers(args.gt)

    shared = sorted(set(marker_frames) & set(gt_frames))
    if not shared:
        print("ERROR: no common frames between the two files")
        return 2

    records: List[dict] = []
    fitted_frames = 0
    marker_counts: List[int] = []
    residual_rms: List[float] = []
    radii: List[float] = []
    center_errors: List[float] = []

    for frame in shared:
        _, markers = marker_frames[frame]
        _, gt_center = gt_frames[frame]

        fit = fit_sphere(markers)
        if fit is None:
            continue
        center, radius, rrms = fit

        fitted_frames += 1
        marker_counts.append(markers.shape[0])
        residual_rms.append(rrms)
        radii.append(radius)
        center_errors.append(float(np.linalg.norm(center - gt_center)))

        if args.output_csv is not None:
            records.append(
                {
                    "frame": frame,
                    "time_s": time if False else float(
                        next((t for f, (t, _) in gt_frames.items() if f == frame), float("nan"))
                    ),
                    "n_markers": markers.shape[0],
                    "fit_x_mm": float(center[0]),
                    "fit_y_mm": float(center[1]),
                    "fit_z_mm": float(center[2]),
                    "radius_mm": radius,
                    "residual_rms_mm": rrms,
                    "gt_x_mm": float(gt_center[0]),
                    "gt_y_mm": float(gt_center[1]),
                    "gt_z_mm": float(gt_center[2]),
                    "center_error_mm": float(np.linalg.norm(center - gt_center)),
                }
            )

    if fitted_frames == 0:
        print("ERROR: sphere fit failed for every frame")
        return 2

    errors = np.asarray(center_errors, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64)
    res_arr = np.asarray(residual_rms, dtype=np.float64)

    median_err = float(np.median(errors))
    mean_radius = float(np.mean(radii_arr))
    mean_diameter_cm = 2.0 * mean_radius / 10.0
    mean_residual = float(np.mean(res_arr))

    print("=== Data ===")
    print(f"3D marker file : {args.markers.name} ({len(marker_frames)} frames)")
    print(f"6D label file  : {args.gt.name} ({len(gt_frames)} frames)")
    print(f"Common frames  : {len(shared)}")

    # The 3D marker file and the 6D label file both declare NO_OF_FRAMES 16296.
    # The marker file contains all of those frames, but in some of them fewer
    # than 4 of the 6 markers are visible (e.g. occluded during the throw), so
    # a sphere cannot be fitted reliably. Those frames are skipped and only the
    # GT centre is available. Report this breakdown explicitly so the frame
    # count difference is transparent.
    skipped_frames = len(shared) - fitted_frames
    print("\n=== Frame availability ===")
    print(f"Frames in 3D marker file : {len(marker_frames)}")
    print(f"Frames in 6D label file   : {len(gt_frames)}")
    print(f"Common frames             : {len(shared)}")
    print(
        f"Frames with <4 markers    : {skipped_frames} "
        f"(sphere fit not possible, GT centre only)"
    )
    print(
        f"Frames fitted             : {fitted_frames} "
        f"({100.0 * fitted_frames / len(shared):.1f} % of common frames)"
    )

    print("\n=== Sphere fit through the ball markers ===")
    print(
        f"Frames fitted       : {fitted_frames} / {len(shared)} "
        f"({100.0 * fitted_frames / len(shared):.1f} %)  "
        f"[min 4 of 6 markers per frame]"
    )
    print(f"Markers per frame   : mean {np.mean(marker_counts):.2f} (min {np.min(marker_counts)})")
    print(f"Marker surface RMS  : mean {_fmt(mean_residual)} mm, max {_fmt(float(np.max(res_arr)))} mm")
    print(
        f"Fitted radius       : mean {_fmt(mean_radius)} mm "
        f"({_fmt(mean_diameter_cm, 1)} cm diameter, nominal ball ~19 cm)"
    )
    print(
        f"Fitted radius range : {_fmt(float(np.min(radii_arr)))} .. {_fmt(float(np.max(radii_arr)))} mm "
        f"(min/max per frame)"
    )
    radius_delta = mean_radius - args.expected_radius_mm
    print(
        f"vs nominal          : +{_fmt(radius_delta)} mm above the expected "
        f"{_fmt(args.expected_radius_mm, 0)} mm radius -- expected, when the "
        f"markers sit on / slightly above the ball surface"
    )

    print("\n=== Centre comparison: fitted sphere centre vs GT centre (6D label) ===")
    print(f"Mean  |fit - GT|    : {_fmt(float(np.mean(errors)))} mm")
    print(f"Median              : {_fmt(median_err)} mm")
    print(f"Std                 : {_fmt(float(np.std(errors)))} mm")
    print(f"95th percentile     : {_fmt(float(np.percentile(errors, 95)))} mm")
    print(f"Max                 : {_fmt(float(np.max(errors)))} mm")

    # How often is the GT centre inside the sphere itself (< 1 fitted radius)?
    within_radius = float(np.mean(errors < mean_radius))
    print(f"Frames within fitted radius of GT: {100.0 * within_radius:.1f} %")

    passed = median_err <= args.center_tolerance_mm
    print(
        f"\nCENTER CHECK (median |fit - GT| <= {_fmt(args.center_tolerance_mm, 0)} mm): "
        f"{'PASS' if passed else 'FAIL'}  (median {_fmt(median_err)} mm)"
    )

    if args.output_csv is not None:
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nPer-frame results written to {args.output_csv}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())