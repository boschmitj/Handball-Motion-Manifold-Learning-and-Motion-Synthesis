#!/usr/bin/env python3
"""Replace the pre-release part of reconstructed_full with Mocap GT centres.

The 6D rigid-body translations are transformed to canonical League coordinates
with the same convention as ``create_throw_representation``:

    league_x = raw_y / 1000 + 12.6
    league_y = -raw_x / 1000
    league_z = raw_z / 1000

For recordings without a 6D centre export, and for individual zero/missing 6D
samples, the existing marker-sphere-fit points in the input are retained.  The
post-release reconstructed League trajectory is translated by the difference
between its old PoR and the GT PoR so the two portions remain continuous.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


TRANSLATION_X_M = 12.6


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    _set_csv_field_limit()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _read_gt_centers(path: Path) -> dict[int, tuple[float, float, float]]:
    centers: dict[int, tuple[float, float, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 5 or row[0].strip() == "Frame":
                continue
            try:
                frame = int(float(row[0]))
                center = tuple(float(value) for value in row[2:5])
            except ValueError:
                continue
            if not all(math.isfinite(value) for value in center):
                continue
            # Qualisys uses an all-zero translation for an untracked body.
            if all(value == 0.0 for value in center):
                continue
            centers[frame] = center
    return centers


def _to_league(center_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    raw_x, raw_y, raw_z = center_mm
    return (
        raw_y / 1000.0 + TRANSLATION_X_M,
        -raw_x / 1000.0,
        raw_z / 1000.0,
    )


def _recording_prefix(throw_id: str) -> str:
    marker = "_seg"
    if marker not in throw_id:
        raise ValueError(f"Cannot determine recording from throw ID {throw_id!r}")
    return throw_id.rsplit(marker, 1)[0]


def _find_gt_files(mocap_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for recording in sorted(mocap_root.glob("throw_*")):
        sixdof = recording / "6DOF"
        candidates = [
            *sixdof.glob("*3D_6D*.tsv"),
            *sixdof.glob("*labeling_done_6D.tsv"),
        ]
        candidates = sorted(
            path for path in set(candidates) if "XY_swapped" not in path.name
        )
        if candidates:
            result[recording.name] = candidates[0]
    return result


def _replace_points(
    points: list[dict[str, Any]],
    *,
    por_frame: int,
    centers: Mapping[int, tuple[float, float, float]],
) -> tuple[list[dict[str, Any]], int, bool]:
    if not points:
        raise ValueError("trajectory_json contains no points")

    gt_por_mm = centers.get(por_frame)
    gt_por = _to_league(gt_por_mm) if gt_por_mm is not None else None
    por_point = next(
        (point for point in points if int(point["frame"]) == 0), None
    )
    has_gt_por = gt_por is not None and por_point is not None
    shift = (
        tuple(gt_por[i] - float(por_point[axis]) for i, axis in enumerate("xyz"))
        if has_gt_por
        else (0.0, 0.0, 0.0)
    )

    replaced = 0
    output: list[dict[str, Any]] = []
    for original in points:
        point = dict(original)
        relative_frame = int(point["frame"])
        if relative_frame < 0:
            center_mm = centers.get(por_frame + relative_frame)
            if center_mm is not None:
                center = _to_league(center_mm)
                for axis, value in zip("xyz", center):
                    point[axis] = value
                replaced += 1
        elif has_gt_por:
            for i, axis in enumerate("xyz"):
                point[axis] = float(point[axis]) + shift[i]
        output.append(point)
    return output, replaced, has_gt_por


def combine_file(
    reconstructed_full: Path,
    throw_index: Path,
    mocap_root: Path,
    output: Path,
) -> tuple[int, int, list[str]]:
    rows = _read_csv(reconstructed_full)
    index_rows = _read_csv(throw_index, delimiter=";")
    index_by_id = {row["throw_id"]: row for row in index_rows}
    gt_files = _find_gt_files(mocap_root)
    centers_by_recording = {
        recording: _read_gt_centers(path) for recording, path in gt_files.items()
    }

    fields = [
        "mocap_throw_id",
        "league_throw_id",
        "trajectory_point_count",
        "start_time_ms",
        "end_time_ms",
        "trajectory_json",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    total_replaced = 0
    fallback_recordings: set[str] = set()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            throw_id = row["mocap_throw_id"]
            recording = _recording_prefix(throw_id)
            centers = centers_by_recording.get(recording)
            points = json.loads(row["trajectory_json"])
            if centers is None:
                fallback_recordings.add(recording)
                combined = points
            else:
                # Stored global indices are zero-based positions. These files
                # have contiguous native frames beginning at frame 1.
                por_frame = int(index_by_id[throw_id]["por_global_idx"]) + 1
                combined, count, _ = _replace_points(
                    points, por_frame=por_frame, centers=centers
                )
                total_replaced += count

            writer.writerow({
                "mocap_throw_id": throw_id,
                "league_throw_id": row["league_throw_id"],
                "trajectory_point_count": len(combined),
                "start_time_ms": combined[0]["t_since_ms"],
                "end_time_ms": combined[-1]["t_since_ms"],
                "trajectory_json": json.dumps(
                    combined, separators=(",", ":"), allow_nan=False
                ),
            })
    return len(rows), total_replaced, sorted(fallback_recordings)


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parents[1]
    default_dir = base / "out" / "throw_features" / "learned_ranker_all_types"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstructed-full",
        type=Path,
        default=default_dir / "reconstructed_full.csv",
    )
    parser.add_argument(
        "--throw-index",
        type=Path,
        default=base / "out" / "throw_features" / "throw_index.csv",
    )
    parser.add_argument(
        "--mocap-root", type=Path, default=base / "mocap_files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "reconstructed_full_gt_ball.csv",
    )
    args = parser.parse_args(argv)
    rows, points, fallbacks = combine_file(
        args.reconstructed_full, args.throw_index, args.mocap_root, args.output
    )
    print(f"Wrote {rows} trajectories to {args.output}")
    print(f"Replaced {points} pre-release points with 6D GT centres")
    if fallbacks:
        print("Used existing marker-fit points for: " + ", ".join(fallbacks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
