#!/usr/bin/env python3
"""Combine pre-PoR Mocap samples with reconstructed PoR-to-goal trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


XYZ = ("x", "y", "z")


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv(path: Path) -> list[dict[str, str]]:
    _set_csv_field_limit()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = max((";", ",", "\t"), key=header.count)
        if header.count(delimiter) == 0:
            raise ValueError(f"Could not detect CSV/TSV delimiter in {path}")
        return list(csv.DictReader(handle, delimiter=delimiter))


def _parse_points(value: str, *, source: str) -> list[dict[str, Any]]:
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"{source} trajectory_json must be a non-empty list")
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(decoded):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source} trajectory point {index} is not an object")
        try:
            point = {
                "frame": float(raw["frame"]),
                "t_since_ms": float(raw["t_since_ms"]),
                **{axis: float(raw[axis]) for axis in XYZ},
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source} trajectory point {index} lacks frame/t/x/y/z") from exc
        if not all(math.isfinite(float(value)) for value in point.values()):
            raise ValueError(f"{source} trajectory point {index} contains non-finite values")
        points.append(point)
    return sorted(points, key=lambda point: point["t_since_ms"])


def combine_trajectory(
    mocap_row: Mapping[str, str], reconstructed_json: str, *, atol: float = 1e-6,
) -> list[dict[str, float | int]]:
    """Return signed-time positions in absolute League/world coordinates."""
    mocap_relative = _parse_points(mocap_row["trajectory_json"], source="Mocap")
    reconstructed = _parse_points(reconstructed_json, source="reconstructed")
    por = {axis: float(mocap_row[f"por_{axis}_m"]) for axis in XYZ}

    if not math.isclose(reconstructed[0]["t_since_ms"], 0.0, abs_tol=atol):
        raise ValueError("reconstructed trajectory must start at PoR (t_since_ms = 0)")
    reconstructed_start = [reconstructed[0][axis] for axis in XYZ]
    expected_start = [por[axis] for axis in XYZ]
    if any(not math.isclose(actual, expected, abs_tol=atol)
           for actual, expected in zip(reconstructed_start, expected_start)):
        raise ValueError(
            "reconstructed trajectory does not start at the matched Mocap PoR: "
            f"got {tuple(reconstructed_start)}, expected {tuple(expected_start)}"
        )

    combined: list[dict[str, float | int]] = []
    for point in mocap_relative:
        if point["t_since_ms"] >= 0:
            continue
        combined.append({
            "frame": int(point["frame"]) if point["frame"].is_integer() else point["frame"],
            "t_since_ms": point["t_since_ms"],
            **{axis: point[axis] + por[axis] for axis in XYZ},
        })
    for point in reconstructed:
        combined.append({
            "frame": int(point["frame"]) if point["frame"].is_integer() else point["frame"],
            "t_since_ms": point["t_since_ms"],
            **{axis: point[axis] for axis in XYZ},
        })
    return combined


def combine_file(raw_mocap: Path, reconstructed: Path, output: Path) -> int:
    mocap_by_id = {row["throw_id"]: row for row in _read_csv(raw_mocap)}
    reconstructed_rows = _read_csv(reconstructed)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mocap_throw_id", "league_throw_id", "trajectory_point_count",
        "start_time_ms", "end_time_ms", "trajectory_json",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_number, row in enumerate(reconstructed_rows, start=2):
            mocap_id = row.get("mocap_throw_id", "")
            league_id = row.get("league_throw_id", "")
            if not mocap_id or not league_id:
                raise ValueError(f"missing Mocap/League ID in {reconstructed}:{row_number}")
            try:
                mocap = mocap_by_id[mocap_id]
            except KeyError as exc:
                raise KeyError(f"Mocap throw {mocap_id!r} not found in {raw_mocap}") from exc
            points = combine_trajectory(mocap, row.get("trajectory_json", ""))
            writer.writerow({
                "mocap_throw_id": mocap_id,
                "league_throw_id": league_id,
                "trajectory_point_count": len(points),
                "start_time_ms": points[0]["t_since_ms"],
                "end_time_ms": points[-1]["t_since_ms"],
                "trajectory_json": json.dumps(points, separators=(",", ":"), allow_nan=False),
            })
    return len(reconstructed_rows)


def default_output_path(reconstructed: Path) -> Path:
    return reconstructed.with_name(f"{reconstructed.stem}_full{reconstructed.suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstructed", type=Path, required=True,
        help="reconstructed_matched_trajectories*.csv to combine",
    )
    parser.add_argument(
        "--raw-mocap", type=Path, default=base / "out" / "throw_features" / "raw_mocap.csv",
        help="raw Mocap CSV/TSV (default: out/throw_features/raw_mocap.csv)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="output CSV (default: <reconstructed stem>_full.csv beside the input)",
    )
    args = parser.parse_args(argv)
    output = args.output or default_output_path(args.reconstructed)
    count = combine_file(args.raw_mocap, args.reconstructed, output)
    print(f"Wrote {count} complete trajectories to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
