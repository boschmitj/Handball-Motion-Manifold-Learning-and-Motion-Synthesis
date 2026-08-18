#!/usr/bin/env python3
"""Plot one geometry-preserving League-to-Mocap reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajectory_reconstruction import (  # noqa: E402
    XYZ, align_trajectory_to_por, load_raw_throw, parse_trajectory_json,
)


def _load_reconstruction(path: Path, mocap_id: str, league_id: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["mocap_throw_id"] == mocap_id and row["league_throw_id"] == league_id:
                return row
    raise KeyError(f"match ({mocap_id!r}, {league_id!r}) not found in {path}")


def main() -> int:
    base = Path(__file__).resolve().parents[1] / "out"
    throw_features = base / "throw_features"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructed", type=Path, default=base / "reconstructed_matched_trajectories.csv")
    parser.add_argument("--raw-league", type=Path, default=base / "raw_files" / "raw_league.csv")
    parser.add_argument("--raw-mocap", type=Path, default=base / "raw_files" / "raw_mocap.csv")
    parser.add_argument("--mocap-throw-id", required=True)
    parser.add_argument("--league-throw-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    row = _load_reconstruction(args.reconstructed, args.mocap_throw_id, args.league_throw_id)
    league = load_raw_throw(args.raw_league, args.league_throw_id)
    mocap = load_raw_throw(args.raw_mocap, args.mocap_throw_id)
    por = np.asarray([float(mocap[f"por_{axis}_m"]) for axis in XYZ])
    league_points = align_trajectory_to_por(parse_trajectory_json(league), por)
    mocap_relative = parse_trajectory_json(mocap)
    mocap_points = np.asarray([[p[a] for a in XYZ] for p in mocap_relative]) + por
    reconstructed = json.loads(row["trajectory_json"])

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*mocap_points.T, color="0.65", label="Mocap trajectory around PoR")
    ax.scatter(*por, marker="*", s=180, color="gold", edgecolor="black", label="Mocap PoR")
    league_xyz = np.asarray([[p[a] for a in XYZ] for p in league_points])
    ax.scatter(*league_xyz.T, s=35, color="tab:blue", label="Original League samples")
    reconstructed_xyz = np.asarray([[p[a] for a in XYZ] for p in reconstructed])
    ax.plot(*reconstructed_xyz.T, color="tab:blue", label="Upsampled League trajectory")
    bounce = next((p for p in reconstructed if p.get("event") == "bounce"), None)
    if bounce:
        incoming = [p for p in reconstructed
                    if bounce["bounce_boundary_start_ms"] <= p["t_since_ms"] <= bounce["t_since_ms"]]
        outgoing = [p for p in reconstructed
                    if bounce["t_since_ms"] <= p["t_since_ms"] <= bounce["bounce_boundary_end_ms"]]
        incoming_xyz = np.asarray([[p[a] for a in XYZ] for p in incoming])
        outgoing_xyz = np.asarray([[p[a] for a in XYZ] for p in outgoing])
        ax.plot(*incoming_xyz.T, color="tab:red", linewidth=2, label="Incoming reconstruction")
        ax.plot(*outgoing_xyz.T, color="tab:orange", linewidth=2, label="Outgoing reconstruction")
        ax.scatter(bounce["x"], bounce["y"], bounce["z"], marker="D", s=65,
                   color="tab:red", label="Estimated bounce")
        boundary_points = [min(reconstructed, key=lambda p: abs(p["t_since_ms"] - boundary))
                           for boundary in (bounce["bounce_boundary_start_ms"],
                                            bounce["bounce_boundary_end_ms"])]
        boundary_xyz = np.asarray([[p[a] for a in XYZ] for p in boundary_points])
        ax.scatter(*boundary_xyz.T, marker="s", s=50, color="black", label="Window boundaries")
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
           title=f"Mocap {args.mocap_throw_id} → League {args.league_throw_id}")
    ax.legend()

    side = fig.add_subplot(122)
    side.plot(reconstructed_xyz[:, 0], reconstructed_xyz[:, 2], color="tab:blue",
              label="Upsampled trajectory")
    side.scatter(league_xyz[:, 0], league_xyz[:, 2], s=35, color="tab:blue",
                 label="Measured League samples", zorder=3)
    if bounce:
        side.plot(incoming_xyz[:, 0], incoming_xyz[:, 2], color="tab:red", linewidth=2,
                  label="Incoming reconstruction")
        side.plot(outgoing_xyz[:, 0], outgoing_xyz[:, 2], color="tab:orange", linewidth=2,
                  label="Outgoing reconstruction")
        side.scatter(bounce["x"], bounce["z"], marker="D", s=65, color="tab:red",
                     label="Estimated PoGC", zorder=4)
        side.scatter(boundary_xyz[:, 0], boundary_xyz[:, 2], marker="s", s=50,
                     color="black", label="Window boundaries", zorder=4)
    side.set(xlabel="x [m]", ylabel="z [m]", title="Side view (bounce cusp)")
    side.grid(alpha=0.25)
    side.legend()
    fig.tight_layout()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
