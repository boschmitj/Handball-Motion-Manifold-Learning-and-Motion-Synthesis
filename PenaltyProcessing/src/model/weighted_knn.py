#!/usr/bin/env python3
"""Interpretable weighted nearest-neighbour retrieval for handball throws.

Scaling is fitted exclusively on League throws. A pair is compared only on
finite features present in both rows; missing time samples are never filled or
extrapolated. Group weights are divided over the selected scalar features in a
group, making them budgets rather than accidentally giving large groups more
importance simply because they contain more columns.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "release_speed": 3.0,
    "release_angles": 3.0,
    "release_height": 1.0,
    "relative_trajectory": 3.0,
    "velocity_evolution": 2.5,
    "trajectory_angles": 2.0,
    "acceleration": 0.25,
    "absolute_por": 0.25,
}

ANGLE_GROUPS = {"release_angles", "trajectory_angles"}
ID_CANDIDATES = ("throw_id", "mocap_throw_id", "league_throw_id")
EPSILON_SCALE = 1e-8


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    scale: float
    weight: float
    angular: bool = False


def _matches(pattern: str, column: str) -> bool:
    return re.fullmatch(pattern, column, flags=re.IGNORECASE) is not None


def select_feature_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    """Select supported feature columns without admitting metadata columns."""
    cols = list(columns)
    groups: dict[str, list[str]] = {name: [] for name in DEFAULT_GROUP_WEIGHTS}

    def add(group: str, patterns: Sequence[str]) -> None:
        groups[group] = [c for c in cols if any(_matches(p, c) for p in patterns)]

    add("release_speed", [r"release_speed(?:_m_s)?"])
    add("release_angles", [r"release_(?:horizontal_)?direction(?:_deg)?",
                           r"release_(?:vertical_angle|elevation)(?:_deg)?"])
    # Current feature extraction uses elevation at t=0 as release elevation.
    if not any("vertical" in c.lower() or "elevation" in c.lower()
               for c in groups["release_angles"]):
        fallback = next((c for c in cols if _matches(r"vert_angle_t0ms", c)), None)
        if fallback:
            groups["release_angles"].append(fallback)
    add("release_height", [r"release_height(?:_m)?"])
    add("relative_trajectory", [r"(?:disp|displacement)_[xyz]_t\d+ms"])
    add("velocity_evolution", [r"velocity_t\d+ms", r"velocity_[xyz]_t\d+ms"])
    add("trajectory_angles", [r"direction_t\d+ms", r"(?:vert_angle|elevation)_t\d+ms"])
    # Avoid counting t=0 elevation twice when it is the release fallback.
    groups["trajectory_angles"] = [
        c for c in groups["trajectory_angles"] if c not in groups["release_angles"]
    ]
    add("acceleration", [r"accel(?:eration)?_.*"])
    add("absolute_por", [r"por_[xyz](?:_m)?"])
    return {g: features for g, features in groups.items() if features}


def circular_difference_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return signed shortest angular difference in [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def _angular_scale(values: np.ndarray) -> float:
    """League standard deviation after unwrapping around its circular mean."""
    radians = np.deg2rad(values)
    mean = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    deviations = circular_difference_degrees(values, np.rad2deg(mean))
    return float(np.std(deviations, ddof=0))


class WeightedKNNRetriever:
    """Fit League-only scales and retrieve the nearest League throws."""

    def __init__(self, group_weights: Mapping[str, float] | None = None) -> None:
        self.group_weights = dict(DEFAULT_GROUP_WEIGHTS)
        if group_weights:
            unknown = set(group_weights) - set(DEFAULT_GROUP_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
            self.group_weights.update(group_weights)
        if any(not np.isfinite(w) or w < 0 for w in self.group_weights.values()):
            raise ValueError("Group weights must be finite and non-negative")
        self.feature_specs: list[FeatureSpec] = []
        self.id_column: str | None = None
        self.league: pd.DataFrame | None = None

    def fit(self, league: pd.DataFrame) -> "WeightedKNNRetriever":
        league = league.copy()
        league.columns = league.columns.str.strip()
        self.id_column = next((c for c in ID_CANDIDATES if c in league.columns), None)
        if self.id_column is None:
            raise ValueError(f"League input needs one of these ID columns: {ID_CANDIDATES}")

        groups = select_feature_groups(league.columns)
        specs: list[FeatureSpec] = []
        for group, features in groups.items():
            group_weight = self.group_weights[group]
            if group_weight == 0:
                continue
            scalar_weight = group_weight / len(features)
            for feature in features:
                numeric = pd.to_numeric(league[feature], errors="coerce").to_numpy(float)
                finite = numeric[np.isfinite(numeric)]
                angular = group in ANGLE_GROUPS
                scale = _angular_scale(finite) if angular and finite.size else (
                    float(np.std(finite, ddof=0)) if finite.size else np.nan
                )
                # Constant League features carry no discriminating information.
                if np.isfinite(scale) and scale > EPSILON_SCALE:
                    specs.append(FeatureSpec(feature, group, scale, scalar_weight, angular))
        if not specs:
            raise ValueError("No usable, non-constant matching features were found")
        self.feature_specs = specs
        self.league = league
        return self

    def retrieve(self, mocap: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        if self.league is None or self.id_column is None:
            raise RuntimeError("Call fit() before retrieve()")
        if k < 1:
            raise ValueError("k must be at least 1")
        mocap = mocap.copy()
        mocap.columns = mocap.columns.str.strip()
        mocap_id = next((c for c in ID_CANDIDATES if c in mocap.columns), None)
        if mocap_id is None:
            raise ValueError(f"Mocap input needs one of these ID columns: {ID_CANDIDATES}")

        specs = [s for s in self.feature_specs if s.name in mocap.columns]
        if not specs:
            raise ValueError("Mocap and League inputs have no usable common features")
        league_ids = self.league[self.id_column].to_numpy()
        rows: list[dict[str, object]] = []

        for _, mrow in mocap.iterrows():
            numerator = np.zeros(len(self.league), dtype=float)
            denominator = np.zeros(len(self.league), dtype=float)
            group_numerators = {
                group: np.zeros(len(self.league), dtype=float) for group in self.group_weights
            }
            group_denominators = {
                group: np.zeros(len(self.league), dtype=float) for group in self.group_weights
            }
            for spec in specs:
                mvalue = pd.to_numeric(pd.Series([mrow[spec.name]]), errors="coerce").iloc[0]
                if not np.isfinite(mvalue):
                    continue
                lvalues = pd.to_numeric(self.league[spec.name], errors="coerce").to_numpy(float)
                valid = np.isfinite(lvalues)
                difference = (circular_difference_degrees(lvalues, mvalue)
                              if spec.angular else lvalues - mvalue)
                term = spec.weight * np.square(difference / spec.scale)
                numerator[valid] += term[valid]
                denominator[valid] += spec.weight
                group_numerators[spec.group][valid] += term[valid]
                group_denominators[spec.group][valid] += spec.weight

            distances = np.divide(numerator, denominator, out=np.full_like(numerator, np.inf),
                                  where=denominator > 0)
            count = min(k, len(distances))
            nearest = np.argsort(distances, kind="stable")[:count]
            for rank, idx in enumerate(nearest, start=1):
                result: dict[str, object] = {
                    "mocap_throw_id": mrow[mocap_id], "rank": rank,
                    "league_throw_id": league_ids[idx],
                    "total_distance": distances[idx],
                    "compared_weight": denominator[idx],
                    "compared_feature_count": sum(
                        1 for s in specs
                        if np.isfinite(pd.to_numeric(pd.Series([mrow[s.name]]), errors="coerce").iloc[0])
                        and np.isfinite(pd.to_numeric(
                            pd.Series([self.league.iloc[idx][s.name]]), errors="coerce").iloc[0])
                    ),
                }
                for group in self.group_weights:
                    gd = group_denominators[group][idx]
                    result[f"{group}_distance"] = (
                        group_numerators[group][idx] / gd if gd > 0 else np.nan
                    )
                    # Additive contribution; these sum exactly to total_distance.
                    result[f"{group}_contribution"] = (
                        group_numerators[group][idx] / denominator[idx]
                        if denominator[idx] > 0 else np.nan
                    )
                rows.append(result)
        return pd.DataFrame(rows)


def read_feature_csv(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited feature CSV and trim headers."""
    frame = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    return frame


def _parse_weights(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Weights must be a JSON object")
    return {str(k): float(v) for k, v in parsed.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2] / "out"
    parser.add_argument("--mocap", type=Path, default=base / "features_mocap.csv")
    parser.add_argument("--league", type=Path, default=base / "features_league.csv")
    parser.add_argument("--output", type=Path, default=base / "weighted_knn_matches.csv")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--weights", help="JSON object or path to a JSON file")
    args = parser.parse_args()

    retriever = WeightedKNNRetriever(_parse_weights(args.weights))
    results = retriever.fit(read_feature_csv(args.league)).retrieve(
        read_feature_csv(args.mocap), k=args.k
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Wrote {len(results)} matches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
