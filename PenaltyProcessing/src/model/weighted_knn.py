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
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Use these as the default weights
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

GROUP_WEIGHTS_HEIGHT_FOCUS: dict[str, float] = {
    "release_speed": 3.0,
    "release_height": 3.0,
    "relative_trajectory": 3.0,
    "velocity_evolution": 2.0,
    "release_angles": 2.0,
    "trajectory_angles": 2.0,
    "absolute_por": 0.75,
    "acceleration": 0.25,
}

BEST_RANDOM_WEIGHTS_CURRENT: dict[str, float] = {
    "release_speed": 3.122191774027964,
    "release_angles": 0.48649669239408283,
    "release_height": 4.239295961166828,
    "relative_trajectory": 0.1734364062696802,
    "velocity_evolution": 0.7678062523986564,
    "trajectory_angles": 0.2854078774284084,
    "acceleration": 0.1348043190299895,
    "absolute_por": 5.790560717284391
}

# Add new named configurations here. They automatically become valid
# --weight-preset choices and are included, with values, in --help.
PREDEFINED_WEIGHT_SETS: dict[str, dict[str, float]] = {
    "default": DEFAULT_GROUP_WEIGHTS,
    "height_focus": GROUP_WEIGHTS_HEIGHT_FOCUS,
    "best_random" : BEST_RANDOM_WEIGHTS_CURRENT,
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
            components = self.distance_components(mrow, specs=specs)
            numerator = components.pop("_numerator")
            denominator = components.pop("_denominator")

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
                    "trajectory_common_point_count": int(
                        components["trajectory_common_point_count"][idx]
                    ),
                    "trajectory_query_point_count": int(
                        components["trajectory_query_point_count"][idx]
                    ),
                    "trajectory_overlap_ratio": components["trajectory_overlap_ratio"][idx],
                    "trajectory_evidence_ratio": components["trajectory_evidence_ratio"][idx],
                }
                for group in self.group_weights:
                    result[f"{group}_distance"] = components[group][idx]
                    # Additive contribution; these sum exactly to total_distance.
                    result[f"{group}_contribution"] = (
                        components[f"_{group}_numerator"][idx] / denominator[idx]
                        if denominator[idx] > 0 else np.nan
                    )
                rows.append(result)
        return pd.DataFrame(rows)

    def distance_components(
        self, query: pd.Series, *, specs: Sequence[FeatureSpec] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the baseline's normalized distances to every League row.

        This is the single shared calculation used by both manual retrieval and
        learned ranking.  Private numerator entries support the legacy additive
        contribution columns; callers should consume the public group names and
        ``z_trajectory`` diagnostic component.
        """
        if self.league is None:
            raise RuntimeError("Call fit() before computing distance components")
        selected = list(specs) if specs is not None else [
            spec for spec in self.feature_specs if spec.name in query.index
        ]
        size = len(self.league)
        total_num = np.zeros(size, dtype=float)
        total_den = np.zeros(size, dtype=float)
        numerators = {group: np.zeros(size) for group in self.group_weights}
        denominators = {group: np.zeros(size) for group in self.group_weights}
        z_num, z_den = np.zeros(size), np.zeros(size)
        # Count elapsed-time points using disp_x only, avoiding counting the
        # multiple coordinates/kinematic values at one time as independent
        # observations. These diagnostics make truncated-query confidence and
        # candidate coverage explicit.
        trajectory_time_names = [
            name for name in self.league.columns
            if name in query.index
            and re.fullmatch(r"disp_x_t\d+ms", name, flags=re.IGNORECASE)
        ]
        query_trajectory_count = 0
        common_trajectory_count = np.zeros(size, dtype=float)
        for name in trajectory_time_names:
            qvalue = pd.to_numeric(pd.Series([query.get(name)]), errors="coerce").iloc[0]
            if not np.isfinite(qvalue):
                continue
            query_trajectory_count += 1
            values = pd.to_numeric(self.league[name], errors="coerce").to_numpy(float)
            common_trajectory_count[np.isfinite(values)] += 1.0
        for spec in selected:
            qvalue = pd.to_numeric(pd.Series([query.get(spec.name)]), errors="coerce").iloc[0]
            if not np.isfinite(qvalue):
                continue
            values = pd.to_numeric(self.league[spec.name], errors="coerce").to_numpy(float)
            valid = np.isfinite(values)
            difference = (circular_difference_degrees(values, qvalue)
                          if spec.angular else values - qvalue)
            term = spec.weight * np.square(difference / spec.scale)
            total_num[valid] += term[valid]
            total_den[valid] += spec.weight
            numerators[spec.group][valid] += term[valid]
            denominators[spec.group][valid] += spec.weight
            if re.fullmatch(r"disp_z_t\d+ms", spec.name, flags=re.IGNORECASE):
                z_num[valid] += term[valid]
                z_den[valid] += spec.weight
        result: dict[str, np.ndarray] = {
            "_numerator": total_num, "_denominator": total_den,
        }
        for group in self.group_weights:
            result[group] = np.divide(
                numerators[group], denominators[group], out=np.full(size, np.nan),
                where=denominators[group] > 0,
            )
            result[f"_{group}_numerator"] = numerators[group]
        result["z_trajectory"] = np.divide(
            z_num, z_den, out=np.full(size, np.nan), where=z_den > 0,
        )
        query_counts = np.full(size, float(query_trajectory_count))
        overlap = np.divide(
            common_trajectory_count, query_counts, out=np.zeros(size),
            where=query_counts > 0,
        )
        possible_count = len(trajectory_time_names)
        evidence = (float(query_trajectory_count) / possible_count
                    if possible_count else 0.0)
        result["trajectory_common_point_count"] = common_trajectory_count
        result["trajectory_query_point_count"] = query_counts
        result["trajectory_overlap_ratio"] = overlap
        result["trajectory_evidence_ratio"] = np.full(size, evidence)
        result["trajectory_coverage"] = 1.0 - overlap
        return result


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


def infer_throw_types(mocap: pd.DataFrame) -> dict[str, str]:
    """Map Mocap IDs to types, preferring an explicit throw_type column."""
    id_column = next((c for c in ID_CANDIDATES if c in mocap.columns), None)
    if id_column is None:
        raise ValueError(f"Mocap input needs one of these ID columns: {ID_CANDIDATES}")
    mapping: dict[str, str] = {}
    for _, row in mocap.iterrows():
        throw_id = str(row[id_column])
        explicit = row.get("throw_type")
        if explicit is not None and pd.notna(explicit) and str(explicit).strip():
            throw_type = str(explicit).strip()
        else:
            match = re.match(r"^(.*?)(?:_seg\d+)$", throw_id, flags=re.IGNORECASE)
            throw_type = match.group(1) if match else "unknown"
        mapping[throw_id] = throw_type
    return mapping


def balanced_top1_score(
    matches: pd.DataFrame, mocap: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    """Return equal-per-type mean top-1 distance and its monotonic score.

    Each throw type contributes equally regardless of how many recorded
    segments it contains. Lower balanced distance and higher score are better.
    """
    top1 = matches[matches["rank"] == 1].copy()
    type_by_id = infer_throw_types(mocap)
    top1["throw_type"] = top1["mocap_throw_id"].astype(str).map(type_by_id).fillna("unknown")
    top1["total_distance"] = pd.to_numeric(top1["total_distance"], errors="coerce")
    finite = top1[np.isfinite(top1["total_distance"])].copy()
    if finite.empty:
        raise ValueError("No finite rank-1 distances are available for scoring")
    by_type = (finite.groupby("throw_type", as_index=False)["total_distance"]
               .agg(mean_top1_distance="mean", throw_count="size"))
    balanced_distance = float(by_type["mean_top1_distance"].mean())
    score = 1.0 / (1.0 + balanced_distance)
    return balanced_distance, score, by_type


def random_group_weights(rng: np.random.Generator) -> dict[str, float]:
    """Draw positive group budgets with the same total as the defaults."""
    names = list(DEFAULT_GROUP_WEIGHTS)
    values = rng.dirichlet(np.ones(len(names))) * sum(DEFAULT_GROUP_WEIGHTS.values())
    return dict(zip(names, map(float, values)))


def _unique_search_directory(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = base / f"weighted_knn_random_{stamp}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_random_weight_search(
    mocap: pd.DataFrame, league: pd.DataFrame, *, runs: int, k: int,
    output_base: Path, seed: int | None = None,
) -> tuple[Path, pd.DataFrame, dict[str, float]]:
    """Run randomized retrievals and persist all results plus the best model."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    search_dir = _unique_search_directory(output_base)
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    best_distance = math.inf
    best_weights: dict[str, float] | None = None
    best_matches: pd.DataFrame | None = None

    for run_number in range(1, runs + 1):
        weights = random_group_weights(rng)
        matches = WeightedKNNRetriever(weights).fit(league).retrieve(mocap, k=k)
        distance, score, by_type = balanced_top1_score(matches, mocap)
        run_dir = search_dir / f"run_{run_number:04d}"
        run_dir.mkdir()
        matches.to_csv(run_dir / "weighted_knn_matches.csv", index=False)
        by_type.to_csv(run_dir / "metrics_by_throw_type.csv", index=False)
        (run_dir / "weights.json").write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
        summary_rows.append({
            "run": run_number, "balanced_mean_top1_distance": distance,
            "balanced_score": score, "weights_json": json.dumps(weights, sort_keys=True), **weights,
        })
        if distance < best_distance:
            best_distance, best_weights, best_matches = distance, weights, matches

    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_mean_top1_distance", "run"], kind="stable"
    )
    summary.to_csv(search_dir / "random_search_summary.csv", index=False)
    assert best_weights is not None and best_matches is not None
    (search_dir / "best_weights.json").write_text(
        json.dumps({"balanced_mean_top1_distance": best_distance,
                    "balanced_score": 1.0 / (1.0 + best_distance),
                    "weights": best_weights}, indent=2) + "\n", encoding="utf-8"
    )
    best_matches.to_csv(search_dir / "best_weighted_knn_matches.csv", index=False)
    return search_dir, summary, best_weights


def main() -> int:
    preset_help = "\n".join(
        f"  {name}: {json.dumps(weights, sort_keys=True)}"
        for name, weights in PREDEFINED_WEIGHT_SETS.items()
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Predefined weight sets:\n{preset_help}",
    )
    base = Path(__file__).resolve().parents[2] / "out" / "throw_features"
    parser.add_argument("--mocap", type=Path, default=base / "features_mocap.csv")
    parser.add_argument("--league", type=Path, default=base / "features_league.csv")
    parser.add_argument("--output", type=Path, default=base / "weighted_knn_matches.csv")
    parser.add_argument("-k", type=int, default=5)
    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument(
        "--weight-preset", choices=tuple(PREDEFINED_WEIGHT_SETS), default="default",
        help="named predefined weight set (default: default; full values shown below)",
    )
    weights_group.add_argument("--weights", help="custom JSON object or path to a JSON file")
    parser.add_argument(
        "--random-weight-runs", type=int, default=0, metavar="N",
        help="run N randomized weight configurations and save all runs in a new folder under out/",
    )
    parser.add_argument("--random-seed", type=int, help="seed for reproducible randomized weights")
    args = parser.parse_args()

    mocap = read_feature_csv(args.mocap)
    league = read_feature_csv(args.league)
    if args.random_weight_runs:
        search_dir, summary, best_weights = run_random_weight_search(
            mocap, league, runs=args.random_weight_runs, k=args.k,
            output_base=base, seed=args.random_seed,
        )
        best = summary.iloc[0]
        print(f"Wrote {args.random_weight_runs} randomized runs to {search_dir}")
        print(f"Best balanced mean top-1 distance: {best['balanced_mean_top1_distance']:.8g}")
        print(f"Best balanced score: {best['balanced_score']:.8g}")
        print(f"Best weights: {json.dumps(best_weights, sort_keys=True)}")
        return 0

    selected_weights = (_parse_weights(args.weights) if args.weights is not None
                        else PREDEFINED_WEIGHT_SETS[args.weight_preset])
    retriever = WeightedKNNRetriever(selected_weights)
    results = retriever.fit(league).retrieve(mocap, k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Wrote {len(results)} matches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
