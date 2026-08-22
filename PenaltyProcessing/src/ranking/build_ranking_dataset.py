"""Leakage-safe synthetic queries and hard-negative ranking pairs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.feature_extraction import extract_features_from_row
    from src.model.weighted_knn import (
        ID_CANDIDATES, PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, read_feature_csv,
    )
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    # Support direct scripts and imports through the top-level ``ranking`` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from feature_extraction import extract_features_from_row
    from model.weighted_knn import (
        ID_CANDIDATES, PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, read_feature_csv,
    )

BASE_COMPONENTS = list(WeightedKNNRetriever().group_weights)
INTERACTIONS = ["bounce_release_height", "bounce_z_trajectory"]
logger = logging.getLogger("build_ranking_dataset")


def setup_logging(verbose: bool = False) -> None:
    """Configure concise timestamped progress logging for command-line runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


@dataclass
class RankingSplit:
    queries: pd.DataFrame
    source_ids: np.ndarray


def split_source_ids(ids: np.ndarray, *, seed: int = 0,
                     fractions: tuple[float, float, float] = (.7, .15, .15)) -> dict[str, np.ndarray]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    shuffled = np.asarray(ids, object).copy()
    np.random.default_rng(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * fractions[0])
    n_valid = int(len(shuffled) * fractions[1])
    return {"train": shuffled[:n_train], "validation": shuffled[n_train:n_train + n_valid],
            "test": shuffled[n_train + n_valid:]}


def is_bounce(row: pd.Series) -> bool:
    explicit = row.get("is_bounce", row.get("bounce_detected"))
    if explicit is not None and pd.notna(explicit):
        return str(explicit).strip().lower() in {"1", "true", "yes"}
    text = " ".join(str(row.get(c, "")) for c in ("throw_type", "throw_id"))
    return "bounce" in text.lower()


def add_interactions(base: np.ndarray, component_names: list[str], bounce: bool,
                     z_trajectory: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = len(base)
    height = base[:, component_names.index("release_height")]
    flag = float(bounce)
    extra = np.column_stack((flag * height, flag * z_trajectory.reshape(n)))
    # Missing vertical evidence remains missing for bounce, but all interactions
    # are exactly zero for non-bounce queries.
    if not bounce:
        extra[:] = 0.0
    return np.column_stack((base, extra)), component_names + INTERACTIONS


def _component_matrix_and_manual_distances(
    retriever: WeightedKNNRetriever, query: pd.Series,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    values = retriever.distance_components(query)
    base = np.column_stack([values[name] for name in BASE_COMPONENTS])
    matrix, names = add_interactions(
        base, BASE_COMPONENTS, is_bounce(query), values["z_trajectory"]
    )
    distances = np.divide(
        values["_numerator"], values["_denominator"],
        out=np.full_like(values["_numerator"], np.inf),
        where=values["_denominator"] > 0,
    )
    return matrix, names, distances


def component_matrix(retriever: WeightedKNNRetriever, query: pd.Series) -> tuple[np.ndarray, list[str]]:
    matrix, names, _ = _component_matrix_and_manual_distances(retriever, query)
    return matrix, names


def _trajectory_times_seconds(points: list[dict]) -> np.ndarray:
    times_ms = pd.to_numeric(
        pd.Series([point.get("t_since_ms") for point in points]), errors="coerce",
    ).to_numpy(float)
    if np.all(np.isfinite(times_ms)):
        return times_ms / 1000.0
    sampling_rate = 20.0
    frames = pd.to_numeric(
        pd.Series([point.get("frame", i) for i, point in enumerate(points)]), errors="coerce",
    ).to_numpy(float)
    return (frames - frames[0]) / sampling_rate


def _recompute_kinematics(positions: np.ndarray, times_s: np.ndarray) -> dict[str, np.ndarray]:
    """Recompute all derived motion values from one perturbed position path."""
    n = len(positions)
    velocity = np.full((n, 3), np.nan)
    for index in range(n):
        before, after = (0, 1) if index == 0 else (
            (n - 2, n - 1) if index == n - 1 else (index - 1, index + 1)
        )
        dt = times_s[after] - times_s[before]
        if dt > 0:
            velocity[index] = (positions[after] - positions[before]) / dt

    acceleration = np.full((n, 3), np.nan)
    for index in range(n):
        before, after = (0, 1) if index == 0 else (
            (n - 2, n - 1) if index == n - 1 else (index - 1, index + 1)
        )
        dt = times_s[after] - times_s[before]
        if dt > 0 and np.isfinite(velocity[[before, after]]).all():
            acceleration[index] = (velocity[after] - velocity[before]) / dt

    speed = np.linalg.norm(velocity, axis=1)
    accel = np.linalg.norm(acceleration, axis=1)
    direction = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
    vertical_angle = np.degrees(np.arctan2(velocity[:, 2], horizontal_speed))
    return {
        "v": speed, "a": accel, "dir": direction,
        "vx": velocity[:, 0], "vy": velocity[:, 1], "vz": velocity[:, 2],
        "vert_angle": vertical_angle,
    }


def perturb_raw_throw(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Perturb a complete path, then derive every kinematic value from that path."""
    out = row.copy()
    points = json.loads(row.get("trajectory_json", "[]") or "[]")
    if len(points) < 2:
        raise ValueError(f"Throw {row.get('throw_id')} has fewer than two trajectory points")

    positions = np.asarray(
        [[point.get(axis, np.nan) for axis in "xyz"] for point in points], dtype=float,
    )
    times_s = _trajectory_times_seconds(points)
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(times_s)
    if valid.sum() < 2:
        raise ValueError(f"Throw {row.get('throw_id')} has fewer than two valid trajectory points")
    points = [point for point, keep in zip(points, valid) if keep]
    positions, times_s = positions[valid], times_s[valid]

    # Remove complete samples so position and every derived value share the
    # same missing-data pattern. Always retain the PoR and the final sample.
    keep = rng.random(len(points)) >= .04
    keep[0] = keep[-1] = True
    if keep.sum() < 2:
        keep[:] = True
    points = [point for point, retain in zip(points, keep) if retain]
    positions, times_s = positions[keep], times_s[keep]

    # A small yaw and axis scaling mimic calibration and speed differences.
    yaw = math.radians(rng.normal(0.0, 2.0))
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)],
                           [math.sin(yaw), math.cos(yaw)]])
    positions[:, :2] = (positions[:, :2] @ rotation.T) * rng.normal(1.0, .04)
    vertical_scale = rng.normal(1.0, .025)
    positions[:, 2] *= vertical_scale

    # Add smooth path deformation anchored at the PoR. Avoid vertical bending
    # for bounce throws so scaling preserves their ground-contact geometry.
    duration = max(times_s[-1] - times_s[0], 1e-6)
    phase = (times_s - times_s[0]) / duration
    linear_offset = phase[:, None] * rng.normal(0.0, [.05, .05, .02], size=3)
    curved_scale = np.asarray([.04, .04, 0.0 if is_bounce(row) else .02])
    curved_offset = np.sin(np.pi * phase)[:, None] * rng.normal(0.0, curved_scale, size=3)
    positions += linear_offset + curved_offset
    positions -= positions[0].copy()

    kinematics = _recompute_kinematics(positions, times_s)
    for index, (point, position) in enumerate(zip(points, positions)):
        for axis, value in zip("xyz", position):
            point[axis] = float(value)
        for name, values in kinematics.items():
            point[name] = float(values[index]) if np.isfinite(values[index]) else None

    por_z = float(row.get("por_z_m", row.get("release_height_m", 0.0))) * vertical_scale
    out["por_x_m"] = float(row.get("por_x_m", 0.0)) + rng.normal(0.0, .03)
    out["por_y_m"] = float(row.get("por_y_m", 0.0)) + rng.normal(0.0, .03)
    out["por_z_m"] = por_z
    out["release_height_m"] = por_z
    out["release_speed_m_s"] = kinematics["v"][0]
    out["release_direction_deg"] = kinematics["dir"][0]
    out["trajectory_json"] = json.dumps(points, ensure_ascii=False, separators=(",", ":"))
    out["trajectory_point_count"] = len(points)
    out["trajectory_duration_ms"] = (times_s[-1] - times_s[0]) * 1000.0
    return out


def build_synthetic_splits(league: pd.DataFrame, raw_league: pd.DataFrame, *,
                           augmentations: int = 3, seed: int = 0) -> dict[str, RankingSplit]:
    id_col = next((c for c in ID_CANDIDATES if c in league), None)
    if id_col is None:
        raise ValueError("League data needs a throw ID")
    raw_id_col = next((c for c in ID_CANDIDATES if c in raw_league), None)
    if raw_id_col is None:
        raise ValueError("Raw League data needs a throw ID")
    raw_by_id = {str(row[raw_id_col]): row for _, row in raw_league.iterrows()}
    missing = [value for value in league[id_col] if str(value) not in raw_by_id]
    if missing:
        raise ValueError(f"Raw League data is missing {len(missing)} feature-row IDs")
    rng = np.random.default_rng(seed)
    partitions = split_source_ids(league[id_col].to_numpy(), seed=seed)
    result = {}
    for split, ids in partitions.items():
        logger.info(
            "Generating %s queries from %d source throws (%d augmentations each)",
            split, len(ids), augmentations,
        )
        rows, sources = [], []
        for source_id in ids:
            source = raw_by_id[str(source_id)]
            for aug in range(augmentations):
                perturbed = perturb_raw_throw(source, rng)
                query = pd.Series(extract_features_from_row(perturbed)).reindex(league.columns)
                query[id_col] = f"synthetic_{source_id}_{aug}"
                rows.append(query)
                sources.append(source_id)
        result[split] = RankingSplit(pd.DataFrame(rows, columns=league.columns), np.asarray(sources))
        logger.info("Generated %d %s queries", len(rows), split)
    return result


def build_pairs(retriever: WeightedKNNRetriever, split: RankingSplit, *,
                hard_negatives: int = 3, random_negatives: int = 1,
                seed: int = 0, split_name: str = "split",
                log_every: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    positives, negatives = [], []
    query_count = len(split.queries)
    started = time.perf_counter()
    logger.info(
        "Mining pairs for %s: %d queries against %d league throws "
        "(%d hard + %d random negatives per query)",
        split_name, query_count, len(league_ids), hard_negatives, random_negatives,
    )
    for query_number, ((_, query), source_id) in enumerate(
        zip(split.queries.iterrows(), split.source_ids), start=1
    ):
        positive_index = int(np.flatnonzero(league_ids == source_id)[0])
        matrix, _, manual_distances = _component_matrix_and_manual_distances(retriever, query)
        ranked_indices = np.argsort(manual_distances, kind="stable")
        hard_indices = ranked_indices[league_ids[ranked_indices] != source_id][:hard_negatives]
        random_pool = np.flatnonzero(league_ids != source_id)
        random_indices = rng.choice(
            random_pool, size=min(random_negatives, len(random_pool)), replace=False,
        )
        negative_indices = np.concatenate((hard_indices, random_indices))
        negatives.extend(matrix[negative_indices])
        positives.extend(np.repeat(matrix[[positive_index]], len(negative_indices), axis=0))
        if query_number == 1 or query_number == query_count or (
            log_every > 0 and query_number % log_every == 0
        ):
            elapsed = time.perf_counter() - started
            rate = query_number / elapsed if elapsed > 0 else 0.0
            remaining = (query_count - query_number) / rate if rate > 0 else 0.0
            logger.info(
                "%s pair mining: %d/%d queries (%.1f%%), %d pairs, "
                "elapsed %.1fs, ETA %.1fs",
                split_name, query_number, query_count,
                100.0 * query_number / query_count if query_count else 100.0,
                len(positives), elapsed, remaining,
            )
    width = len(BASE_COMPONENTS) + len(INTERACTIONS)
    return (np.asarray(positives, float).reshape(-1, width),
            np.asarray(negatives, float).reshape(-1, width))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2] / "out"
    parser.add_argument("--league", type=Path, default=base / "throw_features" / "features_league.csv")
    parser.add_argument(
        "--raw-league", type=Path, default=base / "throw_features" / "raw_league.csv",
        help="Raw League trajectories used to generate physically consistent synthetic queries",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "throw_features" / "ranking_dataset")
    parser.add_argument("--augmentations", type=int, default=6)
    parser.add_argument("--hard-negatives", type=int, default=12)
    parser.add_argument("--random-negatives", type=int, default=3)
    parser.add_argument(
        "--weight-preset", choices=tuple(PREDEFINED_WEIGHT_SETS), default="best_random",
        help="manual kNN weights used to mine hard negatives (default: best_random)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="Report pair-mining progress every N queries; use 0 to disable periodic updates",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    started = time.perf_counter()
    logger.info("Building synthetic ranking dataset")
    logger.info("Reading league features from %s", args.league)
    league = read_feature_csv(args.league)
    logger.info("Loaded %d league throws with %d columns", len(league), len(league.columns))
    logger.info("Reading raw League trajectories from %s", args.raw_league)
    raw_league = pd.read_csv(args.raw_league, sep=";")
    logger.info("Loaded %d raw League trajectories", len(raw_league))
    logger.info("Fitting weighted kNN retriever with weight preset '%s'", args.weight_preset)
    retriever = WeightedKNNRetriever(PREDEFINED_WEIGHT_SETS[args.weight_preset]).fit(league)
    logger.info(
        "Creating leakage-safe splits (augmentations=%d, seed=%d)",
        args.augmentations, args.seed,
    )
    splits = build_synthetic_splits(
        league, raw_league, augmentations=args.augmentations, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing dataset to %s", args.output_dir)
    for name, split in splits.items():
        queries = split.queries.copy()
        queries.insert(1, "source_league_throw_id", split.source_ids)
        queries_path = args.output_dir / f"{name}_queries.csv"
        queries.to_csv(queries_path, index=False)
        logger.info("Wrote %d queries to %s", len(queries), queries_path)
        positive, negative = build_pairs(
            retriever, split, hard_negatives=args.hard_negatives,
            random_negatives=args.random_negatives, seed=args.seed,
            split_name=name, log_every=args.log_every,
        )
        pairs_path = args.output_dir / f"{name}_pairs.npz"
        np.savez_compressed(
            pairs_path, positive=positive, negative=negative,
            feature_names=np.asarray(BASE_COMPONENTS + INTERACTIONS),
        )
        logger.info("Wrote %d ranking pairs to %s", len(positive), pairs_path)
    logger.info(
        "Finished synthetic ranking dataset in %.1fs: %s",
        time.perf_counter() - started, args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
