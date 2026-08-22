"""Leakage-safe synthetic queries and hard-negative ranking pairs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.model.weighted_knn import (
        ID_CANDIDATES, PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, read_feature_csv,
    )
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    # Support direct scripts and imports through the top-level ``ranking`` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.weighted_knn import (
        ID_CANDIDATES, PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, read_feature_csv,
    )

BASE_COMPONENTS = list(WeightedKNNRetriever().group_weights)
INTERACTIONS = ["bounce_release_height", "bounce_z_trajectory"]


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


def component_matrix(retriever: WeightedKNNRetriever, query: pd.Series) -> tuple[np.ndarray, list[str]]:
    values = retriever.distance_components(query)
    base = np.column_stack([values[name] for name in BASE_COMPONENTS])
    return add_interactions(base, BASE_COMPONENTS, is_bounce(query), values["z_trajectory"])


def perturb_throw(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Apply mild feature-space equivalents of measurement/timing distortions."""
    out = row.copy()
    changed = False
    for name in row.index:
        value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            continue
        noise = None
        if name == "release_speed_m_s": noise = .025 * max(abs(value), 1.0)
        elif name == "release_height_m": noise = .02
        elif re.fullmatch(r"por_[xyz](?:_m)?", name): noise = .03
        elif re.fullmatch(r"(?:disp_[xyz]|velocity(?:_[xyz])?)_t\d+ms", name): noise = .02 * max(abs(value), 1.0)
        elif "direction" in name or "angle" in name: noise = 1.5
        if noise is not None:
            out[name] = value + rng.normal(0.0, noise)
            changed = True
        if re.search(r"_t\d+ms", name) and rng.random() < .04:
            out[name] = np.nan  # sparse sample removal
            changed = True
    if not changed:
        numeric = [c for c in row.index if np.isfinite(pd.to_numeric(
            pd.Series([row[c]]), errors="coerce").iloc[0])]
        if numeric:
            out[numeric[0]] = float(row[numeric[0]]) + 1e-3
    return out


def build_synthetic_splits(league: pd.DataFrame, *, augmentations: int = 3,
                           seed: int = 0) -> dict[str, RankingSplit]:
    id_col = next((c for c in ID_CANDIDATES if c in league), None)
    if id_col is None:
        raise ValueError("League data needs a throw ID")
    rng = np.random.default_rng(seed)
    partitions = split_source_ids(league[id_col].to_numpy(), seed=seed)
    result = {}
    for split, ids in partitions.items():
        rows, sources = [], []
        for source_id in ids:
            source = league.loc[league[id_col] == source_id].iloc[0]
            for aug in range(augmentations):
                query = perturb_throw(source, rng)
                query[id_col] = f"synthetic_{source_id}_{aug}"
                rows.append(query)
                sources.append(source_id)
        result[split] = RankingSplit(pd.DataFrame(rows, columns=league.columns), np.asarray(sources))
    return result


def build_pairs(retriever: WeightedKNNRetriever, split: RankingSplit, *,
                hard_negatives: int = 3, random_negatives: int = 1,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    positives, negatives = [], []
    for (_, query), source_id in zip(split.queries.iterrows(), split.source_ids):
        matrix, _ = component_matrix(retriever, query)
        manual = retriever.retrieve(pd.DataFrame([query]), k=len(league_ids))
        ranked_ids = manual["league_throw_id"].to_numpy()
        hard = [x for x in ranked_ids if x != source_id][:hard_negatives]
        pool = league_ids[league_ids != source_id]
        random = rng.choice(pool, size=min(random_negatives, len(pool)), replace=False).tolist()
        positive_index = int(np.flatnonzero(league_ids == source_id)[0])
        for negative_id in hard + random:
            negatives.append(matrix[int(np.flatnonzero(league_ids == negative_id)[0])])
            positives.append(matrix[positive_index])
    width = len(BASE_COMPONENTS) + len(INTERACTIONS)
    return (np.asarray(positives, float).reshape(-1, width),
            np.asarray(negatives, float).reshape(-1, width))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2] / "out"
    parser.add_argument("--league", type=Path, default=base / "features_league.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "ranking_dataset")
    parser.add_argument("--augmentations", type=int, default=6)
    parser.add_argument("--hard-negatives", type=int, default=12)
    parser.add_argument("--random-negatives", type=int, default=3)
    parser.add_argument(
        "--weight-preset", choices=tuple(PREDEFINED_WEIGHT_SETS), default="best_random",
        help="manual kNN weights used to mine hard negatives (default: best_random)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    league = read_feature_csv(args.league)
    retriever = WeightedKNNRetriever(PREDEFINED_WEIGHT_SETS[args.weight_preset]).fit(league)
    splits = build_synthetic_splits(league, augmentations=args.augmentations, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        queries = split.queries.copy()
        queries.insert(1, "source_league_throw_id", split.source_ids)
        queries.to_csv(args.output_dir / f"{name}_queries.csv", index=False)
        positive, negative = build_pairs(
            retriever, split, hard_negatives=args.hard_negatives,
            random_negatives=args.random_negatives, seed=args.seed,
        )
        np.savez_compressed(
            args.output_dir / f"{name}_pairs.npz", positive=positive, negative=negative,
            feature_names=np.asarray(BASE_COMPONENTS + INTERACTIONS),
        )
    print(f"Wrote synthetic ranking dataset to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
