"""Train, evaluate, compare, and export the learned linear throw ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Support ``python src/ranking/evaluate_ranker.py`` from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.weighted_knn import PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, read_feature_csv
    from ranking.build_ranking_dataset import (
        BASE_COMPONENTS, INTERACTIONS, RankingSplit, build_pairs,
        build_synthetic_splits, component_matrix,
    )
    from ranking.linear_pairwise_ranker import LinearPairwiseRanker
else:
    from src.model.weighted_knn import PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, read_feature_csv
    from src.ranking.build_ranking_dataset import (
        BASE_COMPONENTS, INTERACTIONS, RankingSplit, build_pairs,
        build_synthetic_splits, component_matrix,
    )
    from src.ranking.linear_pairwise_ranker import LinearPairwiseRanker


def retrieval_metrics(ranks: list[int]) -> dict[str, float]:
    values = np.asarray(ranks, float)
    if not len(values):
        return {"recall@1": np.nan, "recall@3": np.nan, "recall@5": np.nan, "mrr": np.nan}
    return {"recall@1": float(np.mean(values <= 1)),
            "recall@3": float(np.mean(values <= 3)),
            "recall@5": float(np.mean(values <= 5)),
            "mrr": float(np.mean(1.0 / values))}


def rank_queries(model: LinearPairwiseRanker, retriever: WeightedKNNRetriever,
                 queries: pd.DataFrame, *, k: int | None = None,
                 minimum_common_trajectory_points: int = 2,
                 minimum_overlap_ratio: float = .5) -> pd.DataFrame:
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    query_id = next(c for c in ("throw_id", "mocap_throw_id", "league_throw_id") if c in queries)
    rows = []
    for _, query in queries.iterrows():
        matrix, names = component_matrix(retriever, query)
        matrix = align_component_matrix(matrix, names, model.feature_names)
        scores = np.asarray(model.score(matrix), float)
        diagnostics = retriever.distance_components(query)
        common = diagnostics["trajectory_common_point_count"]
        query_count = diagnostics["trajectory_query_point_count"]
        overlap = diagnostics["trajectory_overlap_ratio"]
        evidence = diagnostics["trajectory_evidence_ratio"]
        # A query with less evidence than requested cannot satisfy the literal
        # threshold; require all of its available points instead. This retains
        # a ranking while making its low confidence explicit.
        required_points = np.minimum(minimum_common_trajectory_points, query_count)
        eligible = ((common >= required_points) & (overlap >= minimum_overlap_ratio))
        if np.any(eligible):
            eligible_indices = np.flatnonzero(eligible)
            order = eligible_indices[np.argsort(scores[eligible_indices], kind="stable")][:k]
        else:
            # Preserve a diagnostic fallback instead of returning no rows. All
            # returned candidates are explicitly marked below as failing the
            # configured overlap requirement.
            order = np.argsort(scores, kind="stable")[:k]
        for rank, index in enumerate(order, 1):
            row = {"mocap_throw_id": query[query_id], "rank": rank,
                   "league_throw_id": league_ids[index], "learned_distance": scores[index],
                   "trajectory_common_point_count": int(common[index]),
                   "trajectory_query_point_count": int(query_count[index]),
                   "trajectory_overlap_ratio": float(overlap[index]),
                   "trajectory_evidence_ratio": float(evidence[index]),
                   "ranking_confidence": float(overlap[index] * evidence[index]),
                   "meets_minimum_overlap": bool(eligible[index])}
            row.update({f"{name}_distance": matrix[index, j]
                        for j, name in enumerate(model.feature_names)})
            rows.append(row)
    return pd.DataFrame(rows)


def align_component_matrix(
    matrix: np.ndarray, available_names: list[str], required_names: list[str],
) -> np.ndarray:
    """Select/reorder components, preserving compatibility with old models."""
    missing = [name for name in required_names if name not in available_names]
    if missing:
        raise ValueError(f"Ranker requires unavailable components: {missing}")
    indices = [available_names.index(name) for name in required_names]
    return matrix[:, indices]


def evaluate_synthetic(model: LinearPairwiseRanker, retriever: WeightedKNNRetriever,
                       split: RankingSplit) -> dict[str, float]:
    return retrieval_metrics(synthetic_ranks(model, retriever, split).tolist())


def synthetic_ranks(model: LinearPairwiseRanker, retriever: WeightedKNNRetriever,
                    split: RankingSplit) -> np.ndarray:
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    ranks = []
    for (_, query), source_id in zip(split.queries.iterrows(), split.source_ids):
        matrix, names = component_matrix(retriever, query)
        matrix = align_component_matrix(matrix, names, model.feature_names)
        order = np.argsort(model.score(matrix), kind="stable")
        ranks.append(int(np.flatnonzero(league_ids[order] == source_id)[0]) + 1)
    return np.asarray(ranks, dtype=int)


def evaluate_manual_synthetic(retriever: WeightedKNNRetriever,
                              split: RankingSplit) -> dict[str, float]:
    return retrieval_metrics(manual_synthetic_ranks(retriever, split).tolist())


def manual_synthetic_ranks(retriever: WeightedKNNRetriever,
                           split: RankingSplit) -> np.ndarray:
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    ranks = []
    for (_, query), source_id in zip(split.queries.iterrows(), split.source_ids):
        components = retriever.distance_components(query)
        distances = np.divide(
            components["_numerator"], components["_denominator"],
            out=np.full_like(components["_numerator"], np.inf),
            where=components["_denominator"] > 0,
        )
        order = np.argsort(distances, kind="stable")
        ranks.append(int(np.flatnonzero(league_ids[order] == source_id)[0]) + 1)
    return np.asarray(ranks, dtype=int)


def load_ranking_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, RankingSplit], dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Load the query splits and pair matrices written by build_ranking_dataset.py."""
    splits: dict[str, RankingSplit] = {}
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_names: list[str] | None = None

    for name in ("train", "validation", "test"):
        queries_path = dataset_dir / f"{name}_queries.csv"
        pairs_path = dataset_dir / f"{name}_pairs.npz"
        if not queries_path.exists() or not pairs_path.exists():
            raise FileNotFoundError(
                f"Ranking dataset needs {queries_path.name} and {pairs_path.name}"
            )

        queries = pd.read_csv(queries_path)
        if "source_league_throw_id" not in queries:
            raise ValueError(f"{queries_path} has no source_league_throw_id column")
        source_ids = queries.pop("source_league_throw_id").to_numpy()
        splits[name] = RankingSplit(queries=queries, source_ids=source_ids)

        with np.load(pairs_path) as archive:
            positive = np.asarray(archive["positive"], float)
            negative = np.asarray(archive["negative"], float)
            names = [str(value) for value in archive["feature_names"].tolist()]
        if positive.shape != negative.shape or positive.ndim != 2:
            raise ValueError(f"Invalid positive/negative shapes in {pairs_path}")
        if positive.shape[1] != len(names):
            raise ValueError(f"Pair width and feature_names disagree in {pairs_path}")
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature names in {pairs_path} differ from the train split")
        pairs[name] = (positive, negative)

    assert feature_names is not None
    return splits, pairs, feature_names


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, float)[:k]
    discounts = np.log2(np.arange(2, len(rel) + 2))
    dcg = np.sum((np.power(2.0, rel) - 1.0) / discounts)
    ideal = np.sort(np.asarray(relevance, float))[::-1][:k]
    idcg = np.sum((np.power(2.0, ideal) - 1.0) /
                  np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_manual_ndcg(ranked: pd.DataFrame, judgments: pd.DataFrame,
                         k: int = 5) -> float:
    required = {"mocap_throw_id", "league_throw_id", "relevance"}
    if not required <= set(judgments):
        raise ValueError(f"Manual relevance CSV needs columns {sorted(required)}")
    scores = []
    for query_id, truth in judgments.groupby("mocap_throw_id"):
        order = ranked[ranked.mocap_throw_id == query_id].sort_values("rank").head(k)
        relevance = dict(zip(truth.league_throw_id, truth.relevance))
        observed = np.asarray([relevance.get(candidate, 0) for candidate in order.league_throw_id])
        ideal = np.sort(pd.to_numeric(truth.relevance, errors="coerce").fillna(0).to_numpy())[::-1][:k]
        dcg = np.sum((np.power(2.0, observed) - 1.0) /
                     np.log2(np.arange(2, len(observed) + 2)))
        idcg = np.sum((np.power(2.0, ideal) - 1.0) /
                      np.log2(np.arange(2, len(ideal) + 2)))
        scores.append(float(dcg / idcg) if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else np.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2] / "out" / "throw_features"
    parser.add_argument("--league", type=Path, default=base / "features_league.csv")
    parser.add_argument(
        "--raw-league", type=Path, default=base / "raw_league.csv",
        help="Raw League trajectories used when --dataset-dir is not supplied",
    )
    parser.add_argument("--mocap", type=Path, default=base / "features_mocap.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "learned_ranker")
    parser.add_argument(
        "--dataset-dir", type=Path,
        help=("Load train/validation/test query CSVs and pair NPZs produced by "
              "build_ranking_dataset.py instead of regenerating them"),
    )
    parser.add_argument("--manual-relevance", type=Path)
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--hard-negatives", type=int, default=3)
    parser.add_argument("--random-negatives", type=int, default=1)
    parser.add_argument(
        "--weight-preset", choices=tuple(PREDEFINED_WEIGHT_SETS), default="best_random",
        help=("manual kNN weights used for hard-negative mining and baseline comparison "
              "(default: best_random)"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--regularization-grid", default="0,0.01,0.1,1,10",
        help="Comma-separated L2 strengths selected by validation pairwise loss",
    )
    parser.add_argument(
        "--minimum-weight", type=float, default=.01,
        help="Floor for every component budget (default: 0.01)",
    )
    parser.add_argument(
        "--minimum-common-trajectory-points", type=int, default=2,
        help="Minimum shared elapsed-time points for real-query candidates",
    )
    parser.add_argument(
        "--minimum-overlap-ratio", type=float, default=.5,
        help="Minimum fraction of observed query points covered by a candidate",
    )
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.minimum_common_trajectory_points < 0:
        parser.error("--minimum-common-trajectory-points must be non-negative")
    if not 0.0 <= args.minimum_overlap_ratio <= 1.0:
        parser.error("--minimum-overlap-ratio must be between zero and one")

    league = read_feature_csv(args.league)
    retriever = WeightedKNNRetriever(PREDEFINED_WEIGHT_SETS[args.weight_preset]).fit(league)
    if args.dataset_dir is not None:
        print(f"Loading ranking dataset from {args.dataset_dir}")
        splits, pairs, names = load_ranking_dataset(args.dataset_dir)
    else:
        print("Generating synthetic ranking dataset in memory")
        raw_league = pd.read_csv(args.raw_league, sep=";")
        splits = build_synthetic_splits(
            league, raw_league, augmentations=args.augmentations, seed=args.seed,
        )
        pairs = {name: build_pairs(
            retriever, split, hard_negatives=args.hard_negatives,
            random_negatives=args.random_negatives, seed=args.seed,
        ) for name, split in splits.items()}
        names = BASE_COMPONENTS + INTERACTIONS
    regularization_grid = [
        float(value) for value in args.regularization_grid.split(",") if value.strip()
    ]
    if not regularization_grid or any(
        not np.isfinite(value) or value < 0 for value in regularization_grid
    ):
        parser.error("--regularization-grid needs non-negative finite values")
    initial_model = LinearPairwiseRanker(names, minimum_weight=args.minimum_weight)
    initial_loss = initial_model.pairwise_loss(*pairs["train"])
    candidates: list[tuple[float, float, LinearPairwiseRanker]] = []
    selection_pairs = (pairs["validation"] if len(pairs["validation"][0])
                       else pairs["train"])
    for strength in regularization_grid:
        candidate = LinearPairwiseRanker(
            names, regularization_strength=strength,
            minimum_weight=args.minimum_weight,
        ).fit(*pairs["train"])
        validation_loss = candidate.pairwise_loss(*selection_pairs)
        candidates.append((validation_loss, strength, candidate))
    validation_loss, selected_strength, model = min(
        candidates, key=lambda item: (item[0], item[1]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "learned_weights.json")
    metrics: dict[str, object] = {
        "train_pairwise_loss_before": initial_loss,
        "selected_regularization_strength": selected_strength,
        "minimum_component_weight": args.minimum_weight,
        "regularization_candidates": [
            {"strength": strength, "validation_pairwise_loss": loss}
            for loss, strength, _ in candidates
        ],
    }
    for name, split in splits.items():
        metrics[f"{name}_pairwise_loss"] = model.pairwise_loss(*pairs[name]) if len(pairs[name][0]) else np.nan
        learned_ranks = synthetic_ranks(model, retriever, split)
        manual_ranks = manual_synthetic_ranks(retriever, split)
        metrics.update({f"learned_{name}_{key}": value for key, value in
                        retrieval_metrics(learned_ranks.tolist()).items()})
        metrics.update({f"manual_{name}_{key}": value for key, value in
                        retrieval_metrics(manual_ranks.tolist()).items()})
        if "augmentation_severity" in split.queries:
            severities = split.queries["augmentation_severity"].astype(str).to_numpy()
            for severity in sorted(set(severities)):
                selected = severities == severity
                metrics.update({
                    f"learned_{name}_{severity}_{key}": value
                    for key, value in retrieval_metrics(
                        learned_ranks[selected].tolist()
                    ).items()
                })
                metrics.update({
                    f"manual_{name}_{severity}_{key}": value
                    for key, value in retrieval_metrics(
                        manual_ranks[selected].tolist()
                    ).items()
                })

    if args.mocap.exists():
        mocap = read_feature_csv(args.mocap)
        learned = rank_queries(
            model, retriever, mocap, k=args.k,
            minimum_common_trajectory_points=args.minimum_common_trajectory_points,
            minimum_overlap_ratio=args.minimum_overlap_ratio,
        )
        manual = retriever.retrieve(mocap, k=args.k)
        learned.to_csv(args.output_dir / "learned_ranked_candidates.csv", index=False)
        manual.to_csv(args.output_dir / "manual_ranked_candidates.csv", index=False)
        if args.manual_relevance:
            judgments = read_feature_csv(args.manual_relevance)
            metrics[f"learned_ndcg@{args.k}"] = evaluate_manual_ndcg(learned, judgments, args.k)
            metrics[f"manual_ndcg@{args.k}"] = evaluate_manual_ndcg(manual, judgments, args.k)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Wrote learned/manual comparison to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
