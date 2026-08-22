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
                 queries: pd.DataFrame, *, k: int | None = None) -> pd.DataFrame:
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    query_id = next(c for c in ("throw_id", "mocap_throw_id", "league_throw_id") if c in queries)
    rows = []
    for _, query in queries.iterrows():
        matrix, names = component_matrix(retriever, query)
        scores = model.score(matrix)
        order = np.argsort(scores, kind="stable")[:k]
        for rank, index in enumerate(order, 1):
            row = {"mocap_throw_id": query[query_id], "rank": rank,
                   "league_throw_id": league_ids[index], "learned_distance": scores[index]}
            row.update({f"{name}_distance": matrix[index, j] for j, name in enumerate(names)})
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_synthetic(model: LinearPairwiseRanker, retriever: WeightedKNNRetriever,
                       split: RankingSplit) -> dict[str, float]:
    league_ids = retriever.league[retriever.id_column].to_numpy()  # type: ignore[index]
    ranks = []
    for (_, query), source_id in zip(split.queries.iterrows(), split.source_ids):
        matrix, _ = component_matrix(retriever, query)
        order = np.argsort(model.score(matrix), kind="stable")
        ranks.append(int(np.flatnonzero(league_ids[order] == source_id)[0]) + 1)
    return retrieval_metrics(ranks)


def evaluate_manual_synthetic(retriever: WeightedKNNRetriever,
                              split: RankingSplit) -> dict[str, float]:
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
    return retrieval_metrics(ranks)


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
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

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
    model = LinearPairwiseRanker(names)
    initial_loss = model.pairwise_loss(*pairs["train"])
    model.fit(*pairs["train"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "learned_weights.json")
    metrics: dict[str, object] = {"train_pairwise_loss_before": initial_loss}
    for name, split in splits.items():
        metrics[f"{name}_pairwise_loss"] = model.pairwise_loss(*pairs[name]) if len(pairs[name][0]) else np.nan
        metrics.update({f"learned_{name}_{key}": value
                        for key, value in evaluate_synthetic(model, retriever, split).items()})
        metrics.update({f"manual_{name}_{key}": value
                        for key, value in evaluate_manual_synthetic(retriever, split).items()})

    if args.mocap.exists():
        mocap = read_feature_csv(args.mocap)
        learned = rank_queries(model, retriever, mocap, k=args.k)
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
