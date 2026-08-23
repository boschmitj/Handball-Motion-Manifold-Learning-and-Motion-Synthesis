#!/usr/bin/env python3
"""Interactive blinded relevance annotation for reconstructed throw matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from src.combine_full_reconstructed_trajectories import combine_trajectory
    from src.trajectory_reconstruction import (
        XYZ, _read_csv, parse_trajectory_json, reconstruct_league_continuation,
    )
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from combine_full_reconstructed_trajectories import combine_trajectory
    from trajectory_reconstruction import (
        XYZ, _read_csv, parse_trajectory_json, reconstruct_league_continuation,
    )

VISUALIZATION_DIR = Path(__file__).resolve().parents[2] / "visualization"
if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))


ANNOTATION_FIELDS = [
    "mocap_throw_id", "league_throw_id", "query_order", "presentation_order",
    "candidate_id", "overall_relevance", "speed_match", "direction_match",
    "trajectory_shape_match", "vertical_match", "throw_type_match",
    "annotation_timestamp", "annotation_order", "session_id", "was_deferred",
    "knn_in_top_k", "knn_rank", "knn_distance", "ranker_in_top_k",
    "ranker_rank", "ranker_distance", "mocap_release_speed_m_s",
    "league_release_speed_m_s", "transition_speed_difference_m_s",
    "mocap_reported_release_speed_m_s", "league_reported_release_speed_m_s",
    "mocap_release_direction_deg", "league_release_direction_deg",
    "direction_difference_deg", "mocap_release_height_m", "league_release_height_m",
    "release_height_difference_m", "mocap_has_ground_contact",
    "league_bounce_detected", "league_original_pogc_z_m", "aligned_pogc_z_m",
    "expected_contact_z_m", "ground_contact_error_m", "bounce_vertical_fit_rmse",
    "bounce_vz_in_m_s", "bounce_vz_out_m_s", "bounce_reconstruction_rmse_m",
    "bounce_reconstruction_max_error_m", "bounce_fit_valid",
]
RATING_FIELDS = [
    "overall_relevance", "speed_match", "direction_match",
    "trajectory_shape_match", "vertical_match", "throw_type_match",
]


@dataclass(frozen=True)
class Candidate:
    mocap_throw_id: str
    league_throw_id: str
    query_order: int
    presentation_order: int
    candidate_id: str
    knn_rank: int | None = None
    knn_distance: float | None = None
    ranker_rank: int | None = None
    ranker_distance: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.mocap_throw_id, self.league_throw_id


@dataclass(frozen=True)
class CandidateMaterial:
    mocap_pre_por: list[dict[str, Any]]
    continuation: list[dict[str, Any]]
    original_league_samples: list[dict[str, Any]]
    por: tuple[float, float, float]
    diagnostics: dict[str, Any]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _score(row: Mapping[str, Any], candidates: Sequence[str]) -> float | None:
    return next((_optional_float(row.get(name)) for name in candidates
                 if _optional_float(row.get(name)) is not None), None)


def _top_by_query(path: Path, top_k: int, *, score_columns: Sequence[str]) -> dict[str, dict[str, tuple[int, float | None]]]:
    rows, _ = _read_csv(path)
    result: dict[str, dict[str, tuple[int, float | None]]] = {}
    for row in rows:
        mocap_id, league_id = str(row.get("mocap_throw_id", "")), str(row.get("league_throw_id", ""))
        rank = _optional_int(row.get("rank", row.get("match_rank")))
        if not mocap_id or not league_id or rank is None or not 1 <= rank <= top_k:
            continue
        by_league = result.setdefault(mocap_id, {})
        previous = by_league.get(league_id)
        entry = (rank, _score(row, score_columns))
        if previous is None or rank < previous[0]:
            by_league[league_id] = entry
    return result


def build_candidate_pool(
    mocap_ids: Sequence[str], knn_results: Path, ranker_results: Path, *,
    top_k: int = 10, exclude: Sequence[str] = (), seed: int = 42,
) -> list[Candidate]:
    """Merge, deduplicate, and independently shuffle top-k candidates per query."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    excluded = set(map(str, exclude))
    knn = _top_by_query(
        knn_results, top_k, score_columns=("total_distance", "distance", "score"),
    )
    ranker = _top_by_query(
        ranker_results, top_k,
        score_columns=("learned_distance", "ranker_distance", "distance", "score"),
    )
    candidates: list[Candidate] = []
    query_order = 0
    for mocap_id in map(str, mocap_ids):
        if mocap_id in excluded:
            continue
        league_ids = sorted(set(knn.get(mocap_id, {})) | set(ranker.get(mocap_id, {})))
        if not league_ids:
            continue
        query_order += 1
        digest = hashlib.sha256(f"{seed}:{mocap_id}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(league_ids)
        for presentation_order, league_id in enumerate(league_ids, start=1):
            knn_value = knn.get(mocap_id, {}).get(league_id)
            ranker_value = ranker.get(mocap_id, {}).get(league_id)
            candidates.append(Candidate(
                mocap_throw_id=mocap_id, league_throw_id=league_id,
                query_order=query_order, presentation_order=presentation_order,
                candidate_id=f"Q{query_order:03d}-C{presentation_order:03d}",
                knn_rank=None if knn_value is None else knn_value[0],
                knn_distance=None if knn_value is None else knn_value[1],
                ranker_rank=None if ranker_value is None else ranker_value[0],
                ranker_distance=None if ranker_value is None else ranker_value[1],
            ))
    return candidates


class AnnotationStore:
    """CSV-backed candidate database with atomic save-after-every-action."""

    def __init__(self, path: Path, candidates: Sequence[Candidate]) -> None:
        self.path = path
        self.active_keys = {candidate.key for candidate in candidates}
        self.rows: dict[tuple[str, str], dict[str, str]] = {}
        if path.exists():
            rows, _ = _read_csv(path)
            for raw in rows:
                row = {field: str(raw.get(field, "") or "") for field in ANNOTATION_FIELDS}
                key = (row["mocap_throw_id"], row["league_throw_id"])
                if all(key):
                    self.rows[key] = row
        existing_session = next((row["session_id"] for row in self.rows.values()
                                 if row["session_id"]), None)
        self.session_id = existing_session or uuid.uuid4().hex[:12]
        maximum_presentation: dict[str, int] = {}
        for row in self.rows.values():
            mocap_id = row["mocap_throw_id"]
            maximum_presentation[mocap_id] = max(
                maximum_presentation.get(mocap_id, 0),
                int(row.get("presentation_order") or 0),
            )
        for candidate in candidates:
            previous = self.rows.get(candidate.key, {})
            base = {field: str(previous.get(field, "") or "") for field in ANNOTATION_FIELDS}
            if base["presentation_order"]:
                presentation_order = int(base["presentation_order"])
            elif candidate.mocap_throw_id in maximum_presentation:
                presentation_order = maximum_presentation[candidate.mocap_throw_id] + 1
            else:
                presentation_order = candidate.presentation_order
            maximum_presentation[candidate.mocap_throw_id] = max(
                maximum_presentation.get(candidate.mocap_throw_id, 0), presentation_order,
            )
            query_order = int(base["query_order"] or candidate.query_order)
            base.update({
                "mocap_throw_id": candidate.mocap_throw_id,
                "league_throw_id": candidate.league_throw_id,
                "query_order": str(query_order),
                "presentation_order": str(presentation_order),
                "candidate_id": base["candidate_id"] or f"Q{query_order:03d}-C{presentation_order:03d}",
                "session_id": base["session_id"] or self.session_id,
                "was_deferred": base["was_deferred"] or "0",
                "knn_in_top_k": "1" if candidate.knn_rank is not None else "0",
                "knn_rank": "" if candidate.knn_rank is None else str(candidate.knn_rank),
                "knn_distance": "" if candidate.knn_distance is None else str(candidate.knn_distance),
                "ranker_in_top_k": "1" if candidate.ranker_rank is not None else "0",
                "ranker_rank": "" if candidate.ranker_rank is None else str(candidate.ranker_rank),
                "ranker_distance": "" if candidate.ranker_distance is None else str(candidate.ranker_distance),
            })
            self.rows[candidate.key] = base
        self.save()

    @staticmethod
    def completed(row: Mapping[str, str]) -> bool:
        return row.get("overall_relevance") in {"0", "1", "2", "3"}

    def ordered_keys(self, *, include_completed: bool = False) -> list[tuple[str, str]]:
        rows = [row for row in self.rows.values()
                if (row["mocap_throw_id"], row["league_throw_id"]) in self.active_keys
                and (include_completed or not self.completed(row))]
        rows.sort(key=lambda row: (
            _as_bool(row.get("was_deferred")),
            int(row.get("query_order") or 0), int(row.get("presentation_order") or 0),
        ))
        return [(row["mocap_throw_id"], row["league_throw_id"]) for row in rows]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.rows.values(), key=lambda row: (
            int(row.get("query_order") or 0), int(row.get("presentation_order") or 0),
        ))
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", newline="", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_name = handle.name
                writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
                writer.writeheader()
                writer.writerows(ordered)
            os.replace(temporary_name, self.path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def update_diagnostics(self, key: tuple[str, str], diagnostics: Mapping[str, Any]) -> None:
        row = self.rows[key]
        for field, value in diagnostics.items():
            if field in ANNOTATION_FIELDS:
                row[field] = "" if value is None else str(value)
        self.save()

    def annotate(self, key: tuple[str, str], ratings: Sequence[int]) -> None:
        row = self.rows[key]
        for field in RATING_FIELDS:
            row[field] = ""
        for field, rating in zip(RATING_FIELDS, ratings):
            row[field] = str(rating)
        row["annotation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if not row["annotation_order"]:
            completed_orders = [int(value["annotation_order"]) for value in self.rows.values()
                                if value.get("annotation_order")]
            row["annotation_order"] = str(max(completed_orders, default=0) + 1)
        row["was_deferred"] = "0"
        self.save()

    def defer(self, key: tuple[str, str]) -> None:
        row = self.rows[key]
        for field in RATING_FIELDS:
            row[field] = ""
        row["was_deferred"] = "1"
        self.save()


def parse_rating(value: str, *, subratings: bool) -> tuple[int, ...] | str:
    normalized = value.strip().lower()
    if normalized in {"q", "b", "s"}:
        return normalized
    tokens = normalized.split()
    expected = 6 if subratings else 1
    if len(tokens) != expected:
        raise ValueError(f"enter exactly {expected} rating value{'s' if expected > 1 else ''}")
    try:
        ratings = tuple(map(int, tokens))
    except ValueError as exc:
        raise ValueError("ratings must be integers") from exc
    if ratings[0] not in range(4):
        raise ValueError("overall relevance must be 0, 1, 2, or 3")
    if any(rating not in range(3) for rating in ratings[1:]):
        raise ValueError("subratings must be 0, 1, or 2")
    return ratings


def _speed(point: Mapping[str, Any]) -> float | None:
    components = [_optional_float(point.get(name)) for name in ("vx", "vy", "vz")]
    if all(value is not None for value in components):
        return float(np.linalg.norm(components))
    return _optional_float(point.get("v"))


def _nearest_speed(points: Sequence[Mapping[str, Any]], *, before_por: bool) -> float | None:
    eligible = [point for point in points
                if (float(point["t_since_ms"]) < 0 if before_por
                    else float(point["t_since_ms"]) > 0)
                and _speed(point) is not None]
    if not eligible:
        return None
    selected = (max(eligible, key=lambda point: float(point["t_since_ms"]))
                if before_por else min(eligible, key=lambda point: float(point["t_since_ms"])))
    return _speed(selected)


def _angle_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_candidate_material(
    mocap: Mapping[str, Any], league: Mapping[str, Any], *, target_hz: float = 300.0,
    ground_z: float = .095, pre_por_ms: float = 250.0,
) -> CandidateMaterial:
    """Reuse the production reconstruction/stitching path and derive diagnostics."""
    continuation = reconstruct_league_continuation(
        league, mocap, target_hz=target_hz, ground_z=ground_z,
    )
    combined = combine_trajectory(
        mocap, json.dumps(continuation, separators=(",", ":"), allow_nan=False),
    )
    mocap_pre = [point for point in combined
                 if -pre_por_ms <= float(point["t_since_ms"]) < 0]
    original_samples = [point for point in continuation if point.get("is_original_sample")]
    por = tuple(float(mocap[f"por_{axis}_m"]) for axis in XYZ)
    mocap_points = parse_trajectory_json(mocap)
    mocap_transition_speed = _nearest_speed(mocap_points, before_por=True)
    league_transition_speed = _nearest_speed(continuation, before_por=False)
    mocap_reported_speed = _optional_float(mocap.get("release_speed_m_s"))
    league_reported_speed = _optional_float(league.get("release_speed_m_s"))
    if mocap_transition_speed is None:
        mocap_transition_speed = mocap_reported_speed
    if league_transition_speed is None:
        league_transition_speed = league_reported_speed
    mocap_direction = _optional_float(mocap.get("release_direction_deg"))
    league_direction = _optional_float(league.get("release_direction_deg"))
    mocap_height = _optional_float(mocap.get("release_height_m"))
    league_height = _optional_float(league.get("release_height_m"))
    bounce = next((point for point in continuation if point.get("event") == "bounce"), None)
    aligned_pogc = None if bounce is None else _optional_float(bounce.get("z"))
    diagnostics: dict[str, Any] = {
        "mocap_release_speed_m_s": mocap_transition_speed,
        "league_release_speed_m_s": league_transition_speed,
        "transition_speed_difference_m_s": (
            None if mocap_transition_speed is None or league_transition_speed is None
            else abs(mocap_transition_speed - league_transition_speed)
        ),
        "mocap_reported_release_speed_m_s": mocap_reported_speed,
        "league_reported_release_speed_m_s": league_reported_speed,
        "mocap_release_direction_deg": mocap_direction,
        "league_release_direction_deg": league_direction,
        "direction_difference_deg": _angle_difference(mocap_direction, league_direction),
        "mocap_release_height_m": mocap_height,
        "league_release_height_m": league_height,
        "release_height_difference_m": (
            None if mocap_height is None or league_height is None
            else abs(mocap_height - league_height)
        ),
        "mocap_has_ground_contact": _as_bool(mocap.get("is_bounce")),
        "league_bounce_detected": bounce is not None,
        "league_original_pogc_z_m": None if bounce is None else ground_z,
        "aligned_pogc_z_m": aligned_pogc,
        "expected_contact_z_m": ground_z,
        "ground_contact_error_m": (
            None if aligned_pogc is None else aligned_pogc - ground_z
        ),
    }
    bounce_fields = (
        "bounce_vertical_fit_rmse", "bounce_vz_in_m_s", "bounce_vz_out_m_s",
        "bounce_reconstruction_rmse_m", "bounce_reconstruction_max_error_m",
        "bounce_fit_valid",
    )
    diagnostics.update({field: None if bounce is None else bounce.get(field)
                        for field in bounce_fields})
    return CandidateMaterial(mocap_pre, continuation, original_samples, por, diagnostics)


def _rubric(subratings: bool) -> str:
    overall = (
        "Overall: 3 natural | 2 plausible, one discrepancy | "
        "1 questionable | 0 incompatible"
    )
    details = ("\nSubratings: speed direction trajectory vertical throw_type; "
               "2 good | 1 noticeable mismatch | 0 poor") if subratings else ""
    return overall + details + "\nCommands: q quit | b edit previous | s defer"


def run_annotation(
    *, mocap_path: Path, league_path: Path, knn_results: Path,
    ranker_results: Path, output: Path, exclude: Sequence[str] = (), top_k: int = 10,
    subratings: bool = True, seed: int = 42, target_hz: float = 300.0,
    ground_z: float = .095, pre_por_ms: float = 250.0,
) -> int:
    # Keep candidate-pool and persistence utilities usable in headless/test
    # contexts without importing a plotting backend.
    import matplotlib.pyplot as plt
    from manual_annotation_view import render_annotation_figure

    mocap_rows, _ = _read_csv(mocap_path)
    league_rows, _ = _read_csv(league_path)
    mocap_by_id = {str(row["throw_id"]): row for row in mocap_rows}
    league_by_id = {str(row["throw_id"]): row for row in league_rows}
    candidates = build_candidate_pool(
        list(mocap_by_id), knn_results, ranker_results, top_k=top_k,
        exclude=exclude, seed=seed,
    )
    missing = sorted({candidate.league_throw_id for candidate in candidates} - set(league_by_id))
    if missing:
        raise KeyError(f"{len(missing)} candidate League IDs are missing from {league_path}")
    store = AnnotationStore(output, candidates)
    queue = deque(store.ordered_keys())
    deferred_this_run: set[tuple[str, str]] = set()
    history: list[tuple[str, str]] = []
    active_query_ids = sorted(
        {key[0] for key in store.active_keys},
        key=lambda mocap_id: min(
            int(store.rows[key]["query_order"])
            for key in store.active_keys if key[0] == mocap_id
        ),
    )
    query_total = len(active_query_ids)
    query_position = {mocap_id: index for index, mocap_id in enumerate(active_query_ids, 1)}
    candidate_position: dict[tuple[str, str], int] = {}
    query_candidate_totals: dict[str, int] = {}
    for mocap_id in active_query_ids:
        keys = sorted(
            (key for key in store.active_keys if key[0] == mocap_id),
            key=lambda key: int(store.rows[key]["presentation_order"]),
        )
        query_candidate_totals[mocap_id] = len(keys)
        candidate_position.update({key: index for index, key in enumerate(keys, 1)})

    while queue:
        key = queue.popleft()
        if key in deferred_this_run:
            continue
        row = store.rows[key]
        material = build_candidate_material(
            mocap_by_id[key[0]], league_by_id[key[1]], target_hz=target_hz,
            ground_z=ground_z, pre_por_ms=pre_por_ms,
        )
        store.update_diagnostics(key, material.diagnostics)
        active_rows = [store.rows[item] for item in store.active_keys]
        completed = sum(store.completed(item) for item in active_rows)
        print("\n" + "=" * 72)
        print(f"Mocap throw {query_position[key[0]]}/{query_total}")
        print(f"Candidate {candidate_position[key]}/{query_candidate_totals[key[0]]} ({row['candidate_id']})")
        print(f"Total annotations {completed}/{len(active_rows)}")
        print(_rubric(subratings))
        previous = [row[field] for field in RATING_FIELDS if row.get(field) != ""]
        if previous:
            print("Current saved rating:", " ".join(previous))

        fig = render_annotation_figure(
            query_label=f"Query {query_position[key[0]]}", candidate_label=row["candidate_id"],
            mocap_pre_por=material.mocap_pre_por, continuation=material.continuation,
            original_league_samples=material.original_league_samples, por=material.por,
            diagnostics=material.diagnostics, ground_z=ground_z,
        )
        plt.show(block=False)
        plt.pause(.05)
        try:
            while True:
                prompt = ("rating [overall speed direction trajectory vertical throw_type]: "
                          if subratings else "overall relevance [0-3]: ")
                try:
                    raw_input = input(prompt)
                except (EOFError, KeyboardInterrupt):
                    print("\nInput closed; quitting cleanly.")
                    action = "q"
                    break
                try:
                    action = parse_rating(raw_input, subratings=subratings)
                except ValueError as error:
                    print(f"Invalid input: {error}")
                    continue
                break
        finally:
            plt.close(fig)

        if action == "q":
            print(f"Saved {completed} completed annotations to {output}")
            return completed
        if action == "s":
            store.defer(key)
            deferred_this_run.add(key)
            print(f"Deferred {row['candidate_id']}; progress saved.")
            continue
        if action == "b":
            if not history:
                print("No previous annotation is available to edit.")
                queue.appendleft(key)
                continue
            previous_key = history.pop()
            queue.appendleft(key)
            queue.appendleft(previous_key)
            print("Returning to the previous annotation.")
            continue

        assert isinstance(action, tuple)
        store.annotate(key, action)
        history.append(key)
        labels = dict(zip(RATING_FIELDS, action))
        print("Recorded:", ", ".join(f"{name}={value}" for name, value in labels.items()))

    active_rows = [store.rows[item] for item in store.active_keys]
    completed = sum(store.completed(item) for item in active_rows)
    deferred = sum(not store.completed(item) for item in active_rows)
    print(f"Annotation pass complete: {completed}/{len(active_rows)} rated; {deferred} deferred.")
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parents[2] / "out" / "throw_features"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mocap", type=Path, default=base / "raw_mocap.csv")
    parser.add_argument("--league", type=Path, default=base / "raw_league.csv")
    parser.add_argument(
        "--knn-results", type=Path,
        default=base / "learned_ranker" / "manual_ranked_candidates.csv",
    )
    parser.add_argument(
        "--ranker-results", type=Path,
        default=base / "learned_ranker" / "learned_ranked_candidates.csv",
    )
    parser.add_argument("--output", type=Path, default=base / "manual_relevance_annotations.csv")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--subratings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-hz", type=float, default=300.0)
    parser.add_argument("--ground-z", type=float, default=.095)
    parser.add_argument("--pre-por-ms", type=float, default=250.0)
    args = parser.parse_args(argv)
    run_annotation(
        mocap_path=args.mocap, league_path=args.league,
        knn_results=args.knn_results, ranker_results=args.ranker_results,
        output=args.output, exclude=args.exclude, top_k=args.top_k,
        subratings=args.subratings, seed=args.seed, target_hz=args.target_hz,
        ground_z=args.ground_z, pre_por_ms=args.pre_por_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
