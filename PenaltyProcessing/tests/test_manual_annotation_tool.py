import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ranking.manual_annotation_tool import (
    ANNOTATION_FIELDS, AnnotationStore, build_candidate_material,
    build_candidate_pool, parse_rating,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ManualAnnotationToolTests(unittest.TestCase):
    def test_candidate_pool_merges_deduplicates_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knn, ranker = root / "knn.csv", root / "ranker.csv"
            _write_csv(knn, [
                {"mocap_throw_id": "m1", "rank": 1, "league_throw_id": "a", "total_distance": .1},
                {"mocap_throw_id": "m1", "rank": 2, "league_throw_id": "b", "total_distance": .2},
                {"mocap_throw_id": "m1", "rank": 3, "league_throw_id": "outside", "total_distance": .3},
                {"mocap_throw_id": "excluded", "rank": 1, "league_throw_id": "z", "total_distance": .1},
            ])
            _write_csv(ranker, [
                {"mocap_throw_id": "m1", "rank": 1, "league_throw_id": "b", "learned_distance": .4},
                {"mocap_throw_id": "m1", "rank": 2, "league_throw_id": "c", "learned_distance": .5},
                {"mocap_throw_id": "excluded", "rank": 1, "league_throw_id": "z", "learned_distance": .2},
            ])
            first = build_candidate_pool(
                ["m1", "excluded"], knn, ranker, top_k=2,
                exclude=["excluded"], seed=7,
            )
            second = build_candidate_pool(
                ["m1", "excluded"], knn, ranker, top_k=2,
                exclude=["excluded"], seed=7,
            )
            self.assertEqual([item.league_throw_id for item in first],
                             [item.league_throw_id for item in second])
            self.assertEqual({item.league_throw_id for item in first}, {"a", "b", "c"})
            common = next(item for item in first if item.league_throw_id == "b")
            self.assertEqual((common.knn_rank, common.ranker_rank), (2, 1))
            self.assertEqual(len({item.key for item in first}), len(first))

    def test_store_saves_schema_ratings_deferral_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knn, ranker, output = root / "knn.csv", root / "ranker.csv", root / "annotations.csv"
            rows = [{"mocap_throw_id": "m1", "rank": 1,
                     "league_throw_id": "a", "total_distance": .1}]
            _write_csv(knn, rows)
            _write_csv(ranker, [{"mocap_throw_id": "m1", "rank": 1,
                                 "league_throw_id": "b", "learned_distance": .2}])
            candidates = build_candidate_pool(["m1"], knn, ranker, seed=3)
            store = AnnotationStore(output, candidates)
            first, second = [candidate.key for candidate in candidates]
            store.annotate(first, (3, 2, 2, 2, 1, 2))
            store.defer(second)
            session_id = store.session_id

            resumed = AnnotationStore(output, candidates)
            self.assertEqual(resumed.session_id, session_id)
            self.assertEqual(resumed.rows[first]["overall_relevance"], "3")
            self.assertEqual(resumed.rows[first]["vertical_match"], "1")
            self.assertEqual(resumed.rows[second]["was_deferred"], "1")
            self.assertEqual(resumed.ordered_keys(), [second])
            with output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), ANNOTATION_FIELDS)

    def test_overall_only_rating_leaves_subratings_empty(self):
        self.assertEqual(parse_rating("2", subratings=False), (2,))
        self.assertEqual(parse_rating("3 2 2 2 1 2", subratings=True), (3, 2, 2, 2, 1, 2))
        self.assertEqual(parse_rating("q", subratings=True), "q")
        with self.assertRaises(ValueError):
            parse_rating("3 3 2 2 2 2", subratings=True)

    def test_material_reuses_reconstruction_and_stitches_pre_por(self):
        mocap = {
            "throw_id": "m1", "por_x_m": "13", "por_y_m": "0", "por_z_m": "1.7",
            "release_speed_m_s": "10", "release_direction_deg": "5",
            "release_height_m": "1.7", "is_bounce": "0",
            "trajectory_json": json.dumps([
                {"frame": -2, "t_since_ms": -50, "x": -.5, "y": 0, "z": 0,
                 "v": 10, "vx": 10, "vy": 0, "vz": 0},
                {"frame": 0, "t_since_ms": 0, "x": 0, "y": 0, "z": 0,
                 "v": 10, "vx": 10, "vy": 0, "vz": 0},
            ]),
        }
        league = {
            "throw_id": "l1", "por_x_m": "14", "por_y_m": ".2", "por_z_m": "1.8",
            "release_speed_m_s": "12", "release_direction_deg": "8",
            "release_height_m": "1.8", "is_bounce": "0",
            "trajectory_json": json.dumps([
                {"frame": 0, "t_since_ms": 0, "x": 0, "y": 0, "z": 0},
                {"frame": 1, "t_since_ms": 50, "x": .6, "y": .03, "z": -.05},
                {"frame": 2, "t_since_ms": 100, "x": 1.2, "y": .06, "z": -.15},
            ]),
        }
        material = build_candidate_material(mocap, league, target_hz=100)
        self.assertEqual(material.por, (13.0, 0.0, 1.7))
        self.assertEqual(material.mocap_pre_por[0]["t_since_ms"], -50)
        self.assertEqual(material.continuation[0]["x"], 13.0)
        self.assertEqual(len(material.original_league_samples), 3)
        self.assertAlmostEqual(material.diagnostics["direction_difference_deg"], 3.0)
        self.assertFalse(material.diagnostics["league_bounce_detected"])
        self.assertIsNone(material.diagnostics["aligned_pogc_z_m"])
        self.assertEqual(material.diagnostics["reconstruction_status"], "validated_reconstruction")
        self.assertIsNone(material.diagnostics["reconstruction_error"])

    def test_material_falls_back_to_aligned_raw_samples_after_reconstruction_error(self):
        mocap = {
            "throw_id": "m1", "por_x_m": "13", "por_y_m": "0", "por_z_m": "1.7",
            "release_speed_m_s": "10", "release_direction_deg": "5",
            "release_height_m": "1.7", "is_bounce": "0",
            "trajectory_json": json.dumps([
                {"frame": 0, "t_since_ms": 0, "x": 0, "y": 0, "z": 0},
                {"frame": 1, "t_since_ms": 50, "x": .5, "y": 0, "z": -.1},
            ]),
        }
        league = {
            "throw_id": "l1", "por_x_m": "14", "por_y_m": ".2", "por_z_m": "1.8",
            "release_speed_m_s": "12", "release_direction_deg": "8",
            "release_height_m": "1.8", "is_bounce": "1",
            "trajectory_json": json.dumps([
                {"frame": 0, "t_since_ms": 0, "x": 0, "y": 0, "z": 0},
                {"frame": 1, "t_since_ms": 50, "x": .6, "y": .03, "z": -.05},
                {"frame": 2, "t_since_ms": 100, "x": 1.2, "y": .06, "z": -.15},
            ]),
        }
        with patch(
            "src.ranking.manual_annotation_tool.reconstruct_league_continuation",
            side_effect=AssertionError(),
        ):
            material = build_candidate_material(mocap, league, target_hz=100)

        self.assertEqual(material.continuation[0]["x"], 13.0)
        self.assertEqual(material.continuation[0]["y"], 0.0)
        self.assertEqual(material.continuation[0]["z"], 1.7)
        self.assertEqual(len(material.original_league_samples), 3)
        self.assertTrue(material.diagnostics["league_bounce_detected"])
        self.assertEqual(material.diagnostics["reconstruction_status"], "fallback_raw_aligned")
        self.assertIn("AssertionError", material.diagnostics["reconstruction_error"])


if __name__ == "__main__":
    unittest.main()
