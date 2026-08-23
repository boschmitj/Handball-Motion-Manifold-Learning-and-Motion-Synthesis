import unittest
import json

import numpy as np
import pandas as pd

from src.model.weighted_knn import WeightedKNNRetriever
from src.ranking.build_ranking_dataset import (
    BASE_COMPONENTS, PERTURBATION_PROFILES, add_interactions,
    build_synthetic_splits, component_matrix, perturb_raw_throw,
)
from src.ranking.evaluate_ranker import rank_queries
from src.ranking.linear_pairwise_ranker import LinearPairwiseRanker


class LearnedRankerTests(unittest.TestCase):
    def test_weights_are_nonnegative_and_sum_to_one(self):
        model = LinearPairwiseRanker(["a", "b", "c"])
        model.logits_ = np.array([-10.0, 2.0, 7.0])
        self.assertTrue(np.all(model.weights_ >= 0))
        self.assertAlmostEqual(model.weights_.sum(), 1.0)

    def test_minimum_weight_prevents_component_collapse(self):
        model = LinearPairwiseRanker(["a", "b", "c"], minimum_weight=.05)
        model.logits_ = np.array([-100.0, 0.0, 100.0])
        self.assertTrue(np.all(model.weights_ >= .05))
        self.assertAlmostEqual(model.weights_.sum(), 1.0)

    def test_regularization_pulls_weights_towards_prior(self):
        positive = np.tile([0.0, 2.0], (30, 1))
        negative = np.tile([2.0, 0.0], (30, 1))
        unregularized = LinearPairwiseRanker(["a", "b"]).fit(positive, negative)
        regularized = LinearPairwiseRanker(
            ["a", "b"], regularization_strength=100.0,
        ).fit(positive, negative)
        uniform = np.array([.5, .5])
        self.assertLess(
            np.linalg.norm(regularized.weights_ - uniform),
            np.linalg.norm(unregularized.weights_ - uniform),
        )

    def test_correct_order_has_lower_pairwise_loss(self):
        model = LinearPairwiseRanker(["d"])
        good = model.pairwise_loss(np.array([[1.0]]), np.array([[2.0]]))
        bad = model.pairwise_loss(np.array([[2.0]]), np.array([[1.0]]))
        self.assertLess(good, bad)

    def test_training_reduces_pairwise_loss(self):
        model = LinearPairwiseRanker(["useful", "misleading"])
        positive = np.tile([0.0, 2.0], (20, 1))
        negative = np.tile([2.0, 0.0], (20, 1))
        before = model.pairwise_loss(positive, negative)
        model.fit(positive, negative)
        self.assertLess(model.pairwise_loss(positive, negative), before)

    def test_lower_distance_sorts_first(self):
        model = LinearPairwiseRanker(["d"])
        order = np.argsort(model.score(np.array([[3.0], [.2], [1.0]])))
        np.testing.assert_array_equal(order, [1, 2, 0])

    def test_synthetic_splits_do_not_leak_sources(self):
        league = pd.DataFrame({"throw_id": [f"l{i}" for i in range(20)],
                               "release_speed_m_s": np.arange(20.0)})
        raw = pd.DataFrame([
            {
                "throw_id": f"l{i}", "source": "league", "sampling_rate_hz": 20.0,
                "por_x_m": 13.0, "por_y_m": 0.0, "por_z_m": 1.5,
                "release_height_m": 1.5, "is_bounce": 0,
                "trajectory_json": json.dumps([
                    {"frame": j, "t_since_ms": j * 50.0,
                     "x": i + j, "y": j * .1, "z": -j * .05}
                    for j in range(4)
                ]),
            }
            for i in range(20)
        ])
        splits = build_synthetic_splits(league, raw, augmentations=2, seed=4)
        source_sets = [set(split.source_ids) for split in splits.values()]
        self.assertFalse(source_sets[0] & source_sets[1])
        self.assertFalse(source_sets[0] & source_sets[2])
        self.assertFalse(source_sets[1] & source_sets[2])

    def test_synthetic_kinematics_are_recomputed_from_perturbed_positions(self):
        raw = pd.Series({
            "throw_id": "l1", "source": "league", "sampling_rate_hz": 20.0,
            "por_x_m": 13.0, "por_y_m": 0.0, "por_z_m": 1.5,
            "release_height_m": 1.5, "is_bounce": 0,
            "trajectory_json": json.dumps([
                {"frame": j, "t_since_ms": j * 50.0,
                 "x": j, "y": j * .1, "z": -j * .05,
                 "v": 999.0, "vx": 999.0, "vy": 999.0, "vz": 999.0,
                 "a": 999.0, "dir": 999.0, "vert_angle": 999.0}
                for j in range(5)
            ]),
        })
        perturbed = perturb_raw_throw(raw, np.random.default_rng(7))
        points = json.loads(perturbed["trajectory_json"])
        first, second = points[:2]
        dt = (second["t_since_ms"] - first["t_since_ms"]) / 1000.0
        expected_velocity = np.asarray([
            (second[axis] - first[axis]) / dt for axis in "xyz"
        ])
        np.testing.assert_allclose(
            [first["vx"], first["vy"], first["vz"]], expected_velocity,
        )
        self.assertAlmostEqual(first["v"], float(np.linalg.norm(expected_velocity)))
        self.assertNotEqual(first["v"], 999.0)
        self.assertNotEqual(first["a"], 999.0)

    def test_severe_augmentation_can_truncate_and_always_retains_two_points(self):
        raw = pd.Series({
            "throw_id": "l1", "source": "league", "sampling_rate_hz": 20.0,
            "por_x_m": 13.0, "por_y_m": 0.0, "por_z_m": 1.5,
            "release_height_m": 1.5, "is_bounce": 0,
            "trajectory_json": json.dumps([
                {"frame": j, "t_since_ms": j * 50.0,
                 "x": j * .5, "y": j * .1, "z": -j * .05}
                for j in range(20)
            ]),
        })
        counts = []
        for seed in range(20):
            perturbed = perturb_raw_throw(
                raw, np.random.default_rng(seed), severity="severe",
            )
            points = json.loads(perturbed["trajectory_json"])
            counts.append(len(points))
            self.assertGreaterEqual(len(points), 2)
            self.assertEqual(points[0]["t_since_ms"], 0.0)
        self.assertLess(min(counts), 20)
        self.assertEqual(set(PERTURBATION_PROFILES), {"mild", "medium", "severe"})

    def test_coverage_component_penalizes_short_candidate(self):
        league = pd.DataFrame({
            "throw_id": ["complete", "short", "far"],
            "release_speed_m_s": [10.0, 10.0, 12.0],
            "disp_x_t0ms": [0.0, 0.0, 0.0],
            "disp_x_t50ms": [1.0, np.nan, 3.0],
            "disp_x_t100ms": [2.0, np.nan, 6.0],
        })
        query = pd.Series({
            "throw_id": "q", "release_speed_m_s": 10.0,
            "disp_x_t0ms": 0.0, "disp_x_t50ms": 1.0, "disp_x_t100ms": 2.0,
        })
        retriever = WeightedKNNRetriever().fit(league)
        matrix, names = component_matrix(retriever, query)
        coverage = matrix[:, names.index("trajectory_coverage")]
        np.testing.assert_allclose(coverage, [0.0, 2.0 / 3.0, 0.0])

        model = LinearPairwiseRanker(names, minimum_weight=.01)
        ranked = rank_queries(model, retriever, pd.DataFrame([query]), k=3)
        self.assertEqual(ranked.iloc[0].league_throw_id, "complete")
        self.assertTrue(ranked.iloc[0].meets_minimum_overlap)

    def test_bounce_interactions_are_zero_for_non_bounce(self):
        base = np.ones((2, len(BASE_COMPONENTS)))
        matrix, _ = add_interactions(base, BASE_COMPONENTS, False, np.ones(2))
        np.testing.assert_array_equal(matrix[:, -2:], 0.0)

    def test_baseline_and_ranker_share_component_calculation(self):
        league = pd.DataFrame({"throw_id": ["a", "b", "c"],
                               "release_speed_m_s": [1.0, 2.0, 4.0]})
        query = pd.DataFrame({"throw_id": ["q"], "release_speed_m_s": [1.2]})
        retriever = WeightedKNNRetriever().fit(league)
        components = retriever.distance_components(query.iloc[0])
        ranked = retriever.retrieve(query, k=3)
        by_id = dict(zip(league.throw_id, components["release_speed"]))
        for _, row in ranked.iterrows():
            self.assertAlmostEqual(row.release_speed_distance, by_id[row.league_throw_id])


if __name__ == "__main__":
    unittest.main()
