import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.weighted_knn import (
    PREDEFINED_WEIGHT_SETS, WeightedKNNRetriever, balanced_top1_score,
    circular_difference_degrees, run_random_weight_search,
)


class WeightedKNNTests(unittest.TestCase):
    def test_circular_difference_wraps_at_180(self):
        result = circular_difference_degrees(np.array([179.0]), -179.0)
        self.assertEqual(abs(result[0]), 2.0)

    def test_retrieval_ignores_metadata_and_handles_pairwise_missing_values(self):
        league = pd.DataFrame({
            "throw_id": ["near", "far", "missing"],
            "release_speed_m_s": [10.0, 20.0, 15.0],
            "release_direction_deg": [179.0, 0.0, 90.0],
            "disp_x_t50ms": [1.0, 3.0, np.nan],
            "disp_x_t100ms": [2.0, 6.0, np.nan],
            "sampling_rate_hz": [20.0, 20.0, 9999.0],
            "trajectory_point_count": [3, 3, 9999],
        })
        mocap = pd.DataFrame({
            "throw_id": ["m1"],
            "release_speed_m_s": [10.1],
            "release_direction_deg": [-179.0],
            "disp_x_t50ms": [1.1],
            "disp_x_t100ms": [np.nan],
            "sampling_rate_hz": [300.0],
            "trajectory_point_count": [1],
        })
        model = WeightedKNNRetriever().fit(league)
        self.assertFalse({"sampling_rate_hz", "trajectory_point_count"} & {
            spec.name for spec in model.feature_specs
        })
        result = model.retrieve(mocap, k=3)
        self.assertEqual(result.iloc[0]["league_throw_id"], "near")
        self.assertTrue(np.isfinite(result["total_distance"]).all())
        contributions = [c for c in result if c.endswith("_contribution")]
        np.testing.assert_allclose(result[contributions].sum(axis=1), result.total_distance)
        near = result[result.league_throw_id == "near"].iloc[0]
        missing = result[result.league_throw_id == "missing"].iloc[0]
        self.assertEqual(near.trajectory_common_point_count, 1)
        self.assertEqual(near.trajectory_overlap_ratio, 1.0)
        self.assertEqual(near.trajectory_evidence_ratio, .5)
        self.assertEqual(missing.trajectory_overlap_ratio, 0.0)

    def test_scales_are_fit_on_league_only(self):
        league = pd.DataFrame({"throw_id": [1, 2], "release_speed_m_s": [0.0, 2.0]})
        mocap = pd.DataFrame({"throw_id": [3], "release_speed_m_s": [1000.0]})
        model = WeightedKNNRetriever().fit(league)
        self.assertEqual(model.feature_specs[0].scale, 1.0)
        model.retrieve(mocap)
        self.assertEqual(model.feature_specs[0].scale, 1.0)

    def test_predefined_weight_registry_contains_complete_sets(self):
        self.assertIn("default", PREDEFINED_WEIGHT_SETS)
        expected = set(PREDEFINED_WEIGHT_SETS["default"])
        for weights in PREDEFINED_WEIGHT_SETS.values():
            self.assertEqual(set(weights), expected)

    def test_balanced_score_gives_each_throw_type_equal_influence(self):
        mocap = pd.DataFrame({
            "throw_id": ["hard_seg1", "hard_seg2", "lob_seg1"],
        })
        matches = pd.DataFrame({
            "mocap_throw_id": ["hard_seg1", "hard_seg2", "lob_seg1"],
            "rank": [1, 1, 1], "total_distance": [1.0, 3.0, 10.0],
        })
        distance, score, by_type = balanced_top1_score(matches, mocap)
        self.assertEqual(distance, 6.0)  # mean(mean(1,3), mean(10))
        self.assertEqual(score, 1.0 / 7.0)
        self.assertEqual(set(by_type.throw_type), {"hard", "lob"})

    def test_random_search_writes_every_run_and_best_summary(self):
        league = pd.DataFrame({
            "throw_id": ["l1", "l2", "l3"],
            "release_speed_m_s": [5.0, 10.0, 15.0],
            "release_height_m": [1.5, 1.7, 1.9],
        })
        mocap = pd.DataFrame({
            "throw_id": ["hard_seg1", "lob_seg1"],
            "release_speed_m_s": [6.0, 14.0],
            "release_height_m": [1.55, 1.85],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            search_dir, summary, best = run_random_weight_search(
                mocap, league, runs=2, k=1, output_base=Path(temp_dir), seed=7,
            )
            self.assertEqual(len(summary), 2)
            self.assertEqual(set(best), set(PREDEFINED_WEIGHT_SETS["default"]))
            self.assertTrue((search_dir / "best_weights.json").is_file())
            self.assertTrue((search_dir / "best_weighted_knn_matches.csv").is_file())
            self.assertTrue((search_dir / "run_0001" / "weighted_knn_matches.csv").is_file())
            self.assertTrue((search_dir / "run_0002" / "metrics_by_throw_type.csv").is_file())


if __name__ == "__main__":
    unittest.main()
