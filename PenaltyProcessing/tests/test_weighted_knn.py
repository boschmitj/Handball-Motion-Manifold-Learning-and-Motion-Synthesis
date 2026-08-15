import unittest

import numpy as np
import pandas as pd

from src.model.weighted_knn import WeightedKNNRetriever, circular_difference_degrees


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

    def test_scales_are_fit_on_league_only(self):
        league = pd.DataFrame({"throw_id": [1, 2], "release_speed_m_s": [0.0, 2.0]})
        mocap = pd.DataFrame({"throw_id": [3], "release_speed_m_s": [1000.0]})
        model = WeightedKNNRetriever().fit(league)
        self.assertEqual(model.feature_specs[0].scale, 1.0)
        model.retrieve(mocap)
        self.assertEqual(model.feature_specs[0].scale, 1.0)


if __name__ == "__main__":
    unittest.main()
