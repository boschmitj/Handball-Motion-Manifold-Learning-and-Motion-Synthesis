"""Tests for predictor_factory helper conversions."""

import numpy as np

from core.predictor_factory import _to_relative_box, _to_relative_points


class TestCoordinateConversion:
    def test_relative_box_basic(self):
        box = [100, 50, 300, 200]
        rel = _to_relative_box(box, width=400, height=400)
        expected = np.array([[0.25, 0.125, 0.75, 0.5]])
        np.testing.assert_allclose(rel, expected, atol=1e-6)

    def test_relative_box_batch(self):
        boxes = [[0, 0, 100, 100], [100, 100, 200, 200]]
        rel = _to_relative_box(boxes, width=200, height=200)
        assert rel.shape == (2, 4)
        np.testing.assert_allclose(rel[0], [0, 0, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(rel[1], [0.5, 0.5, 1.0, 1.0], atol=1e-6)

    def test_relative_points_basic(self):
        points = [[200, 100]]
        rel = _to_relative_points(points, width=400, height=200)
        expected = np.array([[0.5, 0.5]])
        np.testing.assert_allclose(rel, expected, atol=1e-6)

    def test_relative_points_batch(self):
        points = [[0, 0], [400, 200]]
        rel = _to_relative_points(points, width=400, height=200)
        np.testing.assert_allclose(rel[0], [0, 0], atol=1e-6)
        np.testing.assert_allclose(rel[1], [1, 1], atol=1e-6)

    def test_does_not_mutate_input(self):
        box = np.array([100, 50, 300, 200], dtype=np.float64)
        original = box.copy()
        _to_relative_box(box, 400, 400)
        np.testing.assert_array_equal(box, original)
