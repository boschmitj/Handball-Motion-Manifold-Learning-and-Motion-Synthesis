import unittest

import numpy as np

from preprocessing.joint_mapping import CocoToH36MMapper
from preprocessing.normalization import KeypointNormalizer


class TestPreprocessing(unittest.TestCase):
    def test_coco_to_h36m_shape(self) -> None:
        xy = np.zeros((5, 17, 2), dtype=np.float32)
        conf = np.ones((5, 17), dtype=np.float32)
        mapper = CocoToH36MMapper()
        hxy, hconf = mapper.map(xy, conf)
        self.assertEqual(hxy.shape, (5, 17, 2))
        self.assertEqual(hconf.shape, (5, 17))

    def test_normalize_denormalize(self) -> None:
        pts = np.random.randn(10, 17, 2).astype(np.float32)
        norm = KeypointNormalizer(root_index=0, left_shoulder=11, right_shoulder=14)
        normalized, root, scale = norm.normalize(pts)
        recovered = norm.denormalize(normalized, root, scale)
        self.assertTrue(np.allclose(pts, recovered, atol=1e-5))

    def test_mapped_neck_head_not_collapsed(self) -> None:
        xy = np.zeros((2, 17, 2), dtype=np.float32)
        conf = np.ones((2, 17), dtype=np.float32)

        # shoulders
        xy[:, 5] = np.array([100.0, 100.0], dtype=np.float32)
        xy[:, 6] = np.array([140.0, 100.0], dtype=np.float32)
        # face
        xy[:, 0] = np.array([120.0, 70.0], dtype=np.float32)  # nose
        xy[:, 1] = np.array([115.0, 66.0], dtype=np.float32)  # left eye
        xy[:, 2] = np.array([125.0, 66.0], dtype=np.float32)  # right eye
        xy[:, 3] = np.array([110.0, 72.0], dtype=np.float32)  # left ear
        xy[:, 4] = np.array([130.0, 72.0], dtype=np.float32)  # right ear
        # hips for pelvis/thorax pipeline
        xy[:, 11] = np.array([108.0, 150.0], dtype=np.float32)
        xy[:, 12] = np.array([132.0, 150.0], dtype=np.float32)

        hxy, _ = CocoToH36MMapper().map(xy, conf)
        neck = hxy[:, 9]
        head = hxy[:, 10]
        self.assertTrue(np.all(np.linalg.norm(head - neck, axis=1) > 1e-3))


if __name__ == "__main__":
    unittest.main()
