from __future__ import annotations

import numpy as np


class KeypointNormalizer:
    """Centers around root and applies scale normalization per frame."""

    def __init__(self, root_index: int = 0, left_shoulder: int = 11, right_shoulder: int = 14) -> None:
        self.root_index = root_index
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder

    def normalize(self, keypoints_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            keypoints_xy: (T, J, 2)
        Returns:
            normalized: (T, J, 2)
            root: (T, 2)
            scale: (T,)
        """
        root = keypoints_xy[:, self.root_index].copy()
        centered = keypoints_xy - root[:, None, :]

        shoulder_vec = centered[:, self.left_shoulder] - centered[:, self.right_shoulder]
        scale = np.linalg.norm(shoulder_vec, axis=-1)
        scale = np.clip(scale, 1e-6, None)

        normalized = centered / scale[:, None, None]
        return normalized.astype(np.float32), root.astype(np.float32), scale.astype(np.float32)

    @staticmethod
    def denormalize(
        normalized_xy: np.ndarray,
        root: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        return normalized_xy * scale[:, None, None] + root[:, None, :]
