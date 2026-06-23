from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


class Pose3DPostprocessor:
    """Applies temporal smoothing and basic geometric normalization."""

    def __init__(
        self,
        smooth_window: int = 11,
        smooth_polyorder: int = 2,
        foot_joint_indices: tuple[int, int] = (3, 6),
    ) -> None:
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.foot_joint_indices = foot_joint_indices
        self.parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int32)

    def smooth(self, joints_xyz: np.ndarray) -> np.ndarray:
        if joints_xyz.ndim != 3 or joints_xyz.shape[-1] != 3:
            raise ValueError("Expected shape (T, J, 3)")

        t = joints_xyz.shape[0]
        win = min(self.smooth_window, t if t % 2 == 1 else t - 1)
        if win < 3:
            return joints_xyz.copy()

        smoothed = savgol_filter(joints_xyz, window_length=win, polyorder=self.smooth_polyorder, axis=0)
        return smoothed.astype(np.float32)

    def align_ground(self, joints_xyz: np.ndarray) -> np.ndarray:
        grounded = joints_xyz.copy()
        feet = grounded[:, self.foot_joint_indices, 1]
        min_y = float(np.min(feet))
        grounded[:, :, 1] -= min_y
        return grounded

    @staticmethod
    def normalize_scale(joints_xyz: np.ndarray, desired_height: float = 1.75) -> np.ndarray:
        # Use median per-frame body height for robust scaling.
        per_frame_height = np.max(joints_xyz[:, :, 1], axis=1) - np.min(joints_xyz[:, :, 1], axis=1)
        current_height = float(np.median(per_frame_height))
        scale = desired_height / max(current_height, 1e-6)
        return (joints_xyz * scale).astype(np.float32)

    def enforce_constant_bone_lengths(self, joints_xyz: np.ndarray) -> np.ndarray:
        """Preserve pose directions but enforce fixed bone lengths over time."""
        if joints_xyz.ndim != 3 or joints_xyz.shape[-1] != 3 or joints_xyz.shape[1] != len(self.parents):
            raise ValueError("Expected shape (T, 17, 3)")

        t, j, _ = joints_xyz.shape
        lengths = np.zeros((j,), dtype=np.float32)
        for child in range(1, j):
            parent = self.parents[child]
            seg = joints_xyz[:, child] - joints_xyz[:, parent]
            lengths[child] = float(np.median(np.linalg.norm(seg, axis=1)))

        stabilized = np.zeros_like(joints_xyz, dtype=np.float32)
        stabilized[:, 0] = joints_xyz[:, 0]

        for frame_idx in range(t):
            for child in range(1, j):
                parent = self.parents[child]
                direction = joints_xyz[frame_idx, child] - joints_xyz[frame_idx, parent]
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    if frame_idx > 0:
                        direction = stabilized[frame_idx - 1, child] - stabilized[frame_idx - 1, parent]
                        norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    norm = 1.0
                direction = direction / norm
                stabilized[frame_idx, child] = stabilized[frame_idx, parent] + direction * lengths[child]

        return stabilized

    @staticmethod
    def velocity(joints_xyz: np.ndarray, dt: float) -> np.ndarray:
        return np.gradient(joints_xyz, dt, axis=0)
