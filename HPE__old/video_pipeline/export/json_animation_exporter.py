from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from retargeting.mixamo_retargeter import RetargetedAnimation


class JsonAnimationExporter:
    """Exports retargeted animation as Unity-ready JSON."""

    def __init__(self, frame_rate: float, center_root_xz: bool = True) -> None:
        self.frame_rate = float(frame_rate)
        self.center_root_xz = center_root_xz
        # Source model space -> Unity (Y-up, left-handed, Z-forward).
        self._basis = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self._basis_inv = np.linalg.inv(self._basis).astype(np.float32)

    def _to_unity_basis_vec(self, vec: np.ndarray) -> np.ndarray:
        return np.einsum("ij,...j->...i", self._basis, vec).astype(np.float32)

    def _to_unity_basis_quat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        rot = R.from_quat(quat_xyzw).as_matrix()
        rot_u = self._basis @ rot @ self._basis_inv
        return R.from_matrix(rot_u).as_quat().astype(np.float32)

    @staticmethod
    def _compute_rest_world_offsets(offsets: np.ndarray, parents: list[int]) -> np.ndarray:
        world = np.zeros_like(offsets, dtype=np.float32)
        for idx, parent in enumerate(parents):
            if parent < 0:
                world[idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                world[idx] = world[parent] + offsets[idx]
        return world

    def export(self, animation: RetargetedAnimation, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        offsets = self._to_unity_basis_vec(animation.bone_offsets)
        root_positions = self._to_unity_basis_vec(animation.root_positions)
        quats = np.zeros_like(animation.local_quaternions, dtype=np.float32)

        for f in range(animation.local_quaternions.shape[0]):
            for j in range(animation.local_quaternions.shape[1]):
                quats[f, j] = self._to_unity_basis_quat(animation.local_quaternions[f, j])

        if root_positions.shape[0] > 0:
            if self.center_root_xz:
                root_positions[:, 0] -= root_positions[0, 0]
                root_positions[:, 2] -= root_positions[0, 2]

            root_positions[:, 1] -= float(np.min(root_positions[:, 1]))
            rest_world = self._compute_rest_world_offsets(offsets, animation.parent_indices)
            min_rest_y = float(np.min(rest_world[:, 1]))
            if min_rest_y < 0.0:
                root_positions[:, 1] += -min_rest_y

        frames = []
        for f in range(root_positions.shape[0]):
            flat_quats = quats[f].reshape(-1).tolist()
            frames.append(
                {
                    "root_position": root_positions[f].tolist(),
                    "rotations_flat": flat_quats,
                    "rotations": quats[f].tolist(),
                }
            )

        payload = {
            "frame_rate": self.frame_rate,
            "coordinate_system": "unity_y_up_left_handed",
            "quaternion_format": "xyzw",
            "root_joint": "Hips",
            "joint_names": list(animation.joint_names),
            "frames": frames,
        }

        out_path.write_text(json.dumps(payload), encoding="utf-8")
