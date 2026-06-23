from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from retargeting.mixamo_retargeter import RetargetedAnimation


class BVHExporter:
    """Exports retargeted animation to BVH for Unity import."""

    def __init__(self, frame_time: float = 1.0 / 30.0, center_root_xz: bool = True) -> None:
        self.frame_time = frame_time
        self.center_root_xz = center_root_xz
        # MotionBERT output in this pipeline behaves as Y-down camera coordinates.
        # Unity expects Y-up, Z-forward. Flipping Y converts vectors accordingly.
        self._basis = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self._basis_inv = np.linalg.inv(self._basis).astype(np.float32)

    def _to_unity_basis_vec(self, vec: np.ndarray) -> np.ndarray:
        return np.einsum("ij,...j->...i", self._basis, vec).astype(np.float32)

    def _to_unity_basis_quat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        rot = R.from_quat(quat_xyzw).as_matrix()
        # Basis change: R' = B * R * B^{-1}
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

        names = animation.joint_names
        parents = animation.parent_indices
        offsets = self._to_unity_basis_vec(animation.bone_offsets)
        root_positions = self._to_unity_basis_vec(animation.root_positions)
        if root_positions.shape[0] > 0:
            # Normalize root trajectory: center horizontal drift at clip start and ground vertical motion.
            if self.center_root_xz:
                root_positions[:, 0] -= root_positions[0, 0]
                root_positions[:, 2] -= root_positions[0, 2]

            root_positions[:, 1] -= float(np.min(root_positions[:, 1]))
            rest_world = self._compute_rest_world_offsets(offsets, animation.parent_indices)
            min_rest_y = float(np.min(rest_world[:, 1]))
            if min_rest_y < 0.0:
                root_positions[:, 1] += -min_rest_y

        quats = np.zeros_like(animation.local_quaternions, dtype=np.float32)
        for f in range(animation.local_quaternions.shape[0]):
            for j in range(animation.local_quaternions.shape[1]):
                quats[f, j] = self._to_unity_basis_quat(animation.local_quaternions[f, j])

        children: dict[int, list[int]] = {i: [] for i in range(len(names))}
        root_idx = 0
        for idx, p in enumerate(parents):
            if p == -1:
                root_idx = idx
            else:
                children[p].append(idx)

        lines: list[str] = []
        lines.append("HIERARCHY")

        def write_joint(idx: int, indent: int) -> None:
            prefix = "  " * indent
            is_root = parents[idx] == -1
            keyword = "ROOT" if is_root else "JOINT"
            lines.append(f"{prefix}{keyword} {names[idx]}")
            lines.append(f"{prefix}{{")
            off = offsets[idx]
            lines.append(f"{prefix}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
            if is_root:
                lines.append(
                    f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
                )
            else:
                lines.append(f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation")

            if len(children[idx]) == 0:
                lines.append(f"{prefix}  End Site")
                lines.append(f"{prefix}  {{")
                lines.append(f"{prefix}    OFFSET 0.000000 0.050000 0.000000")
                lines.append(f"{prefix}  }}")
            else:
                for c in children[idx]:
                    write_joint(c, indent + 1)

            lines.append(f"{prefix}}}")

        write_joint(root_idx, 0)

        lines.append("MOTION")
        n_frames = root_positions.shape[0]
        lines.append(f"Frames: {n_frames}")
        lines.append(f"Frame Time: {self.frame_time:.8f}")

        for frame in range(n_frames):
            channels: list[float] = []
            root = root_positions[frame]
            channels.extend([float(root[0]), float(root[1]), float(root[2])])

            def append_euler(joint_idx: int) -> None:
                quat = quats[frame, joint_idx]
                euler = R.from_quat(quat).as_euler("ZXY", degrees=True)
                channels.extend([float(euler[0]), float(euler[1]), float(euler[2])])
                for c_idx in children[joint_idx]:
                    append_euler(c_idx)

            append_euler(root_idx)
            lines.append(" ".join(f"{v:.6f}" for v in channels))

        out_path.write_text("\n".join(lines), encoding="utf-8")
