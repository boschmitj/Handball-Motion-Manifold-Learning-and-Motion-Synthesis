from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from skeleton.humanoid_skeleton import MIXAMO_SKELETON


@dataclass
class RetargetedAnimation:
    joint_names: list[str]
    parent_indices: list[int]
    root_positions: np.ndarray  # (T, 3)
    local_quaternions: np.ndarray  # (T, J, 4) xyzw
    bone_offsets: np.ndarray  # (J, 3)


class MixamoRetargeter:
    """Converts 3D joints into Mixamo-compatible local joint rotations."""

    def __init__(self) -> None:
        self.joint_names = [joint.name for joint in MIXAMO_SKELETON]
        self.parent_indices = [self.joint_names.index(j.parent) if j.parent else -1 for j in MIXAMO_SKELETON]
        self.children_indices: list[list[int]] = [[] for _ in self.joint_names]
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx >= 0:
                self.children_indices[parent_idx].append(child_idx)

    @staticmethod
    def _align_character_facing(mapped_xyz: np.ndarray) -> np.ndarray:
        """Rotate animation around global Y so frame 0 faces +Z (Unity forward)."""
        if mapped_xyz.shape[0] == 0:
            return mapped_xyz

        # Use shoulder/hip lateral axis and torso up axis to estimate facing direction.
        hips = mapped_xyz[0, 0]
        spine1 = mapped_xyz[0, 8]
        left_side = 0.5 * (mapped_xyz[0, 11] + mapped_xyz[0, 4])
        right_side = 0.5 * (mapped_xyz[0, 15] + mapped_xyz[0, 1])

        lateral = left_side - right_side
        up = spine1 - hips
        forward = np.cross(lateral, up)

        forward_xz = np.array([forward[0], 0.0, forward[2]], dtype=np.float32)
        norm = float(np.linalg.norm(forward_xz))
        if norm < 1e-6:
            return mapped_xyz

        forward_xz /= norm
        yaw = float(np.arctan2(forward_xz[0], forward_xz[2]))
        rot = R.from_euler("y", -yaw)

        centered = mapped_xyz - hips[None, None, :]
        rotated = rot.apply(centered.reshape(-1, 3)).reshape(mapped_xyz.shape)
        rotated += hips[None, None, :]
        return rotated.astype(np.float32)

    def _compute_rest_offsets(self, joints_xyz: np.ndarray) -> np.ndarray:
        rest = np.mean(joints_xyz[: min(10, len(joints_xyz))], axis=0)
        offsets = np.zeros_like(rest)
        for j, parent in enumerate(self.parent_indices):
            if parent < 0:
                offsets[j] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                offsets[j] = rest[j] - rest[parent]
        return offsets.astype(np.float32)

    @staticmethod
    def _align_vectors(v_from: np.ndarray, v_to: np.ndarray) -> R:
        a = v_from / (np.linalg.norm(v_from) + 1e-8)
        b = v_to / (np.linalg.norm(v_to) + 1e-8)
        cross = np.cross(a, b)
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        if np.linalg.norm(cross) < 1e-8 and dot > 0.9999:
            return R.identity()
        if np.linalg.norm(cross) < 1e-8 and dot < -0.9999:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return R.from_rotvec(np.pi * axis)
        skew = np.array(
            [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]],
            dtype=np.float32,
        )
        rot_m = np.eye(3, dtype=np.float32) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2 + 1e-8))
        return R.from_matrix(rot_m)

    @staticmethod
    def _synthesize_hands(h36m_xyz: np.ndarray) -> np.ndarray:
        """Create 19-joint array in MIXAMO_SKELETON order, including synthetic hands."""
        left_elbow = h36m_xyz[:, 12]
        left_wrist = h36m_xyz[:, 13]
        right_elbow = h36m_xyz[:, 15]
        right_wrist = h36m_xyz[:, 16]

        left_forearm = left_wrist - left_elbow
        right_forearm = right_wrist - right_elbow

        left_len = np.linalg.norm(left_forearm, axis=1, keepdims=True)
        right_len = np.linalg.norm(right_forearm, axis=1, keepdims=True)

        left_dir = left_forearm / (left_len + 1e-8)
        right_dir = right_forearm / (right_len + 1e-8)

        left_hand = left_wrist + left_dir * np.maximum(left_len * 0.4, 1e-4)
        right_hand = right_wrist + right_dir * np.maximum(right_len * 0.4, 1e-4)

        # H36M (17):
        # 0 Hips, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle,
        # 7 Spine, 8 Thorax, 9 Neck, 10 Head, 11 LShoulder, 12 LElbow, 13 LWrist,
        # 14 RShoulder, 15 RElbow, 16 RWrist
        # Target MIXAMO_SKELETON (19):
        # 0 Hips,1 RightUpLeg,2 RightLeg,3 RightFoot,4 LeftUpLeg,5 LeftLeg,6 LeftFoot,
        # 7 Spine,8 Spine1,9 Neck,10 Head,11 LeftShoulder,12 LeftArm,13 LeftForeArm,14 LeftHand,
        # 15 RightShoulder,16 RightArm,17 RightForeArm,18 RightHand
        out = np.zeros((h36m_xyz.shape[0], 19, 3), dtype=np.float32)
        out[:, 0:9] = h36m_xyz[:, 0:9]
        # MotionBERT's H36M convention uses index 9 as nose and 10 as head.
        # Build a stable neck between thorax (8) and head (10) for Mixamo.
        out[:, 9] = 0.5 * (h36m_xyz[:, 8] + h36m_xyz[:, 10])
        out[:, 10] = h36m_xyz[:, 10]
        out[:, 11:14] = h36m_xyz[:, 11:14]
        out[:, 14] = left_hand
        out[:, 15:18] = h36m_xyz[:, 14:17]
        out[:, 18] = right_hand
        return out

    @staticmethod
    def _best_fit_rotation(rest_dirs: np.ndarray, pose_dirs: np.ndarray) -> R:
        if rest_dirs.shape[0] == 0:
            return R.identity()
        if rest_dirs.shape[0] == 1:
            return MixamoRetargeter._align_vectors(rest_dirs[0], pose_dirs[0])

        rest_norm = rest_dirs / (np.linalg.norm(rest_dirs, axis=1, keepdims=True) + 1e-8)
        pose_norm = pose_dirs / (np.linalg.norm(pose_dirs, axis=1, keepdims=True) + 1e-8)
        try:
            rot, _ = R.align_vectors(pose_norm, rest_norm)
            return rot
        except Exception:
            return MixamoRetargeter._align_vectors(rest_norm[0], pose_norm[0])

    def retarget(self, h36m_xyz: np.ndarray) -> RetargetedAnimation:
        """
        Args:
            h36m_xyz: (T, 17, 3) in H36M order.
        """
        if h36m_xyz.shape[1] != 17:
            raise ValueError("Expected 17 H36M joints")

        # Name-aligned joints plus synthesized hands for Unity humanoid requirements.
        mapped_xyz = self._synthesize_hands(h36m_xyz.astype(np.float32))
        mapped_xyz = self._align_character_facing(mapped_xyz)
        bone_offsets = self._compute_rest_offsets(mapped_xyz)

        t, j, _ = mapped_xyz.shape
        local_quats = np.zeros((t, j, 4), dtype=np.float32)
        local_quats[..., 3] = 1.0
        root_positions = mapped_xyz[:, 0].astype(np.float32)

        for frame_idx in range(t):
            world_rots: list[R] = [R.identity() for _ in range(j)]

            for joint_idx in range(j):
                child_ids = self.children_indices[joint_idx]

                rest_dirs = []
                pose_dirs = []
                for child_idx in child_ids:
                    rest_vec = bone_offsets[child_idx]
                    pose_vec = mapped_xyz[frame_idx, child_idx] - mapped_xyz[frame_idx, joint_idx]
                    if np.linalg.norm(rest_vec) < 1e-8 or np.linalg.norm(pose_vec) < 1e-8:
                        continue
                    rest_dirs.append(rest_vec)
                    pose_dirs.append(pose_vec)

                if rest_dirs:
                    world_rots[joint_idx] = self._best_fit_rotation(
                        np.asarray(rest_dirs, dtype=np.float32),
                        np.asarray(pose_dirs, dtype=np.float32),
                    )
                else:
                    world_rots[joint_idx] = R.identity()

                parent_idx = self.parent_indices[joint_idx]
                if parent_idx < 0:
                    local_rot = world_rots[joint_idx]
                else:
                    local_rot = world_rots[parent_idx].inv() * world_rots[joint_idx]
                local_quats[frame_idx, joint_idx] = local_rot.as_quat().astype(np.float32)

        return RetargetedAnimation(
            joint_names=list(self.joint_names),
            parent_indices=self.parent_indices,
            root_positions=root_positions,
            local_quaternions=local_quats,
            bone_offsets=bone_offsets,
        )
