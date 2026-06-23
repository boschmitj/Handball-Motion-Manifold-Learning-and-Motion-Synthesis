from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointFormatSpec:
    name: str
    joints: tuple[str, ...]


COCO_SPEC = JointFormatSpec(
    name="coco17",
    joints=(
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
)

H36M_SPEC = JointFormatSpec(
    name="h36m17",
    joints=(
        "pelvis",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "spine",
        "thorax",
        "neck",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ),
)


class CocoToH36MMapper:
    """Maps COCO 17-keypoints to Human3.6M 17-keypoints."""

    # H36M index -> source COCO index. None means computed/interpolated.
    mapping: tuple[int | None, ...] = (
        None,  # pelvis from hips midpoint
        12,  # right_hip
        14,  # right_knee
        16,  # right_ankle
        11,  # left_hip
        13,  # left_knee
        15,  # left_ankle
        None,  # spine from pelvis/thorax midpoint
        None,  # thorax midpoint shoulders
        0,  # neck approximated from nose (fallback)
        0,  # head from nose
        5,  # left_shoulder
        7,  # left_elbow
        9,  # left_wrist
        6,  # right_shoulder
        8,  # right_elbow
        10,  # right_wrist
    )

    def map(self, coco_xy: np.ndarray, coco_conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            coco_xy: (T, 17, 2)
            coco_conf: (T, 17)
        Returns:
            h36m_xy: (T, 17, 2)
            h36m_conf: (T, 17)
        """
        if coco_xy.shape[1] != 17:
            raise ValueError(f"Expected COCO with 17 joints, got {coco_xy.shape[1]}")

        t = coco_xy.shape[0]
        h36m_xy = np.zeros((t, 17, 2), dtype=np.float32)
        h36m_conf = np.zeros((t, 17), dtype=np.float32)

        for dst_idx, src_idx in enumerate(self.mapping):
            if src_idx is not None:
                h36m_xy[:, dst_idx] = coco_xy[:, src_idx]
                h36m_conf[:, dst_idx] = coco_conf[:, src_idx]

        # pelvis = midpoint(left_hip, right_hip)
        left_hip, right_hip = 11, 12
        h36m_xy[:, 0] = 0.5 * (coco_xy[:, left_hip] + coco_xy[:, right_hip])
        h36m_conf[:, 0] = np.minimum(coco_conf[:, left_hip], coco_conf[:, right_hip])

        # thorax = midpoint(left_shoulder, right_shoulder)
        left_shoulder, right_shoulder = 5, 6
        h36m_xy[:, 8] = 0.5 * (coco_xy[:, left_shoulder] + coco_xy[:, right_shoulder])
        h36m_conf[:, 8] = np.minimum(coco_conf[:, left_shoulder], coco_conf[:, right_shoulder])

        # spine midpoint between pelvis and thorax
        h36m_xy[:, 7] = 0.5 * (h36m_xy[:, 0] + h36m_xy[:, 8])
        h36m_conf[:, 7] = np.minimum(h36m_conf[:, 0], h36m_conf[:, 8])

        # Build a stable neck/head from available COCO face + shoulder cues.
        nose, l_eye, r_eye, l_ear, r_ear = 0, 1, 2, 3, 4
        face_center = 0.2 * coco_xy[:, nose] + 0.2 * coco_xy[:, l_eye] + 0.2 * coco_xy[:, r_eye]
        face_center += 0.2 * coco_xy[:, l_ear] + 0.2 * coco_xy[:, r_ear]

        # Neck sits between thorax and face center to avoid collapse with head.
        h36m_xy[:, 9] = 0.65 * h36m_xy[:, 8] + 0.35 * face_center
        h36m_conf[:, 9] = np.minimum(
            h36m_conf[:, 8],
            np.mean(coco_conf[:, [nose, l_eye, r_eye, l_ear, r_ear]], axis=1),
        )

        # Keep head close to nose/face center for MotionBERT compatibility.
        h36m_xy[:, 10] = 0.7 * coco_xy[:, nose] + 0.3 * face_center
        h36m_conf[:, 10] = np.mean(coco_conf[:, [nose, l_eye, r_eye]], axis=1)

        return h36m_xy, h36m_conf
