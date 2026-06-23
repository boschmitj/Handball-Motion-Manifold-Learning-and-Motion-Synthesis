from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


COCO_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

COCO_JOINT_NAMES = (
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
)

H36M_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


def render_pose2d_overlay_video(
    frames: np.ndarray,
    keypoints_xy: np.ndarray,
    confidence: np.ndarray,
    out_path: Path,
    fps: float,
    threshold: float = 0.2,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for idx, frame_rgb in enumerate(frames):
        canvas = frame_rgb.copy()
        pts = keypoints_xy[idx]
        conf = confidence[idx]

        for i, (x, y) in enumerate(pts):
            if conf[i] < threshold:
                continue
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 255), -1)
            label = COCO_JOINT_NAMES[i] if i < len(COCO_JOINT_NAMES) else f"J{i}"
            text_pos = (int(x) + 4, int(y) - 4)
            cv2.putText(canvas, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        for a, b in COCO_EDGES:
            if conf[a] < threshold or conf[b] < threshold:
                continue
            p0 = tuple(np.int32(pts[a]))
            p1 = tuple(np.int32(pts[b]))
            cv2.line(canvas, p0, p1, (0, 255, 0), 2)

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()


def save_3d_skeleton_plot(joints_xyz: np.ndarray, out_path: Path, frame_idx: int = 0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    pts = joints_xyz[frame_idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=20)
    for a, b in H36M_EDGES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            [pts[a, 2], pts[b, 2]],
            c="blue",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Pose (H36M)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_joint_trajectory_plot(joints_xyz: np.ndarray, joint_idx: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traj = joints_xyz[:, joint_idx]
    t = np.arange(len(traj))

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(t, traj[:, i])
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("frame")
    fig.suptitle(f"Joint {joint_idx} trajectory")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
