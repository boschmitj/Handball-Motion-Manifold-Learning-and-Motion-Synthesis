from __future__ import annotations

from pathlib import Path

import numpy as np


def _print_axis_stats(name: str, arr: np.ndarray) -> None:
    print(f"{name} min xyz: {arr.min(axis=0)}")
    print(f"{name} max xyz: {arr.max(axis=0)}")
    print(f"{name} mean xyz: {arr.mean(axis=0)}")


def main() -> None:
    out = Path("outputs/run06")
    pose2d = np.load(out / "pose2d_h36m.npy")
    pose2d_conf = np.load(out / "pose2d_h36m_conf.npy")
    pose3d = np.load(out / "pose3d_smoothed.npy")

    print("=== Shape and confidence checks ===")
    print("pose2d_h36m:", pose2d.shape)
    print("pose2d_h36m_conf:", pose2d_conf.shape)
    print("pose3d_smoothed:", pose3d.shape)
    print("2D confidence mean/min:", float(pose2d_conf.mean()), float(pose2d_conf.min()))

    # H36M indices
    hips, rhip, rank, lhip, lank, spine, head = 0, 1, 3, 4, 6, 7, 10

    v_head = pose3d[:, head] - pose3d[:, hips]
    v_spine = pose3d[:, spine] - pose3d[:, hips]
    v_rleg = pose3d[:, rank] - pose3d[:, rhip]
    v_lleg = pose3d[:, lank] - pose3d[:, lhip]

    print("\n=== Anatomical direction checks (source 3D) ===")
    print("hips->head mean xyz:", v_head.mean(axis=0))
    print("hips->spine mean xyz:", v_spine.mean(axis=0))
    print("rhip->rank mean xyz:", v_rleg.mean(axis=0))
    print("lhip->lank mean xyz:", v_lleg.mean(axis=0))

    root = pose3d[:, hips]
    print("\n=== Root trajectory (source 3D) ===")
    _print_axis_stats("root", root)

    bvh_path = out / "animation_mixamo.bvh"
    text = bvh_path.read_text(encoding="utf-8")
    print("\n=== BVH hierarchy markers ===")
    for marker in ["ROOT Hips", "JOINT Spine", "JOINT LeftHand", "JOINT RightHand"]:
        print(marker, "->", marker in text)

    lines = text.splitlines()
    motion_idx = lines.index("MOTION")
    frame_line = lines[motion_idx + 1]
    first_frame = np.fromstring(lines[motion_idx + 3], sep=" ")
    root_xyz = first_frame[:3]
    print("\n=== BVH motion checks ===")
    print(frame_line)
    print("first-frame root xyz:", root_xyz)


if __name__ == "__main__":
    main()
