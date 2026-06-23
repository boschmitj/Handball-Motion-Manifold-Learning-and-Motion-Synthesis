from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from pose_lifting_motionbert.motionbert_lifter import MotionBERTLifter


def _stats(arr: np.ndarray) -> tuple[float, float, float]:
    d_neck_head = np.linalg.norm(arr[:, 10] - arr[:, 9], axis=1).mean()
    d_thorax_neck = np.linalg.norm(arr[:, 9] - arr[:, 8], axis=1).mean()
    d_hips_head = np.linalg.norm(arr[:, 10] - arr[:, 0], axis=1).mean()
    return float(d_neck_head), float(d_thorax_neck), float(d_hips_head)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_old = root / "outputs" / "run09"
    out_new = root / "outputs" / "run10_liftonly"

    if out_new.exists():
        shutil.rmtree(out_new)
    out_new.mkdir(parents=True, exist_ok=True)

    h36_xy = np.load(out_old / "pose2d_h36m.npy")
    h36_cf = np.load(out_old / "pose2d_h36m_conf.npy")
    coco_payload = np.load(out_old / "pose2d_coco.npy", allow_pickle=True).item()

    lifter = MotionBERTLifter(
        repo_root=root / "third_party" / "MotionBERT",
        checkpoint_path=root
        / "third_party"
        / "MotionBERT"
        / "checkpoint"
        / "pose3d"
        / "MB_train_h36m"
        / "best_epoch.bin",
        config_path=root / "third_party" / "MotionBERT" / "configs" / "pose3d" / "MB_train_h36m.yaml",
    )

    result = lifter.lift(
        h36_xy,
        h36_cf,
        video_path=root / "data" / "samples" / "sample_video2.mp4",
        output_dir=out_new,
        coco_xy=coco_payload["keypoints_xy"],
        coco_conf=coco_payload["confidence"],
    )

    x_old = np.load(out_old / "motionbert" / "X3D.npy")
    x_new = result.joints_xyz

    print("old raw neck-head, thorax-neck, hips-head:", _stats(x_old))
    print("new raw neck-head, thorax-neck, hips-head:", _stats(x_new))
    print("mean abs diff all joints:", float(np.abs(x_new - x_old).mean()))


if __name__ == "__main__":
    main()
