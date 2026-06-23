from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "MMPose is not installed. Install dependencies from environment.yml first."
    ) from exc


@dataclass
class Pose2DResult:
    keypoints_xy: np.ndarray  # (T, J, 2)
    confidence: np.ndarray  # (T, J)
    format_name: str = "coco"


class ViTPoseEstimator:
    """Runs ViTPose using MMPose top-down API on a single-person full-frame box."""

    def __init__(self, config_path: Path, checkpoint_path: Path, device: str = "cuda:0") -> None:
        register_all_modules()
        self.model = init_model(str(config_path), str(checkpoint_path), device=device)

    def infer(self, frames: np.ndarray) -> Pose2DResult:
        if frames.ndim != 4:
            raise ValueError("Expected frames with shape (num_frames, H, W, 3)")

        all_xy: list[np.ndarray] = []
        all_conf: list[np.ndarray] = []

        for frame_rgb in frames:
            h, w = frame_rgb.shape[:2]
            person_bbox = np.array([0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            preds = inference_topdown(self.model, frame_bgr, bboxes=[person_bbox], bbox_format="xyxy")

            if len(preds) == 0:
                n_joints = self.model.dataset_meta["num_keypoints"]
                all_xy.append(np.zeros((n_joints, 2), dtype=np.float32))
                all_conf.append(np.zeros((n_joints,), dtype=np.float32))
                continue

            sample = preds[0]
            instances: Any = sample.pred_instances
            if hasattr(instances, "keypoint_scores"):
                conf = np.asarray(instances.keypoint_scores[0], dtype=np.float32)
            else:
                conf = np.ones((instances.keypoints.shape[1],), dtype=np.float32)

            xy = np.asarray(instances.keypoints[0], dtype=np.float32)
            all_xy.append(xy)
            all_conf.append(conf)

        keypoints_xy = np.stack(all_xy, axis=0)
        confidence = np.stack(all_conf, axis=0)
        return Pose2DResult(keypoints_xy=keypoints_xy, confidence=confidence)

    @staticmethod
    def save(result: Pose2DResult, npy_path: Path, json_path: Path) -> None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(npy_path, {"keypoints_xy": result.keypoints_xy, "confidence": result.confidence})
        payload = {
            "format": result.format_name,
            "shape": {
                "keypoints_xy": list(result.keypoints_xy.shape),
                "confidence": list(result.confidence.shape),
            },
            "keypoints_xy": result.keypoints_xy.tolist(),
            "confidence": result.confidence.tolist(),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
