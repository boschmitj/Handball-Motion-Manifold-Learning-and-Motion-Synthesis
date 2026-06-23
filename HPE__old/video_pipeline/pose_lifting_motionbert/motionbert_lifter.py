from __future__ import annotations

import json
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pose3DResult:
    joints_xyz: np.ndarray  # (T, J, 3)


class MotionBERTLifter:
    """
    Wrapper for MotionBERT inference.

    This module expects a cloned MotionBERT repository and uses its official
    `infer_wild.py` script to avoid hard-coding internal classes that can vary
    across revisions.
    """

    def __init__(
        self,
        repo_root: Path,
        checkpoint_path: Path,
        config_path: Path,
        device: str = "cuda:0",
    ) -> None:
        self.repo_root = repo_root
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device

        self.infer_script = self.repo_root / "infer_wild.py"
        if not self.infer_script.exists():
            raise FileNotFoundError(
                f"MotionBERT infer script not found at {self.infer_script}. "
                "Clone MotionBERT repo and keep default file layout."
            )

    @staticmethod
    def _coco_to_halpe26_frame(frame_xy: np.ndarray, frame_conf: np.ndarray) -> np.ndarray:
        """Map COCO-17 directly to Halpe-26, preserving face keypoints."""
        if frame_xy.shape != (17, 2) or frame_conf.shape != (17,):
            raise ValueError("Expected a single COCO frame with shapes (17, 2) and (17,)")

        halpe = np.zeros((26, 3), dtype=np.float32)

        # COCO 0..16 matches Halpe body ordering for these joints.
        halpe[:17, :2] = frame_xy
        halpe[:17, 2] = frame_conf

        # Halpe extras required by MotionBERT's halpe2h36m conversion:
        # 17=head, 18=neck, 19=hip-center.
        face_center = (frame_xy[0] + frame_xy[1] + frame_xy[2] + frame_xy[3] + frame_xy[4]) / 5.0
        face_conf = float(np.mean(frame_conf[[0, 1, 2, 3, 4]]))
        shoulder_center = 0.5 * (frame_xy[5] + frame_xy[6])
        shoulder_conf = min(float(frame_conf[5]), float(frame_conf[6]))
        hip_center = 0.5 * (frame_xy[11] + frame_xy[12])
        hip_conf = min(float(frame_conf[11]), float(frame_conf[12]))

        # In MotionBERT's halpe2h36m, y9 <- x0 (nose) and y10 <- x17 (head).
        # Synthesize x17 above the nose along the neck->nose direction.
        nose_xy = frame_xy[0]
        neck_to_nose = nose_xy - shoulder_center
        head_xy = nose_xy + 0.8 * neck_to_nose
        halpe[17, :2] = 0.7 * head_xy + 0.3 * face_center
        halpe[17, 2] = float(np.mean(frame_conf[[0, 1, 2, 3, 4]]))
        halpe[18, :2] = shoulder_center
        halpe[18, 2] = shoulder_conf
        halpe[19, :2] = hip_center
        halpe[19, 2] = hip_conf

        # Unused by halpe2h36m, but keep semantically plausible placeholders.
        halpe[20, :2] = frame_xy[15]
        halpe[20, 2] = frame_conf[15]
        halpe[21, :2] = frame_xy[16]
        halpe[21, 2] = frame_conf[16]
        halpe[22, :2] = frame_xy[15]
        halpe[22, 2] = frame_conf[15]
        halpe[23, :2] = frame_xy[16]
        halpe[23, 2] = frame_conf[16]
        halpe[24, :2] = frame_xy[15]
        halpe[24, 2] = frame_conf[15]
        halpe[25, :2] = frame_xy[16]
        halpe[25, 2] = frame_conf[16]

        return halpe

    @staticmethod
    def _h36m_to_halpe26_frame(frame_xy: np.ndarray, frame_conf: np.ndarray) -> np.ndarray:
        if frame_xy.shape != (17, 2) or frame_conf.shape != (17,):
            raise ValueError("Expected a single H36M frame with shapes (17, 2) and (17,)")

        halpe = np.zeros((26, 3), dtype=np.float32)
        halpe[0, :2] = frame_xy[10]
        halpe[0, 2] = frame_conf[10]
        halpe[5, :2] = frame_xy[11]
        halpe[5, 2] = frame_conf[11]
        halpe[6, :2] = frame_xy[14]
        halpe[6, 2] = frame_conf[14]
        halpe[7, :2] = frame_xy[12]
        halpe[7, 2] = frame_conf[12]
        halpe[8, :2] = frame_xy[15]
        halpe[8, 2] = frame_conf[15]
        halpe[9, :2] = frame_xy[13]
        halpe[9, 2] = frame_conf[13]
        halpe[10, :2] = frame_xy[16]
        halpe[10, 2] = frame_conf[16]
        halpe[11, :2] = frame_xy[4]
        halpe[11, 2] = frame_conf[4]
        halpe[12, :2] = frame_xy[1]
        halpe[12, 2] = frame_conf[1]
        halpe[13, :2] = frame_xy[5]
        halpe[13, 2] = frame_conf[5]
        halpe[14, :2] = frame_xy[2]
        halpe[14, 2] = frame_conf[2]
        halpe[15, :2] = frame_xy[6]
        halpe[15, 2] = frame_conf[6]
        halpe[16, :2] = frame_xy[3]
        halpe[16, 2] = frame_conf[3]
        halpe[17, :2] = frame_xy[0]
        halpe[17, 2] = frame_conf[0]
        halpe[18, :2] = 0.5 * (frame_xy[11] + frame_xy[14])
        halpe[18, 2] = min(float(frame_conf[11]), float(frame_conf[14]))
        halpe[19, :2] = 0.5 * (frame_xy[11] + frame_xy[14])
        halpe[19, 2] = min(float(frame_conf[11]), float(frame_conf[14]))
        halpe[20, :2] = frame_xy[15]
        halpe[20, 2] = frame_conf[15]
        halpe[21, :2] = frame_xy[16]
        halpe[21, 2] = frame_conf[16]
        halpe[22, :2] = frame_xy[15]
        halpe[22, 2] = frame_conf[15]
        halpe[23, :2] = frame_xy[16]
        halpe[23, 2] = frame_conf[16]
        halpe[24, :2] = frame_xy[15]
        halpe[24, 2] = frame_conf[15]
        halpe[25, :2] = frame_xy[16]
        halpe[25, 2] = frame_conf[16]
        return halpe

    def lift(
        self,
        h36m_xy: np.ndarray,
        h36m_conf: np.ndarray,
        video_path: Path,
        output_dir: Path,
        coco_xy: np.ndarray | None = None,
        coco_conf: np.ndarray | None = None,
    ) -> Pose3DResult:
        """
        Args:
            h36m_xy: (T, 17, 2), expected Human3.6M order in pixel coordinates.
            h36m_conf: (T, 17) confidence scores.
            video_path: source video path passed to MotionBERT for fps metadata.
        Returns:
            Pose3DResult with shape (T, 17, 3).
        """
        if h36m_xy.ndim != 3 or h36m_xy.shape[-1] != 2:
            raise ValueError("Expected input shape (T, J, 2)")
        if h36m_conf.shape != h36m_xy.shape[:2]:
            raise ValueError("Expected confidence shape (T, J)")

        use_coco_input = coco_xy is not None and coco_conf is not None
        if use_coco_input:
            if coco_xy.shape != h36m_xy.shape[:2] + (2,):
                raise ValueError("Expected COCO keypoints shape (T, 17, 2)")
            if coco_conf.shape != h36m_xy.shape[:2]:
                raise ValueError("Expected COCO confidence shape (T, 17)")

        output_dir.mkdir(parents=True, exist_ok=True)
        motionbert_out_dir = output_dir / "motionbert"
        motionbert_out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="motionbert_") as tmp:
            tmp_dir = Path(tmp)
            input_json = tmp_dir / "alphapose_results.json"

            items = []
            for frame_idx in range(h36m_xy.shape[0]):
                if use_coco_input:
                    halpe = self._coco_to_halpe26_frame(coco_xy[frame_idx], coco_conf[frame_idx])
                else:
                    halpe = self._h36m_to_halpe26_frame(h36m_xy[frame_idx], h36m_conf[frame_idx])
                items.append(
                    {
                        "idx": 0,
                        "image_id": frame_idx,
                        "keypoints": halpe.reshape(-1).tolist(),
                    }
                )

            input_json.write_text(json.dumps(items), encoding="utf-8")

            cmd = [
                sys.executable,
                str(self.infer_script),
                "--config",
                str(self.config_path),
                "--evaluate",
                str(self.checkpoint_path),
                "--vid_path",
                str(video_path),
                "--json_path",
                str(input_json),
                "--out_path",
                str(motionbert_out_dir),
            ]

            process = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    "MotionBERT inference failed. "
                    f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}\n"
                    "The current wrapper uses MotionBERT's infer_wild.py contract."
                )

            output_npy = motionbert_out_dir / "X3D.npy"
            if not output_npy.exists():
                raise RuntimeError(
                    "MotionBERT completed but no output numpy file was created. "
                    "Check infer_wild.py output path handling in your repository revision."
                )

            joints_xyz = np.asarray(np.load(output_npy), dtype=np.float32)

        np.save(output_dir / "pose3d_motionbert.npy", joints_xyz)
        return Pose3DResult(joints_xyz=joints_xyz)
