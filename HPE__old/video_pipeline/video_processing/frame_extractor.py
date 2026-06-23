from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameBatch:
    frames: np.ndarray
    fps: float
    original_size: tuple[int, int]


class FrameExtractor:
    """Reads MP4 files and returns a normalized frame tensor."""

    def __init__(self, target_size: tuple[int, int] = (256, 256)) -> None:
        self.target_size = target_size

    def extract(self, video_path: Path) -> FrameBatch:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_AREA)
            frames.append(resized)

        capture.release()

        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path}")

        return FrameBatch(
            frames=np.asarray(frames, dtype=np.uint8),
            fps=fps,
            original_size=(width, height),
        )

    @staticmethod
    def save_preview_video(frames: np.ndarray, out_path: Path, fps: float) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Expected frames with shape (num_frames, H, W, 3)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        for frame_rgb in frames:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        writer.release()
