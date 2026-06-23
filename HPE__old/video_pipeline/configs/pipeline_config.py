from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoConfig:
    input_video: Path
    output_dir: Path
    target_size: tuple[int, int] = (256, 256)


@dataclass
class ViTPoseConfig:
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    keypoint_format: str = "coco"


@dataclass
class MotionBERTConfig:
    repo_root: Path
    checkpoint_path: Path
    config_path: Path
    device: str = "cuda:0"


@dataclass
class PostprocessConfig:
    smooth_window: int = 9
    smooth_polyorder: int = 2
    foot_joint_indices: tuple[int, int] = (3, 6)


@dataclass
class PipelineConfig:
    video: VideoConfig
    vitpose: ViTPoseConfig
    motionbert: MotionBERTConfig
    postprocess: PostprocessConfig
