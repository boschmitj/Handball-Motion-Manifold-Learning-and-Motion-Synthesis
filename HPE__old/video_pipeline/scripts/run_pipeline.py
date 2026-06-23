from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from export.bvh_exporter import BVHExporter
from export.fbx_converter import BlenderFBXConverter
from export.json_animation_exporter import JsonAnimationExporter
from pose_estimation_vitpose.vitpose_estimator import ViTPoseEstimator
from pose_lifting_motionbert.motionbert_lifter import MotionBERTLifter
from postprocessing.smoothing import Pose3DPostprocessor
from postprocessing.visualization import (
    render_pose2d_overlay_video,
    save_3d_skeleton_plot,
    save_joint_trajectory_plot,
)
from preprocessing.joint_mapping import CocoToH36MMapper
from preprocessing.normalization import KeypointNormalizer
from retargeting.mixamo_retargeter import MixamoRetargeter
from video_processing.frame_extractor import FrameExtractor


@dataclass
class PipelineRunConfig:
    video: Path
    output_dir: Path
    vitpose_config: Path
    vitpose_checkpoint: Path
    vitpose_device: str
    motionbert_repo: Path
    motionbert_config: Path
    motionbert_checkpoint: Path
    motionbert_device: str
    target_width: int = 256
    target_height: int = 256
    blender_executable: str = "blender"
    export_fbx: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monocular video to Unity BVH pipeline")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--vitpose_config", type=Path, required=True)
    parser.add_argument("--vitpose_checkpoint", type=Path, required=True)
    parser.add_argument("--vitpose_device", type=str, default="cuda:0")

    parser.add_argument("--motionbert_repo", type=Path, required=True)
    parser.add_argument("--motionbert_config", type=Path, required=True)
    parser.add_argument("--motionbert_checkpoint", type=Path, required=True)
    parser.add_argument("--motionbert_device", type=str, default="cuda:0")

    parser.add_argument("--target_width", type=int, default=256)
    parser.add_argument("--target_height", type=int, default=256)
    parser.add_argument("--blender_executable", type=str, default="blender")
    parser.add_argument("--disable_fbx_export", action="store_true")
    return parser.parse_args()


def run(config: PipelineRunConfig) -> dict[str, str]:
    config = PipelineRunConfig(
        video=config.video.resolve(),
        output_dir=config.output_dir.resolve(),
        vitpose_config=config.vitpose_config.resolve(),
        vitpose_checkpoint=config.vitpose_checkpoint.resolve(),
        vitpose_device=config.vitpose_device,
        motionbert_repo=config.motionbert_repo.resolve(),
        motionbert_config=config.motionbert_config.resolve(),
        motionbert_checkpoint=config.motionbert_checkpoint.resolve(),
        motionbert_device=config.motionbert_device,
        target_width=config.target_width,
        target_height=config.target_height,
        blender_executable=config.blender_executable,
        export_fbx=config.export_fbx,
    )

    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not config.video.exists():
        raise FileNotFoundError(f"Input video not found: {config.video}")
    if not config.vitpose_config.exists():
        raise FileNotFoundError(f"ViTPose config not found: {config.vitpose_config}")
    if not config.vitpose_checkpoint.exists():
        raise FileNotFoundError(f"ViTPose checkpoint not found: {config.vitpose_checkpoint}")
    if not config.motionbert_repo.exists():
        raise FileNotFoundError(f"MotionBERT repo not found: {config.motionbert_repo}")
    if not config.motionbert_config.exists():
        raise FileNotFoundError(f"MotionBERT config not found: {config.motionbert_config}")
    if not config.motionbert_checkpoint.exists():
        raise FileNotFoundError(f"MotionBERT checkpoint not found: {config.motionbert_checkpoint}")

    # 1) Environment setup is documented in README/environment.yml.

    # 2) Video loading + frame extraction
    extractor = FrameExtractor(target_size=(config.target_width, config.target_height))
    frame_batch = extractor.extract(config.video)
    np.save(out / "frames.npy", frame_batch.frames)

    # 3) ViTPose inference (visual verification via overlay video)
    vitpose = ViTPoseEstimator(
        config_path=config.vitpose_config,
        checkpoint_path=config.vitpose_checkpoint,
        device=config.vitpose_device,
    )
    pose2d = vitpose.infer(frame_batch.frames)
    vitpose.save(
        pose2d,
        npy_path=out / "pose2d_coco.npy",
        json_path=out / "pose2d_coco.json",
    )
    render_pose2d_overlay_video(
        frame_batch.frames,
        pose2d.keypoints_xy,
        pose2d.confidence,
        out_path=out / "pose2d_overlay.mp4",
        fps=frame_batch.fps,
    )

    # 4) Joint mapping COCO -> H36M and normalization
    mapper = CocoToH36MMapper()
    h36m_xy, h36m_conf = mapper.map(pose2d.keypoints_xy, pose2d.confidence)
    normalizer = KeypointNormalizer(root_index=0, left_shoulder=11, right_shoulder=14)
    h36m_norm, root_xy, scale_xy = normalizer.normalize(h36m_xy)

    np.save(out / "pose2d_h36m.npy", h36m_xy)
    np.save(out / "pose2d_h36m_norm.npy", h36m_norm)
    np.save(out / "pose2d_h36m_conf.npy", h36m_conf)

    # 5) MotionBERT inference
    lifter = MotionBERTLifter(
        repo_root=config.motionbert_repo,
        checkpoint_path=config.motionbert_checkpoint,
        config_path=config.motionbert_config,
    )
    pose3d = lifter.lift(
        h36m_xy,
        h36m_conf,
        config.video,
        output_dir=out,
        coco_xy=pose2d.keypoints_xy,
        coco_conf=pose2d.confidence,
    )

    # 6) Visualize 3D skeleton
    save_3d_skeleton_plot(pose3d.joints_xyz, out_path=out / "pose3d_frame0.png", frame_idx=0)

    # 7) Smoothing and normalization
    post = Pose3DPostprocessor()
    smooth_xyz = post.smooth(pose3d.joints_xyz)
    smooth_xyz = post.enforce_constant_bone_lengths(smooth_xyz)
    smooth_xyz = post.normalize_scale(smooth_xyz, desired_height=1.8)
    smooth_xyz = post.align_ground(smooth_xyz)
    np.save(out / "pose3d_smoothed.npy", smooth_xyz)
    vel = post.velocity(smooth_xyz, dt=1.0 / frame_batch.fps)
    np.save(out / "pose3d_velocity.npy", vel)
    save_joint_trajectory_plot(smooth_xyz, joint_idx=0, out_path=out / "traj_joint0.png")

    # 8) Retargeting
    retargeter = MixamoRetargeter()
    anim = retargeter.retarget(smooth_xyz)

    # 8.1) Export direct JSON animation for Unity runtime playback
    json_path = out / "animation_mixamo.json"
    json_exporter = JsonAnimationExporter(frame_rate=frame_batch.fps)
    json_exporter.export(anim, json_path)

    # 9) Export BVH
    exporter = BVHExporter(frame_time=1.0 / frame_batch.fps)
    bvh_path = out / "animation_mixamo.bvh"
    exporter.export(anim, bvh_path)

    # 9.1) Export FBX via Blender CLI
    fbx_path = out / "animation_mixamo.fbx"
    if config.export_fbx:
        fbx_converter = BlenderFBXConverter(blender_executable=config.blender_executable)
        fbx_converter.convert(bvh_path, fbx_path)

    # 10) Frontend provided in frontend/app.py

    summary = {
        "frames": str(out / "frames.npy"),
        "pose2d_overlay": str(out / "pose2d_overlay.mp4"),
        "pose3d_plot": str(out / "pose3d_frame0.png"),
        "trajectory_plot": str(out / "traj_joint0.png"),
        "json_animation": str(json_path),
        "bvh": str(bvh_path),
        "fbx": str(fbx_path) if config.export_fbx else "",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        PipelineRunConfig(
            video=args.video,
            output_dir=args.output_dir,
            vitpose_config=args.vitpose_config,
            vitpose_checkpoint=args.vitpose_checkpoint,
            vitpose_device=args.vitpose_device,
            motionbert_repo=args.motionbert_repo,
            motionbert_config=args.motionbert_config,
            motionbert_checkpoint=args.motionbert_checkpoint,
            motionbert_device=args.motionbert_device,
            target_width=args.target_width,
            target_height=args.target_height,
            blender_executable=args.blender_executable,
            export_fbx=not args.disable_fbx_export,
        )
    )
    print("Pipeline completed.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
