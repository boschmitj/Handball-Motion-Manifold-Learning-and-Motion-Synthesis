from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None


# Mixamo-like human bone naming (close to Unity Humanoid mapping).
MIXAMO_SKELETON: tuple[Joint, ...] = (
    Joint("Hips", None),
    Joint("RightUpLeg", "Hips"),
    Joint("RightLeg", "RightUpLeg"),
    Joint("RightFoot", "RightLeg"),
    Joint("LeftUpLeg", "Hips"),
    Joint("LeftLeg", "LeftUpLeg"),
    Joint("LeftFoot", "LeftLeg"),
    Joint("Spine", "Hips"),
    Joint("Spine1", "Spine"),
    Joint("Neck", "Spine1"),
    Joint("Head", "Neck"),
    Joint("LeftShoulder", "Spine1"),
    Joint("LeftArm", "LeftShoulder"),
    Joint("LeftForeArm", "LeftArm"),
    Joint("LeftHand", "LeftForeArm"),
    Joint("RightShoulder", "Spine1"),
    Joint("RightArm", "RightShoulder"),
    Joint("RightForeArm", "RightArm"),
    Joint("RightHand", "RightForeArm"),
)


# MotionBERT/H36M index to mixamo-like names, index aligned to 17-joint H36M order.
H36M_TO_MIXAMO_NAMES: tuple[str, ...] = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
)
