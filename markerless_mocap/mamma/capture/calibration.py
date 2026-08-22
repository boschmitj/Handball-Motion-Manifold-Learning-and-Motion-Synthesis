"""Calibration data model and format-dispatching loader.

Three calibration file formats are supported:

* ``.yaml`` / ``.yml`` — pinhole + radtan, Hamilton ``[w,x,y,z]`` quaternion,
  meters, single-rig per file. The format we recommend for new users.
* ``.xcp`` — Vicon XML calibration export. Vicon 5-parameter radial
  distortion, JPL ``[x,y,z,w]`` quaternion, millimetre units (converted
  to metres at parse time).
* ``.json`` — OpenCV-style 3x3 intrinsics + 3x4 extrinsics + 5-param
  Brown-Conrady distortion. Both nested ``{seq: {cam: ...}}`` and flat
  ``{cam: ...}`` layouts are accepted, plus a legacy flat layout that
  uses ``focal`` / ``princpt`` / ``rotation`` / ``position`` keys.

All loaders return the same in-memory representation (:class:`Calibration`
holding a mapping of :class:`Camera`). Both ``T_cam_world`` (world->cam,
the projection-ready transform) and ``T_world_cam`` (camera pose in
the world) are stored verbatim — every downstream consumer wants one or
the other and round-tripping needs both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


VALID_DISTORTION_MODELS: Tuple[str, ...] = (
    "radtan",            # YAML: 4-param Brown-Conrady (k1, k2, p1, p2)
    "opencv_brown",      # OpenCV JSON: 5-param (k1, k2, p1, p2, k3)
    "vicon_radial_2",    # Vicon XCP / legacy JSON: (pp_x, pp_y, rad_1, rad_2, rad_3)
)


class CalibrationError(ValueError):
    """Raised by any loader when a calibration file is malformed or unsupported."""


@dataclass(frozen=True)
class Camera:
    """A single calibrated camera.

    Attributes:
        name: Camera identifier (must be unique within a :class:`Calibration`).
        width: Image width in pixels.
        height: Image height in pixels.
        intrinsics: 3x3 ``K`` matrix in float64. ``K[0,0] = fx``, ``K[1,1] = fy``,
            ``K[0,2] = cx``, ``K[1,2] = cy``.
        distortion_model: One of :data:`VALID_DISTORTION_MODELS`.
        distortion_coeffs: Tuple of distortion coefficients. Length depends on
            the model (4 for ``radtan``, 5 for ``opencv_brown`` and
            ``vicon_radial_2``).
        T_cam_world: 4x4 float64 transform. Multiplying a homogeneous world
            point by this gives the point in the camera frame
            (``p_cam = T_cam_world @ p_world``). This is what most projection
            code expects.
        T_world_cam: 4x4 float64 transform. The camera's pose in the world
            (its inverse is ``T_cam_world``). Stored verbatim so we never
            lose the "natural" representation of a given source format.
    """

    name: str
    width: int
    height: int
    intrinsics: np.ndarray
    distortion_model: str
    distortion_coeffs: Tuple[float, ...]
    T_cam_world: np.ndarray
    T_world_cam: np.ndarray


@dataclass(frozen=True)
class Calibration:
    """A multi-camera rig loaded from a calibration file."""

    cameras: Mapping[str, Camera]
    source_format: str            # "yaml" | "xcp" | "json"
    source_path: Path


_EXT_DISPATCH = {
    ".yaml": ("yaml", "yaml_loader"),
    ".yml":  ("yaml", "yaml_loader"),
    ".xcp":  ("xcp",  "xcp_loader"),
    ".json": ("json", "json_loader"),
}


def load_calibration(path) -> Calibration:
    """Load a calibration file by path. Format is selected from the extension.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` / ``.xcp`` / ``.json`` file.
            Both ``str`` and :class:`pathlib.Path` are accepted; extension
            matching is case-insensitive.

    Returns:
        A :class:`Calibration` whose ``cameras`` map is sorted by camera name.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        CalibrationError: For any other failure (unsupported extension,
            malformed file, missing required field, bad quaternion, ...).
    """
    p = Path(os.fspath(path))
    if not p.exists():
        raise FileNotFoundError(f"calibration file not found: {p}")

    ext = p.suffix.lower()
    if ext not in _EXT_DISPATCH:
        raise CalibrationError(
            f"unsupported calibration extension: {ext!r}; "
            f"supported: .yaml/.yml/.xcp/.json"
        )

    source_format, module_name = _EXT_DISPATCH[ext]
    # Lazy import so an optional dep (pyyaml) only matters when YAML is used.
    if module_name == "yaml_loader":
        from .loaders import yaml_loader as loader
    elif module_name == "xcp_loader":
        from .loaders import xcp_loader as loader
    else:
        from .loaders import json_loader as loader

    cameras = loader.load(p)
    return Calibration(
        cameras=dict(sorted(cameras.items())),
        source_format=source_format,
        source_path=p,
    )
