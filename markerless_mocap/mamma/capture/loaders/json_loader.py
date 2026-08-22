"""JSON calibration loader.

Vendored from ``capture/cameras/opencv_json.py``
(``parse_opencv_json``) and ``cameras/utils.py`` (``load_cam_json``),
rewritten to drop torch coupling and to merge the two formats behind
a single ``load`` entry point that picks the right code path based on
the file's structure.

Three layouts are accepted:

* **OpenCV nested**: ``{ "<seq_name>": { "<cam_name>": { intrinsic_matrix,
  distortions, extrinsics_matrix, image_size } } }`` — the format produced
  by the cluster's calibration scripts. Multiple sequences are supported;
  a warning is emitted and the first one is used (matching upstream).
* **OpenCV flat**: ``{ "<cam_name>": { intrinsic_matrix, ... } }`` — the
  same per-camera shape as nested but with no sequence wrapper.
* **Legacy flat**: ``{ "<cam_name>": { focal, princpt, rotation, position,
  sensor_size [, vicon_radial_2] } }`` — older calibrations.

Distortion-model normalization:

* OpenCV layouts produce :data:`Camera.distortion_model = "opencv_brown"`
  with 5 params ``[k1, k2, p1, p2, k3]``.
* Legacy entries with a ``vicon_radial_2`` key produce
  :data:`Camera.distortion_model = "vicon_radial_2"` with 5 params; legacy
  entries without that key fall back to ``"opencv_brown"`` with all-zero
  coefficients (no distortion).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..calibration import Camera, CalibrationError


def load(path: Path) -> Dict[str, Camera]:
    """Parse a JSON calibration file. Returns ``{cam_name: Camera}``."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CalibrationError(f"{path}: invalid JSON: {e}") from e

    if not isinstance(data, dict) or not data:
        raise CalibrationError(f"{path}: JSON must be a non-empty object")

    first_val = next(iter(data.values()))
    if not isinstance(first_val, dict):
        raise CalibrationError(
            f"{path}: top-level JSON values must be objects"
        )

    if "intrinsic_matrix" in first_val:
        return {name: cam for name, cam in
                _parse_opencv_flat(path, data).items()}

    second_val = next(iter(first_val.values()), None)
    if isinstance(second_val, dict) and "intrinsic_matrix" in second_val:
        return _parse_opencv_nested(path, data)

    if "focal" in first_val or "rotation" in first_val:
        return _parse_legacy_flat(path, data)

    raise CalibrationError(
        f"{path}: unrecognized JSON layout; expected OpenCV "
        "(nested or flat) or legacy flat"
    )


# ---------------------------------------------------------------------------
# OpenCV layouts
# ---------------------------------------------------------------------------

def _parse_opencv_nested(path: Path, data: Dict[str, Any]) -> Dict[str, Camera]:
    if len(data) > 1:
        warnings.warn(
            f"{path}: contains multiple sequences ({list(data.keys())}); "
            "loading the first one. Split per-sequence files for clarity.",
            stacklevel=3,
        )
    cameras_raw = next(iter(data.values()))
    return _parse_opencv_flat(path, cameras_raw)


def _parse_opencv_flat(path: Path, cameras_raw: Dict[str, Any]) -> Dict[str, Camera]:
    cameras: Dict[str, Camera] = {}
    for cam_name, cam in cameras_raw.items():
        if not isinstance(cam, dict):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} is not an object"
            )
        try:
            K = np.asarray(cam["intrinsic_matrix"], dtype=np.float64)
            ext_3x4 = np.asarray(cam["extrinsics_matrix"], dtype=np.float64)
            distortions = list(cam["distortions"])
            img_size = cam["image_size"]
        except KeyError as e:
            raise CalibrationError(
                f"{path}: camera {cam_name!r} missing field {e.args[0]!r}"
            ) from e

        if K.shape != (3, 3):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} intrinsic_matrix must be 3x3, "
                f"got shape {K.shape}"
            )
        if ext_3x4.shape != (3, 4):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} extrinsics_matrix must be 3x4, "
                f"got shape {ext_3x4.shape}"
            )
        if len(distortions) != 5:
            raise CalibrationError(
                f"{path}: camera {cam_name!r} distortions must have length 5 "
                f"[k1,k2,p1,p2,k3], got {len(distortions)}"
            )

        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :] = ext_3x4

        # Invert the rigid-body transform without forming a full inverse.
        R = T_cam_world[:3, :3]
        t = T_cam_world[:3, 3]
        T_world_cam = np.eye(4, dtype=np.float64)
        T_world_cam[:3, :3] = R.T
        T_world_cam[:3, 3] = -R.T @ t

        width, height = _read_image_size(img_size, path, cam_name)

        cameras[str(cam_name)] = Camera(
            name=str(cam_name),
            width=width,
            height=height,
            intrinsics=K,
            distortion_model="opencv_brown",
            distortion_coeffs=tuple(float(v) for v in distortions),
            T_cam_world=T_cam_world,
            T_world_cam=T_world_cam,
        )
    return cameras


def _read_image_size(img_size: Any, path: Path, cam_name: str) -> tuple[int, int]:
    """Accept ``[[W, H]]`` (legacy) or ``[W, H]``."""
    if (isinstance(img_size, list) and len(img_size) == 1
            and isinstance(img_size[0], (list, tuple))):
        img_size = img_size[0]
    if not (isinstance(img_size, (list, tuple)) and len(img_size) == 2):
        raise CalibrationError(
            f"{path}: camera {cam_name!r} image_size must be [W, H], "
            f"got {img_size!r}"
        )
    try:
        w = int(img_size[0])
        h = int(img_size[1])
    except (TypeError, ValueError) as e:
        raise CalibrationError(
            f"{path}: camera {cam_name!r} image_size must contain ints, "
            f"got {img_size!r}"
        ) from e
    if w <= 0 or h <= 0:
        raise CalibrationError(
            f"{path}: camera {cam_name!r} image_size must be positive, "
            f"got [{w}, {h}]"
        )
    return w, h


# ---------------------------------------------------------------------------
# Legacy flat layout
# ---------------------------------------------------------------------------

def _parse_legacy_flat(path: Path, cameras_raw: Dict[str, Any]) -> Dict[str, Camera]:
    cameras: Dict[str, Camera] = {}
    for cam_name, cam in cameras_raw.items():
        if not isinstance(cam, dict):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} is not an object"
            )
        try:
            rot = np.asarray(cam["rotation"], dtype=np.float64)
            position = np.asarray(cam["position"], dtype=np.float64)
            focal = float(cam["focal"])
            princpt = [float(p) for p in cam["princpt"]]
            sensor_size = cam["sensor_size"]
        except KeyError as e:
            raise CalibrationError(
                f"{path}: camera {cam_name!r} missing field {e.args[0]!r}"
            ) from e

        if rot.shape != (3, 3):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} rotation must be 3x3, "
                f"got shape {rot.shape}"
            )
        if position.shape != (3,):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} position must be length 3, "
                f"got shape {position.shape}"
            )

        if "intrinsics" in cam:
            K = np.asarray(cam["intrinsics"], dtype=np.float64)
            if K.shape != (3, 3):
                raise CalibrationError(
                    f"{path}: camera {cam_name!r} intrinsics must be 3x3, "
                    f"got shape {K.shape}"
                )
        else:
            K = np.array(
                [[focal, 0.0, princpt[0]],
                 [0.0, focal, princpt[1]],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

        if "extrinsics" in cam:
            T_cam_world = np.asarray(cam["extrinsics"], dtype=np.float64)
            if T_cam_world.shape != (4, 4):
                raise CalibrationError(
                    f"{path}: camera {cam_name!r} extrinsics must be 4x4, "
                    f"got shape {T_cam_world.shape}"
                )
            R = T_cam_world[:3, :3]
            t = T_cam_world[:3, 3]
            T_world_cam = np.eye(4, dtype=np.float64)
            T_world_cam[:3, :3] = R.T
            T_world_cam[:3, 3] = -R.T @ t
        else:
            T_cam_world = np.eye(4, dtype=np.float64)
            T_cam_world[:3, :3] = rot
            T_cam_world[:3, 3] = -rot @ position
            T_world_cam = np.eye(4, dtype=np.float64)
            T_world_cam[:3, :3] = rot.T
            T_world_cam[:3, 3] = position

        if not (isinstance(sensor_size, (list, tuple)) and len(sensor_size) == 2):
            raise CalibrationError(
                f"{path}: camera {cam_name!r} sensor_size must be [W, H], "
                f"got {sensor_size!r}"
            )
        try:
            width = int(sensor_size[0])
            height = int(sensor_size[1])
        except (TypeError, ValueError) as e:
            raise CalibrationError(
                f"{path}: camera {cam_name!r} sensor_size must contain ints, "
                f"got {sensor_size!r}"
            ) from e

        if "vicon_radial_2" in cam:
            vr2 = list(cam["vicon_radial_2"])
            if len(vr2) != 5:
                raise CalibrationError(
                    f"{path}: camera {cam_name!r} vicon_radial_2 must have "
                    f"length 5, got {len(vr2)}"
                )
            distortion_model = "vicon_radial_2"
            distortion_coeffs = tuple(float(v) for v in vr2)
        else:
            distortion_model = "opencv_brown"
            distortion_coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)

        cameras[str(cam_name)] = Camera(
            name=str(cam_name),
            width=width,
            height=height,
            intrinsics=K,
            distortion_model=distortion_model,
            distortion_coeffs=distortion_coeffs,
            T_cam_world=T_cam_world,
            T_world_cam=T_world_cam,
        )
    return cameras
