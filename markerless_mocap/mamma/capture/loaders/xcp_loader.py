"""Vicon ``.xcp`` calibration loader.

Vendored from ``capture/cameras/xcp.py::parse_xcp``
(lines 9-86), rewritten to:

* drop torch / device coupling (return numpy float64 throughout),
* drop the toolkit's Windows-path shim,
* coerce ``SENSOR_SIZE`` to ints (the upstream returns floats which break
  downstream mask indexing),
* always read positions in millimetres and convert to metres (single
  consumer, no ``out_unit`` ambiguity),
* skip non-``VideoFile`` cameras silently (matches upstream behaviour),
* keep using ``scipy.spatial.transform.Rotation`` for the JPL ``[x,y,z,w]``
  quaternion -- the well-tested path, scalar-last is the scipy default.

Vicon distortion is the proprietary 5-parameter ``Vicon3Parameter``
``[pp_x, pp_y, rad_1, rad_2, rad_3]`` model. We pass it through
verbatim under :data:`Camera.distortion_model = "vicon_radial_2"`;
no lossy conversion to OpenCV's Brown-Conrady is performed.
"""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np

from ..calibration import Camera, CalibrationError

_MM_PER_M = 1000.0


def load(path: Path, *, cam_prefix: str = "IOI", use_deviceid: bool = False) -> Dict[str, Camera]:
    """Parse a Vicon ``.xcp`` file. Returns ``{cam_name: Camera}``."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as e:  # pragma: no cover
        raise CalibrationError(
            "XCP calibration support requires scipy "
            "(usually installed transitively via the numpy/torch stack)"
        ) from e

    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise CalibrationError(f"{path}: invalid XCP/XML: {e}") from e

    cameras: Dict[str, Camera] = {}
    for camera_node in tree.getroot():
        if camera_node.attrib.get("TYPE") != "VideoFile":
            continue

        try:
            device_id = int(camera_node.attrib["DEVICEID"])
            name = camera_node.attrib["NAME"]
            sensor_size = camera_node.attrib["SENSOR_SIZE"].split()
            aspect_ratio = float(camera_node.attrib["PIXEL_ASPECT_RATIO"])

            kf = camera_node.find(".//KeyFrame")
            if kf is None:
                raise CalibrationError(
                    f"{path}: camera {name!r} has no <KeyFrame>"
                )
            position_mm = [float(v) for v in kf.attrib["POSITION"].split()]
            quat_xyzw = [float(v) for v in kf.attrib["ORIENTATION"].split()]
            principal_point = [float(v) for v in kf.attrib["PRINCIPAL_POINT"].split()]
            focal_length_px = float(kf.attrib["FOCAL_LENGTH"])
            vicon_radial_2 = [float(v) for v in kf.attrib["VICON_RADIAL2"].split()[1:]]
        except (KeyError, ValueError) as e:
            raise CalibrationError(f"{path}: malformed camera node: {e}") from e

        if use_deviceid:
            name = f"{cam_prefix}_{(device_id + 1):02d}"
        elif len(name) < 3:
            warnings.warn(
                f"Camera name {name!r} is too short -- you may need to load "
                "with use_deviceid=True (older calibrations).",
                stacklevel=2,
            )

        width = int(round(float(sensor_size[0])))
        height = int(round(float(sensor_size[1])))

        position_m = np.asarray(position_mm, dtype=np.float64) / _MM_PER_M

        # Vicon's quaternion is JPL [x, y, z, w] in scipy's scalar-last order.
        rot = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)

        K = np.array(
            [[focal_length_px, 0.0, principal_point[0]],
             [0.0, focal_length_px / aspect_ratio, principal_point[1]],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # Upstream stores cam_ext = [R | -R @ position]; that's T_cam_world.
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = rot
        T_cam_world[:3, 3] = -rot @ position_m

        # T_world_cam is its inverse; for a rigid body that's R^T and the
        # original world-frame position.
        T_world_cam = np.eye(4, dtype=np.float64)
        T_world_cam[:3, :3] = rot.T
        T_world_cam[:3, 3] = position_m

        if len(vicon_radial_2) != 5:
            raise CalibrationError(
                f"{path}: camera {name!r} has VICON_RADIAL2 with "
                f"{len(vicon_radial_2)} params, expected 5"
            )

        cameras[name] = Camera(
            name=name,
            width=width,
            height=height,
            intrinsics=K,
            distortion_model="vicon_radial_2",
            distortion_coeffs=tuple(vicon_radial_2),
            T_cam_world=T_cam_world,
            T_world_cam=T_world_cam,
        )

    if not cameras:
        raise CalibrationError(
            f"{path}: no VideoFile cameras found in XCP"
        )

    return cameras
