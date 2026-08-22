#!/usr/bin/env python3
"""Convert a Qualisys QTM calibration XML/QCA export to MAMMA YAML.

QTM stores image coordinates in 1/64-pixel subpixels, camera positions in
millimetres, and its camera axes differ from OpenCV's by a 180-degree rotation
around the local X axis.  MAMMA's YAML format stores the OpenCV camera pose in
the world as a Hamilton quaternion ``[w, x, y, z]`` and metres.

Camera names default to the Qualisys serial numbers.  Name the corresponding
videos ``<serial>.mp4``, or use ``--name-mode index`` to emit cam01, cam02, ...

Only Miqus cameras are converted by default for the current handball capture.
Pass ``--camera-model all`` if the other calibrated camera models are needed
later.

This converter deliberately records, but does not apply, QTM's ``viewrotation``.
That setting describes display orientation.  If exported videos have swapped
width/height, rotate the videos back to the sensor orientation before using
this calibration.
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


SUBPIXELS_PER_PIXEL = 64.0
MILLIMETRES_PER_METRE = 1000.0
QTM_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


class ConversionError(ValueError):
    """Raised when a QTM calibration cannot be converted safely."""


def _required_float(element: ET.Element, attribute: str) -> float:
    value = element.get(attribute)
    if value is None:
        raise ConversionError(
            f"<{element.tag}> is missing required attribute {attribute!r}"
        )
    try:
        result = float(value)
    except ValueError as exc:
        raise ConversionError(
            f"<{element.tag}> attribute {attribute!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ConversionError(
            f"<{element.tag}> attribute {attribute!r} is not finite: {value!r}"
        )
    return result


def _nearest_rotation(matrix: np.ndarray, camera_name: str) -> np.ndarray:
    """Remove harmless QTM text-rounding error from a rotation matrix."""
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    error = float(np.linalg.norm(matrix - rotation))
    if error > 1e-3:
        raise ConversionError(
            f"camera {camera_name}: transform is not a rotation matrix "
            f"(distance to nearest rotation: {error:.6g})"
        )
    return rotation


def _camera_name(camera: ET.Element, index: int, mode: str, prefix: str) -> str:
    if mode == "serial":
        serial = (camera.get("serial") or "").strip()
        if not serial:
            raise ConversionError(f"camera {index + 1}: missing serial number")
        return serial
    return f"{prefix}{index + 1:02d}"


def _convert_camera(
    camera: ET.Element,
    index: int,
    *,
    name_mode: str,
    name_prefix: str,
    binning_factor: float,
) -> tuple[str, dict, dict]:
    name = _camera_name(camera, index, name_mode, name_prefix)
    transform = camera.find("transform")
    intrinsic = camera.find("intrinsic")
    fov = camera.find("fov_video")
    if transform is None or intrinsic is None or fov is None:
        raise ConversionError(
            f"camera {name}: expected transform, intrinsic, and fov_video elements"
        )

    left = _required_float(fov, "left")
    top = _required_float(fov, "top")
    right = _required_float(fov, "right")
    bottom = _required_float(fov, "bottom")
    width = int(round((right - left + 1.0) / binning_factor))
    height = int(round((bottom - top + 1.0) / binning_factor))
    if width <= 0 or height <= 0:
        raise ConversionError(f"camera {name}: invalid FOV resolution")

    fx = _required_float(intrinsic, "focalLengthU") / (
        SUBPIXELS_PER_PIXEL * binning_factor
    )
    fy = _required_float(intrinsic, "focalLengthV") / (
        SUBPIXELS_PER_PIXEL * binning_factor
    )
    cx = (
        _required_float(intrinsic, "centerPointU") / SUBPIXELS_PER_PIXEL - left
    ) / binning_factor
    cy = (
        _required_float(intrinsic, "centerPointV") / SUBPIXELS_PER_PIXEL - top
    ) / binning_factor

    # This scaling follows the established Qualisys-QCA to OpenCV conversion
    # used by Pose2Sim.  QTM reports these values in its subpixel convention.
    distortion = [
        _required_float(intrinsic, "radialDistortion1")
        / (SUBPIXELS_PER_PIXEL * binning_factor),
        _required_float(intrinsic, "radialDistortion2")
        / (SUBPIXELS_PER_PIXEL * binning_factor),
        _required_float(intrinsic, "tangentalDistortion1")
        / (SUBPIXELS_PER_PIXEL * binning_factor),
        _required_float(intrinsic, "tangentalDistortion2")
        / (SUBPIXELS_PER_PIXEL * binning_factor),
    ]

    xml_rotation = np.array(
        [
            [_required_float(transform, f"r{row}{col}") for col in range(1, 4)]
            for row in range(1, 4)
        ],
        dtype=np.float64,
    )

    # QTM matrix columns describe its camera axes in world coordinates.
    # Convert that camera pose to OpenCV camera axes.  The camera centre is
    # unchanged by this local coordinate-system conversion.
    rotation_world_camera = _nearest_rotation(
        xml_rotation.T @ QTM_TO_OPENCV, name
    )
    quat_xyzw = Rotation.from_matrix(rotation_world_camera).as_quat()
    quaternion_wxyz = [
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]
    translation_metres = [
        _required_float(transform, axis) / MILLIMETRES_PER_METRE
        for axis in ("x", "y", "z")
    ]

    output = {
        "camera_model": "pinhole",
        "distortion_model": "radtan",
        "intrinsics": [float(fx), float(fy), float(cx), float(cy)],
        "distortion_coeffs": [float(v) for v in distortion],
        "resolution": [width, height],
        "translation": translation_metres,
        "rotation_quaternion": quaternion_wxyz,
    }
    provenance = {
        "serial": camera.get("serial", ""),
        "model": camera.get("model", ""),
        "viewrotation_degrees_not_applied": int(camera.get("viewrotation", "0")),
    }
    return name, output, provenance


def convert(
    input_path: Path,
    *,
    name_mode: str = "serial",
    name_prefix: str = "cam",
    binning_factor: float = 1.0,
    camera_model: str = "Miqus",
) -> dict:
    if binning_factor <= 0 or not math.isfinite(binning_factor):
        raise ConversionError("binning factor must be a positive finite number")
    try:
        root = ET.parse(input_path).getroot()
    except ET.ParseError as exc:
        raise ConversionError(f"invalid XML: {exc}") from exc
    if root.tag.lower() != "calibration":
        raise ConversionError(
            f"expected <calibration> root, found <{root.tag}>"
        )

    camera_elements = root.findall("./cameras/camera")
    if not camera_elements:
        raise ConversionError("no <cameras>/<camera> entries found")

    cameras: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for index, camera in enumerate(camera_elements):
        if camera.get("active", "1") != "1" or camera.get("calibrated") == "false":
            continue
        # For now the handball dataset only contains Miqus videos.  Keep the
        # generic conversion code intact so Arqus/other cameras can be enabled
        # later with ``--camera-model all`` instead of maintaining two scripts.
        source_model = (camera.get("model") or "").strip()
        if camera_model.lower() != "all" and camera_model.lower() not in source_model.lower():
            continue
        name, converted, source = _convert_camera(
            camera,
            index,
            name_mode=name_mode,
            name_prefix=name_prefix,
            binning_factor=binning_factor,
        )
        if name in cameras:
            raise ConversionError(f"duplicate output camera name: {name!r}")
        cameras[name] = converted
        provenance[name] = source

    if not cameras:
        raise ConversionError("no active, calibrated cameras found")

    return {
        "metadata": {
            "source_format": "qualisys_qtm_calibration_xml",
            "source_file": str(input_path),
            "qtm_version": root.get("qtm-version", ""),
            "binning_factor": float(binning_factor),
            "camera_model_filter": camera_model,
            "camera_provenance": provenance,
        },
        "cameras": cameras,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Qualisys Calibration.xml/QCA XML")
    parser.add_argument("output", type=Path, help="MAMMA calibration YAML to write")
    parser.add_argument(
        "--name-mode",
        choices=("serial", "index"),
        default="serial",
        help="use QTM serials (default) or generated camera indices",
    )
    parser.add_argument(
        "--name-prefix",
        default="cam",
        help="prefix for --name-mode index (default: cam -> cam01, cam02, ...)",
    )
    parser.add_argument(
        "--binning-factor",
        type=float,
        default=1.0,
        help="divide resolution/intrinsics for binned or downscaled footage",
    )
    parser.add_argument(
        "--camera-model",
        default="Miqus",
        help="case-insensitive model substring to include (default: Miqus); use all for every model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a non-empty output file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        print(
            f"error: output already exists and is non-empty: {args.output} "
            "(pass --force to overwrite)",
            file=sys.stderr,
        )
        return 2
    try:
        result = convert(
            args.input,
            name_mode=args.name_mode,
            name_prefix=args.name_prefix,
            binning_factor=args.binning_factor,
            camera_model=args.camera_model,
        )
    except (OSError, ConversionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False, default_flow_style=False)
    print(f"wrote {len(result['cameras'])} cameras to {args.output}")
    for name, camera in result["cameras"].items():
        print(f"  {name}: {camera['resolution'][0]}x{camera['resolution'][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
