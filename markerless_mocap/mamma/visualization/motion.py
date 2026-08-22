"""Predicted-motion + 2D-landmark loaders.

Reads outputs from the prior pipeline steps:

* ``ma_3d`` writes one ``verts_joints_body_id-<n>.npz`` per person under
  ``<seq>/`` with a ``pred_vertices`` key, shape ``(F, V, 3)`` in metres.
* ``ma_2d`` writes one ``<cam_name>.npz`` per camera under ``<seq>/`` with
  ``landmarks`` (``(F, P, K, 2)`` or ``(F, P, K, 3)`` -- the third channel
  is per-keypoint log-variance) and ``visibilities`` (``(F, P, K)`` or
  ``(F, P, K, 1)``).

Vendored from the upstream ``run_ma_vis.py`` (the per-person
glob in ``execute_rerun``) and ``engine/systems_mv.py::_load_markers``.
"""
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PersonMotion:
    """Per-person predicted SMPL-X vertex sequence."""

    body_id: int
    vertices: np.ndarray   # (F, V, 3) float


def load_predicted_vertices(motion_dir) -> List[PersonMotion]:
    """Load every ``verts_joints_body_id-*.npz`` from ``motion_dir``.

    Returns persons in body-id order (``body_id-0``, ``body_id-1``, ...).
    Raises ``FileNotFoundError`` if no files match.
    """
    motion = Path(motion_dir)
    if not motion.is_dir():
        raise FileNotFoundError(f"motion dir not found: {motion}")

    paths = sorted(glob(str(motion / "verts_joints_body_id*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"no verts_joints_body_id*.npz files in: {motion}"
        )

    out: List[PersonMotion] = []
    for path in paths:
        body_id = _parse_body_id(path)
        data = np.load(path, allow_pickle=True)
        try:
            if "pred_vertices" not in data.files:
                raise KeyError(f"{path}: missing 'pred_vertices' key")
            verts = np.asarray(data["pred_vertices"])
        finally:
            data.close()
        if verts.ndim != 3 or verts.shape[-1] != 3:
            raise ValueError(
                f"{path}: pred_vertices must be (F, V, 3), got {verts.shape}"
            )
        out.append(PersonMotion(body_id=body_id, vertices=verts))
    out.sort(key=lambda p: p.body_id)
    return out


def _parse_body_id(path: str) -> int:
    """``verts_joints_body_id-7.npz`` -> ``7``. Falls back to the file index."""
    name = Path(path).stem
    if "body_id-" in name:
        try:
            return int(name.split("body_id-")[-1])
        except ValueError:
            pass
    return 0


@dataclass(frozen=True)
class LandmarkData:
    """Per-camera 2D landmark predictions.

    ``landmarks`` stays in *original* camera-pixel coordinates -- the
    upstream code in ``ma_2d`` writes them at the source image resolution.
    Downstream consumers rescale them as needed.

    Attributes:
        landmarks: ``(F, P, K, 2)`` xy pixel coordinates.
        visibility: ``(F, P, K)`` per-keypoint visibility in ``[0, 1]``.
        log_variance: ``(F, P, K)`` per-keypoint log-variance from the
            uncertainty head, or ``None`` if the upstream output didn't
            include it.
    """

    landmarks: np.ndarray
    visibility: np.ndarray
    log_variance: Optional[np.ndarray] = None


def load_landmarks(
    landmarks_dir, *, cam_names: Optional[Sequence[str]] = None
) -> Dict[str, LandmarkData]:
    """Load per-camera 2D landmark npz files from a ``ma_2d`` output dir.

    Args:
        landmarks_dir: Path to ``<seq>/`` containing one ``<cam>.npz`` per
            camera.
        cam_names: Optional whitelist of camera names; missing names raise
            ``FileNotFoundError``. ``None`` returns every camera found.

    Returns:
        ``{cam_name: LandmarkData}``. Cameras whose npz files exist but have
        an unexpected schema raise ``KeyError`` -- never silently skipped.
        Filenames that end with ``_diff.npz`` are skipped (those are
        diff/auxiliary outputs from the upstream model).
    """
    ldir = Path(landmarks_dir)
    if not ldir.is_dir():
        raise FileNotFoundError(f"landmarks dir not found: {ldir}")

    paths = sorted(
        p for p in glob(str(ldir / "*.npz"))
        if not p.rsplit("_", 1)[-1].lower() == "diff.npz"
    )
    by_cam: Dict[str, LandmarkData] = {}
    for path in paths:
        cam = Path(path).stem
        by_cam[cam] = _load_one_landmark(path)

    if cam_names is not None:
        wanted = list(cam_names)
        missing = [n for n in wanted if n not in by_cam]
        if missing:
            raise FileNotFoundError(
                f"landmark predictions missing for {missing} in {ldir}"
            )
        by_cam = {n: by_cam[n] for n in wanted}

    return by_cam


def _load_one_landmark(path: str) -> LandmarkData:
    data = np.load(path, allow_pickle=True)
    try:
        if "landmarks" not in data.files or "visibilities" not in data.files:
            raise KeyError(f"{path}: missing 'landmarks' or 'visibilities' keys")
        ldmks = np.asarray(data["landmarks"])
        vis = np.asarray(data["visibilities"])
    finally:
        data.close()

    if ldmks.ndim < 3 or ldmks.shape[-1] not in (2, 3):
        raise ValueError(
            f"{path}: landmarks must end in 2 or 3 channels (xy or xy+logvar), "
            f"got shape {ldmks.shape}"
        )
    if ldmks.shape[-1] == 3:
        log_var = ldmks[..., 2]
        ldmks_xy = ldmks[..., :2]
    else:
        log_var = None
        ldmks_xy = ldmks

    if vis.ndim == 4 and vis.shape[-1] == 1:
        vis = vis.squeeze(-1)

    return LandmarkData(
        landmarks=ldmks_xy.astype(np.float64),
        visibility=vis.astype(np.float64),
        log_variance=None if log_var is None else log_var.astype(np.float64),
    )


def uncertainty_from_log_variance(log_var: np.ndarray) -> np.ndarray:
    """Map per-keypoint log-variance into a ``[0, 1]`` uncertainty score.

    Mirrors the upstream conversion (``mv-rerun/engine/systems_mv.py``
    ``_load_markers`` lines 105--107): ``sqrt(exp(log_var)) / 2 * 512``,
    then clip to ``[0, 50]`` and divide by 50.
    """
    sigma = np.sqrt(np.exp(log_var)) / 2.0 * 512.0
    return np.clip(sigma, 0.0, 50.0) / 50.0
