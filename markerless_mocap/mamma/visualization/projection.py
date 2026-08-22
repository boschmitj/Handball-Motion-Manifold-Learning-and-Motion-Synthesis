"""3D-to-2D projection helpers.

Vendored from the upstream ``calib/utils.py``. Pure numpy.
"""
from __future__ import annotations

import numpy as np


def project_points(points_3d: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Project N world-frame 3D points onto a camera image plane.

    Args:
        points_3d: ``(N, 3)`` float array of world-frame points.
        intrinsics: ``(3, 3)`` ``K`` matrix.
        extrinsics: ``(4, 4)`` or ``(3, 4)`` world->cam transform (``T_cam_world``).

    Returns:
        ``(N, 2)`` float array of pixel coordinates.
    """
    if extrinsics.shape == (4, 4):
        ext_3x4 = extrinsics[:3, :]
    elif extrinsics.shape == (3, 4):
        ext_3x4 = extrinsics
    else:
        raise ValueError(f"extrinsics must be (3,4) or (4,4), got {extrinsics.shape}")

    proj_mat = intrinsics @ ext_3x4
    points_h = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)], axis=1
    )
    points_2d_h = (proj_mat @ points_h.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    return points_2d


def keep_inside_image(points_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return only the rows of ``points_2d`` that fall within ``[0, w) x [0, h)``."""
    mask = (
        (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < width)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < height)
    )
    return points_2d[mask]


def world_to_cam(points_3d: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Transform N world-frame points into the camera frame (``T_cam_world @ p``).

    Args:
        points_3d: ``(N, 3)`` float array.
        extrinsics: ``(4, 4)`` ``T_cam_world``.

    Returns:
        ``(N, 3)`` float array in the camera frame.
    """
    points_h = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)], axis=1
    )
    return (extrinsics @ points_h.T)[:3, :].T
