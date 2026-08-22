"""Quaternion <-> rotation-matrix helper.

Hand-rolled (no scipy / no transforms3d) so the convention is auditable
in one place. Hamilton convention, ``[w, x, y, z]`` order.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

from .calibration import CalibrationError

log = logging.getLogger(__name__)

_NORM_TOL = 1e-3   # warn if the quaternion deviates from unit norm by more than this


def hamilton_quat_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """Convert a Hamilton ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix.

    The input is renormalized defensively. A warning is logged if the
    pre-normalization norm deviates from 1 by more than ``1e-3``; users
    are expected to ship unit quaternions, but a tiny round-off is fine.

    Args:
        q: Length-4 sequence ``[w, x, y, z]``, Hamilton convention.

    Returns:
        ``(3, 3)`` float64 rotation matrix.

    Raises:
        CalibrationError: If ``q`` is not length 4, or its norm is
            non-finite, or its norm is below ``1e-12``.
    """
    if len(q) != 4:
        raise CalibrationError(
            f"quaternion must have length 4 [w,x,y,z], got length {len(q)}"
        )
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-12:
        raise CalibrationError(f"quaternion has near-zero or non-finite norm: {q}")
    if abs(norm - 1.0) > _NORM_TOL:
        log.warning(
            "quaternion not unit norm (|q|=%.6f); renormalizing", norm
        )
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
