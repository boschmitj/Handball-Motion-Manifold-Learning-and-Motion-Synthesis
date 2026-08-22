"""MAMMA visualization step: scene Rerun log + per-camera SMPL-X overlay videos.

Public entry point::

    from visualization import run_visualization

The pipeline reads outputs from ``ma_cap`` (per-camera npz),
``ma_3d`` (predicted SMPL-X vertices), and (optionally) ``ma_2d``
(2D landmarks), and writes ``scene.rrd``, ``overlay/<cam>.mp4``,
and ``preview.mp4`` under ``<out>/<seq>/``.

This module is the polished, library-quality replacement for the hot
path of the upstream ``mv-rerun``. The ``inference/`` runner subprocesses
``visualization/run_ma_vis.py`` (a thin shim around the polished CLI),
so no upstream copy is required at runtime.
"""
__version__ = "0.1.0"

from .cameras import Camera, MultiViewCameras, load_cameras  # noqa: F401
from .motion import (  # noqa: F401
    LandmarkData,
    PersonMotion,
    load_landmarks,
    load_predicted_vertices,
)
from .pipeline import run_visualization  # noqa: F401
from .projection import keep_inside_image, project_points, world_to_cam  # noqa: F401
