"""Backcompat shim: ``VideoFrameReader`` lives in :mod:`capture.video_reader` now.

This file exists so that legacy imports of the form
``from core.video_reader import …`` continue to resolve. The canonical
home is :mod:`capture.video_reader` in the superproject.
"""
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from capture.video_reader import *  # noqa: E402,F401,F403
from capture.video_reader import (  # noqa: E402,F401  (explicit re-exports)
    VideoFrameReader,
    cam_data_from_video,
)
