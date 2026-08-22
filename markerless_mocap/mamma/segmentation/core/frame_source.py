"""Backcompat shim: ``FrameSource`` lives in :mod:`capture.frame_source` now.

This file exists so that legacy imports of the form
``from core.frame_source import …`` (segmentation submodule's local
package layout when invoked with ``cwd=segmentation/``) continue to
resolve. The canonical home is :mod:`capture.frame_source` in the
superproject.
"""
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from capture.frame_source import *  # noqa: E402,F401,F403
from capture.frame_source import (  # noqa: E402,F401  (explicit re-exports)
    FrameSource,
    ImageFileSource,
    VideoSource,
    frame_source_from_cam_data,
)
