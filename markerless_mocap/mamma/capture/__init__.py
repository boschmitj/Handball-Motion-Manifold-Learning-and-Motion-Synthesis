"""Shared input / capture-data package for the MAMMA pipeline.

Two responsibilities:

1. **Calibration loaders** (``.yaml`` / ``.xcp`` / ``.json``) — read camera
   intrinsics/extrinsics into a normalized :class:`Calibration` object.
2. **Frame sources** — uniform random-access over directories of images
   or MP4 videos, so every step can ingest either layout.

Public surface::

    from capture import (
        load_calibration, Calibration, Camera, CalibrationError,
        FrameSource, ImageFileSource, VideoSource,
        VideoFrameReader, cam_data_from_video,
        frame_source_from_cam_data,
        find_video_files, find_image_cam_dirs, cam_data_from_image_dir,
    )
"""
__version__ = "0.2.0"

from .calibration import (  # noqa: F401  (re-export)
    Calibration,
    CalibrationError,
    Camera,
    VALID_DISTORTION_MODELS,
    load_calibration,
)
from .frame_source import (  # noqa: F401
    FrameSource,
    ImageFileSource,
    VideoSource,
    frame_source_from_cam_data,
)
from .video_reader import (  # noqa: F401
    VideoFrameReader,
    cam_data_from_video,
)
from .discovery import (  # noqa: F401
    IMAGE_EXTENSIONS,
    cam_data_from_image_dir,
    find_image_cam_dirs,
    find_video_files,
    normalize_cam_name,
    normalize_cam_names,
)
from .undistort import (  # noqa: F401
    undistort_rgb,
)
