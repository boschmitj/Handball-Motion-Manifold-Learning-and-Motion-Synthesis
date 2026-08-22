"""Tests for video_reader utilities."""

import os
import shutil
import tempfile

import numpy as np
import pytest

from core.video_reader import VideoFrameReader, cam_data_from_video


def _cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


CV2_AVAILABLE = _cv2_available()
CV2_SKIP_REASON = "OpenCV runtime dependencies are unavailable in this environment"


@pytest.fixture(scope="module")
def synthetic_video():
    """Create a short synthetic MP4 video (10 frames, 64x64, solid colors)."""
    cv2 = pytest.importorskip("cv2", reason=CV2_SKIP_REASON)

    tmpdir = tempfile.mkdtemp(prefix="mamma_test_")
    path = os.path.join(tmpdir, "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (64, 64))
    for i in range(10):
        frame = np.full((64, 64, 3), fill_value=(i * 25, 0, 0), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not CV2_AVAILABLE, reason=CV2_SKIP_REASON)
class TestVideoFrameReader:
    def test_full_video(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        assert len(reader) == 10
        assert reader.width == 64
        assert reader.height == 64

    def test_frame_range(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video, start=2, end=7)
        assert len(reader) == 5

    def test_read_bgr_shape(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        frame = reader.read_bgr(0)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_read_rgb_channel_swap(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        bgr = reader.read_bgr(0)
        rgb = reader.read_rgb(0)
        np.testing.assert_array_equal(bgr[:, :, 0], rgb[:, :, 2])

    def test_read_pil(self, synthetic_video):
        from PIL import Image

        reader = VideoFrameReader(synthetic_video)
        img = reader.read_pil(0)
        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)

    def test_out_of_range_raises(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        with pytest.raises(IndexError):
            reader.read_bgr(10)
        with pytest.raises(IndexError):
            reader.read_bgr(-1)

    def test_frame_range_clamped(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video, start=-5, end=999)
        assert reader.start == 0
        assert reader.end == 10
        assert len(reader) == 10

    def test_invalid_path(self):
        with pytest.raises(RuntimeError, match="Cannot open video"):
            VideoFrameReader("/nonexistent/path.mp4")


@pytest.mark.skipif(not CV2_AVAILABLE, reason=CV2_SKIP_REASON)
class TestCamDataFromVideo:
    def test_basic(self, synthetic_video):
        cd = cam_data_from_video(synthetic_video)
        assert str(cd["cam_name"]) == "test_video"
        assert "frame_reader" in cd
        assert len(cd["frame_reader"]) == 10

    def test_with_frame_range(self, synthetic_video):
        cd = cam_data_from_video(synthetic_video, start=2, end=7)
        assert len(cd["frame_reader"]) == 5

    def test_video_path_is_absolute(self, synthetic_video):
        cd = cam_data_from_video(synthetic_video)
        assert os.path.isabs(str(cd["video_path"]))
