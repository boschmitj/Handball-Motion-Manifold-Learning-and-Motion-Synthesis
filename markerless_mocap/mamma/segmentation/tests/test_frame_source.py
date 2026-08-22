"""Tests for frame source abstractions."""

import numpy as np
import pytest

from core.frame_source import ImageFileSource, VideoSource, frame_source_from_cam_data
from core.video_reader import VideoFrameReader
from tests.test_video_reader import CV2_AVAILABLE, CV2_SKIP_REASON, synthetic_video


class TestImageFileSource:
    @pytest.fixture
    def image_dir(self, tmp_path):
        from PIL import Image as PILImage

        paths = []
        for i in range(5):
            p = tmp_path / f"{i:06d}.jpg"
            PILImage.new("RGB", (100, 80), color=(i * 50, 0, 0)).save(str(p))
            paths.append(str(p))
        return paths

    def test_len(self, image_dir):
        src = ImageFileSource(image_dir)
        assert len(src) == 5

    def test_read_pil(self, image_dir):
        from PIL import Image as PILImage

        src = ImageFileSource(image_dir)
        img = src.read_pil(0)
        assert isinstance(img, PILImage.Image)
        assert img.size == (100, 80)

    def test_read_rgb(self, image_dir):
        src = ImageFileSource(image_dir)
        rgb = src.read_rgb(0)
        assert rgb.shape == (80, 100, 3)
        assert rgb.dtype == np.uint8

    def test_frame_names(self, image_dir):
        src = ImageFileSource(image_dir)
        assert src.frame_names == [f"{i:06d}.jpg" for i in range(5)]

    def test_image_size(self, image_dir):
        src = ImageFileSource(image_dir)
        assert src.image_size == (100, 80)

    def test_cam_name_from_dir(self, image_dir):
        src = ImageFileSource(image_dir)
        assert isinstance(src.cam_name, str)
        assert len(src.cam_name) > 0

    def test_out_of_range(self, image_dir):
        src = ImageFileSource(image_dir)
        with pytest.raises(IndexError):
            src.read_pil(5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ImageFileSource([])


@pytest.mark.skipif(not CV2_AVAILABLE, reason=CV2_SKIP_REASON)
class TestVideoSource:
    def test_len(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        src = VideoSource(reader)
        assert len(src) == 10

    def test_read_pil(self, synthetic_video):
        from PIL import Image as PILImage

        reader = VideoFrameReader(synthetic_video)
        src = VideoSource(reader)
        img = src.read_pil(0)
        assert isinstance(img, PILImage.Image)
        assert img.size == (64, 64)

    def test_frame_names_synthetic(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video, start=5, end=8)
        src = VideoSource(reader)
        assert src.frame_names == ["000005.jpg", "000006.jpg", "000007.jpg"]

    def test_image_size(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        src = VideoSource(reader)
        assert src.image_size == (64, 64)

    def test_cam_name_from_filename(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        src = VideoSource(reader)
        assert src.cam_name == "test_video"

    def test_cam_name_override(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        src = VideoSource(reader, camera_name="IOI_09")
        assert src.cam_name == "IOI_09"


class TestFrameSourceFactory:
    @pytest.mark.skipif(not CV2_AVAILABLE, reason=CV2_SKIP_REASON)
    def test_from_cam_data_video(self, synthetic_video):
        reader = VideoFrameReader(synthetic_video)
        cam_data = {
            "cam_name": np.array("test_cam"),
            "video_path": np.array(synthetic_video),
            "frame_reader": reader,
        }
        src = frame_source_from_cam_data(cam_data)
        assert isinstance(src, VideoSource)
        assert len(src) == 10

    def test_from_cam_data_images(self, tmp_path):
        from PIL import Image as PILImage

        paths = []
        for i in range(3):
            p = tmp_path / f"{i:06d}.jpg"
            PILImage.new("RGB", (50, 50)).save(str(p))
            paths.append(str(p))
        cam_data = {
            "cam_name": np.array("test_cam"),
            "img_abs_path": np.array(paths),
        }
        src = frame_source_from_cam_data(cam_data)
        assert isinstance(src, ImageFileSource)
        assert len(src) == 3
