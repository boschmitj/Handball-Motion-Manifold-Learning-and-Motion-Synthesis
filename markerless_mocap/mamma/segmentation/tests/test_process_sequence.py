"""Tests for process_sequence helpers."""

import os

import pytest

from process_sequence import (
    cam_data_from_image_dir,
    find_image_cam_dirs,
    find_video_files,
    load_assignment_config,
    normalize_cam_name,
    normalize_cam_names,
    reorder_ioi_paths,
)


class TestNormalizeCamNames:
    def test_none_input(self):
        assert normalize_cam_names(None) is None

    def test_empty_list(self):
        assert normalize_cam_names([]) is None

    def test_single_name(self):
        assert normalize_cam_names(["IOI_09"]) == ["IOI_09"]

    def test_comma_separated_string(self):
        assert normalize_cam_names(["IOI_09,IOI_10"]) == ["IOI_09", "IOI_10"]

    def test_multiple_names(self):
        assert normalize_cam_names(["IOI_09", "IOI_10"]) == ["IOI_09", "IOI_10"]

    def test_whitespace_stripping(self):
        assert normalize_cam_names([" IOI_09 ", " IOI_10 "]) == ["IOI_09", "IOI_10"]

    def test_empty_strings_filtered(self):
        assert normalize_cam_names(["", " ", "IOI_09"]) == ["IOI_09"]

    def test_all_empty_returns_none(self):
        assert normalize_cam_names(["", " "]) is None


class TestNormalizeCamName:
    def test_strip_npz(self):
        assert normalize_cam_name("IOI_09.npz") == "IOI_09"

    def test_strip_mp4(self):
        assert normalize_cam_name("IOI_09.mp4") == "IOI_09"

    def test_no_extension(self):
        assert normalize_cam_name("IOI_09") == "IOI_09"

    def test_whitespace(self):
        assert normalize_cam_name("  IOI_09.NPZ  ") == "IOI_09"

    def test_case_insensitive_extension(self):
        assert normalize_cam_name("IOI_09.MP4") == "IOI_09"


class TestFindVideoFiles:
    @pytest.fixture
    def video_dir(self, tmp_path):
        for name in ["IOI_09.mp4", "IOI_10.mp4", "IOI_11.mp4"]:
            (tmp_path / name).write_bytes(b"\x00")
        return str(tmp_path)

    def test_discovers_all(self, video_dir):
        found = find_video_files(video_dir)
        assert len(found) == 3

    def test_filter_by_cam_names(self, video_dir):
        found = find_video_files(video_dir, cam_names=["IOI_09", "IOI_11"])
        basenames = [os.path.basename(f) for f in found]
        assert "IOI_09.mp4" in basenames
        assert "IOI_11.mp4" in basenames
        assert "IOI_10.mp4" not in basenames

    def test_empty_dir(self, tmp_path):
        found = find_video_files(str(tmp_path))
        assert found == []

    def test_recursive_discovery(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "cam01.mp4").write_bytes(b"\x00")
        found = find_video_files(str(tmp_path))
        assert len(found) == 1


class TestLoadAssignmentConfig:
    def test_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("matching:\n  start_frame: 5\n")
        result = load_assignment_config(str(cfg_file))
        assert result == {"matching": {"start_frame": 5}}

    def test_missing_file(self):
        result = load_assignment_config("/nonexistent/path.yaml")
        assert result == {}

    def test_none_path(self):
        result = load_assignment_config(None)
        assert result == {}

    def test_empty_yaml(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        result = load_assignment_config(str(cfg_file))
        assert result == {}


class TestReorderIoiPaths:
    def test_cam_init_first(self):
        paths = ["/data/gt/IOI_01.npz", "/data/gt/IOI_09.npz", "/data/gt/IOI_15.npz"]
        result = reorder_ioi_paths(paths, "IOI_09")
        assert result[0] == "/data/gt/IOI_09.npz"
        assert len(result) == 3

    def test_cam_init_not_found(self):
        paths = ["/data/gt/IOI_01.npz", "/data/gt/IOI_09.npz"]
        result = reorder_ioi_paths(paths, "IOI_99")
        assert result == paths

    def test_no_cam_init(self):
        paths = ["/data/gt/IOI_01.npz", "/data/gt/IOI_09.npz"]
        result = reorder_ioi_paths(paths, None)
        assert result == paths

    def test_case_insensitive(self):
        paths = ["/data/gt/IOI_01.npz", "/data/gt/IOI_09.npz"]
        result = reorder_ioi_paths(paths, "ioi_09")
        assert result[0] == "/data/gt/IOI_09.npz"


class TestImageDirInput:
    @pytest.fixture
    def image_root(self, tmp_path):
        from PIL import Image as PILImage

        for cam in ["cam_01", "cam_02", "cam_03"]:
            cam_dir = tmp_path / cam
            cam_dir.mkdir()
            for i in range(5):
                PILImage.new("RGB", (100, 80), color=(i * 50, 0, 0)).save(
                    str(cam_dir / f"{i:06d}.jpg")
                )
        (tmp_path / "logs").mkdir()
        return str(tmp_path)

    def test_find_all_cam_dirs(self, image_root):
        dirs = find_image_cam_dirs(image_root)
        assert len(dirs) == 3
        names = [os.path.basename(d) for d in dirs]
        assert "cam_01" in names
        assert "logs" not in names

    def test_find_filtered_cam_dirs(self, image_root):
        dirs = find_image_cam_dirs(image_root, cam_names=["cam_01", "cam_03"])
        assert len(dirs) == 2

    def test_find_empty_dir(self, tmp_path):
        dirs = find_image_cam_dirs(str(tmp_path))
        assert dirs == []

    def test_cam_data_from_image_dir(self, image_root):
        cam_dir = os.path.join(image_root, "cam_01")
        cd = cam_data_from_image_dir(cam_dir)
        assert str(cd["cam_name"]) == "cam_01"
        assert len(cd["img_abs_path"]) == 5
        assert all(os.path.isabs(p) for p in cd["img_abs_path"])

    def test_cam_data_frame_range(self, image_root):
        cam_dir = os.path.join(image_root, "cam_01")
        cd = cam_data_from_image_dir(cam_dir, start=1, end=4)
        assert len(cd["img_abs_path"]) == 3

    def test_cam_data_empty_dir(self, tmp_path):
        empty = tmp_path / "empty_cam"
        empty.mkdir()
        with pytest.raises(ValueError, match="No image files"):
            cam_data_from_image_dir(str(empty))
