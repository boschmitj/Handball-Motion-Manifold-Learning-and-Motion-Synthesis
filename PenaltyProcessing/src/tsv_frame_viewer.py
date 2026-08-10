"""Interactive frame-by-frame viewer for Mo_Wurfvariation4_ul0251 TSV exports.

The viewer overlays the body skeleton TSV with either the fitted ball centre
from the 6DOF marker spheres or the ground-truth ball centre from the 6D TSV.
It is designed to stay responsive by indexing frame offsets first and then
loading only the selected frame on demand.

Usage
-----
python3 src/tsv_frame_viewer.py
python3 src/tsv_frame_viewer.py --no-bones --ball-mode gt
python3 src/tsv_frame_viewer.py --frame 500
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider


BALL_DIAMETER_MM = 190.0
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0
BODY_POINT_DIAMETER_MM = 10.0


def _parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_valid_xyz(values: Sequence[Optional[float]]) -> bool:
    if len(values) != 3 or any(value is None for value in values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return np.isfinite(array).all() and not np.allclose(array, 0.0)


def fit_sphere(markers: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    """Least-squares fit of a sphere through marker points."""
    n = markers.shape[0]
    if n < 4:
        return None

    design = np.column_stack((2.0 * markers, np.ones(n)))
    rhs = np.sum(markers * markers, axis=1)
    try:
        coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    center = coeffs[:3]
    radius_sq = coeffs[3] + float(center @ center)
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        return None

    radius = float(np.sqrt(radius_sq))
    distances = np.linalg.norm(markers - center, axis=1)
    residual_rms = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return center, radius, residual_rms


@dataclass(frozen=True)
class FrameIndex:
    path: Path
    frame_to_offset: Dict[int, int]
    frames: List[int]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    segment_names: Optional[List[str]] = None


def _read_lines(path: Path) -> Iterable[Tuple[int, str, int]]:
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            yield offset, raw.decode("utf-8", errors="replace").rstrip("\r\n"), handle.tell()


def index_skeleton_file(path: Path) -> FrameIndex:
    frame_to_offset: Dict[int, int] = {}
    frames: List[int] = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    segment_names: Optional[List[str]] = None

    for offset, line, _ in _read_lines(path):
        if not line:
            continue
        row = line.split("\t")
        if row[0].strip() == "Frame":
            names: List[str] = []
            for idx in range(2, len(row), 8):
                if row[idx].strip():
                    names.append(row[idx].strip())
            segment_names = names
            continue

        if len(row) < 10:
            continue
        frame = _parse_float(row[0])
        if frame is None:
            continue
        frame_number = int(frame)
        frame_to_offset[frame_number] = offset
        frames.append(frame_number)

        for idx in range(2, len(row), 8):
            x = _parse_float(row[idx + 1]) if idx + 1 < len(row) else None
            y = _parse_float(row[idx + 2]) if idx + 2 < len(row) else None
            z = _parse_float(row[idx + 3]) if idx + 3 < len(row) else None
            if _is_valid_xyz((x, y, z)):
                values = np.array([x, y, z], dtype=np.float64)
                bounds_min = np.minimum(bounds_min, values)
                bounds_max = np.maximum(bounds_max, values)

    return FrameIndex(path, frame_to_offset, sorted(set(frames)), bounds_min, bounds_max, segment_names)


def index_ball_3d_file(path: Path) -> FrameIndex:
    frame_to_offset: Dict[int, int] = {}
    frames: List[int] = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for offset, line, _ in _read_lines(path):
        row = line.split("\t")
        if len(row) < 20 or row[0].strip() == "Frame":
            continue
        frame = _parse_float(row[0])
        if frame is None:
            continue
        frame_number = int(frame)
        frame_to_offset[frame_number] = offset
        frames.append(frame_number)

        markers = parse_ball_markers_row(row)
        if markers.size:
            bounds_min = np.minimum(bounds_min, np.min(markers, axis=0))
            bounds_max = np.maximum(bounds_max, np.max(markers, axis=0))

    return FrameIndex(path, frame_to_offset, sorted(set(frames)), bounds_min, bounds_max)


def index_ball_6d_file(path: Path) -> FrameIndex:
    frame_to_offset: Dict[int, int] = {}
    frames: List[int] = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for offset, line, _ in _read_lines(path):
        row = line.split("\t")
        if len(row) < 5 or row[0].strip() == "Frame":
            continue
        frame = _parse_float(row[0])
        if frame is None:
            continue
        frame_number = int(frame)
        frame_to_offset[frame_number] = offset
        frames.append(frame_number)

        center = parse_ball_6d_row(row)
        if center is not None:
            bounds_min = np.minimum(bounds_min, center)
            bounds_max = np.maximum(bounds_max, center)

    return FrameIndex(path, frame_to_offset, sorted(set(frames)), bounds_min, bounds_max)


@lru_cache(maxsize=256)
def parse_skeleton_row_cached(
    path_str: str,
    offset: int,
    segment_names: Tuple[str, ...],
) -> Tuple[int, float, Dict[str, np.ndarray]]:
    path = Path(path_str)
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline().decode("utf-8", errors="replace").rstrip("\r\n")

    row = line.split("\t")
    frame = int(float(row[0]))
    time_s = float(row[1])
    positions: Dict[str, np.ndarray] = {}
    for segment_idx, name in enumerate(segment_names):
        base_idx = 2 + segment_idx * 8
        if base_idx + 3 >= len(row):
            continue
        x = _parse_float(row[base_idx + 1])
        y = _parse_float(row[base_idx + 2])
        z = _parse_float(row[base_idx + 3])
        if name and _is_valid_xyz((x, y, z)):
            positions[name] = np.array([x, y, z], dtype=np.float64)
    return frame, time_s, positions


@lru_cache(maxsize=256)
def parse_ball_markers_cached(path_str: str, offset: int) -> Tuple[int, float, np.ndarray]:
    path = Path(path_str)
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline().decode("utf-8", errors="replace").rstrip("\r\n")

    row = line.split("\t")
    frame = int(float(row[0]))
    time_s = float(row[1])
    markers = parse_ball_markers_row(row)
    return frame, time_s, markers


@lru_cache(maxsize=256)
def parse_ball_6d_cached(path_str: str, offset: int) -> Tuple[int, float, Optional[np.ndarray]]:
    path = Path(path_str)
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline().decode("utf-8", errors="replace").rstrip("\r\n")

    row = line.split("\t")
    frame = int(float(row[0]))
    time_s = float(row[1])
    return frame, time_s, parse_ball_6d_row(row)


def parse_ball_markers_row(row: Sequence[str]) -> np.ndarray:
    values: List[List[float]] = []
    for idx in range(2, min(len(row), 20), 3):
        coords = [_parse_float(row[idx]), _parse_float(row[idx + 1]) if idx + 1 < len(row) else None, _parse_float(row[idx + 2]) if idx + 2 < len(row) else None]
        if _is_valid_xyz(coords):
            values.append([float(coords[0]), float(coords[1]), float(coords[2])])
    if not values:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def parse_ball_6d_row(row: Sequence[str]) -> Optional[np.ndarray]:
    if len(row) < 5:
        return None
    center = np.array([
        _parse_float(row[2]),
        _parse_float(row[3]),
        _parse_float(row[4]),
    ], dtype=np.float64)
    if not np.isfinite(center).all() or np.allclose(center, 0.0):
        return None
    return center


def choose_nearest_frame(frames: Sequence[int], requested: Optional[int]) -> int:
    if not frames:
        raise ValueError("No frames available")
    if requested is None:
        return 0
    if requested in frames:
        return frames.index(requested)
    diffs = [abs(frame - requested) for frame in frames]
    return int(np.argmin(diffs))


def segment_connections() -> List[Tuple[str, str]]:
    chains = [
        ("Hips", "Spine"),
        ("Spine", "Spine1"),
        ("Spine1", "Spine2"),
        ("Spine2", "Neck"),
        ("Neck", "Head"),
        ("Neck", "LeftShoulder"),
        ("LeftShoulder", "LeftArm"),
        ("LeftArm", "LeftForeArm"),
        ("LeftForeArm", "LeftForeArmRoll"),
        ("LeftForeArmRoll", "LeftHand"),
        ("Neck", "RightShoulder"),
        ("RightShoulder", "RightArm"),
        ("RightArm", "RightForeArm"),
        ("RightForeArm", "RightForeArmRoll"),
        ("RightForeArmRoll", "RightHand"),
        ("Hips", "LeftUpLeg"),
        ("LeftUpLeg", "LeftLeg"),
        ("LeftLeg", "LeftFoot"),
        ("LeftFoot", "LeftToeBase"),
        ("Hips", "RightUpLeg"),
        ("RightUpLeg", "RightLeg"),
        ("RightLeg", "RightFoot"),
        ("RightFoot", "RightToeBase"),
        ("LeftHand", "LeftInHandThumb"),
        ("LeftInHandThumb", "LeftHandThumb1"),
        ("LeftHandThumb1", "LeftHandThumb2"),
        ("LeftHandThumb2", "LeftHandThumb3"),
        ("LeftHand", "LeftInHandIndex"),
        ("LeftInHandIndex", "LeftHandIndex1"),
        ("LeftHandIndex1", "LeftHandIndex2"),
        ("LeftHandIndex2", "LeftHandIndex3"),
        ("LeftHand", "LeftInHandMiddle"),
        ("LeftInHandMiddle", "LeftHandMiddle1"),
        ("LeftHandMiddle1", "LeftHandMiddle2"),
        ("LeftHandMiddle2", "LeftHandMiddle3"),
        ("LeftHand", "LeftInHandRing"),
        ("LeftInHandRing", "LeftHandRing1"),
        ("LeftHandRing1", "LeftHandRing2"),
        ("LeftHandRing2", "LeftHandRing3"),
        ("LeftHand", "LeftInHandPinky"),
        ("LeftInHandPinky", "LeftHandPinky1"),
        ("LeftHandPinky1", "LeftHandPinky2"),
        ("LeftHandPinky2", "LeftHandPinky3"),
        ("RightHand", "RightInHandThumb"),
        ("RightInHandThumb", "RightHandThumb1"),
        ("RightHandThumb1", "RightHandThumb2"),
        ("RightHandThumb2", "RightHandThumb3"),
        ("RightHand", "RightInHandIndex"),
        ("RightInHandIndex", "RightHandIndex1"),
        ("RightHandIndex1", "RightHandIndex2"),
        ("RightHandIndex2", "RightHandIndex3"),
        ("RightHand", "RightInHandMiddle"),
        ("RightInHandMiddle", "RightHandMiddle1"),
        ("RightHandMiddle1", "RightHandMiddle2"),
        ("RightHandMiddle2", "RightHandMiddle3"),
        ("RightHand", "RightInHandRing"),
        ("RightInHandRing", "RightHandRing1"),
        ("RightHandRing1", "RightHandRing2"),
        ("RightHandRing2", "RightHandRing3"),
        ("RightHand", "RightInHandPinky"),
        ("RightInHandPinky", "RightHandPinky1"),
        ("RightHandPinky1", "RightHandPinky2"),
        ("RightHandPinky2", "RightHandPinky3"),
    ]
    return chains


def set_axes_equal(ax: plt.Axes, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
    center = (bounds_min + bounds_max) / 2.0
    span = np.maximum(bounds_max - bounds_min, 1.0)
    radius = float(np.max(span) / 2.0)
    margin = max(radius * 0.15, 50.0)
    radius += margin
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_sphere(ax: plt.Axes, center: np.ndarray, radius_mm: float, color: str, alpha: float, resolution: int = 10) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = center[0] + radius_mm * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius_mm * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius_mm * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


class FrameViewer:
    def __init__(
        self,
        skeleton_index: FrameIndex,
        ball_3d_index: FrameIndex,
        ball_6d_index: FrameIndex,
        use_bones: bool,
        ball_mode: str,
        start_frame: Optional[int],
    ) -> None:
        self.skeleton_index = skeleton_index
        self.ball_3d_index = ball_3d_index
        self.ball_6d_index = ball_6d_index
        self.use_bones = use_bones
        self.ball_mode = ball_mode
        self.shared_frames = sorted(
            set(skeleton_index.frames) & set(ball_3d_index.frames) & set(ball_6d_index.frames)
        )
        if not self.shared_frames:
            raise ValueError("No common frames across skeleton, ball 3D, and ball 6D files")

        self.frame_idx = choose_nearest_frame(self.shared_frames, start_frame)

        self.fig = plt.figure(figsize=(13, 9))
        self.ax = self.fig.add_axes([0.05, 0.14, 0.7, 0.8], projection="3d")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")

        self.prev_button_ax = self.fig.add_axes([0.79, 0.10, 0.08, 0.05])
        self.next_button_ax = self.fig.add_axes([0.88, 0.10, 0.08, 0.05])
        self.slider_ax = self.fig.add_axes([0.10, 0.05, 0.60, 0.03])
        self.check_ax = self.fig.add_axes([0.80, 0.25, 0.16, 0.14])
        self.info_ax = self.fig.add_axes([0.77, 0.42, 0.20, 0.55])
        self.info_ax.axis("off")

        self.prev_button = Button(self.prev_button_ax, "Prev")
        self.next_button = Button(self.next_button_ax, "Next")
        self.slider = Slider(self.slider_ax, "Frame", 0, len(self.shared_frames) - 1, valinit=self.frame_idx, valstep=1)
        self.checks = CheckButtons(self.check_ax, ["Bones", "GT ball"], [self.use_bones, self.ball_mode == "gt"])

        self.prev_button.on_clicked(lambda _event: self.step(-1))
        self.next_button.on_clicked(lambda _event: self.step(1))
        self.slider.on_changed(self.on_slider)
        self.checks.on_clicked(self.on_toggle)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.color_map = {
            "body": "#2e86de",
            "bones": "#1b1b1b",
            "ball": "#ff6b35",
            "fallback": "#8e44ad",
        }

        self.base_bounds_min = np.array(self._bound_min(), dtype=np.float64)
        self.base_bounds_max = np.array(self._bound_max(), dtype=np.float64)
        self.update()

    def _bound_min(self) -> np.ndarray:
        candidates = [
            self.skeleton_index.bounds_min,
            self.ball_3d_index.bounds_min,
            self.ball_6d_index.bounds_min,
        ]
        finite_candidates = [candidate for candidate in candidates if np.isfinite(candidate).all()]
        if not finite_candidates:
            return np.array([-1000.0, -1000.0, -1000.0], dtype=np.float64)
        values = np.vstack(finite_candidates)
        return np.min(values, axis=0)

    def _bound_max(self) -> np.ndarray:
        candidates = [
            self.skeleton_index.bounds_max,
            self.ball_3d_index.bounds_max,
            self.ball_6d_index.bounds_max,
        ]
        finite_candidates = [candidate for candidate in candidates if np.isfinite(candidate).all()]
        if not finite_candidates:
            return np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
        values = np.vstack(finite_candidates)
        return np.max(values, axis=0)

    def on_slider(self, value: float) -> None:
        idx = int(round(value))
        if idx != self.frame_idx:
            self.frame_idx = max(0, min(idx, len(self.shared_frames) - 1))
            self.update()

    def on_toggle(self, label: str) -> None:
        active = {text.get_text(): self.checks.get_status()[idx] for idx, text in enumerate(self.checks.labels)}
        self.use_bones = active.get("Bones", self.use_bones)
        self.ball_mode = "gt" if active.get("GT ball", False) else "fit"
        self.update()

    def on_key(self, event) -> None:
        if event.key in {"right", "down", "pagedown", " ", "n"}:
            self.step(1)
        elif event.key in {"left", "up", "pageup", "p", "backspace"}:
            self.step(-1)
        elif event.key == "home":
            self.goto_index(0)
        elif event.key == "end":
            self.goto_index(len(self.shared_frames) - 1)

    def step(self, delta: int) -> None:
        self.goto_index(self.frame_idx + delta)

    def goto_index(self, idx: int) -> None:
        idx = max(0, min(idx, len(self.shared_frames) - 1))
        if idx == self.frame_idx:
            return
        self.frame_idx = idx
        self.slider.set_val(idx)
        self.update()

    def _frame_offset(self, index: FrameIndex, frame: int) -> int:
        try:
            return index.frame_to_offset[frame]
        except KeyError as exc:
            raise KeyError(f"Frame {frame} not found in {index.path.name}") from exc

    def _body_positions(self, frame: int) -> Tuple[float, Dict[str, np.ndarray]]:
        offset = self._frame_offset(self.skeleton_index, frame)
        segment_names = tuple(self.skeleton_index.segment_names or ())
        _, time_s, positions = parse_skeleton_row_cached(str(self.skeleton_index.path), offset, segment_names)
        return time_s, positions

    def _ball_from_fit(self, frame: int) -> Tuple[Optional[np.ndarray], str]:
        offset = self._frame_offset(self.ball_3d_index, frame)
        _, _, markers = parse_ball_markers_cached(str(self.ball_3d_index.path), offset)
        fit = fit_sphere(markers)
        if fit is None:
            return None, "ball markers insufficient for a fit"
        center, _, _ = fit
        if not np.isfinite(center).all() or np.allclose(center, 0.0):
            return None, "fitted ball centre is invalid"
        return center, f"fit from {markers.shape[0]} ball markers"

    def _ball_from_gt(self, frame: int) -> Tuple[Optional[np.ndarray], str]:
        offset = self._frame_offset(self.ball_6d_index, frame)
        _, _, center = parse_ball_6d_cached(str(self.ball_6d_index.path), offset)
        if center is None:
            return None, "ground-truth ball centre missing"
        return center, "ground-truth centre from 6D TSV"

    def _fallback_ball(self, positions: Dict[str, np.ndarray]) -> np.ndarray:
        if "Hips" in positions and np.isfinite(positions["Hips"]).all():
            return positions["Hips"]
        for name in ("Spine", "Spine1", "Spine2"):
            if name in positions and np.isfinite(positions[name]).all():
                return positions[name]
        if positions:
            return np.mean(np.stack(list(positions.values()), axis=0), axis=0)
        return np.zeros(3, dtype=np.float64)

    def _active_ball_center(self, frame: int, positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
        if self.ball_mode == "gt":
            center, reason = self._ball_from_gt(frame)
        else:
            center, reason = self._ball_from_fit(frame)

        if center is None:
            return self._fallback_ball(positions), f"fallback to hips because {reason}"
        return center, reason

    def _draw_connections(self, positions: Dict[str, np.ndarray]) -> None:
        for start_name, end_name in segment_connections():
            if start_name not in positions or end_name not in positions:
                continue
            start = positions[start_name]
            end = positions[end_name]
            self.ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=self.color_map["bones"],
                linewidth=1.4,
                alpha=0.8,
            )

    def _draw_body_points(self, positions: Dict[str, np.ndarray]) -> None:
        radius = BODY_POINT_DIAMETER_MM / 2.0
        for name, position in positions.items():
            draw_sphere(self.ax, position, radius, self.color_map["body"], alpha=0.55, resolution=8)

    def _draw_ball(self, center: np.ndarray) -> None:
        draw_sphere(self.ax, center, BALL_RADIUS_MM, self.color_map["ball"], alpha=0.75, resolution=12)

    def update(self) -> None:
        frame_number = self.shared_frames[self.frame_idx]
        time_s, positions = self._body_positions(frame_number)
        ball_center, ball_reason = self._active_ball_center(frame_number, positions)

        elev = getattr(self.ax, "elev", 20.0)
        azim = getattr(self.ax, "azim", -60.0)
        self.ax.cla()
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")

        if self.use_bones:
            self._draw_connections(positions)
        else:
            self._draw_body_points(positions)

        self._draw_ball(ball_center)

        set_axes_equal(self.ax, self.base_bounds_min, self.base_bounds_max)
        self.ax.set_title(
            f"Frame {frame_number} | time {time_s:.5f} s | bones={'on' if self.use_bones else 'off'} | ball={self.ball_mode}"
        )

        info_lines = [
            f"Frame: {frame_number}",
            f"Time: {time_s:.5f} s", 
            f"Segments: {len(positions)}",
            f"Ball mode: {self.ball_mode}",
            f"Ball center: {ball_reason}",
            f"Bones: {'on' if self.use_bones else 'off'}",
            "",
            "Keys:",
            "Left/Right: previous / next",
            "Home/End: first / last",
            "",
            "Mouse drag in 3D space",
        ]
        self.info_ax.cla()
        self.info_ax.axis("off")
        self.info_ax.text(0.0, 1.0, "\n".join(info_lines), va="top", ha="left", fontsize=10)
        self.fig.canvas.draw_idle()


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent.parent
    default_skeleton = default_root / "mocap_files" / "skeleton" / "Mo_Wurfvariation4_ul0251_reprocessing_labeling_done_s_Josh.tsv"
    default_ball_3d = default_root / "mocap_files" / "6DOF" / "Mo_Wurfvariation4_ul0251_reprocessing_6DOF_3D.tsv"
    default_ball_6d = default_root / "mocap_files" / "6DOF" / "Mo_Wurfvariation4_ul0251_reprocessing_labeling_done_6D.tsv"

    parser = argparse.ArgumentParser(description="Interactive frame-by-frame TSV viewer for body and ball mocap data.")
    parser.add_argument("--skeleton", type=Path, default=default_skeleton, help="Skeleton TSV with body segments")
    parser.add_argument("--ball-3d", type=Path, default=default_ball_3d, help="6DOF marker TSV for the ball")
    parser.add_argument("--ball-6d", type=Path, default=default_ball_6d, help="6D TSV with the ground-truth ball center")
    parser.add_argument("--frame", type=int, default=None, help="Initial frame to show")
    parser.add_argument("--ball-mode", choices=("fit", "gt"), default="fit", help="Initial ball placement mode")
    parser.add_argument("--bones", dest="bones", action="store_true", default=True, help="Draw skeleton bones")
    parser.add_argument("--no-bones", dest="bones", action="store_false", help="Draw body segments as points instead of bones")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    skeleton_index = index_skeleton_file(args.skeleton)
    ball_3d_index = index_ball_3d_file(args.ball_3d)
    ball_6d_index = index_ball_6d_file(args.ball_6d)

    viewer = FrameViewer(
        skeleton_index=skeleton_index,
        ball_3d_index=ball_3d_index,
        ball_6d_index=ball_6d_index,
        use_bones=args.bones,
        ball_mode=args.ball_mode,
        start_frame=args.frame,
    )

    print(f"Loaded {len(viewer.shared_frames)} shared frames")
    print(f"Start frame: {viewer.shared_frames[viewer.frame_idx]}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())