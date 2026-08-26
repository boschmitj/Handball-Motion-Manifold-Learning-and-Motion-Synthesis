"""Browse a short motion-capture recording frame by frame."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np


HERE = Path(__file__).parent
DATA = HERE / "data"
BONES = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Neck"), ("Neck", "Head"),
    ("Neck", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Neck", "RightShoulder"), ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
]


def rows_after_header(path):
    """Return the TSV header and all numeric rows below it."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Frame\t"))
    return lines[header_index].split("\t"), [line.split("\t") for line in lines[header_index + 1:] if line]


def read_skeleton(path):
    header, rows = rows_after_header(path)
    names = [header[i] for i in range(2, len(header), 8) if header[i]]
    frames = {}
    for row in rows:
        points = {}
        for i, name in enumerate(names):
            start = 3 + 8 * i
            try:
                xyz = np.array(row[start:start + 3], dtype=float)
            except ValueError:
                continue
            if np.isfinite(xyz).all() and not np.allclose(xyz, 0):
                points[name] = xyz
        frames[int(float(row[0]))] = (float(row[1]), points)
    return frames


def read_ball_markers(path):
    _, rows = rows_after_header(path)
    frames = {}
    for row in rows:
        markers = []
        for start in range(2, min(len(row) - 2, 20), 3):
            try:
                xyz = np.array(row[start:start + 3], dtype=float)
            except ValueError:
                continue
            if np.isfinite(xyz).all() and not np.allclose(xyz, 0):
                markers.append(xyz)
        frames[int(float(row[0]))] = np.array(markers)
    return frames


def read_ball_centres(path):
    _, rows = rows_after_header(path)
    return {int(float(row[0])): np.array(row[2:5], dtype=float) for row in rows}


def fit_sphere(points):
    """Least-squares sphere centre for the six markers on the ball."""
    matrix = np.column_stack((2 * points, np.ones(len(points))))
    target = np.sum(points * points, axis=1)
    return np.linalg.lstsq(matrix, target, rcond=None)[0][:3]


def equal_axes(ax, points):
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2
    radius = max(high - low) * 0.58
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, help="frame to show first")
    args = parser.parse_args()

    skeleton = read_skeleton(DATA / "skeleton.tsv")
    markers = read_ball_markers(DATA / "ball_markers.tsv")
    centres = read_ball_centres(DATA / "ball_centre.tsv")
    frames = sorted(set(skeleton) & set(markers) & set(centres))
    start = min(range(len(frames)), key=lambda i: abs(frames[i] - (args.frame or frames[0])))

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_axes([0.05, 0.15, 0.72, 0.8], projection="3d")
    slider = Slider(figure.add_axes([0.12, 0.06, 0.58, 0.04]), "Frame", 0, len(frames) - 1,
                    valinit=start, valstep=1)
    choice = CheckButtons(figure.add_axes([0.81, 0.72, 0.16, 0.12]), ["Ground truth"], [False])

    def draw(_=None):
        frame = frames[int(slider.val)]
        time, body = skeleton[frame]
        ball = centres[frame] if choice.get_status()[0] else fit_sphere(markers[frame])
        view = axis.elev, axis.azim
        axis.clear()
        axis.view_init(*view)
        for first, second in BONES:
            if first in body and second in body:
                line = np.vstack((body[first], body[second]))
                axis.plot(*line.T, color="#333333", linewidth=1.5)
        body_points = np.vstack(list(body.values()))
        axis.scatter(*body_points.T, s=9, color="#2878b5")
        axis.scatter(*ball, s=180, color="#e45756", label="ball")
        equal_axes(axis, np.vstack((body_points, ball)))
        axis.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)",
                 title=f"Frame {frame} — {time:.3f} s")
        axis.legend()
        figure.canvas.draw_idle()

    def key(event):
        if event.key in {"left", "right"}:
            step = -1 if event.key == "left" else 1
            slider.set_val(np.clip(slider.val + step, 0, len(frames) - 1))

    slider.on_changed(draw)
    choice.on_clicked(draw)
    figure.canvas.mpl_connect("key_press_event", key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
