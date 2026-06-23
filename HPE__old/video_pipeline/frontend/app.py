from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scripts.run_pipeline import PipelineRunConfig, run


H36M_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


def _build_skeleton_animation(joints_xyz: np.ndarray) -> go.Figure:
    def frame_to_traces(pts: np.ndarray) -> list[go.Scatter3d]:
        traces: list[go.Scatter3d] = [
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 4, "color": "red"},
                showlegend=False,
            )
        ]
        for a, b in H36M_EDGES:
            traces.append(
                go.Scatter3d(
                    x=[pts[a, 0], pts[b, 0]],
                    y=[pts[a, 1], pts[b, 1]],
                    z=[pts[a, 2], pts[b, 2]],
                    mode="lines",
                    line={"width": 3, "color": "blue"},
                    showlegend=False,
                )
            )
        return traces

    frames = [go.Frame(data=frame_to_traces(joints_xyz[i]), name=str(i)) for i in range(joints_xyz.shape[0])]

    fig = go.Figure(
        data=frame_to_traces(joints_xyz[0]),
        frames=frames,
        layout=go.Layout(
            height=650,
            scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "data"},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True}],
                        },
                        {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                    ],
                }
            ],
        ),
    )
    return fig


st.set_page_config(page_title="Handball Throw -> Unity BVH", layout="wide")
st.title("Monocular Handball Throw to 3D Unity Animation")

st.markdown(
    """
Upload an MP4, run ViTPose + MotionBERT, inspect 2D/3D outputs, and export BVH for Unity Humanoid.
"""
)

uploaded = st.file_uploader("Upload MP4", type=["mp4"])

with st.sidebar:
    st.header("Model Paths")
    vitpose_config = st.text_input("ViTPose config path")
    vitpose_ckpt = st.text_input("ViTPose checkpoint path")
    motionbert_repo = st.text_input("MotionBERT repo root")
    motionbert_config = st.text_input("MotionBERT config path")
    motionbert_ckpt = st.text_input("MotionBERT checkpoint path")
    device = st.text_input("Device", value="cuda:0")

run_btn = st.button("Run Pipeline", type="primary", disabled=uploaded is None)

if run_btn and uploaded is not None:
    with tempfile.TemporaryDirectory(prefix="handball_ui_") as tmp:
        tmp_dir = Path(tmp)
        video_path = tmp_dir / "input.mp4"
        output_dir = tmp_dir / "output"

        video_path.write_bytes(uploaded.read())

        cfg = PipelineRunConfig(
            video=video_path,
            output_dir=output_dir,
            vitpose_config=Path(vitpose_config),
            vitpose_checkpoint=Path(vitpose_ckpt),
            vitpose_device=device,
            motionbert_repo=Path(motionbert_repo),
            motionbert_config=Path(motionbert_config),
            motionbert_checkpoint=Path(motionbert_ckpt),
            motionbert_device=device,
        )

        with st.spinner("Running pipeline..."):
            summary = run(cfg)

        st.success("Pipeline finished")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Video")
            st.video(str(video_path))
        with col2:
            st.subheader("2D Pose Overlay")
            st.video(summary["pose2d_overlay"])

        st.subheader("3D Skeleton (Frame 0)")
        st.image(summary["pose3d_plot"])

        pose3d_smoothed = np.load(output_dir / "pose3d_smoothed.npy")

        st.subheader("3D Skeleton Animation")
        st.plotly_chart(_build_skeleton_animation(pose3d_smoothed), use_container_width=True)

        st.subheader("Joint Trajectories")
        joint_idx = st.slider("Joint index", min_value=0, max_value=pose3d_smoothed.shape[1] - 1, value=0)
        t = np.arange(pose3d_smoothed.shape[0])
        traj = pose3d_smoothed[:, joint_idx]
        traj_fig = go.Figure()
        traj_fig.add_trace(go.Scatter(x=t, y=traj[:, 0], mode="lines", name="x"))
        traj_fig.add_trace(go.Scatter(x=t, y=traj[:, 1], mode="lines", name="y"))
        traj_fig.add_trace(go.Scatter(x=t, y=traj[:, 2], mode="lines", name="z"))
        traj_fig.update_layout(height=400, xaxis_title="Frame", yaxis_title="Position")
        st.plotly_chart(traj_fig, use_container_width=True)

        bvh_bytes = Path(summary["bvh"]).read_bytes()
        st.download_button(
            label="Download BVH",
            data=bvh_bytes,
            file_name="animation_mixamo.bvh",
            mime="application/octet-stream",
        )

        if summary.get("fbx"):
            fbx_bytes = Path(summary["fbx"]).read_bytes()
            st.download_button(
                label="Download FBX",
                data=fbx_bytes,
                file_name="animation_mixamo.fbx",
                mime="application/octet-stream",
            )

        st.subheader("Raw Arrays")
        st.write({"pose3d_smoothed_shape": list(pose3d_smoothed.shape)})
