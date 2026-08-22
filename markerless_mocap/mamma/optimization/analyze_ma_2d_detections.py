import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANALYSIS_IMG_SIZE_PX = 512.0


def _ensure_landmarks_shape(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks shape to [T, P, L, C].
    """
    if landmarks.ndim == 4:
        return landmarks
    if landmarks.ndim == 3:
        return landmarks[:, None, :, :]
    raise ValueError(f"Unsupported landmarks shape: {landmarks.shape}")


def _ensure_vis_shape(vis: Optional[np.ndarray], t: int, p: int, l: int) -> Optional[np.ndarray]:
    if vis is None:
        return None
    if vis.ndim == 3:
        return vis
    if vis.ndim == 2:
        return vis[:, None, :]
    raise ValueError(f"Unsupported visibilities shape: {vis.shape}")


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() == 0:
        return out
    finite_values = values[finite_mask]
    v_min = float(np.min(finite_values))
    v_max = float(np.max(finite_values))
    if abs(v_max - v_min) < 1e-10:
        out[finite_mask] = 0.0
        return out
    out[finite_mask] = (finite_values - v_min) / (v_max - v_min)
    return out


def _safe_nanmean(values: np.ndarray, axis=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(values, axis=axis)


def _sigma_model_from_raw(unc_raw: np.ndarray) -> np.ndarray:
    """
    sigma_model = sqrt(exp(raw_uncertainty)).
    """
    unc_raw = np.clip(unc_raw, -30.0, 30.0)
    sigma_model = np.sqrt(np.exp(unc_raw))
    sigma_model = np.where(np.isfinite(sigma_model), sigma_model, np.nan)
    return sigma_model


def _sigma_eff_px_from_sigma_model(sigma_model: np.ndarray, img_size_px: float = _ANALYSIS_IMG_SIZE_PX) -> np.ndarray:
    """
    Mirror losses.proj_pts_loss mapping:
      sigma_eff_px = clip((sigma_model / 2) * img_size_px, 1, 50)
      if global minimum > 15, divide by 2 before clipping.
    """
    sigma_eff_px = (sigma_model / 2.0) * float(img_size_px)
    finite_vals = sigma_eff_px[np.isfinite(sigma_eff_px)]
    if finite_vals.size > 0 and float(np.min(finite_vals)) > 15.0:
        sigma_eff_px = sigma_eff_px / 2.0
    sigma_eff_px = np.clip(sigma_eff_px, 1.0, 50.0)
    sigma_eff_px = np.where(np.isfinite(sigma_eff_px), sigma_eff_px, np.nan)
    return sigma_eff_px


def _optimizer_weight_from_sigma_eff(sigma_eff_px: np.ndarray) -> np.ndarray:
    """
    Reprojection MSE divides residual by (2 * sigma_eff_px), so effective weight
    is proportional to 1 / (2 * sigma_eff_px)^2.
    Higher value means stronger optimization influence for that keypoint.
    """
    weight = 1.0 / np.square(2.0 * sigma_eff_px)
    weight = np.where(np.isfinite(weight), weight, np.nan)
    return weight


def _find_pred_files(pred_dir: str, cam_names: List[str], cam_name_prefix: str) -> List[str]:
    all_pred_fns = [
        f for f in glob.glob(os.path.join(pred_dir, "*.npz"))
        if os.path.basename(f).rsplit("_", 1)[-1] != "diff.npz"
    ]
    all_pred_fns = sorted(all_pred_fns)

    if cam_names:
        pred_fns = []
        for cam_name in cam_names:
            pred_fns.extend([f for f in all_pred_fns if cam_name in os.path.basename(f)])
        pred_fns = sorted(pred_fns)
    else:
        pred_fns = [f for f in all_pred_fns if os.path.basename(f).startswith(cam_name_prefix)]
        pred_fns = sorted(pred_fns)
        if not pred_fns:
            pred_fns = all_pred_fns

    return pred_fns


def _camera_id_from_file(pred_fn: str) -> str:
    return os.path.splitext(os.path.basename(pred_fn))[0]


def _save_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    colorbar_label: str,
    out_fn: str,
    cmap: str = "viridis",
):
    plt.figure(figsize=(max(8, len(col_labels) * 0.8), max(5, len(row_labels) * 0.45)))
    im = plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_fn, dpi=160)
    plt.close()


def _save_camera_component_bars(camera_summary: pd.DataFrame, out_fn: str):
    x = np.arange(len(camera_summary))
    labels = camera_summary["camera_id"].tolist()

    missing = camera_summary["missing_ratio"].to_numpy(dtype=np.float64)
    vis_component = np.where(
        np.isfinite(camera_summary["mean_visibility"].to_numpy(dtype=np.float64)),
        1.0 - camera_summary["mean_visibility"].to_numpy(dtype=np.float64),
        0.0,
    )
    if "mean_opt_weight" in camera_summary.columns:
        mean_opt_weight = np.nan_to_num(camera_summary["mean_opt_weight"].to_numpy(dtype=np.float64), nan=0.0)
        unc_component = 1.0 - _minmax_normalize(mean_opt_weight)
        unc_label = "1-opt_weight_norm"
    else:
        unc_component = _minmax_normalize(
            np.nan_to_num(camera_summary["mean_uncertainty_used"].to_numpy(dtype=np.float64), nan=0.0)
        )
        unc_label = "uncertainty_norm"

    width = 0.28
    plt.figure(figsize=(max(10, len(labels) * 0.55), 5.5))
    plt.bar(x - width, missing, width=width, label="missing_ratio")
    plt.bar(x, vis_component, width=width, label="1-mean_visibility")
    plt.bar(x + width, unc_component, width=width, label=unc_label)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Component value")
    plt.title("Detection Quality Components Per Camera (optimizer-aligned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fn, dpi=160)
    plt.close()


def _save_opt_weight_histograms_by_camera(df_frame_cam_sub: pd.DataFrame, out_fn: str, bins: int = 30):
    """
    Save small-multiples histograms of mean_opt_weight per camera.
    Higher mean_opt_weight means stronger optimizer influence from 2D reprojection.
    """
    if "mean_opt_weight" not in df_frame_cam_sub.columns:
        return
    df = df_frame_cam_sub[["camera_id", "mean_opt_weight"]].copy()
    df = df[np.isfinite(df["mean_opt_weight"].to_numpy(dtype=np.float64))]
    if df.empty:
        return

    cameras = sorted(df["camera_id"].unique().tolist())
    n_cams = len(cameras)
    n_cols = 4
    n_rows = int(np.ceil(n_cams / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.6, max(3.2, n_rows * 2.9)), squeeze=False)

    for idx, cam in enumerate(cameras):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        vals = df.loc[df["camera_id"] == cam, "mean_opt_weight"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.set_title(cam, fontsize=9)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_xlim(0.0, 0.26)
            continue
        ax.hist(vals, bins=bins, color="#4f7da7", alpha=0.85)
        ax.axvline(float(np.median(vals)), color="#cc4b37", linewidth=1.4)
        ax.set_title(cam, fontsize=9)
        ax.set_xlim(0.0, 0.26)
        ax.grid(alpha=0.18)

    for idx in range(n_cams, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    fig.suptitle("Per-Camera Distribution of mean_opt_weight (higher = stronger optimizer influence)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_fn, dpi=160)
    plt.close(fig)


def _save_camera_influence_ranking(camera_summary: pd.DataFrame, out_fn: str):
    """
    Camera influence proxies:
    - mean_opt_weight: uncertainty-driven influence from reprojection weighting.
    - influence_proxy: combines detection availability and visibility:
        influence_proxy = (1-missing_ratio) * clip(mean_visibility, 0, 1) * mean_opt_weight
      Higher values indicate cameras that are expected to affect optimization more.
    """
    if "mean_opt_weight" not in camera_summary.columns:
        return
    df = camera_summary.copy()
    df = df[np.isfinite(df["mean_opt_weight"].to_numpy(dtype=np.float64))]
    if df.empty:
        return

    vis = np.clip(np.nan_to_num(df["mean_visibility"].to_numpy(dtype=np.float64), nan=1.0), 0.0, 1.0)
    avail = 1.0 - np.clip(np.nan_to_num(df["missing_ratio"].to_numpy(dtype=np.float64), nan=1.0), 0.0, 1.0)
    mean_opt = np.nan_to_num(df["mean_opt_weight"].to_numpy(dtype=np.float64), nan=0.0)
    influence_proxy = avail * vis * mean_opt
    df["influence_proxy"] = influence_proxy

    df = df.sort_values("influence_proxy", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    labels = df["camera_id"].tolist()

    plt.figure(figsize=(max(10, len(labels) * 0.55), 5.8))
    plt.bar(x, df["influence_proxy"].to_numpy(dtype=np.float64), label="influence_proxy=(1-missing)*visibility*mean_opt_weight")
    plt.plot(x, df["mean_opt_weight"].to_numpy(dtype=np.float64), color="#cc4b37", marker="o", linewidth=1.3, label="mean_opt_weight")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Influence proxy value")
    plt.title("Camera Influence Ranking (higher = more influential in optimization)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=160)
    plt.close()


def _save_subject_timeline(df_frame_subject: pd.DataFrame, top_k: int, out_fn: str):
    subject_ids = sorted(df_frame_subject["subject_id"].unique().tolist())
    n_sub = len(subject_ids)
    fig, axes = plt.subplots(n_sub, 1, figsize=(14, max(3, n_sub * 2.7)), sharex=True)
    if n_sub == 1:
        axes = [axes]

    for ax, subject_id in zip(axes, subject_ids):
        df_sub = df_frame_subject[df_frame_subject["subject_id"] == subject_id].sort_values("frame_idx")
        frames = df_sub["frame_idx"].to_numpy(dtype=int)
        missing = df_sub["missing_ratio"].to_numpy(dtype=np.float64)
        vis_comp = df_sub["visibility_component"].to_numpy(dtype=np.float64)
        if "opt_weight_component" in df_sub.columns:
            unc_comp = df_sub["opt_weight_component"].to_numpy(dtype=np.float64)
            unc_label = "opt_weight_component"
        else:
            unc_comp = df_sub["uncertainty_component"].to_numpy(dtype=np.float64)
            unc_label = "uncertainty_component"
        score = df_sub["confidence_score"].to_numpy(dtype=np.float64)

        ax.plot(frames, score, label="score", linewidth=2.0)
        ax.plot(frames, missing, label="missing_ratio", alpha=0.7)
        ax.plot(frames, vis_comp, label="visibility_component", alpha=0.7)
        ax.plot(frames, unc_comp, label=unc_label, alpha=0.7)

        worst = df_sub.nlargest(min(top_k, len(df_sub)), "confidence_score")
        ax.scatter(
            worst["frame_idx"].to_numpy(dtype=int),
            worst["confidence_score"].to_numpy(dtype=np.float64),
            s=16,
            marker="x",
            label="worst frames",
        )

        ax.set_ylabel(f"S{subject_id}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Per-Subject Frame Confidence Diagnostics (optimizer-aligned uncertainty)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_fn, dpi=160)
    plt.close(fig)


def run_analysis(pred_dir: str, out_dir: str, cam_names: List[str], cam_name_prefix: str, top_k_frames: int):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pred_fns = _find_pred_files(pred_dir, cam_names, cam_name_prefix)
    if not pred_fns:
        raise ValueError(f"No prediction npz files found in {pred_dir}")

    loaded = []
    for pred_fn in pred_fns:
        data = np.load(pred_fn, allow_pickle=True)
        if "landmarks" not in data:
            continue
        landmarks = _ensure_landmarks_shape(np.asarray(data["landmarks"]))
        vis = _ensure_vis_shape(np.asarray(data["visibilities"]) if "visibilities" in data else None, *landmarks.shape[:3])
        loaded.append(
            {
                "camera_id": _camera_id_from_file(pred_fn),
                "pred_fn": pred_fn,
                "landmarks": landmarks,
                "vis": vis,
            }
        )

    if not loaded:
        raise ValueError("No usable npz files with 'landmarks' key were found.")

    min_t = min(item["landmarks"].shape[0] for item in loaded)
    min_p = min(item["landmarks"].shape[1] for item in loaded)
    min_l = min(item["landmarks"].shape[2] for item in loaded)

    print(f"Using aligned shape: T={min_t}, P={min_p}, L={min_l}, cams={len(loaded)}")

    for item in loaded:
        item["landmarks"] = item["landmarks"][:min_t, :min_p, :min_l, :]
        if item["vis"] is not None:
            item["vis"] = item["vis"][:min_t, :min_p, :min_l]

    camera_subject_rows = []
    frame_cam_subject_rows = []

    for item in loaded:
        camera_id = item["camera_id"]
        landmarks = item["landmarks"]
        vis = item["vis"]

        xy = landmarks[..., :2]
        valid = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1]) & (xy[..., 0] > 1.0) & (xy[..., 1] > 1.0)

        sigma_model = None
        sigma_eff_px = None
        opt_weight_sigma = None
        if landmarks.shape[-1] >= 3:
            unc_raw = landmarks[..., 2]
            sigma_model = _sigma_model_from_raw(unc_raw)
            sigma_eff_px = _sigma_eff_px_from_sigma_model(sigma_model, img_size_px=_ANALYSIS_IMG_SIZE_PX)
            opt_weight_sigma = _optimizer_weight_from_sigma_eff(sigma_eff_px)

        for subject_id in range(min_p):
            valid_sub = valid[:, subject_id, :]
            # missing_ratio in [0,1]. Lower is better.
            missing_ratio = 1.0 - float(valid_sub.mean())

            if vis is not None:
                vis_sub = vis[:, subject_id, :]
                mean_visibility = float(_safe_nanmean(vis_sub))
            else:
                mean_visibility = np.nan

            if sigma_model is not None:
                sigma_model_sub = sigma_model[:, subject_id, :]
                sigma_eff_sub = sigma_eff_px[:, subject_id, :]
                opt_weight_sub = opt_weight_sigma[:, subject_id, :]
                mean_sigma_model = float(_safe_nanmean(sigma_model_sub))
                mean_sigma_eff_px = float(_safe_nanmean(sigma_eff_sub))
                mean_opt_weight = float(_safe_nanmean(opt_weight_sub))
            else:
                mean_sigma_model = np.nan
                mean_sigma_eff_px = np.nan
                mean_opt_weight = np.nan

            # Legacy alias kept for compatibility with previous outputs.
            mean_uncertainty_used = mean_sigma_model

            camera_subject_rows.append(
                {
                    "camera_id": camera_id,
                    "subject_id": subject_id,
                    "missing_ratio": missing_ratio,
                    "mean_visibility": mean_visibility,
                    "mean_uncertainty_used": mean_uncertainty_used,
                    "mean_sigma_model": mean_sigma_model,
                    "mean_sigma_eff_px": mean_sigma_eff_px,
                    "mean_opt_weight": mean_opt_weight,
                    "frames": min_t,
                    "landmarks": min_l,
                }
            )

            missing_frame = 1.0 - valid_sub.mean(axis=1)

            if vis is not None:
                vis_frame = _safe_nanmean(vis[:, subject_id, :], axis=1)
                # visibility_component in [0,1]. Lower is better.
                vis_component = 1.0 - np.clip(np.nan_to_num(vis_frame, nan=0.0), 0.0, 1.0)
            else:
                vis_frame = np.full(min_t, np.nan, dtype=np.float64)
                vis_component = np.zeros(min_t, dtype=np.float64)

            if sigma_model is not None:
                sigma_model_frame = _safe_nanmean(sigma_model[:, subject_id, :], axis=1)
                sigma_eff_frame = _safe_nanmean(sigma_eff_px[:, subject_id, :], axis=1)
                opt_weight_frame = _safe_nanmean(opt_weight_sigma[:, subject_id, :], axis=1)
                # Convert "stronger optimization weight = better" into a "worse-is-higher"
                # component for unified ranking together with missing/visibility components.
                opt_weight_component = 1.0 - _minmax_normalize(np.nan_to_num(opt_weight_frame, nan=0.0))
            else:
                sigma_model_frame = np.full(min_t, np.nan, dtype=np.float64)
                sigma_eff_frame = np.full(min_t, np.nan, dtype=np.float64)
                opt_weight_frame = np.full(min_t, np.nan, dtype=np.float64)
                opt_weight_component = np.zeros(min_t, dtype=np.float64)

            # confidence_score: higher means worse 2D evidence for fitting.
            score = missing_frame + vis_component + opt_weight_component

            for frame_idx in range(min_t):
                frame_cam_subject_rows.append(
                    {
                        "camera_id": camera_id,
                        "subject_id": subject_id,
                        "frame_idx": frame_idx,
                        "missing_ratio": float(missing_frame[frame_idx]),
                        "mean_visibility": float(vis_frame[frame_idx]) if np.isfinite(vis_frame[frame_idx]) else np.nan,
                        "mean_uncertainty_used": float(sigma_model_frame[frame_idx]) if np.isfinite(sigma_model_frame[frame_idx]) else np.nan,
                        "mean_sigma_model": float(sigma_model_frame[frame_idx]) if np.isfinite(sigma_model_frame[frame_idx]) else np.nan,
                        "mean_sigma_eff_px": float(sigma_eff_frame[frame_idx]) if np.isfinite(sigma_eff_frame[frame_idx]) else np.nan,
                        "mean_opt_weight": float(opt_weight_frame[frame_idx]) if np.isfinite(opt_weight_frame[frame_idx]) else np.nan,
                        "visibility_component": float(vis_component[frame_idx]),
                        "uncertainty_component": float(opt_weight_component[frame_idx]),
                        "opt_weight_component": float(opt_weight_component[frame_idx]),
                        "confidence_score": float(score[frame_idx]),
                    }
                )

    df_cam_sub = pd.DataFrame(camera_subject_rows).sort_values(["camera_id", "subject_id"])
    df_frame_cam_sub = pd.DataFrame(frame_cam_subject_rows).sort_values(["camera_id", "subject_id", "frame_idx"])

    df_cam_sub.to_csv(os.path.join(out_dir, "camera_subject_summary.csv"), index=False)

    df_camera = (
        df_cam_sub.groupby("camera_id", as_index=False)[
            ["missing_ratio", "mean_visibility", "mean_uncertainty_used", "mean_sigma_model", "mean_sigma_eff_px", "mean_opt_weight"]
        ]
        .mean()
        .sort_values("camera_id")
    )
    df_camera.to_csv(os.path.join(out_dir, "camera_summary.csv"), index=False)

    df_subject = (
        df_cam_sub.groupby("subject_id", as_index=False)[
            ["missing_ratio", "mean_visibility", "mean_uncertainty_used", "mean_sigma_model", "mean_sigma_eff_px", "mean_opt_weight"]
        ]
        .mean()
        .sort_values("subject_id")
    )
    df_subject.to_csv(os.path.join(out_dir, "subject_summary.csv"), index=False)

    df_frame_cam_sub_topk = df_frame_cam_sub.nlargest(top_k_frames, "confidence_score")
    df_frame_cam_sub_topk.to_csv(os.path.join(out_dir, "least_confident_frames_camera_subject.csv"), index=False)

    df_frame_subject = (
        df_frame_cam_sub.groupby(["subject_id", "frame_idx"], as_index=False)[
            ["missing_ratio", "visibility_component", "uncertainty_component", "opt_weight_component", "confidence_score"]
        ]
        .mean()
        .sort_values(["subject_id", "frame_idx"])
    )
    df_frame_subject.to_csv(os.path.join(out_dir, "frame_subject_scores.csv"), index=False)
    df_frame_subject.nlargest(top_k_frames, "confidence_score").to_csv(
        os.path.join(out_dir, "least_confident_frames_subject.csv"), index=False
    )

    # Camera-frame ranking (used to visualize weakest mesh overlays)
    df_worst_subject_per_cam_frame = (
        df_frame_cam_sub.sort_values("confidence_score")
        .groupby(["camera_id", "frame_idx"], as_index=False)
        .tail(1)[["camera_id", "frame_idx", "subject_id", "confidence_score"]]
        .rename(
            columns={
                "subject_id": "worst_subject_id",
                "confidence_score": "max_confidence_score",
            }
        )
    )

    df_camera_frame = (
        df_frame_cam_sub.groupby(["camera_id", "frame_idx"], as_index=False)[
            [
                "missing_ratio",
                "visibility_component",
                "uncertainty_component",
                "opt_weight_component",
                "mean_sigma_model",
                "mean_sigma_eff_px",
                "mean_opt_weight",
                "confidence_score",
            ]
        ]
        .mean()
        .rename(columns={"confidence_score": "mean_confidence_score"})
        .merge(df_worst_subject_per_cam_frame, on=["camera_id", "frame_idx"], how="left")
        .sort_values(["camera_id", "frame_idx"])
    )
    df_camera_frame.to_csv(os.path.join(out_dir, "camera_frame_scores.csv"), index=False)
    df_camera_frame.nlargest(top_k_frames, "max_confidence_score").to_csv(
        os.path.join(out_dir, "least_confident_camera_frames.csv"), index=False
    )

    # Heatmaps (camera x subject)
    cameras = sorted(df_cam_sub["camera_id"].unique().tolist())
    subjects = sorted(df_cam_sub["subject_id"].unique().tolist())

    missing_mat = (
        df_cam_sub.pivot(index="camera_id", columns="subject_id", values="missing_ratio")
        .reindex(index=cameras, columns=subjects)
        .to_numpy(dtype=np.float64)
    )
    _save_heatmap(
        missing_mat,
        row_labels=cameras,
        col_labels=[f"S{s}" for s in subjects],
        title="Missing Detection Ratio (Camera x Subject)",
        colorbar_label="missing_ratio",
        out_fn=os.path.join(plots_dir, "heatmap_missing_ratio.png"),
        cmap="magma",
    )

    if df_cam_sub["mean_visibility"].notna().any():
        vis_mat = (
            df_cam_sub.pivot(index="camera_id", columns="subject_id", values="mean_visibility")
            .reindex(index=cameras, columns=subjects)
            .to_numpy(dtype=np.float64)
        )
        _save_heatmap(
            vis_mat,
            row_labels=cameras,
            col_labels=[f"S{s}" for s in subjects],
            title="Mean Visibility (Camera x Subject)",
            colorbar_label="mean_visibility",
            out_fn=os.path.join(plots_dir, "heatmap_mean_visibility.png"),
            cmap="viridis",
        )

    if df_cam_sub["mean_uncertainty_used"].notna().any():
        unc_mat = (
            df_cam_sub.pivot(index="camera_id", columns="subject_id", values="mean_uncertainty_used")
            .reindex(index=cameras, columns=subjects)
            .to_numpy(dtype=np.float64)
        )
        _save_heatmap(
            unc_mat,
            row_labels=cameras,
            col_labels=[f"S{s}" for s in subjects],
            title="Mean sigma_model = sqrt(exp(raw_uncertainty)) (Camera x Subject)",
            colorbar_label="sigma_model",
            out_fn=os.path.join(plots_dir, "heatmap_mean_uncertainty_used.png"),
            cmap="cividis",
        )

    if df_cam_sub["mean_sigma_eff_px"].notna().any():
        sigma_eff_mat = (
            df_cam_sub.pivot(index="camera_id", columns="subject_id", values="mean_sigma_eff_px")
            .reindex(index=cameras, columns=subjects)
            .to_numpy(dtype=np.float64)
        )
        _save_heatmap(
            sigma_eff_mat,
            row_labels=cameras,
            col_labels=[f"S{s}" for s in subjects],
            title="Mean sigma_eff_px used by reprojection loss (Camera x Subject)",
            colorbar_label="sigma_eff_px",
            out_fn=os.path.join(plots_dir, "heatmap_mean_sigma_eff_px.png"),
            cmap="plasma",
        )

    if df_cam_sub["mean_opt_weight"].notna().any():
        opt_weight_mat = (
            df_cam_sub.pivot(index="camera_id", columns="subject_id", values="mean_opt_weight")
            .reindex(index=cameras, columns=subjects)
            .to_numpy(dtype=np.float64)
        )
        _save_heatmap(
            opt_weight_mat,
            row_labels=cameras,
            col_labels=[f"S{s}" for s in subjects],
            title="Mean optimization-weight proxy 1/(2*sigma_eff_px)^2 (Camera x Subject)",
            colorbar_label="opt_weight_proxy",
            out_fn=os.path.join(plots_dir, "heatmap_mean_opt_weight.png"),
            cmap="viridis",
        )

    _save_camera_component_bars(df_camera, os.path.join(plots_dir, "camera_quality_components.png"))
    _save_opt_weight_histograms_by_camera(
        df_frame_cam_sub,
        os.path.join(plots_dir, "hist_mean_opt_weight_by_camera.png"),
        bins=30,
    )
    _save_camera_influence_ranking(
        df_camera,
        os.path.join(plots_dir, "camera_influence_ranking.png"),
    )
    _save_subject_timeline(df_frame_subject, top_k=top_k_frames, out_fn=os.path.join(plots_dir, "subject_frame_timeline.png"))

    # Global frame score over subjects/cameras
    df_global_frame = (
        df_frame_cam_sub.groupby("frame_idx", as_index=False)["confidence_score"].mean().sort_values("frame_idx")
    )
    plt.figure(figsize=(12, 4.5))
    plt.plot(df_global_frame["frame_idx"], df_global_frame["confidence_score"], linewidth=1.8)
    worst = df_global_frame.nlargest(min(top_k_frames, len(df_global_frame)), "confidence_score")
    plt.scatter(worst["frame_idx"], worst["confidence_score"], s=16, marker="x")
    plt.title("Global Frame Confidence Score (higher = worse detections)")
    plt.xlabel("Frame index")
    plt.ylabel("mean confidence_score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "global_frame_score.png"), dpi=160)
    plt.close()

    print(f"Analysis completed. Output directory: {out_dir}")
    print(f"Loaded cameras: {[item['camera_id'] for item in loaded]}")
    print(
        "Uncertainty interpretation used: sigma_model=sqrt(exp(raw)); "
        "sigma_eff_px=clip((sigma_model/2)*512, 1, 50) with global /2 when min>15; "
        "optimizer-weight proxy is proportional to 1/(2*sigma_eff_px)^2."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Meta analysis for multi-view 2D detections used in optimization.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to sequence 2D detections folder (camera npz files).")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder for CSVs and plots. Default: <pred_dir>/analysis_2d_detections")
    parser.add_argument("--cam_names", type=str, nargs="+", default=[], help="Optional camera filter (e.g., IOI_01 IOI_02).")
    parser.add_argument("--cam_name_prefix", type=str, default="IOI_", help="Camera filename prefix when --cam_names is empty.")
    parser.add_argument("--top_k_frames", type=int, default=30, help="How many least-confident frames to report.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.pred_dir, "analysis_2d_detections")
    run_analysis(
        pred_dir=args.pred_dir,
        out_dir=out_dir,
        cam_names=args.cam_names,
        cam_name_prefix=args.cam_name_prefix,
        top_k_frames=max(1, int(args.top_k_frames)),
    )


if __name__ == "__main__":
    main()
