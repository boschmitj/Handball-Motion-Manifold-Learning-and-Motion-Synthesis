"""Blinded visualization for one manual Mocap-to-League annotation pair."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

XYZ = ("x", "y", "z")


def _xyz(points: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([[float(point[axis]) for axis in XYZ] for point in points], dtype=float)


def _diagnostic_text(values: Mapping[str, Any]) -> str:
    def value(name: str, digits: int = 3) -> str:
        raw = values.get(name)
        if raw in (None, ""):
            return "n/a"
        if isinstance(raw, bool):
            return "yes" if raw else "no"
        try:
            return f"{float(raw):.{digits}f}"
        except (TypeError, ValueError):
            return str(raw)

    reconstruction_error = values.get("reconstruction_error")
    warning = []
    if reconstruction_error not in (None, ""):
        compact_error = " ".join(str(reconstruction_error).split())
        if len(compact_error) > 220:
            compact_error = compact_error[:217] + "..."
        warning = [
            "!!! RECONSTRUCTION ERROR !!!",
            "Showing aligned measured League samples as fallback.",
            compact_error,
            "",
        ]

    return "\n".join((*warning,
        "Transition diagnostics",
        f"Mocap speed before PoR: {value('mocap_release_speed_m_s')} m/s",
        f"League speed after PoR: {value('league_release_speed_m_s')} m/s",
        f"|speed difference|: {value('transition_speed_difference_m_s')} m/s",
        "",
        f"Mocap direction: {value('mocap_release_direction_deg', 1)}°",
        f"League direction: {value('league_release_direction_deg', 1)}°",
        f"direction difference: {value('direction_difference_deg', 1)}°",
        "",
        f"Mocap release height: {value('mocap_release_height_m')} m",
        f"League release height: {value('league_release_height_m')} m",
        f"height difference: {value('release_height_difference_m')} m",
        "",
        f"Mocap ground contact: {value('mocap_has_ground_contact')} ",
        f"League bounce detected: {value('league_bounce_detected')}",
        f"original League PoGC z: {value('league_original_pogc_z_m')} m",
        f"aligned PoGC z: {value('aligned_pogc_z_m')} m",
        f"expected contact z: {value('expected_contact_z_m')} m",
        f"contact z error: {value('ground_contact_error_m')} m",
        "",
        f"bounce fit valid: {value('bounce_fit_valid')}",
        f"vertical fit RMSE: {value('bounce_vertical_fit_rmse')} m",
        f"reconstruction RMSE: {value('bounce_reconstruction_rmse_m')} m",
        f"reconstruction max error: {value('bounce_reconstruction_max_error_m')} m",
    ))


def render_annotation_figure(
    *, query_label: str, candidate_label: str,
    mocap_pre_por: Sequence[Mapping[str, Any]],
    continuation: Sequence[Mapping[str, Any]],
    original_league_samples: Sequence[Mapping[str, Any]],
    por: Sequence[float], diagnostics: Mapping[str, Any], ground_z: float,
) -> plt.Figure:
    """Render geometry and diagnostics without accepting model metadata."""
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 1.25, .9))
    ax = fig.add_subplot(grid[:, 0:2], projection="3d")
    side = fig.add_subplot(grid[0, 2])
    text = fig.add_subplot(grid[1, 2])

    continuation_xyz = _xyz(continuation)
    original_xyz = _xyz(original_league_samples)
    por_xyz = np.asarray(por, dtype=float)
    pre_xyz = _xyz(mocap_pre_por) if mocap_pre_por else np.empty((0, 3))

    if len(pre_xyz):
        ax.plot(*pre_xyz.T, color="0.35", linewidth=2.2, label="Mocap before PoR")
        side.plot(pre_xyz[:, 0], pre_xyz[:, 2], color="0.35", linewidth=2.2,
                  label="Mocap before PoR")
    reconstruction_failed = diagnostics.get("reconstruction_error") not in (None, "")
    continuation_label = (
        "FALLBACK: aligned measured trajectory" if reconstruction_failed
        else "Reconstructed continuation"
    )
    continuation_color = "tab:orange" if reconstruction_failed else "tab:blue"
    continuation_style = "--" if reconstruction_failed else "-"
    ax.plot(*continuation_xyz.T, color=continuation_color, linestyle=continuation_style,
            linewidth=2, label=continuation_label)
    side.plot(continuation_xyz[:, 0], continuation_xyz[:, 2],
              color=continuation_color, linestyle=continuation_style,
              linewidth=2, label=continuation_label)
    ax.scatter(*original_xyz.T, s=38, marker="o", facecolors="none",
               edgecolors="tab:cyan", linewidths=1.5, label="Original League samples")
    side.scatter(original_xyz[:, 0], original_xyz[:, 2], s=38, marker="o",
                 facecolors="none", edgecolors="tab:cyan", linewidths=1.5,
                 label="Original League samples", zorder=4)
    ax.scatter(*por_xyz, marker="*", s=210, color="gold", edgecolor="black", label="PoR")
    side.scatter(por_xyz[0], por_xyz[2], marker="*", s=180, color="gold",
                 edgecolor="black", label="PoR", zorder=5)

    bounce = next((point for point in continuation if point.get("event") == "bounce"), None)
    if bounce is not None:
        ax.scatter(float(bounce["x"]), float(bounce["y"]), float(bounce["z"]),
                   marker="D", s=75, color="tab:red", label="Estimated PoGC")
        side.scatter(float(bounce["x"]), float(bounce["z"]), marker="D", s=75,
                     color="tab:red", label="Estimated PoGC", zorder=5)

    all_xyz = np.vstack(tuple(array for array in (pre_xyz, continuation_xyz) if len(array)))
    x_min, x_max = float(all_xyz[:, 0].min()), float(all_xyz[:, 0].max())
    y_min, y_max = float(all_xyz[:, 1].min()), float(all_xyz[:, 1].max())
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    ax.plot_surface(xx, yy, np.full_like(xx, ground_z), alpha=.12, color="0.4")
    side.axhline(ground_z, color="0.35", linestyle="--", linewidth=1,
                 label="Ball-center contact height")

    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
           title=f"{query_label} · {candidate_label}: stitched continuation")
    side.set(xlabel="horizontal progression x [m]", ylabel="height z [m]",
             title="Side view")
    side.grid(alpha=.25)
    ax.legend(loc="best", fontsize=8)
    side.legend(loc="best", fontsize=7)
    text.axis("off")
    text.text(0, 1, _diagnostic_text(diagnostics), va="top", ha="left",
              family="monospace", fontsize=9)
    if reconstruction_failed:
        fig.suptitle(
            "WARNING: reconstruction failed — raw aligned League samples shown",
            fontsize=13, color="darkred", fontweight="bold",
        )
        ax.set_title(f"{query_label} · {candidate_label}: reconstruction-error fallback")
    else:
        fig.suptitle("Manual continuation relevance (model identity and rank hidden)", fontsize=13)
    fig.tight_layout()
    return fig
