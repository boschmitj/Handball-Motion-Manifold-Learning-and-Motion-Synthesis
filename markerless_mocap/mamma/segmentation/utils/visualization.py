"""Visualization utilities for ma_masks pipeline outputs."""

from pathlib import Path


def generate_cross_camera_summary(seq_out_path, log_fn=None):
    """Generate per-person cross-camera consistency images.

    For each person, stacks their person_XX_crop_summary.png from all cameras
    into one tall image. Makes it easy to visually verify that the same person
    has consistent ID across all cameras.

    Saves to: seq_out_path/cross_camera_summary/person_XX.png

    Args:
        seq_out_path: Sequence output directory containing camera subdirs.
        log_fn: Optional logging function (level, message). Falls back to print.
    """
    def _log(level, msg):
        if log_fn:
            log_fn(level, msg)
        else:
            print(f"[{level}] {msg}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        seq_dir = Path(seq_out_path)
        cam_dirs = sorted([
            d for d in seq_dir.iterdir()
            if d.is_dir() and (d / "masks").is_dir()
        ])

        if len(cam_dirs) < 2:
            return

        # Find all person IDs across cameras
        all_person_ids = set()
        for cam_dir in cam_dirs:
            for f in cam_dir.glob("person_*_crop_summary.png"):
                name = f.stem
                parts = name.split("_")
                if len(parts) >= 2:
                    try:
                        all_person_ids.add(int(parts[1]))
                    except ValueError:
                        pass

        if not all_person_ids:
            return

        out_dir = seq_dir / "cross_camera_summary"
        out_dir.mkdir(exist_ok=True)

        for pid in sorted(all_person_ids):
            cam_images = []
            cam_labels = []
            for cam_dir in cam_dirs:
                summary_path = cam_dir / f"person_{pid:02d}_crop_summary.png"
                if summary_path.exists():
                    cam_images.append(PILImage.open(summary_path))
                    cam_labels.append(cam_dir.name)

            if not cam_images:
                continue

            n_cams = len(cam_images)
            fig, axes = plt.subplots(n_cams, 1, figsize=(16, 4 * n_cams))
            if n_cams == 1:
                axes = [axes]

            for ax, img, label in zip(axes, cam_images, cam_labels):
                ax.imshow(img)
                ax.set_ylabel(label, fontsize=14, fontweight='bold', rotation=0,
                              labelpad=80, va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"Person {pid} — Cross-Camera Consistency",
                         fontsize=16, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(out_dir / f"person_{pid:02d}.png", dpi=100,
                        bbox_inches='tight', facecolor='white')
            plt.close()

        _log("INFO", f"Saved cross-camera summaries to '{out_dir}' ({len(all_person_ids)} people).")
    except Exception as exc:
        _log("WARN", f"Failed to generate cross-camera summary: {exc}")
