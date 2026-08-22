#!/usr/bin/env python
"""Generate the two canonical preset YAMLs.

Emits exactly two **capture-independent** preset files:

* ``configs/examples/presets/full.yaml`` — the canonical preset.
  Runs the full DAG (``ma_cap → ma_masks → ma_2d → ma_3d → ma_vis``)
  end-to-end against any capture.

* ``configs/examples/presets/quick.yaml`` — same DAG with a
  fixed frame slice (``start_frame: 60``, ``end_frame: 90``).
  Designed for variety smoke-testing: the full DAG completes in
  a few minutes per capture.

A preset describes *how* to run; it knows nothing about which
capture, cameras, sequences, or data root the run targets. Those
fields are filled in at submit/run time by
:func:`inference.config.materialize_run_config` from the
``--capture`` argument. Specifically:

* ``global.dataset_name`` — derived from the capture filename.
* ``global.cam_names`` — derived from ``capture.cams``.
* ``ma_cap.videos_dir`` — derived from ``capture.capture_root``
  + ``capture.videos_subdir`` (default ``"videos_crf24"``).
* ``ma_cap.calibration`` — derived from ``capture.calib``.

So a single ``full.yaml`` works against *every* shipped capture
(`140725_Breakdance.json`, `iphones_indoors.json`, etc.). Run::

    python scripts/generate_task_configs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRESETS_DIR = _REPO_ROOT / "configs" / "examples" / "presets"

# Quick-mode constants: small frame range so the full DAG completes
# in a few minutes for variety smoke-testing across captures.
_QUICK_START_FRAME = 60
_QUICK_END_FRAME = 90


def _base_preset() -> dict:
    """The capture-independent template, before quick-mode tweaks."""
    return {
        "global": {
            "version": 1.0,
            "username": "",
            # NB: ``capture_json``, ``dataset_name``, ``cam_names``,
            # ``seq_ids`` are intentionally NOT emitted here — they're
            # capture-coupled. The materializer fills them in at
            # submit/run time when a preset is bound to a capture.
            "out_dir": "output",
            "jobs_log_dir": "output/logs/jobs",
            "bind": [],
            "conda_env": "mamma",
        },
        "ma_cap": {
            "engine": "conda",
            "enabled": True,
            "dependencies": [],
            "script": "run_ma_cap.py",
            "repo_path": "capture",
            # videos_dir and calibration are intentionally omitted —
            # the materializer derives them from the capture JSON
            # (see _derive_videos_dir / _derive_calibration in
            # inference/config.py). Set ma_cap.videos_dir here only
            # if you want to override the convention for a specific
            # workflow.
            "flags": [],
        },
        "ma_masks": {
            "engine": "conda",
            "enabled": True,
            "dependencies": ["ma_cap"],
            "script": "run_ma_masks.py",
            "repo_path": "segmentation",
            # No videos_dir / start / end here — ma_masks reads the
            # ma_cap NPZ which carries video_path + frame_start/end.
            "flags": ["--sam_version sam3_prompt"],
            "undistort": False,
        },
        "ma_2d": {
            "engine": "conda",
            "enabled": True,
            "dependencies": ["ma_masks"],
            "script": "run_ma_2d.py",
            "repo_path": "landmarks",
            "config_path": "configs/train/models_2d/config_mammanet_mask_512.yaml",
            "weights": "data/weights/ma_2d/mamma_mask_full_cvpr.ckpt",
            "flags": [],
            "undistort": False,
        },
        "ma_3d": {
            "engine": "conda",
            "enabled": True,
            "dependencies": ["ma_2d"],
            "script": "run_ma_3d.py",
            "repo_path": "optimization",
            "config_file": "config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml",
            "flags": [],
        },
        "ma_vis": {
            "engine": "conda",
            "enabled": True,
            "dependencies": ["ma_2d", "ma_3d"],
            "script": "run_ma_vis.py",
            "repo_path": "visualization",
            "flags": [],
            "undistort": False,
        },
    }


def _build_full() -> dict:
    return _base_preset()


def _build_quick() -> dict:
    cfg = _base_preset()
    # Frame-range plumbing flows ONLY through ma_cap: it writes
    # frame_start/frame_end into per-camera NPZs, and downstream steps
    # inherit the range via the FrameSource that reads the NPZ's
    # video_path + frame range. No per-step --start/--end here.
    cfg["global"]["start_frame"] = _QUICK_START_FRAME
    cfg["global"]["end_frame"] = _QUICK_END_FRAME
    return cfg


def _emit(cfg: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # default_flow_style=False = block style for readability;
        # sort_keys=False to preserve the logical ordering above.
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"  wrote {out_path.relative_to(_REPO_ROOT)}")


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(f"Generating capture-independent presets under {_PRESETS_DIR} ...")
    _emit(_build_full(), _PRESETS_DIR / "full.yaml")
    _emit(_build_quick(), _PRESETS_DIR / "quick.yaml")
    print("\nDone. 2 preset YAMLs generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
