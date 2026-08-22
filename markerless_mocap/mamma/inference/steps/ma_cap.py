"""ma_cap (capture / input) — extract per-camera frames from a capture session."""
from __future__ import annotations

from typing import List

from .base import StepBuilder


class MaCapBuilder(StepBuilder):
    step_name = "ma_cap"

    def python_argv(self, seq_name: str) -> List[str]:
        argv: List[str] = [self.script]
        frame_source_flags = self._frame_source_flags()

        # Tri-mode dispatch: --videos_dir or --images_root_dir (with
        # --calibration) override the default --json capture-mode.
        if frame_source_flags:
            argv += frame_source_flags
            argv += self._calibration_flag()
        else:
            # Anchored so subprocess (cwd=capture/) can open it from anywhere.
            argv += ["--json", self._resolve("capture_json")]
            # Allow an explicit calibration override on top of capture.json.
            argv += self._calibration_flag()

        if self.cam_names:
            argv += ["--cam_names", *self.cam_names]
        argv += ["--out", self.step_out_dir(with_dataset=True)]
        argv += ["--seq_name", seq_name]
        # ma_cap owns the canonical frame range for the run. Downstream
        # steps inherit it via per-camera NPZ (frame_start / frame_end),
        # so only this builder emits --start / --end.
        argv += self._frame_range_flags()
        argv += self.flags
        return argv
