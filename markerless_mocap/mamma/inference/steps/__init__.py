"""Per-step command builders.

Each module exports a :class:`StepBuilder` subclass. The registry below maps
step names from ``task.json`` (e.g. ``"ma_cap"``) to the corresponding builder.
"""
from .base import StepBuilder
from . import ma_2d, ma_3d, ma_cap, ma_masks, ma_vis

BUILDERS: dict[str, type[StepBuilder]] = {
    "ma_cap": ma_cap.MaCapBuilder,
    "ma_masks": ma_masks.MaMasksBuilder,
    "ma_2d": ma_2d.Ma2dBuilder,
    "ma_3d": ma_3d.Ma3dBuilder,
    "ma_vis": ma_vis.MaVisBuilder,
}


def get_builder(step_name: str, step_cfg: dict, global_cfg: dict, tag: str) -> StepBuilder:
    """Instantiate the builder registered for ``step_name``.

    Raises ``ValueError`` if the step name is unknown.
    """
    if step_name not in BUILDERS:
        raise ValueError(
            f"No builder registered for step {step_name!r}. "
            f"Known steps: {sorted(BUILDERS)}"
        )
    return BUILDERS[step_name](step_cfg, global_cfg, tag)
