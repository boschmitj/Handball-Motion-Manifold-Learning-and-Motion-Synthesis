"""Per-installation paths consumed by ma_3d.

These were previously read at module-import time via bare
``os.environ["MAMMA_*"]`` calls, which crashed the entire pipeline
with an opaque ``KeyError`` if any required var was missing. They
now flow through argparse as explicit ``--flag`` arguments at the
process boundary; this dataclass bundles them so they can be passed
through the optimization call graph as a single value instead of a
handful of separate parameters.

The runner (``inference/steps/ma_3d.py``) translates the corresponding
``MAMMA_*`` env vars into command-line flags before spawning the
subprocess. Direct ``python run_ma_3d.py`` users pass the flags
themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PathsConfig:
    """Installation paths required (or optionally required) by ma_3d."""

    # Required for every ma_3d run.
    smplx_lockhead_models: str
    downsampled_verts_pkl: str

    # Optional. Used only when the algorithm config_file selects them.
    bun_models: Optional[str] = None        # use_bun_model: True
    part_mesh_path: Optional[str] = None    # SDF-based loss

    @classmethod
    def from_args(cls, args) -> "PathsConfig":
        """Build a PathsConfig from an argparse Namespace.

        Looks up flags using both ``--kebab-case`` and ``--snake_case``
        forms (argparse converts ``--smplx-models`` to ``args.smplx_models``).
        """
        return cls(
            smplx_lockhead_models=args.smplx_models,
            downsampled_verts_pkl=args.downsampled_verts,
            bun_models=getattr(args, "bun_models", None) or None,
            part_mesh_path=getattr(args, "part_mesh", None) or None,
        )
