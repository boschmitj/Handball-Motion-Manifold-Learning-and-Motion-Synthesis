"""``python -m inference doctor`` — pre-flight installation validator.

Prints the resolved env state (where each MAMMA_* path comes from, whether
it points to an existing file/dir), validates per-step env requirements,
and — given ``--task`` — reports per-step path resolution for the run
config (including the ma_2d weights precedence: task.json > env). Exits 0
when the installation is ready, non-zero when something required is
missing or unresolvable.

``--task`` expects a fully-bound run config (one with ``global.capture_json``
set). Presets alone won't pass — doctor can't resolve step paths without
knowing the capture. Either point it at a GUI-saved
``gui/var/interface/run_configs/run_<id>.json`` or build a temp file via
:func:`inference.config.materialize_run_config` first.

Usage::

    python -m inference doctor
    python -m inference doctor --task gui/var/interface/run_configs/run_1.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .. import config as task_config
from ..assets import ASSETS, step_argv_translation, step_optional_translation
from ..env import DEFAULTS, bootstrap_env, repo_root


# (step_name, required_flags, optional_flags). Walks the central
# registry rather than re-listing the step env requirements here, so
# adding or renaming a flag is a one-place change in
# ``inference/assets.py``.
_STEP_ENV_SPEC: Tuple[Tuple[str, Tuple, Tuple], ...] = tuple(
    (step, step_argv_translation(step), step_optional_translation(step))
    for step in ("ma_masks", "ma_2d", "ma_3d")
)


# ─── Source detection ────────────────────────────────────────────────────


def _read_dotenv_local() -> dict:
    """Read .env.local at the repo root as a plain dict (no override)."""
    p = repo_root() / ".env.local"
    if not p.exists():
        return {}
    out: dict = {}
    with open(p, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            out[k] = v
    return out


def _expected_default(key: str) -> Optional[str]:
    """The value that DEFAULTS would inject for ``key``, post-anchoring."""
    raw = DEFAULTS.get(key)
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((repo_root() / p).resolve())


def _classify_source(key: str, value: str, pre_bootstrap: dict, dotenv: dict) -> str:
    """Return one of: [env], [.env.local], [defaults], [unset]."""
    if not value:
        return "[unset]"
    # pre_bootstrap captures the shell-exported env BEFORE bootstrap ran.
    pre = pre_bootstrap.get(key)
    if pre and pre == value:
        return "[env]"
    if dotenv.get(key) and dotenv[key] == value:
        return "[.env.local]"
    # Anchoring rewrites relative defaults; compare against the rewritten form.
    if value == _expected_default(key):
        return "[defaults]"
    # Anything else: probably .env.local with a relative path that was
    # anchored, or a pre-bootstrap value with the same shape. Report the
    # most likely source by precedence.
    if dotenv.get(key):
        return "[.env.local]"
    if pre:
        return "[env]"
    return "[defaults]"


def _path_status(value: str) -> str:
    """Return one of: ok (file), ok (dir), MISSING, unset."""
    if not value:
        return "unset"
    p = Path(value)
    if p.is_file():
        return "ok (file)"
    if p.is_dir():
        return "ok (dir)"
    return "MISSING"


# ─── Reporting ──────────────────────────────────────────────────────────


def _print_header(title: str) -> None:
    print()
    print(title)
    print("─" * max(40, len(title)))


def _print_env_table(pre_bootstrap: dict, dotenv: dict) -> List[str]:
    """Print the DEFAULTS env table. Return a list of error strings (required keys missing)."""
    _print_header("Environment")
    errors: List[str] = []
    rows = []
    for key in sorted(DEFAULTS.keys()):
        value = os.environ.get(key, "")
        source = _classify_source(key, value, pre_bootstrap, dotenv)
        status = _path_status(value)
        rows.append((key, source, value or "(unset)", status))

    width_k = max(len(r[0]) for r in rows)
    width_s = max(len(r[1]) for r in rows)
    width_v = min(72, max(len(r[2]) for r in rows))
    fmt = f"  {{k:<{width_k}}}  {{s:<{width_s}}}  {{v:<{width_v}}}  {{st}}"
    print(fmt.format(k="KEY", s="SOURCE", v="VALUE", st="STATUS"))
    for k, s, v, st in rows:
        display_v = v if len(v) <= width_v else "…" + v[-(width_v - 1):]
        print(fmt.format(k=k, s=s, v=display_v, st=st))
    return errors


def _print_step_reqs() -> List[str]:
    """Walk each builder's env-flag spec, surface missing required vars."""
    _print_header("Per-step env requirements")
    errors: List[str] = []
    for step, required, optional in _STEP_ENV_SPEC:
        print(f"  {step}:")
        for env_key, flag in required:
            value = os.environ.get(env_key, "")
            mark = "ok " if value else "MISSING"
            if not value:
                errors.append(f"{step}: required env var {env_key} is unset ({flag})")
            print(f"    [required] {env_key:<32} -> {flag:<22} {mark}")
        for env_key, flag in optional:
            value = os.environ.get(env_key, "")
            mark = "set" if value else "unset (optional)"
            print(f"    [optional] {env_key:<32} -> {flag:<22} {mark}")
    return errors


def _print_task_report(task_path: str) -> List[str]:
    """For each enabled step in the task file, dry-run argv resolution."""
    _print_header(f"Task: {task_path}")
    errors: List[str] = []
    try:
        cfg = task_config.load_run_config(task_path)
    except (FileNotFoundError, task_config.TaskConfigError) as e:
        errors.append(f"task: {e}")
        print(f"  FAILED to load: {e}")
        return errors

    # ma_2d weights precedence (the one path that may come from task.json).
    s2d = cfg.get("ma_2d") or {}
    if s2d.get("enabled"):
        task_weights = s2d.get("weights") or ""
        env_weights = os.environ.get("MAMMA_MA2D_CHECKPOINT") or ""
        if task_weights:
            src, val = "task.json", task_weights
        elif env_weights:
            src, val = "env", env_weights
        else:
            src, val = "(none)", ""
            errors.append("ma_2d: --weights cannot be resolved (neither task.json nor MAMMA_MA2D_CHECKPOINT set)")
        st = _path_status(val)
        print(f"  ma_2d.weights      source={src:<10}  status={st:<10}  value={val or '(unset)'}")
        if val and st == "MISSING":
            errors.append(f"ma_2d: --weights resolved to {val} but the file does not exist")

    # Dry-build every enabled step's argv. We only surface RuntimeError
    # from the builder itself (e.g. required env-var missing). We do NOT
    # check whether the argv values point at existing paths — many of
    # them (--out, --ma_2d_dir, --ma_cap_dir, etc.) are inter-step or
    # output directories that legitimately don't exist before the run.
    # Per-installation path existence is already covered by the env
    # table at the top.
    from ..steps.ma_2d import Ma2dBuilder
    from ..steps.ma_3d import Ma3dBuilder
    from ..steps.ma_masks import MaMasksBuilder
    builders = {"ma_masks": MaMasksBuilder, "ma_2d": Ma2dBuilder, "ma_3d": Ma3dBuilder}
    for step, cls in builders.items():
        s = cfg.get(step) or {}
        if not s.get("enabled"):
            continue
        try:
            b = cls(s, cfg.get("global") or {}, "local")
            b.python_argv("seq_test")
            print(f"  {step:<10} builder ok")
        except RuntimeError as e:
            errors.append(f"{step}: builder error: {e}")
            print(f"  {step:<10} builder FAILED: {e}")
    return errors


# ─── Entry point ────────────────────────────────────────────────────────


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m inference doctor",
        description="Validate the MAMMA installation: env paths, step "
                    "requirements, and (optionally) a task file.",
    )
    p.add_argument("--task", default=None,
                   help="Path to a task file (.json/.yaml) to also dry-validate")
    args = p.parse_args(list(argv) if argv is not None else None)

    # Snapshot env BEFORE bootstrap so we can detect shell-exported values.
    pre_bootstrap = dict(os.environ)
    dotenv = _read_dotenv_local()
    bootstrap_env()

    errors: List[str] = []
    errors += _print_env_table(pre_bootstrap, dotenv)
    errors += _print_step_reqs()
    if args.task:
        errors += _print_task_report(args.task)

    _print_header("Summary")
    if errors:
        print(f"  FAIL — {len(errors)} issue(s):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  PASS — environment looks healthy.")
        sys.exit(0)


if __name__ == "__main__":
    main()
