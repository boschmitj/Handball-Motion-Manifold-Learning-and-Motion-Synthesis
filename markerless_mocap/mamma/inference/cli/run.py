"""``python -m inference run`` — execute the full pipeline DAG locally.

Two input forms are supported:

1. ``--task <run_config.json>`` — back-compat. The path points at a
   fully-bound run config (capture_json embedded). Typically used to
   re-run a config persisted by the GUI under ``run_configs/``.

2. ``--preset <preset.yaml> --capture <capture.json>`` — preferred.
   The preset is a capture-independent template; the capture supplies
   the binding. They are merged in memory and the resulting run config
   is written to a temp file before dispatch (the runner re-reads it
   by path).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile

from .. import config, runner
from ..env import bootstrap_env
from ..status import make_sink

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference run",
        description="Run the full MAMMA pipeline DAG locally.",
    )
    p.add_argument("--task", default=None,
                   help="Path to a fully-bound run config (.json / .yaml). "
                        "Mutually exclusive with --cfg/--capture/--footage.")
    # --cfg is the canonical name; --preset is a deprecated alias kept
    # for backward compatibility with scripts / saved commands.
    p.add_argument("--cfg", "--preset", dest="cfg", default=None,
                   help="Path to a capture-independent preset (.yaml / .json). "
                        "Requires either --capture, or the trio "
                        "--footage + --seq_name + --calib. "
                        "(Deprecated alias: --preset.)")
    p.add_argument("--capture", default=None,
                   help="Path to a capture JSON. Requires --cfg. "
                        "Mutually exclusive with --footage/--seq_name/--calib.")
    # Alternative-run mode: skip the capture JSON entirely by passing
    # a footage dataset root + the sequence subdir + a calibration.
    # The runner synthesizes an in-memory capture for that single
    # sequence (matches what the GUI's New-task form does).
    p.add_argument("--footage", default=None,
                   help="Dataset directory containing the sequence subdir. "
                        "When set, --seq_name and --calib are required and "
                        "--capture must be omitted.")
    p.add_argument("--seq_name", default=None,
                   help="Sequence subdirectory name under --footage to run.")
    p.add_argument("--calib", default=None,
                   help="Calibration file (.yaml / .xcp / OpenCV .json) for the "
                        "alternative-run mode.")
    p.add_argument("--out-tag", default=None,
                   help="Output sub-directory tag (default: 'local')")
    p.add_argument("--log-tag", default=None,
                   help="Log sub-directory tag (default: same as --out-tag)")
    p.add_argument("--out-dir", default=None,
                   help="Override global.out_dir from the config")
    p.add_argument("--seqs", default=None,
                   help="Comma-separated seq_ids overriding global.seq_ids "
                        "(e.g. '0,3,7')")
    p.add_argument("--force", action="store_true",
                   help="Re-run (step, seq) pairs even if a DONE sentinel exists")
    p.add_argument("--status-jsonl", default=None,
                   help="Append a JSONL log of status transitions to this path")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for INFO, -vv for DEBUG (default WARNING)")
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _validate_input_flags(args: argparse.Namespace) -> None:
    """Enforce the three valid input combos:

      (1) --task <run_config>
      (2) --cfg <preset> --capture <capture.json>
      (3) --cfg <preset> --footage <dir> --seq_name <name> --calib <file>
    """
    has_task = args.task is not None
    has_cfg = args.cfg is not None
    has_capture = args.capture is not None
    alt_flags = {
        "--footage": args.footage,
        "--seq_name": args.seq_name,
        "--calib": args.calib,
    }
    alt_set = {k: v for k, v in alt_flags.items() if v is not None}
    has_alt_any = bool(alt_set)
    has_alt_all = len(alt_set) == len(alt_flags)

    if has_task and (has_cfg or has_capture or has_alt_any):
        raise SystemExit(
            "error: --task is mutually exclusive with "
            "--cfg / --capture / --footage / --seq_name / --calib"
        )
    if has_capture and has_alt_any:
        raise SystemExit(
            "error: --capture is mutually exclusive with "
            "--footage / --seq_name / --calib (pick one input mode)"
        )
    if has_alt_any and not has_alt_all:
        missing = [k for k, v in alt_flags.items() if v is None]
        raise SystemExit(
            f"error: --footage / --seq_name / --calib must be supplied "
            f"together. Missing: {', '.join(missing)}"
        )
    if has_cfg and not (has_capture or has_alt_all):
        raise SystemExit(
            "error: --cfg requires either --capture <capture.json> or "
            "--footage <dir> --seq_name <name> --calib <file>"
        )
    if (has_capture or has_alt_any) and not has_cfg:
        raise SystemExit(
            "error: --capture / --footage requires --cfg <preset>"
        )
    if not has_task and not has_cfg:
        raise SystemExit(
            "error: provide one of: "
            "--task <run_config>; "
            "--cfg <preset> --capture <capture.json>; "
            "--cfg <preset> --footage <dir> --seq_name <name> --calib <file>"
        )


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> bool:
    """Apply CLI overrides to the loaded config in place.

    Returns True if any override actually mutated the config (which
    means the caller must persist a temp copy before handing the path
    to the runner).
    """
    g = cfg.setdefault("global", {})
    changed = False
    if args.out_dir is not None:
        g["out_dir"] = args.out_dir
        changed = True
    if args.seqs is not None:
        try:
            g["seq_ids"] = [int(s) for s in args.seqs.split(",") if s.strip()]
            changed = True
        except ValueError as e:
            raise SystemExit(f"--seqs: must be comma-separated integers ({e})")
    return changed


def _persist_temp(cfg: dict) -> str:
    """Write ``cfg`` to a temp ``.json`` file and return the path.

    The runner re-loads the config by path, so any in-memory mutation
    (overrides; or the preset+capture merge) has to be materialized to
    disk first.
    """
    fd, path = tempfile.mkstemp(prefix="mamma_run_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main(argv=None) -> None:
    bootstrap_env()
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    _validate_input_flags(args)

    # Load (or materialize) the run config in memory.
    try:
        if args.task is not None:
            cfg = config.load_run_config(args.task)
            source_path = args.task
            materialized = False
        elif args.capture is not None:
            cfg = config.materialize_run_config(args.cfg, args.capture)
            source_path = None
            materialized = True
        else:
            # Alternative-run mode: synthesize a capture for the single
            # sequence, write it to a temp file, and feed the existing
            # materialize_run_config flow with that path. dataset_name
            # is forced to the footage dir's basename — without that
            # override the materializer derives it from the temp
            # file's stem, which would land outputs under e.g.
            # output/ma_*/run01/tmp_abc123/<seq>/...
            synth = config.synthesize_capture(
                args.footage, args.calib, args.seq_name,
            )
            synth_path = _persist_temp(synth)
            dataset_name = os.path.basename(args.footage.rstrip("/")) or "dataset"
            cfg = config.materialize_run_config(
                args.cfg, synth_path, dataset_name=dataset_name,
            )
            source_path = None
            materialized = True
    except (FileNotFoundError, ValueError, config.TaskConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    overrode = _apply_overrides(cfg, args)

    # Always validate the final, post-override config. Materialized configs
    # haven't been validated yet (load_run_config validates on read; the
    # materializer does not).
    try:
        config.validate(cfg)
    except config.TaskConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    # The runner re-loads by path. If we mutated the dict or materialized
    # from a preset, persist to a temp file; otherwise just hand back the
    # original on-disk path.
    if materialized or overrode:
        task_path = _persist_temp(cfg)
    else:
        task_path = source_path

    sink = make_sink(args.status_jsonl)
    rc = runner.run_dag(
        task_path,
        out_tag=args.out_tag,
        log_tag=args.log_tag,
        sink=sink,
        force=args.force,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
