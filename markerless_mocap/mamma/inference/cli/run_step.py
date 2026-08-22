"""``python -m inference run-step`` — execute a single pipeline step locally."""
from __future__ import annotations

import argparse
import logging
import sys

from .. import config, runner
from ..env import bootstrap_env
from ..status import make_sink
from .run import _apply_overrides, _configure_logging

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference run-step",
        description="Run a single MAMMA pipeline step locally.",
    )
    p.add_argument("--task", required=True, help="Path to task.json")
    p.add_argument("--step", required=True, choices=runner.ALL_STEPS,
                   help="Which step to run")
    p.add_argument("--out-tag", default=None)
    p.add_argument("--log-tag", default=None)
    p.add_argument("--out-dir", default=None,
                   help="Override global.out_dir from the task config")
    p.add_argument("--seqs", default=None,
                   help="Comma-separated seq_ids overriding global.seq_ids")
    p.add_argument("--force", action="store_true",
                   help="Re-run sequences even if a DONE sentinel exists")
    p.add_argument("--status-jsonl", default=None,
                   help="Append a JSONL log of status transitions to this path")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv=None) -> None:
    bootstrap_env()
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        cfg = config.load_task(args.task)
    except (FileNotFoundError, config.TaskConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    _apply_overrides(cfg, args)

    task_path = args.task
    if args.out_dir is not None or args.seqs is not None:
        import json
        import os
        import tempfile
        fd, task_path = tempfile.mkstemp(prefix="mamma_task_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)

    sink = make_sink(args.status_jsonl)
    rc = runner.run_step(
        task_path,
        args.step,
        out_tag=args.out_tag,
        log_tag=args.log_tag,
        sink=sink,
        force=args.force,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
