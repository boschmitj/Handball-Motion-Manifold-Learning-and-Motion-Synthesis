"""Runner entry point spawned by the GUI's Flask app.

This is a thin shim that:

1. Adds the parent repo (``mamma_release/``) to ``sys.path`` so the
   ``inference`` package imports cleanly.
2. Calls :func:`inference.env.bootstrap_env` to populate ``MAMMA_*``
   defaults and apply any ``.env.local`` override.
3. Parses argv from the Flask layer.
4. Builds a :class:`SqliteSink` bound to the submitted ``task_id`` so
   the runner's status transitions land in ``gui/var/mamma.sqlite``.
5. Hands off to :func:`inference.runner.run_dag` (or
   :func:`inference.runner.run_step` when ``--step`` is set).

Invoked from ``gui/backend/app.py`` as::

    python -m runner_main --task <path> --task-id <int> \
        --out-tag <output_id> --log-tag <task_id>

with ``cwd=gui/backend`` so module imports (``db``, ``sinks``,
``objects.*``) resolve via the cwd-on-path convention the GUI already
uses.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make ``inference`` importable from the parent repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.env import bootstrap_env  # noqa: E402

bootstrap_env()

from inference.runner import run_dag, run_step  # noqa: E402
from sinks import SqliteSink  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GUI-side runner shim.")
    p.add_argument("--task", required=True, help="Path to a task.json / task.yaml.")
    p.add_argument("--task-id", type=int, required=True, dest="task_id",
                   help="DB task_id; the SqliteSink writes statuses against it.")
    p.add_argument("--out-tag", default=None, dest="out_tag",
                   help="Tag used in output/<step>/<tag>/... (typically task.output_id).")
    p.add_argument("--log-tag", default=None, dest="log_tag",
                   help="Tag used in jobs_log_dir/<user>/<tag>/... (typically str(task_id)).")
    p.add_argument("--step", default=None,
                   help="If set, run a single step instead of the full DAG.")
    p.add_argument("--force", action="store_true",
                   help="Ignore DONE sentinels and re-run every (step, seq).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sink = SqliteSink(args.task_id)
    if args.step:
        return run_step(
            args.task,
            args.step,
            out_tag=args.out_tag,
            log_tag=args.log_tag,
            sink=sink,
            force=args.force,
        )
    return run_dag(
        args.task,
        out_tag=args.out_tag,
        log_tag=args.log_tag,
        sink=sink,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
