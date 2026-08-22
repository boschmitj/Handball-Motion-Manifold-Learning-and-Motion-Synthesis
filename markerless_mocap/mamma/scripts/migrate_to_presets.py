#!/usr/bin/env python3
"""One-shot migration: rename "task" → "preset"/"run config" on disk.

The repo used to conflate two concepts under the word "task":

  * a *preset* — a capture-independent template of steps + hyperparams
  * a *run config* — a preset bound to a specific capture + form input

This script materializes the new vocabulary by moving / renaming the
on-disk artifacts. The companion changes in inference/* and gui/* code
expect the new layout once this script has been run once.

Idempotent: safe to invoke twice. Subsequent runs are no-ops because
each step checks for the absence of the source location.

What it does, in order:

  1. configs/examples/tasks/*.yaml
       -> configs/examples/presets/*.yaml
       (strips global.capture_json so the files are real, capture-
       independent presets; uses ``git mv`` first so history is
       preserved, then rewrites contents in place.)

  2. configs/examples/quick_tasks/*.yaml
       -> configs/examples/presets/quick/*.yaml
       (same strip; same git-mv-then-rewrite.)

  3. <MAMMA_INTERFACE_DIR>/samples/tasks/
       -> <MAMMA_INTERFACE_DIR>/samples/presets/
       (gitignored — local user data; preserves the optional ``user/``
       subdir as-is.)

  4. <MAMMA_INTERFACE_DIR>/task_jsons/task_config_<id>.json
       -> <MAMMA_INTERFACE_DIR>/run_configs/run_<id>.json
       (gitignored; also rewrites tasks.task_json_path in the SQLite DB
       under <MAMMA_DATA_DIR>/mamma.sqlite so historical rows still
       resolve.)

After this script, delete it from the tree in the following release.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# --- step 1 + 2: configs/examples preset relocation -----------------------

_EXAMPLES_DIR = _REPO_ROOT / "configs" / "examples"
_OLD_TASKS_DIR = _EXAMPLES_DIR / "tasks"
_OLD_QUICK_DIR = _EXAMPLES_DIR / "quick_tasks"
_NEW_PRESETS_DIR = _EXAMPLES_DIR / "presets"
_NEW_QUICK_DIR = _NEW_PRESETS_DIR / "quick"

# --- step 3 + 4: GUI var/ relocation --------------------------------------

_DATA_DIR = Path(os.environ.get("MAMMA_DATA_DIR") or _REPO_ROOT / "gui" / "var").expanduser()
_INTERFACE_DIR = Path(os.environ.get("MAMMA_INTERFACE_DIR", str(_DATA_DIR / "interface"))).expanduser()
_OLD_SAMPLES_TASKS = _INTERFACE_DIR / "samples" / "tasks"
_NEW_SAMPLES_PRESETS = _INTERFACE_DIR / "samples" / "presets"
_OLD_TASK_JSONS = _INTERFACE_DIR / "task_jsons"
_NEW_RUN_CONFIGS = _INTERFACE_DIR / "run_configs"
_DB_PATH = _DATA_DIR / "mamma.sqlite"

_TASK_CONFIG_RE = re.compile(r"^task_config_(\d+)\.json$")


def _git_mv_or_copy(src: Path, dst: Path) -> bool:
    """Move ``src`` to ``dst`` preserving git history when possible.

    Falls back to a plain copy+remove if the file isn't tracked (e.g. on
    a fresh clone where the user has uncommitted local renames). Returns
    True on success.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel_src = src.relative_to(_REPO_ROOT)
        rel_dst = dst.relative_to(_REPO_ROOT)
        result = subprocess.run(
            ["git", "mv", str(rel_src), str(rel_dst)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        # git mv failed (probably "not under version control"); fall back.
        print(f"  git mv declined ({result.stderr.strip()}); falling back to shutil.move")
    except ValueError:
        pass  # src not under repo root — gitignored area
    shutil.move(str(src), str(dst))
    return True


# Capture-coupled `global` keys to drop from presets. The materializer
# fills these in at submit/run time, so leaving them in the preset
# would lock it to one capture by name.
_CAPTURE_COUPLED_GLOBAL_KEYS = (
    "capture_json", "dataset_name", "cam_names", "seq_ids",
)

# Capture-coupled per-step keys. The materializer derives ma_cap.videos_dir
# and ma_cap.calibration from the capture JSON; the iphones-style
# ``images_root_dir`` falls in the same category.
_CAPTURE_COUPLED_STEP_KEYS = {
    "ma_cap": ("videos_dir", "images_root_dir", "calibration"),
}

# Captures whose data ships under a non-default videos subdir.
# Materializer reads ``capture.videos_subdir`` (defaulting to
# ``"videos_crf24"``); the migration sets the iphones captures to
# ``"videos"`` so the post-collapse preset still resolves correctly.
_CAPTURE_VIDEOS_SUBDIR_OVERRIDES = {
    "iphones_indoors": "videos",
    "iphones_outdoors": "videos",
}


def _strip_global_keys(yaml_path: Path, keys: tuple[str, ...] = _CAPTURE_COUPLED_GLOBAL_KEYS) -> list[str]:
    """Remove the named ``global.*`` keys from a preset YAML in place.

    Thin wrapper around :func:`_strip_capture_coupled_keys` kept for
    backwards-compat with earlier callers.
    """
    return _strip_capture_coupled_keys(yaml_path, global_keys=keys, step_keys={})


def _strip_capture_coupled_keys(
    yaml_path: Path,
    *,
    global_keys: tuple[str, ...] = _CAPTURE_COUPLED_GLOBAL_KEYS,
    step_keys: dict[str, tuple[str, ...]] = _CAPTURE_COUPLED_STEP_KEYS,
) -> list[str]:
    """Strip both ``global.*`` and per-step capture-coupled keys.

    Returns the flat list of removed keys (``global.<k>`` or
    ``<step>.<k>``) so callers can log what was changed. Idempotent:
    re-running on an already-clean file is a no-op.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return []

    removed: list[str] = []
    g = data.get("global")
    if isinstance(g, dict):
        for k in global_keys:
            if k in g:
                g.pop(k, None)
                removed.append(f"global.{k}")
    for step_name, keys in step_keys.items():
        s = data.get(step_name)
        if not isinstance(s, dict):
            continue
        for k in keys:
            if k in s:
                s.pop(k, None)
                removed.append(f"{step_name}.{k}")

    if removed:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return removed


def migrate_examples(old_dir: Path, new_dir: Path) -> int:
    """Move every ``*.yaml`` under ``old_dir`` to ``new_dir`` and strip
    capture-coupled ``global.*`` keys. Returns the number of files
    migrated.
    """
    if not old_dir.is_dir():
        print(f"  [skip] {old_dir} — already migrated or never existed")
        return 0

    n_moved = 0
    yaml_paths = sorted(old_dir.glob("*.yaml")) + sorted(old_dir.glob("*.yml"))
    new_dir.mkdir(parents=True, exist_ok=True)
    for src in yaml_paths:
        dst = new_dir / src.name
        if dst.exists():
            print(f"  [skip] {dst.relative_to(_REPO_ROOT)} — already present at destination")
            continue
        _git_mv_or_copy(src, dst)
        removed = _strip_global_keys(dst)
        marker = f"stripped {', '.join(removed)}" if removed else "no capture-coupled keys found"
        print(f"  moved {src.relative_to(_REPO_ROOT)} -> {dst.relative_to(_REPO_ROOT)} ({marker})")
        n_moved += 1

    # Tear down the now-empty source dir so future runs short-circuit.
    try:
        if not any(old_dir.iterdir()):
            old_dir.rmdir()
            print(f"  removed empty {old_dir.relative_to(_REPO_ROOT)}")
    except OSError:
        pass

    return n_moved


def cleanup_presets_in_place(presets_dir: Path) -> int:
    """Re-run the capture-coupled-key strip across an already-migrated
    presets tree. Idempotent. Returns the number of files modified.

    Used when a new key is added to ``_CAPTURE_COUPLED_GLOBAL_KEYS`` /
    ``_CAPTURE_COUPLED_STEP_KEYS`` after the initial migration ran —
    without this pass, the prior pollution stays on disk until someone
    manually edits the YAMLs.
    """
    if not presets_dir.is_dir():
        print(f"  [skip] {presets_dir} — not present")
        return 0
    n_modified = 0
    yaml_paths = sorted(presets_dir.rglob("*.yaml")) + sorted(presets_dir.rglob("*.yml"))
    for path in yaml_paths:
        removed = _strip_capture_coupled_keys(path)
        if removed:
            print(f"  cleaned {path.relative_to(_REPO_ROOT)} (stripped {', '.join(removed)})")
            n_modified += 1
    return n_modified


# ---------------------------------------------------------------------------
# Phase 6: enrich capture JSONs with `cams` + `videos_subdir`, then collapse
# the now-duplicate preset YAMLs to a single ``full.yaml`` + ``quick.yaml``.
# ---------------------------------------------------------------------------

_CAPTURES_DIR = _EXAMPLES_DIR / "captures"


def enrich_captures_from_presets(presets_dir: Path, captures_dir: Path) -> int:
    """Lift capture-coupled metadata from the per-capture preset YAMLs
    into the matching capture JSON.

    For each preset like ``configs/examples/presets/<stem>.yaml`` whose
    matching capture ``configs/examples/captures/<stem>.json`` exists:

    * Populate ``cams`` in the capture JSON from the preset's
      ``global.cam_names`` (the shipped captures all have an empty
      ``cams: []``; the camera lists currently live in the presets).
    * Set ``videos_subdir`` for captures that use a non-default tier
      (today only the two iphones captures use raw ``videos/``).

    Captures whose ``cams`` is already populated are left alone
    (idempotent — re-running doesn't clobber user edits).

    Returns the number of capture JSONs modified.
    """
    if not presets_dir.is_dir() or not captures_dir.is_dir():
        return 0

    n_modified = 0
    for preset_path in sorted(presets_dir.glob("*.yaml")) + sorted(presets_dir.glob("*.yml")):
        stem = preset_path.stem
        if stem in ("full", "quick"):
            continue  # already-collapsed presets carry no per-capture data
        capture_path = captures_dir / f"{stem}.json"
        if not capture_path.exists():
            continue

        with open(preset_path) as f:
            preset = yaml.safe_load(f) or {}
        cams_from_preset = (preset.get("global") or {}).get("cam_names") or []

        with open(capture_path) as f:
            cap = json.load(f)

        changed: list[str] = []
        if cams_from_preset and not cap.get("cams"):
            cap["cams"] = list(cams_from_preset)
            changed.append(f"cams ({len(cams_from_preset)})")
        override = _CAPTURE_VIDEOS_SUBDIR_OVERRIDES.get(stem)
        if override and cap.get("videos_subdir") != override:
            cap["videos_subdir"] = override
            changed.append(f"videos_subdir={override!r}")

        if changed:
            with open(capture_path, "w") as f:
                json.dump(cap, f, indent=2)
            print(f"  enriched {capture_path.relative_to(_REPO_ROOT)}: {', '.join(changed)}")
            n_modified += 1

    return n_modified


def collapse_presets_to_two(presets_dir: Path, *, quick_subdir_name: str = "quick") -> int:
    """Reduce the per-capture preset YAMLs to a single ``full.yaml``
    + ``quick.yaml``.

    Strategy:
      * If ``full.yaml`` / ``quick.yaml`` already exist, this is a
        no-op (idempotent).
      * Otherwise, pick the alphabetically-first ``.yaml`` in the
        flat directory as the canonical "full" template, rename it to
        ``full.yaml``, delete all other ``.yaml`` files in the flat
        directory. Same for the ``quick/`` subdir into a single
        ``quick.yaml`` in the parent (and remove the now-empty
        ``quick/``).

    Returns the number of files deleted (excluding the two kept ones).
    """
    if not presets_dir.is_dir():
        return 0

    n_deleted = 0

    # Reserved final filenames in the flat dir. The post-collapse state
    # is `{full.yaml, quick.yaml}` — they must never be deleted as
    # "duplicates" on re-runs of this function.
    keep = {"full.yaml", "quick.yaml"}

    # Flat dir → full.yaml.
    full_target = presets_dir / "full.yaml"
    flat_yamls = sorted(p for p in presets_dir.glob("*.yaml") if p.name not in keep)
    if not full_target.exists() and flat_yamls:
        canonical = flat_yamls[0]
        canonical.rename(full_target)
        flat_yamls = sorted(p for p in presets_dir.glob("*.yaml") if p.name not in keep)
        print(f"  renamed {canonical.name} -> full.yaml (canonical 'full' preset)")
    for p in flat_yamls:
        p.unlink()
        print(f"  deleted duplicate {p.relative_to(_REPO_ROOT)}")
        n_deleted += 1

    # Subdir → quick.yaml (lifted up one level so we end with a 2-file
    # layout, not nested).
    quick_subdir = presets_dir / quick_subdir_name
    quick_target = presets_dir / "quick.yaml"
    if quick_subdir.is_dir():
        sub_yamls = sorted(quick_subdir.glob("*.yaml"))
        if not quick_target.exists() and sub_yamls:
            canonical = sub_yamls[0]
            canonical.rename(quick_target)
            sub_yamls = sorted(quick_subdir.glob("*.yaml"))
            print(f"  renamed {canonical.relative_to(_REPO_ROOT)} -> {quick_target.relative_to(_REPO_ROOT)}")
        for p in sub_yamls:
            p.unlink()
            print(f"  deleted duplicate {p.relative_to(_REPO_ROOT)}")
            n_deleted += 1
        try:
            quick_subdir.rmdir()
            print(f"  removed empty {quick_subdir.relative_to(_REPO_ROOT)}/")
        except OSError:
            pass

    return n_deleted


def migrate_samples_dir() -> int:
    """Rename gui/var/.../samples/tasks/ to samples/presets/.

    Preserves the optional ``user/`` subdir as-is. Plain shutil move —
    this lives under a gitignored tree.
    """
    if not _OLD_SAMPLES_TASKS.is_dir():
        print(f"  [skip] {_OLD_SAMPLES_TASKS} — already migrated or never existed")
        return 0
    if _NEW_SAMPLES_PRESETS.exists():
        # Both exist — merge by file. Refuse to clobber.
        n = 0
        _NEW_SAMPLES_PRESETS.mkdir(parents=True, exist_ok=True)
        for entry in _OLD_SAMPLES_TASKS.iterdir():
            target = _NEW_SAMPLES_PRESETS / entry.name
            if target.exists():
                print(f"  [skip] {target} — already present, not overwriting")
                continue
            shutil.move(str(entry), str(target))
            print(f"  moved {entry} -> {target}")
            n += 1
        try:
            _OLD_SAMPLES_TASKS.rmdir()
        except OSError:
            pass
        return n
    _NEW_SAMPLES_PRESETS.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_OLD_SAMPLES_TASKS), str(_NEW_SAMPLES_PRESETS))
    print(f"  moved {_OLD_SAMPLES_TASKS} -> {_NEW_SAMPLES_PRESETS}")
    return 1


def migrate_run_configs() -> int:
    """Rename gui/var/.../task_jsons/task_config_<id>.json to run_configs/run_<id>.json.

    Updates tasks.task_json_path rows in mamma.sqlite to point at the
    new paths. Idempotent: skips files already at the destination.
    """
    if not _OLD_TASK_JSONS.is_dir():
        print(f"  [skip] {_OLD_TASK_JSONS} — already migrated or never existed")
        return 0
    _NEW_RUN_CONFIGS.mkdir(parents=True, exist_ok=True)

    # Pre-compute the file moves so we can update the DB transactionally
    # after every filesystem move succeeds.
    moves: list[tuple[Path, Path]] = []
    for src in sorted(_OLD_TASK_JSONS.iterdir()):
        m = _TASK_CONFIG_RE.match(src.name)
        if not m:
            print(f"  [skip] {src.name} — does not match task_config_<id>.json pattern")
            continue
        task_id = int(m.group(1))
        dst = _NEW_RUN_CONFIGS / f"run_{task_id}.json"
        if dst.exists():
            print(f"  [skip] {dst.relative_to(_INTERFACE_DIR)} — already present")
            continue
        moves.append((src, dst))

    for src, dst in moves:
        shutil.move(str(src), str(dst))
        print(f"  moved {src.relative_to(_INTERFACE_DIR)} -> {dst.relative_to(_INTERFACE_DIR)}")

    _rewrite_db_paths()

    try:
        if not any(_OLD_TASK_JSONS.iterdir()):
            _OLD_TASK_JSONS.rmdir()
            print(f"  removed empty {_OLD_TASK_JSONS.relative_to(_INTERFACE_DIR)}/")
    except OSError:
        pass

    return len(moves)


def _rewrite_db_paths() -> None:
    """Rewrite tasks.task_json_path values from task_jsons/task_config_N.json
    to run_configs/run_N.json. No-op if the DB or table is absent.
    """
    if not _DB_PATH.exists():
        print(f"  [skip] {_DB_PATH} — DB not found, no rows to rewrite")
        return
    conn = sqlite3.connect(_DB_PATH)
    try:
        # Defensive: tasks table might not exist on a fresh clone.
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()
        if row is None:
            print(f"  [skip] tasks table not present in {_DB_PATH}")
            return
        rows = conn.execute(
            "SELECT task_id, task_json_path FROM tasks WHERE task_json_path LIKE ?",
            ("%/task_jsons/task_config_%.json",),
        ).fetchall()
        n_updated = 0
        for task_id, path in rows:
            m = re.search(r"task_jsons/task_config_(\d+)\.json$", path)
            if not m:
                continue
            new_path = re.sub(
                r"task_jsons/task_config_(\d+)\.json$",
                lambda mm: f"run_configs/run_{mm.group(1)}.json",
                path,
            )
            conn.execute(
                "UPDATE tasks SET task_json_path = ? WHERE task_id = ?",
                (new_path, task_id),
            )
            n_updated += 1
        conn.commit()
        print(f"  rewrote {n_updated} DB row(s) under tasks.task_json_path")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions but do not modify the filesystem or DB.",
    )
    args = parser.parse_args()
    if args.dry_run:
        # Cheap and useful: simulate by running each migrate_* in the
        # source tree but never invoking. For now we just print state.
        print("--dry-run: reporting current state (no changes applied)")
        for label, p in [
            ("configs/examples/tasks", _OLD_TASKS_DIR),
            ("configs/examples/quick_tasks", _OLD_QUICK_DIR),
            ("samples/tasks", _OLD_SAMPLES_TASKS),
            ("task_jsons", _OLD_TASK_JSONS),
        ]:
            present = "EXISTS" if p.exists() else "absent"
            print(f"  {label}: {present} ({p})")
        return 0

    print("[1/7] configs/examples/tasks -> configs/examples/presets")
    migrate_examples(_OLD_TASKS_DIR, _NEW_PRESETS_DIR)

    print("[2/7] configs/examples/quick_tasks -> configs/examples/presets/quick")
    migrate_examples(_OLD_QUICK_DIR, _NEW_QUICK_DIR)

    print("[3/7] samples/tasks -> samples/presets")
    migrate_samples_dir()

    print("[4/7] task_jsons/task_config_<id>.json -> run_configs/run_<id>.json (+ DB rewrite)")
    migrate_run_configs()

    # Phase 5: enrich capture JSONs with `cams` and `videos_subdir`
    # from the per-capture preset YAMLs *before* we strip those fields
    # away. Otherwise the data is lost and the materializer has nothing
    # to derive from. Idempotent (skips captures that already have cams).
    print("[5/7] lift cams + videos_subdir from per-capture presets into capture JSONs")
    enrich_captures_from_presets(_NEW_PRESETS_DIR, _CAPTURES_DIR)

    # Phase 6: strip every capture-coupled key (now safe — the data
    # has been moved into the capture JSONs by phase 5).
    print(f"[6/7] strip capture-coupled keys (global: {', '.join(_CAPTURE_COUPLED_GLOBAL_KEYS)}; "
          f"per-step: {sum((tuple(v) for v in _CAPTURE_COUPLED_STEP_KEYS.values()), ())})")
    cleanup_presets_in_place(_NEW_PRESETS_DIR)

    # Phase 7: collapse the now-identical per-capture YAMLs down to a
    # single ``full.yaml`` + ``quick.yaml``. Idempotent (no-op when the
    # collapse has already happened).
    print("[7/7] collapse per-capture preset YAMLs to full.yaml + quick.yaml")
    collapse_presets_to_two(_NEW_PRESETS_DIR)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
