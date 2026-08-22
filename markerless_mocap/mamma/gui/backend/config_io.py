"""Format-agnostic loader for capture, preset, and run-config files.

Captures, presets, and run configs may be written in either JSON or
YAML; the format choice is per file (extension-driven), not per-purpose.
Convention:

- ``configs/examples/captures/*.json`` (machine-generated; sequences-heavy).
- ``configs/examples/presets/*.yaml`` (human-edited presets; comments + multi-line lists).
- GUI-saved user files: ``gui/var/interface/.../*.{json,yaml}``.
- GUI-frozen run configs: ``gui/var/interface/run_configs/run_<id>.json``.

Both the inference runner (:mod:`inference.config`) and the GUI use
this dispatch so the same file works in either entry point.
"""
from __future__ import annotations

import json
import os
from typing import Any


_JSON_SUFFIXES = (".json",)
_YAML_SUFFIXES = (".yaml", ".yml")


def load_config_file(path: str) -> dict[str, Any]:
    """Load a config file by extension. Returns the parsed dict.

    Raises:
        ValueError: unknown suffix.
        OSError: file not readable.
        json.JSONDecodeError / yaml.YAMLError: parse failure.
    """
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if suffix in _YAML_SUFFIXES:
            import yaml  # local import: pyyaml is a runtime dep
            data = yaml.safe_load(f)
        elif suffix in _JSON_SUFFIXES:
            data = json.load(f)
        else:
            raise ValueError(
                f"{path}: unrecognized suffix {suffix!r}; "
                f"expected one of {_JSON_SUFFIXES + _YAML_SUFFIXES}"
            )
    if data is None:
        # YAML empty file → None; JSON empty file → JSONDecodeError.
        # Normalize both to empty dict so callers can do safe .get().
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping, got {type(data).__name__}"
        )
    return data


def is_config_file(name: str) -> bool:
    """True iff the filename has a config-file extension we accept."""
    return name.lower().endswith(_JSON_SUFFIXES + _YAML_SUFFIXES)
