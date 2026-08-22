"""Capture-config JSON loader.

Resolves sequence ids (integers from ``task.json``'s ``global.seq_ids``) to
sequence names (strings) by indexing into a capture-config JSON whose
``sequences`` map is keyed by zero-padded ids.

Sequence-name field: prefers ``name`` (current schema), falls back to
``ioi`` (legacy; "IOI" was a brand name of the original capture rig).
"""
from __future__ import annotations

import json
from typing import Iterable


def _seq_name(entry: dict) -> str | None:
    """Extract the sequence name from a sequences[*] entry.

    Prefers the new ``name`` key; falls back to the legacy ``ioi`` key.
    """
    if not isinstance(entry, dict):
        return None
    return entry.get("name") or entry.get("ioi")


class CaptureCfgJsonLoader:
    """Read a capture-config JSON and resolve sequence ids ↔ names."""

    def __init__(self, json_path: str) -> None:
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.sequences: dict = self.data.get("sequences", {})

    def get_sequence(self, seq_id: int) -> dict | None:
        return self.sequences.get(f"{seq_id:03d}")

    def get_seq_names(self, seq_ids: Iterable[int]) -> list[str]:
        names: list[str] = []
        for sid in seq_ids:
            seq = self.get_sequence(int(sid))
            name = _seq_name(seq) if seq else None
            if name:
                names.append(name)
        return names

    def all_seq_names(self) -> list[str]:
        return sorted(
            n for n in (_seq_name(v) for v in self.sequences.values()) if n
        )
