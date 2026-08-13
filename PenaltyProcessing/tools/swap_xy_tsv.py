"""Rotate X/Y coordinates by 90 degrees around the Z axis in Qualisys TSV files.

Reads a TSV file, applies the rotation
    newX = oldY
    newY = -oldX
    newZ = oldZ
to the X/Y columns in the Frame header row and in every data row, and writes
the result to a new file with the suffix ``_XY_swapped`` appended to the stem.

This is a 90-degree rotation around the Z axis that preserves right-handedness,
so the thrower ends up throwing in the positive X direction.

The script is format-agnostic: it detects X/Y columns from the header names
(``... X`` / ``... Y`` or bare ``X`` / ``Y``) and transforms the corresponding
values in each data row. This works for skeleton (8 columns per segment),
body markers (3 columns per marker), 6DOF ball markers (3 columns per marker),
and 6D body files (X/Y/Z followed by Roll/Pitch/Yaw/Residual/Rot).

Usage
-----
python3 tools/swap_xy_tsv.py path/to/file.tsv
python3 tools/swap_xy_tsv.py path/to/file1.tsv path/to/file2.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _is_x_header(name: str) -> bool:
    stripped = name.strip()
    return stripped.endswith(" X") or stripped == "X"


def _is_y_header(name: str) -> bool:
    stripped = name.strip()
    return stripped.endswith(" Y") or stripped == "Y"


def _swap_header_name(name: str) -> str:
    stripped = name.strip()
    if _is_x_header(stripped):
        if stripped.endswith(" X"):
            return stripped[:-2] + " Y"
        return "Y"
    if _is_y_header(stripped):
        if stripped.endswith(" Y"):
            return stripped[:-2] + " X"
        return "X"
    return name


def _find_xy_columns(header_row: Sequence[str]) -> Tuple[List[int], List[int]]:
    x_cols: List[int] = []
    y_cols: List[int] = []
    for idx, name in enumerate(header_row):
        if _is_x_header(name):
            x_cols.append(idx)
        elif _is_y_header(name):
            y_cols.append(idx)
    return x_cols, y_cols


def swap_xy_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Rotate X/Y by 90 degrees around Z in a TSV file and write the result.

    Applies ``newX = oldY``, ``newY = -oldX``, ``newZ = oldZ`` to every data
    row. If ``output_path`` is None, the output is written next to the input
    with the suffix ``_XY_swapped`` appended to the stem.
    """
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_XY_swapped" + input_path.suffix)

    x_cols: List[int] = []
    y_cols: List[int] = []

    with input_path.open("r", encoding="utf-8", errors="replace") as src, \
         output_path.open("w", encoding="utf-8", newline="") as dst:
        for line in src:
            line = line.rstrip("\r\n")
            if not line:
                dst.write("\n")
                continue
            row = line.split("\t")
            if row[0].strip() == "Frame":
                x_cols, y_cols = _find_xy_columns(row)
                for idx in x_cols:
                    row[idx] = _swap_header_name(row[idx])
                for idx in y_cols:
                    row[idx] = _swap_header_name(row[idx])
                dst.write("\t".join(row) + "\n")
                continue

            if x_cols and y_cols:
                for x_idx, y_idx in zip(x_cols, y_cols):
                    if x_idx < len(row) and y_idx < len(row):
                        try:
                            old_x = float(row[x_idx])
                            old_y = float(row[y_idx])
                        except ValueError:
                            continue
                        row[x_idx] = str(old_y)
                        row[y_idx] = str(-old_x)
            dst.write("\t".join(row) + "\n")

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Swap X and Y coordinates in Qualisys TSV mocap files."
    )
    parser.add_argument("files", nargs="+", type=Path, help="Input TSV files to swap")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in args.files:
        output = swap_xy_file(path)
        print(f"Swapped {path} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())