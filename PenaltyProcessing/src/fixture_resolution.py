"""
Fixture resolution: map penalty rows to their position-tracking files.

This module is responsible for figuring out which ``*_2_phases_positions.csv``
file corresponds to a given penalty row in ``penalties.csv``. It does this by
canonicalizing the home/away team names from the penalty row and matching them
against the team names embedded in the position file names.

It also provides helpers for:
- Building an index of all available fixture files.
- Loading the penalties CSV.
- Filtering out penalty rows that are known to be unreliable.
- Handling edge-case fixture names that do not follow the standard format.
- Creating the numbered output run folder.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from penalty_time_utils import canonical_team_name


def build_fixture_index(positions_dir: Path) -> Tuple[Dict[Tuple[str, str], List[Path]], List[str]]:
    """Build ordered fixture index from files in positions dir.

    Scans ``positions_dir`` for files matching the pattern
    ``*_2_phases_positions.csv`` and parses the home/away team names from the
    filename stem (the part before ``_2_phases_positions.csv``). The stem is
    expected to look like ``<home>_vs_<away>``.

    Args:
        positions_dir: Directory containing the position-tracking CSV files.

    Returns:
        A tuple ``(index, issues)`` where:
        - ``index`` maps a ``(home_key, away_key)`` tuple of canonical team
          names to a list of matching file paths (usually length 1).
        - ``issues`` is a list of human-readable strings describing files that
          could not be parsed (e.g. missing "_vs_" or empty team names).
    """
    index: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    issues: List[str] = []

    for path in sorted(positions_dir.glob("*_2_phases_positions.csv")):
        # Strip the fixed suffix to get the fixture part of the filename.
        stem = path.name[:-len("_2_phases_positions.csv")]
        if "_vs_" not in stem:
            issues.append(f"invalid_fixture_filename:{path.name}")
            continue
        # Split on the first "_vs_" -> home team and away team.
        home_raw, away_raw = stem.split("_vs_", 1)
        home_key = canonical_team_name(home_raw)
        away_key = canonical_team_name(away_raw)
        if not home_key or not away_key:
            issues.append(f"invalid_fixture_parts:{path.name}")
            continue
        index[(home_key, away_key)].append(path)

    return index, issues


def load_penalties(penalties_file: Path) -> List[Dict[str, str]]:
    """Load all rows from the penalties CSV as a list of dicts.

    Args:
        penalties_file: Path to the penalties.csv file (semicolon-delimited).

    Returns:
        A list of dicts, one per row, keyed by column name.
    """
    with penalties_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = [row for row in reader]
    return rows


def is_successful_penalty(row: Dict[str, str]) -> bool:
    """Return True if the penalty row represents a successful (scored) throw.

    A throw is considered successful when the "success" column is exactly "1".
    """
    return row.get("success", "").strip() == "1"


def should_skip_penalty_row(row: Dict[str, str]) -> bool:
    """Skip penalties with clock times that are known to be tracked incorrectly.

    Some penalties are recorded at clock times that are known to be unreliable
    in the source data (e.g. the very start of a period). These rows are
    filtered out to avoid producing garbage trajectories.

    Args:
        row: A single penalty row.

    Returns:
        True if the row should be skipped.
    """
    return row.get("game_clock", "").strip() in {"00:00", "0:00", "30:00", "60:00"}


def build_edge_case_mappings() -> Dict[Tuple[str, str], str]:
    """Build manual mappings for edge-case fixture names with unusual formatting.

    Some fixture files in the source data do not follow the standard
    ``<home>_vs_<away>`` naming convention (e.g. they are truncated or contain
    unusual characters). This function returns a hard-coded mapping from a
    canonical ``(home, away)`` team pair to a filename prefix that can be used
    to locate the correct file.

    Returns:
        A dict mapping ``(home_key, away_key)`` to a filename prefix.
    """
    return {
        ("hbw_balingen_weilstetten", "thsv_eisenach"): "HBW_Balingen-Weilstetten_vs_T",
        ("hbw_balingen_weilstetten", "frisch_auf_goeppingen"): "HBW_Balingen-Weilstetten_vs_F",
        ("hbw_balingen_weilstetten", "hsv_hamburg"): "HBW_Balingen-Weilstetten_vs_H",
        ("hbw_balingen_weilstetten", "sg_flensburg_handewitt"): "HBW_Balingen-Weilstetten_vs_S",
        ("sg_flensburg_handewitt", "sc_dhfk_leipzig"): "SG_Flensburg-Handewitt_vs_SC_",
        ("frisch_auf_goeppingen", "sc_dhfk_leipzig"): "Frisch_Auf!_Goeppingen_vs_SC_",
    }


def resolve_fixture_file(
    row: Dict[str, str],
    fixture_index: Dict[Tuple[str, str], List[Path]],
    edge_cases: Dict[Tuple[str, str], str],
) -> Tuple[Optional[Path], List[str]]:
    """Resolve the position file for a single penalty row.

    Steps:
    1. Canonicalize the home/away team names from the row.
    2. Look up the canonical pair in the fixture index.
    3. If exactly one file matches, return it.
    4. If no file matches, try the edge-case mappings (prefix search).
    5. Otherwise return an error describing why resolution failed.

    Args:
        row: A single penalty row (must contain "home_team" and "away_team").
        fixture_index: Index built by ``build_fixture_index``.
        edge_cases: Mapping built by ``build_edge_case_mappings``.

    Returns:
        A tuple ``(path, errors)`` where ``path`` is the resolved file (or
        None) and ``errors`` is a list of issue strings (empty on success).
    """
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    key = (canonical_team_name(home), canonical_team_name(away))
    candidates = fixture_index.get(key, [])

    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) == 0:
        # No exact match -> try the manual edge-case prefix mapping.
        if key in edge_cases:
            prefix = edge_cases[key]
            for all_paths in fixture_index.values():
                for path in all_paths:
                    if prefix in path.name:
                        return path, []
        return None, ["no_fixture_match"]
    return None, [f"fixture_not_unique:{'|'.join(p.name for p in candidates)}"]


def create_run_folder(base_dir: Path) -> Path:
    """Create the next numbered run folder inside penalty_trajectories directory.

    The folder is named ``run_<N>__<timestamp>`` where ``N`` is the next
    available integer (max existing run id + 1) and ``timestamp`` is the
    current local time formatted as ``DD_MM_YYYY_HH_MM``.

    Args:
        base_dir: Base output directory. The run folder is created directly
            inside it.

    Returns:
        The path to the newly created run folder.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Collect all existing run ids from folder names like "run_3__01_02_2025_10_30".
    existing_ids = []
    for path in base_dir.glob("run_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"run_(\d+)(?:__.*)?", path.name)
        if match:
            existing_ids.append(int(match.group(1)))

    now = datetime.now().strftime("%d_%m_%Y_%H_%M")

    next_id = max(existing_ids, default=0) + 1
    run_dir = base_dir / f"run_{next_id}__{now}"
    run_dir.mkdir(exist_ok=True)

    return run_dir