from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from penalty_time_utils import canonical_team_name


def build_fixture_index(positions_dir: Path) -> Tuple[Dict[Tuple[str, str], List[Path]], List[str]]:
    """Build ordered fixture index from files in positions dir."""
    index: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    issues: List[str] = []

    for path in sorted(positions_dir.glob("*_2_phases_positions.csv")):
        stem = path.name[:-len("_2_phases_positions.csv")]
        if "_vs_" not in stem:
            issues.append(f"invalid_fixture_filename:{path.name}")
            continue
        home_raw, away_raw = stem.split("_vs_", 1)
        home_key = canonical_team_name(home_raw)
        away_key = canonical_team_name(away_raw)
        if not home_key or not away_key:
            issues.append(f"invalid_fixture_parts:{path.name}")
            continue
        index[(home_key, away_key)].append(path)

    return index, issues


def load_penalties(penalties_file: Path) -> List[Dict[str, str]]:
    with penalties_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = [row for row in reader]
    return rows


def is_successful_penalty(row: Dict[str, str]) -> bool:
    return row.get("success", "").strip() == "1"


def should_skip_penalty_row(row: Dict[str, str]) -> bool:
    """Skip penalties with clock times that are known to be tracked incorrectly."""
    return row.get("game_clock", "").strip() in {"30:00", "60:00"}


def build_edge_case_mappings() -> Dict[Tuple[str, str], str]:
    """Build manual mappings for edge-case fixture names with unusual formatting."""
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
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    key = (canonical_team_name(home), canonical_team_name(away))
    candidates = fixture_index.get(key, [])

    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) == 0:
        if key in edge_cases:
            prefix = edge_cases[key]
            for all_paths in fixture_index.values():
                for path in all_paths:
                    if prefix in path.name:
                        return path, []
        return None, ["no_fixture_match"]
    return None, [f"fixture_not_unique:{'|'.join(p.name for p in candidates)}"]


def create_run_folder(base_dir: Path) -> Path:
    """Create the next numbered run folder inside penalty_trajectories directory."""
    base_dir.mkdir(exist_ok=True)

    existing_ids = []
    for path in base_dir.glob("run_*"):
        if not path.is_dir():
            continue
        suffix = path.name[len("run_"):]
        if suffix.isdigit():
            existing_ids.append(int(suffix))
            
    now = datetime.now().strftime("%d_%m_%Y_%H_%M")

    next_id = max(existing_ids, default=0) + 1
    run_dir = base_dir / f"run_{next_id}__{now}"
    run_dir.mkdir(exist_ok=True)

    return run_dir