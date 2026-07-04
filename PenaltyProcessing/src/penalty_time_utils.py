from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional


def canonical_team_name(name: str) -> str:
    """Canonicalize team names for filename matching."""
    if name is None:
        return ""

    s = name.strip()
    s = (
        s.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )
    s = s.replace("-", "_").replace(" ", "_")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def parse_penalty_local_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_position_local_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip().strip('"')
    for fmt in (
        "%m/%d/%Y, %I:%M:%S.%f %p",
        "%m/%d/%Y, %I:%M:%S %p",
        "%d.%m.%Y, %H:%M:%S.%f",
        "%d.%m.%Y, %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def try_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().strip('"')
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def try_int(value: str) -> Optional[int]:
    f = try_float(value)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError):
        return None