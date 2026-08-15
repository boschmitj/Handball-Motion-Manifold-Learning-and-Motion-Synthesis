"""
Shared parsing and normalization utilities.

This module provides small, reusable helpers used across the penalty pipeline:

- ``canonical_team_name``: normalizes team names so they can be matched
  between the penalties CSV and the position-file names.
- ``parse_penalty_local_time`` / ``parse_position_local_time``: parse the
  different timestamp formats used in the penalties CSV and the position
  tracking files.
- ``try_float`` / ``try_int``: safely parse numeric CSV fields, returning
  None for empty or invalid values.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional


def canonical_team_name(name: str) -> str:
    """Canonicalize team names for filename matching.

    Normalizes a team name so that equivalent names (e.g. with umlauts,
    hyphens, spaces, or different capitalization) map to the same key. This is
    needed because the penalties CSV and the position-file names may spell the
    same team differently.

    Transformations applied:
    - Umlauts are transliterated (ä -> ae, ö -> oe, ü -> ue, ß -> ss).
    - Hyphens and spaces become underscores.
    - Accented characters are decomposed and diacritics removed.
    - Any remaining non-alphanumeric characters become underscores.
    - Runs of underscores are collapsed and leading/trailing ones removed.
    - The result is lowercased.

    Args:
        name: Raw team name (may be None).

    Returns:
        The canonical lowercase key, or "" if the input was None/empty.
    """
    if name is None:
        return ""

    s = name.strip()
    # Transliterate German umlauts so "Balingen-Weilstetten" and
    # "Balingen_Weilstetten" match.
    s = (
        s.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )
    # Unify separators: hyphens and spaces become underscores.
    s = s.replace("-", "_").replace(" ", "_")
    # Decompose accented characters and strip combining diacritics
    # (e.g. "é" -> "e").
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Replace any remaining non-alphanumeric characters with underscores.
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    # Collapse repeated underscores and strip leading/trailing ones.
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def parse_penalty_local_time(value: str) -> Optional[datetime]:
    """Parse a timestamp from the penalties CSV.

    The penalties CSV uses ISO-like timestamps such as
    ``2025-01-15 14:30:00.123`` (with or without fractional seconds).

    Args:
        value: Raw timestamp string (may be empty).

    Returns:
        A datetime, or None if the value could not be parsed.
    """
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
    """Parse a timestamp from a position-tracking CSV.

    The position files use several possible formats, e.g. US-style
    ``01/15/2025, 2:30:00.123 PM`` or European-style
    ``15.01.2025, 14:30:00.123``. This tries each known format in order.

    Also handles ISO-8601 timestamps (e.g. ``2023-08-24T19:03:18.150``) which
    appear in the ``t_local`` field of serialized trajectory JSON.

    Args:
        value: Raw timestamp string (may be empty or wrapped in quotes).

    Returns:
        A datetime, or None if the value could not be parsed.
    """
    if not value:
        return None
    value = value.strip().strip('"')
    for fmt in (
        "%m/%d/%Y, %I:%M:%S.%f %p",
        "%m/%d/%Y, %I:%M:%S %p",
        "%d.%m.%Y, %H:%M:%S.%f",
        "%d.%m.%Y, %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def try_float(value: str) -> Optional[float]:
    """Safely parse a numeric CSV field as a float.

    Handles empty strings, surrounding quotes, and comma decimal separators
    (e.g. "12,5" -> 12.5). Returns None for empty or unparseable values.

    Args:
        value: Raw string from a CSV cell.

    Returns:
        The parsed float, or None if it could not be parsed.
    """
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
    """Safely parse a numeric CSV field as an int.

    First parses as a float (to handle decimal separators), then truncates to
    an int. Returns None for empty or unparseable values.

    Args:
        value: Raw string from a CSV cell.

    Returns:
        The parsed int, or None if it could not be parsed.
    """
    f = try_float(value)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError):
        return None