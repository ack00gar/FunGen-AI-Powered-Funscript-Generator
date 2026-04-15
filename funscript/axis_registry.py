"""
Canonical axis registry for multi-axis funscript support.

Single source of truth mapping between axis identifiers across all layers:
funscript files, TCode protocol, timeline numbers, and device control.
"""

from enum import Enum
from typing import Optional, Dict


class FunscriptAxis(Enum):
    """Canonical axis names matching community conventions."""
    STROKE = "stroke"   # Primary linear (L0) — default T1
    ROLL   = "roll"     # R1 — default T2
    PITCH  = "pitch"    # R2
    TWIST  = "twist"    # R0
    SWAY   = "sway"     # L1 (left/right)
    SURGE  = "surge"    # L2 (front/back)
    VIB    = "vib"      # V0
    PUMP   = "pump"     # V1 (suction/pump)


# axis file suffix (inserted between basename and .funscript)
AXIS_FILE_SUFFIX: Dict[FunscriptAxis, str] = {
    FunscriptAxis.STROKE: "",            # basename.funscript (no suffix for primary)
    FunscriptAxis.ROLL:   ".roll",       # basename.roll.funscript
    FunscriptAxis.PITCH:  ".pitch",
    FunscriptAxis.TWIST:  ".twist",
    FunscriptAxis.SWAY:   ".sway",
    FunscriptAxis.SURGE:  ".surge",
    FunscriptAxis.VIB:    ".vib",
    FunscriptAxis.PUMP:   ".pump",
}

# TCode protocol mapping
AXIS_TCODE: Dict[FunscriptAxis, str] = {
    FunscriptAxis.STROKE: "L0",
    FunscriptAxis.SWAY:   "L1",
    FunscriptAxis.SURGE:  "L2",
    FunscriptAxis.TWIST:  "R0",
    FunscriptAxis.ROLL:   "R1",
    FunscriptAxis.PITCH:  "R2",
    FunscriptAxis.VIB:    "V0",
    FunscriptAxis.PUMP:   "V1",
}

# Default timeline assignment — used at startup before any tracker is selected.
# When a tracker is activated, TrackerManager applies the tracker's declared
# primary_axis / secondary_axis from its metadata, overriding these defaults.
DEFAULT_TIMELINE_AXIS: Dict[int, FunscriptAxis] = {
    1: FunscriptAxis.STROKE,
    2: FunscriptAxis.ROLL,
}

# All known axis file suffixes for discovery
_SUFFIX_TO_AXIS: Dict[str, FunscriptAxis] = {v: k for k, v in AXIS_FILE_SUFFIX.items() if v}
_TCODE_TO_AXIS: Dict[str, FunscriptAxis] = {v: k for k, v in AXIS_TCODE.items()}


def axis_from_file_suffix(suffix: str) -> Optional[FunscriptAxis]:
    """Look up a FunscriptAxis from its axis file suffix (e.g. '.roll' -> ROLL).

    Returns None for unrecognized suffixes.
    """
    return _SUFFIX_TO_AXIS.get(suffix)


def axis_from_tcode(tcode: str) -> Optional[FunscriptAxis]:
    """Look up a FunscriptAxis from a TCode channel ID (e.g. 'R1' -> ROLL)."""
    return _TCODE_TO_AXIS.get(tcode)


def file_suffix_for_axis(axis_name: str) -> str:
    """Return the axis file suffix for a given axis name string.

    For known FunscriptAxis values, returns the standard suffix.
    For custom/unknown axis names, returns '.{axis_name}'.
    """
    try:
        axis = FunscriptAxis(axis_name)
        return AXIS_FILE_SUFFIX[axis]
    except (ValueError, KeyError):
        return f".{axis_name}"


def tcode_for_axis(axis_name: str) -> Optional[str]:
    """Return the TCode channel ID for a given axis name string, or None if unknown."""
    try:
        axis = FunscriptAxis(axis_name)
        return AXIS_TCODE.get(axis)
    except ValueError:
        return None


def all_known_suffixes() -> list:
    """Return all known axis file suffixes (excluding empty string for stroke)."""
    return [s for s in AXIS_FILE_SUFFIX.values() if s]
