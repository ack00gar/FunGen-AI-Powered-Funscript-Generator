"""Tiny dedicated helpers for timeline edit ops.

Each operation lives in its own module so the giant
interactive_timeline.py only delegates and stays slim.

Helpers take the InteractiveFunscriptTimeline as `tl` so they can
read selection, mutate the action list, push undo, and trigger refresh
through the same channels the legacy in-class methods used.
"""
from .add_point import add_point
from .chapter_selection import select_points_in_chapter
from .clear_all import clear_all_points
from .clipboard import (
    copy_selection, copy_to_other,
    paste_actions, paste_actions_exact, paste_actions_replace,
)
from .delete_selected import delete_selected
from .equalize import equalize_selection
from .filter_selection import filter_selection
from .isolate import isolate_selection
from .live_tools import begin_live_drag, end_live_drag, live_range_extend, live_rdp_simplify
from .nudge import (
    nudge_all_time, nudge_chapter_time,
    nudge_selection_time, nudge_selection_value,
)
from .repeat_last_stroke import repeat_last_stroke
from .select_relative_to_playhead import select_relative_to_playhead
from .snap_to_playhead import snap_to_playhead

__all__ = [
    "add_point",
    "clear_all_points",
    "copy_selection",
    "copy_to_other",
    "paste_actions",
    "paste_actions_exact",
    "paste_actions_replace",
    "delete_selected",
    "equalize_selection",
    "filter_selection",
    "isolate_selection",
    "begin_live_drag",
    "end_live_drag",
    "live_range_extend",
    "live_rdp_simplify",
    "nudge_all_time",
    "nudge_chapter_time",
    "nudge_selection_time",
    "nudge_selection_value",
    "repeat_last_stroke",
    "select_points_in_chapter",
    "select_relative_to_playhead",
    "snap_to_playhead",
]
