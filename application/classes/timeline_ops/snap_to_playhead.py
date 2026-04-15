"""Snap an action onto the current playhead.

If a selection exists, snaps the selected point closest to the playhead.
Otherwise snaps the nearest action overall. Bound to the M key.
"""
from bisect import bisect_left

from common.frame_utils import frame_to_ms


def snap_to_playhead(tl) -> None:
    actions = tl._get_actions()
    if not actions:
        return

    processor = tl.app.processor
    if not processor or processor.fps <= 0:
        return
    playhead_ms = frame_to_ms(processor.current_frame_index, processor.fps)

    resolved = tl._resolve_selected_indices()
    if resolved:
        best_idx = min(resolved, key=lambda i: abs(actions[i]['at'] - playhead_ms))
    else:
        timestamps = [a['at'] for a in actions]
        idx = bisect_left(timestamps, playhead_ms)
        best_idx = None
        best_dist = float('inf')
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(actions):
                dist = abs(actions[candidate]['at'] - playhead_ms)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = candidate
        if best_idx is None:
            return

    new_at = int(round(playhead_ms))
    if actions[best_idx]['at'] == new_at:
        return

    prev_limit = actions[best_idx - 1]['at'] + 1 if best_idx > 0 else 0
    next_limit = actions[best_idx + 1]['at'] - 1 if best_idx < len(actions) - 1 else float('inf')
    new_at = int(max(prev_limit, min(next_limit, new_at)))

    old_at = actions[best_idx]['at']
    actions[best_idx]['at'] = new_at

    fs, axis = tl._get_target_funscript_details()
    if fs:
        fs._invalidate_cache(axis or 'both')
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Snap to Playhead")
    tl.invalidate_cache()
    tl.app.project_manager.project_dirty = True
    tl.app.notify("Point snapped to playhead", "info", 1.5)

    from application.classes.undo_manager import SnapToPlayheadCmd
    tl.app.undo_manager.push_done(SnapToPlayheadCmd(tl.timeline_num, best_idx, old_at, new_at))
