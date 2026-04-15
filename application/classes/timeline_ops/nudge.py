"""Nudge actions by value (pos delta) or time (ms delta).

Three scopes:
- ``nudge_selection_value`` / ``nudge_selection_time``: only the multi-selected points.
- ``nudge_all_time``: every point on the timeline.
- ``nudge_chapter_time``: every point inside the context-selected chapter(s).
"""

from bisect import bisect_left, bisect_right

from common.frame_utils import frame_to_ms


def nudge_selection_value(tl, delta: int) -> None:
    actions = tl._get_actions()
    if not actions:
        return
    snap = tl.app.app_state_ui.snap_to_grid_pos
    actual_delta = delta * (snap if snap > 0 else 1)
    resolved = tl._resolve_selected_indices()
    for idx in resolved:
        actions[idx]['pos'] = max(0, min(100, actions[idx]['pos'] + actual_delta))
    tl.multi_selected_action_indices = {tl._action_key(actions[idx]) for idx in resolved}
    fs, axis = tl._get_target_funscript_details()
    if fs:
        fs._invalidate_cache(axis or 'both')
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Nudge Value")
    tl.invalidate_cache()
    from application.classes.undo_manager import NudgeValuesCmd
    tl.app.undo_manager.push_done(NudgeValuesCmd(tl.timeline_num, resolved, actual_delta))


def nudge_selection_time(tl, delta_ms: int) -> None:
    actions = tl._get_actions()
    if not actions:
        return
    resolved = tl._resolve_selected_indices()
    indices = sorted(resolved, reverse=(delta_ms > 0))
    for idx in indices:
        if idx < len(actions):
            prev_limit = actions[idx - 1]['at'] + 1 if idx > 0 else 0
            next_limit = actions[idx + 1]['at'] - 1 if idx < len(actions) - 1 else float('inf')
            new_at = actions[idx]['at'] + delta_ms
            actions[idx]['at'] = int(max(prev_limit, min(next_limit, new_at)))
    tl.multi_selected_action_indices = {tl._action_key(actions[idx]) for idx in resolved}
    fs, axis = tl._get_target_funscript_details()
    if fs:
        fs._invalidate_cache(axis or 'both')
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Nudge Time")
    tl.invalidate_cache()
    from application.classes.undo_manager import NudgeTimesCmd
    tl.app.undo_manager.push_done(NudgeTimesCmd(tl.timeline_num, resolved, delta_ms))


def nudge_all_time(tl, frames: int) -> None:
    """Shift every point on the timeline by ``frames`` frames (undoable)."""
    actions = tl._get_actions()
    if not actions:
        return
    proc = tl.app.processor
    if not proc or not proc.video_info or proc.fps <= 0:
        return
    fps = proc.fps
    delta_ms = frame_to_ms(frames, fps)
    actions_before = list(actions)
    for a in actions:
        a['at'] = max(0, a['at'] + delta_ms)
    fs, axis = tl._get_target_funscript_details()
    if fs and axis:
        fs._invalidate_cache(axis)
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Nudge All Points")
    tl.invalidate_cache()
    actions_after = list(tl._get_actions() or [])
    from application.classes.undo_manager import BulkReplaceCmd
    tl.app.undo_manager.push_done(BulkReplaceCmd(
        tl.timeline_num, actions_before, actions_after,
        f"Nudge All Points (T{tl.timeline_num})"))


def nudge_chapter_time(tl, frames: int) -> None:
    """Shift points inside the context-selected chapter(s) by ``frames``."""
    selected_chapters = []
    gui = tl.app.gui_instance
    if gui and hasattr(gui, 'video_navigation_ui'):
        nav_ui = gui.video_navigation_ui
        if nav_ui and hasattr(nav_ui, 'context_selected_chapters'):
            selected_chapters = nav_ui.context_selected_chapters
    if not selected_chapters:
        if tl.logger:
            tl.logger.info("No chapter selected for nudging",
                           extra={'status_message': True})
        return
    actions = tl._get_actions()
    if not actions:
        return
    proc = tl.app.processor
    if not proc or not proc.video_info or proc.fps <= 0:
        return
    fps = proc.fps
    actions_before = list(actions)
    delta_ms = frame_to_ms(frames, fps)
    total = 0
    for chapter in selected_chapters:
        start_ms = frame_to_ms(chapter.start_frame_id, fps)
        end_ms = frame_to_ms(chapter.end_frame_id, fps)
        timestamps = tl._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s = bisect_left(timestamps, start_ms)
        e = bisect_right(timestamps, end_ms)
        for i in range(s, e):
            actions[i]['at'] = max(0, actions[i]['at'] + delta_ms)
            total += 1
    if total == 0:
        if tl.logger:
            tl.logger.info("No points found in selected chapter(s)",
                           extra={'status_message': True})
        return
    fs, axis = tl._get_target_funscript_details()
    if fs and axis:
        fs._invalidate_cache(axis)
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Nudge Chapter Points")
    tl.invalidate_cache()
    actions_after = list(tl._get_actions() or [])
    from application.classes.undo_manager import BulkReplaceCmd
    tl.app.undo_manager.push_done(BulkReplaceCmd(
        tl.timeline_num, actions_before, actions_after,
        f"Nudge Chapter Points (T{tl.timeline_num})"))
