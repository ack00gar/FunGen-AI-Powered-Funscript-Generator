"""Select peaks, valleys, or both within a range.

Range priority: existing selection > context-selected chapter(s) > full timeline.
"""

from bisect import bisect_left, bisect_right

from common.frame_utils import frame_to_ms


def _resolve_range(tl):
    actions = tl._get_actions()
    if not actions:
        return None, None
    selected = tl._resolve_selected_indices() if tl.multi_selected_action_indices else None
    if selected and len(selected) >= 2:
        return min(selected), max(selected) + 1
    gui = tl.app.gui_instance
    selected_chapters = []
    if gui and hasattr(gui, 'video_navigation_ui'):
        nav_ui = gui.video_navigation_ui
        if nav_ui and hasattr(nav_ui, 'context_selected_chapters'):
            selected_chapters = nav_ui.context_selected_chapters
    proc = tl.app.processor
    if selected_chapters and proc and proc.fps > 0:
        timestamps = tl._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_min, e_max = len(actions), 0
        for ch in selected_chapters:
            start_ms = frame_to_ms(ch.start_frame_id, proc.fps)
            end_ms = frame_to_ms(ch.end_frame_id, proc.fps)
            s = bisect_left(timestamps, start_ms)
            e = bisect_right(timestamps, end_ms)
            if e > s:
                s_min = min(s_min, s)
                e_max = max(e_max, e)
        if e_max > s_min:
            return s_min, e_max
    return 0, len(actions)


def select_extrema(tl, mode: str) -> None:
    if mode not in ('top', 'bottom', 'both'):
        return
    actions = tl._get_actions()
    if not actions or len(actions) < 3:
        return
    s, e = _resolve_range(tl)
    if s is None:
        return
    s = max(1, s)
    e = min(len(actions) - 1, e)
    keys = set()
    for i in range(s, e):
        prev_v = actions[i - 1]['pos']
        cur_v = actions[i]['pos']
        next_v = actions[i + 1]['pos']
        is_peak = (cur_v > prev_v) and (cur_v >= next_v)
        is_valley = (cur_v < prev_v) and (cur_v <= next_v)
        if mode == 'top' and is_peak:
            keys.add(tl._action_key(actions[i]))
        elif mode == 'bottom' and is_valley:
            keys.add(tl._action_key(actions[i]))
        elif mode == 'both' and (is_peak or is_valley):
            keys.add(tl._action_key(actions[i]))
    tl.multi_selected_action_indices = keys
    if tl.logger:
        label = {'top': 'peaks', 'bottom': 'valleys', 'both': 'extrema'}[mode]
        tl.logger.info(
            f"Selected {len(keys)} {label}",
            extra={'status_message': True},
        )
