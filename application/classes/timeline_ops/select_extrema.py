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
    if not actions or len(actions) < 2:
        return
    s, e = _resolve_range(tl)
    if s is None:
        return
    s = max(0, s)
    e = min(len(actions), e)
    keys = set()
    n = len(actions)
    for i in range(s, e):
        cur_v = actions[i]['pos']
        prev_v = actions[i - 1]['pos'] if i > 0 else None
        next_v = actions[i + 1]['pos'] if i < n - 1 else None
        # Endpoints: only the one neighbor decides. Compares strictly so
        # a flat run isn't all peaks AND all valleys.
        if prev_v is None:
            is_peak = next_v is not None and cur_v > next_v
            is_valley = next_v is not None and cur_v < next_v
        elif next_v is None:
            is_peak = cur_v > prev_v
            is_valley = cur_v < prev_v
        else:
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
