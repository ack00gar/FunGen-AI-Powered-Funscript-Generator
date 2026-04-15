"""Select every point inside the context-selected chapter(s)."""

from bisect import bisect_left, bisect_right

from common.frame_utils import frame_to_ms


def select_points_in_chapter(tl) -> None:
    gui = tl.app.gui_instance
    selected_chapters = []
    if gui and hasattr(gui, 'video_navigation_ui'):
        nav_ui = gui.video_navigation_ui
        if nav_ui and hasattr(nav_ui, 'context_selected_chapters'):
            selected_chapters = nav_ui.context_selected_chapters
    if not selected_chapters:
        if tl.logger:
            tl.logger.info("No chapter selected", extra={'status_message': True})
        return
    actions = tl._get_actions()
    if not actions:
        return
    proc = tl.app.processor
    if not proc or not proc.video_info or proc.fps <= 0:
        return
    fps = proc.fps
    new_selection = set()
    for chapter in selected_chapters:
        start_ms = frame_to_ms(chapter.start_frame_id, fps)
        end_ms = frame_to_ms(chapter.end_frame_id, fps)
        timestamps = tl._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s = bisect_left(timestamps, start_ms)
        e = bisect_right(timestamps, end_ms)
        for i in range(s, e):
            new_selection.add(tl._action_key(actions[i]))
    tl.multi_selected_action_indices = new_selection
    if tl.logger:
        tl.logger.info(
            f"Selected {len(new_selection)} points in {len(selected_chapters)} chapter(s)",
            extra={'status_message': True},
        )
