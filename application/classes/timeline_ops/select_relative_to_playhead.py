"""Select all points before or after the playhead."""


def select_relative_to_playhead(tl, before: bool) -> None:
    actions = tl._get_actions()
    if not actions:
        return
    ph = tl._playhead_ms()
    if ph is None:
        return
    if before:
        keys = {tl._action_key(a) for a in actions if a['at'] <= ph}
        label = "left of playhead"
    else:
        keys = {tl._action_key(a) for a in actions if a['at'] >= ph}
        label = "right of playhead"
    tl.multi_selected_action_indices = keys
    tl.app.logger.info(
        f"Selected {len(keys)} point(s) {label} (Timeline {tl.timeline_num})",
        extra={'status_message': True})
