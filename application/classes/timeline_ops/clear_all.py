"""Delete every point on the timeline (undoable)."""


def clear_all_points(tl) -> None:
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return
    actions = tl._get_actions()
    num_points = len(actions) if actions else 0
    if num_points == 0:
        return
    actions_before = list(actions)
    fs.clear_points(axis=axis, selected_indices=list(range(num_points)))
    tl.multi_selected_action_indices.clear()
    tl.selected_action_idx = -1
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Clear All Points")
    tl.invalidate_cache()
    actions_after = list(tl._get_actions() or [])
    from application.classes.undo_manager import BulkReplaceCmd
    tl.app.undo_manager.push_done(BulkReplaceCmd(
        tl.timeline_num, actions_before, actions_after,
        f"Clear All Points (T{tl.timeline_num})"))
