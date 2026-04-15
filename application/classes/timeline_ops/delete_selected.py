"""Delete the currently selected points (undoable)."""


def delete_selected(tl) -> None:
    if not tl.multi_selected_action_indices:
        return
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        tl.logger.error(f"Could not get funscript details for timeline {tl.timeline_num}")
        return
    actions = tl._get_actions()
    resolved = tl._resolve_selected_indices()
    deleted_info = [{'index': idx, 'action': actions[idx]} for idx in resolved]
    fs.clear_points(axis=axis, selected_indices=resolved)
    tl.multi_selected_action_indices.clear()
    tl.selected_action_idx = -1
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Delete Points")
    tl.invalidate_cache()
    if deleted_info:
        from application.classes.undo_manager import DeletePointsCmd
        tl.app.undo_manager.push_done(DeletePointsCmd(tl.timeline_num, deleted_info))
