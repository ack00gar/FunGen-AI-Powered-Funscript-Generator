"""Isolate selection: keep selected points, delete others within their interval."""


def isolate_selection(tl) -> None:
    resolved = sorted(tl._resolve_selected_indices())
    if not resolved:
        return
    actions = tl._get_actions()
    keep_keys = {tl._action_key(actions[i]) for i in resolved}
    t0, t1 = actions[resolved[0]]['at'], actions[resolved[-1]]['at']
    to_delete = [
        i for i, a in enumerate(actions)
        if t0 <= a['at'] <= t1 and tl._action_key(a) not in keep_keys
    ]
    if not to_delete:
        return
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return
    deleted_info = [{'index': i, 'action': dict(actions[i])} for i in to_delete]
    fs.clear_points(axis=axis, selected_indices=to_delete)
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Isolate Selection")
    tl.invalidate_cache()
    try:
        from application.classes.undo_manager import DeletePointsCmd
        tl.app.undo_manager.push_done(DeletePointsCmd(tl.timeline_num, deleted_info))
    except Exception:
        pass
