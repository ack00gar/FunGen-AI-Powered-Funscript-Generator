"""Equalize selection: redistribute selected points evenly in time."""


def equalize_selection(tl) -> None:
    resolved = sorted(tl._resolve_selected_indices())
    if len(resolved) < 3:
        return
    actions = tl._get_actions()
    first_idx, last_idx = resolved[0], resolved[-1]
    t0, t1 = actions[first_idx]['at'], actions[last_idx]['at']
    span = t1 - t0
    if span <= 0:
        return
    step = span / (len(resolved) - 1)
    old = [(idx, actions[idx]['at']) for idx in resolved]
    for i, idx in enumerate(resolved):
        actions[idx]['at'] = int(round(t0 + i * step))
    tl.multi_selected_action_indices = {tl._action_key(actions[idx]) for idx in resolved}
    fs, axis = tl._get_target_funscript_details()
    if fs:
        fs._invalidate_cache(axis or 'both')
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Equalize Selection")
    tl.invalidate_cache()
    try:
        from application.classes.undo_manager import NudgeTimesCmd
        for idx, t_old in old:
            delta = actions[idx]['at'] - t_old
            if delta:
                tl.app.undo_manager.push_done(NudgeTimesCmd(tl.timeline_num, [idx], delta))
    except Exception:
        pass
