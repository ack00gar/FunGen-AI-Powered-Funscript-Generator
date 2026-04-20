"""Add a single point on the timeline (undoable)."""


def add_point(tl, t: float, v: float, snap_time: bool = True) -> None:
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return
    t = int(tl._snap_time(t)) if snap_time else int(t)
    snap_v = tl.app.app_state_ui.snap_to_grid_pos
    v = int(round(v / snap_v) * snap_v) if snap_v > 0 else int(v)
    fs.add_action(
        t,
        v if axis == 'primary' else None,
        v if axis == 'secondary' else None,
    )
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Add Point")
    tl.invalidate_cache()
    from application.classes.undo_manager import AddPointCmd
    tl.app.undo_manager.push_done(AddPointCmd(tl.timeline_num, {'at': t, 'pos': v}))
