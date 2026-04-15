"""Clipboard ops: copy selection + three paste modes.

Paste modes:
- relative (default, Ctrl+V): drop clipboard at playhead, preserving inter-point timing.
- replace (Ctrl+Shift+V): same as relative, but first deletes existing actions
  inside the destination interval. Avoids doubled points.
- exact (Ctrl+Alt+V): paste at original absolute timestamps from when the
  selection was copied. Useful for moving content between projects /
  timelines without manual realignment.
"""


def copy_selection(tl) -> None:
    actions = tl._get_actions()
    if not tl.multi_selected_action_indices:
        return
    indices = tl._resolve_selected_indices()
    selection = [actions[i] for i in indices]
    if not selection:
        return
    base_time = selection[0]['at']
    clipboard_data = [
        {'relative_at': a['at'] - base_time, 'pos': a['pos'], 'original_at': a['at']}
        for a in selection
    ]
    tl.app.funscript_processor.set_clipboard_actions(clipboard_data)
    tl.logger.info(f"Copied {len(clipboard_data)} points.")


def _paste_common(tl, target_times):
    """Shared paste body. `target_times` is a list of (t_ms, pos) tuples."""
    clip = tl.app.funscript_processor.get_clipboard_actions()
    if not clip or not target_times:
        return False
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return False
    actions_before = list(tl._get_actions() or [])
    new_actions = [
        {
            'timestamp_ms': int(t),
            'primary_pos': int(v) if axis == 'primary' else None,
            'secondary_pos': int(v) if axis == 'secondary' else None,
        }
        for t, v in target_times
    ]
    fs.add_actions_batch(new_actions, is_from_live_tracker=False)
    return (fs, axis, actions_before)


def _push_paste_undo(tl, before, label):
    actions_after = list(tl._get_actions() or [])
    from application.classes.undo_manager import BulkReplaceCmd
    tl.app.undo_manager.push_done(
        BulkReplaceCmd(tl.timeline_num, before, actions_after,
                       f"{label} (T{tl.timeline_num})"))


def paste_actions(tl, paste_at_ms: float) -> None:
    """Relative paste: drop clipboard at playhead, preserving spacing."""
    clip = tl.app.funscript_processor.get_clipboard_actions()
    if not clip:
        return
    targets = [(paste_at_ms + item['relative_at'], item['pos']) for item in clip]
    result = _paste_common(tl, targets)
    if not result:
        return
    _, _, before = result
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Paste")
    tl.invalidate_cache()
    _push_paste_undo(tl, before, "Paste")


def paste_actions_replace(tl, paste_at_ms: float) -> None:
    """Relative paste, but clear existing actions in the destination interval first."""
    clip = tl.app.funscript_processor.get_clipboard_actions()
    if not clip:
        return
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return

    # Compute destination interval before mutating
    rels = [item['relative_at'] for item in clip]
    t0 = paste_at_ms + min(rels)
    t1 = paste_at_ms + max(rels)
    actions_before = list(tl._get_actions() or [])

    # Clear actions in [t0, t1] then paste relative
    fs.clear_actions_in_time_range(int(t0), int(t1), axis=axis)
    targets = [(paste_at_ms + item['relative_at'], item['pos']) for item in clip]
    new_actions = [
        {
            'timestamp_ms': int(t),
            'primary_pos': int(v) if axis == 'primary' else None,
            'secondary_pos': int(v) if axis == 'secondary' else None,
        }
        for t, v in targets
    ]
    fs.add_actions_batch(new_actions, is_from_live_tracker=False)
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Paste (Replace)")
    tl.invalidate_cache()
    _push_paste_undo(tl, actions_before, "Paste Replace")


def paste_actions_exact(tl) -> None:
    """Paste at the original absolute timestamps from when the selection was copied."""
    clip = tl.app.funscript_processor.get_clipboard_actions()
    if not clip:
        return
    # Fall back to relative-at-zero if old clipboard format (no original_at)
    targets = [(item.get('original_at', item['relative_at']), item['pos']) for item in clip]
    result = _paste_common(tl, targets)
    if not result:
        return
    _, _, before = result
    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, "Paste (Exact)")
    tl.invalidate_cache()
    _push_paste_undo(tl, before, "Paste Exact")


def copy_to_other(tl, target_num=None) -> None:
    """Copy the selection to the *other* timeline (or an explicit target)."""
    actions = tl._get_actions()
    if not tl.multi_selected_action_indices:
        return
    other_num = target_num if target_num is not None else (2 if tl.timeline_num == 1 else 1)
    fs_other, axis_other = tl.app.funscript_processor._get_target_funscript_object_and_axis(other_num)
    if not fs_other:
        return
    other_before = list(fs_other.get_axis_actions(axis_other) or [])
    indices = tl._resolve_selected_indices()
    points = [actions[i] for i in indices]
    if axis_other in ('primary', 'secondary'):
        batch = [
            {
                'timestamp_ms': p['at'],
                'primary_pos': p['pos'] if axis_other == 'primary' else None,
                'secondary_pos': p['pos'] if axis_other == 'secondary' else None,
            }
            for p in points
        ]
        fs_other.add_actions_batch(batch, is_from_live_tracker=False)
    else:
        for p in points:
            fs_other.add_action_to_axis(axis_other, p['at'], p['pos'])
    tl.app.funscript_processor._post_mutation_refresh(other_num, f"Copy from T{tl.timeline_num}")
    other_after = list(fs_other.get_axis_actions(axis_other) or [])
    from application.classes.undo_manager import BulkReplaceCmd
    tl.app.undo_manager.push_done(BulkReplaceCmd(
        other_num, other_before, other_after,
        f"Copy from T{tl.timeline_num} (T{other_num})"))
