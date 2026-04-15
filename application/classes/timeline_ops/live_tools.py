"""Live-drag wrappers for batch plugins.

Live-drag pattern: while the user drags a slider, every tick:
  1. If the previous undo entry was this same op, undo + pop it.
  2. Apply the plugin with the current parameter value.
  3. Push a single LiveDragCmd snapshot.

Result: the entire drag collapses into one undo entry on release.

Helpers also stash a `_baseline_actions` snapshot when a drag starts so
each tick re-applies from the original state (not from the previous
tick's output), otherwise the transformation would compound on every
slider movement.
"""
from typing import Optional

from application.classes.undo_manager import LiveDragCmd


def begin_live_drag(tl, op_key: str) -> None:
    """Capture the baseline action snapshot at the start of a drag.

    `op_key` distinguishes simultaneous tools (e.g. 'range_extend' vs 'rdp')
    so a user switching tools doesn't accidentally extend the wrong baseline.
    """
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return
    actions = list(fs.get_axis_actions(axis) or [])
    tl._live_drag_state = {
        'op_key': op_key,
        'baseline': [dict(a) for a in actions],
        'axis': axis,
    }


def end_live_drag(tl) -> None:
    tl._live_drag_state = None


def _restore_baseline(tl, fs, axis) -> Optional[list]:
    state = getattr(tl, '_live_drag_state', None)
    if not state:
        return None
    baseline = state.get('baseline')
    if baseline is None:
        return None
    actions = fs.get_axis_actions(axis)
    if actions is None:
        return None
    actions.clear()
    actions.extend(dict(a) for a in baseline)
    fs._invalidate_cache(axis or 'both')
    return baseline


def _coalesced_apply(tl, plugin_name: str, label: str, **params) -> None:
    """Re-apply `plugin_name` from baseline with `params`, coalescing undo."""
    fs, axis = tl._get_target_funscript_details()
    if not fs:
        return
    pre = _restore_baseline(tl, fs, axis)
    if pre is None:
        # No baseline yet, initialize lazily
        begin_live_drag(tl, plugin_name)
        pre = list(fs.get_axis_actions(axis) or [])

    fs.apply_plugin(plugin_name, axis=axis, **params)
    post = list(fs.get_axis_actions(axis) or [])

    # Coalesce: if previous entry was a live-drag, drop it.
    if tl.app.undo_manager.match_top(LiveDragCmd):
        tl.app.undo_manager.pop_top(tl.app)
    tl.app.undo_manager.push_done(LiveDragCmd(tl.timeline_num, pre, post, label))

    tl.app.funscript_processor._post_mutation_refresh(tl.timeline_num, label)
    tl.invalidate_cache()


def live_range_extend(tl, extend_amount: int) -> None:
    """Apply Range Extender on the current selection (or all) with `extend_amount`."""
    selected = sorted(tl._resolve_selected_indices()) if tl.multi_selected_action_indices else None
    _coalesced_apply(tl, "Range Extender",
                     f"Range Extend ({extend_amount:+d})",
                     extend_amount=int(extend_amount),
                     selected_indices=selected)


def live_rdp_simplify(tl, epsilon: float) -> None:
    """Apply RDP Simplification with `epsilon`. Operates on selection or all."""
    _coalesced_apply(tl, "RDP Simplification",
                     f"RDP Simplify (eps={epsilon:.2f})",
                     epsilon=float(epsilon))
