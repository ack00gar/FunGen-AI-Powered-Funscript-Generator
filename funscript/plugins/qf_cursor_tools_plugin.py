"""
Cursor-based tools - Quickfix Tools.

Four tools that operate relative to the video playhead position:
  1. Adjust Timing at Cursor - snaps closest point, remaps proportionally
  2. Center Points in Time - sets each point to midpoint of its neighbors
  3. Fill Gap at Cursor - auto-fills a gap in the action with interpolated data
  4. Clip Peaks at Cursor - clips peaks above the interpolated level at cursor

Algorithms by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


# ---------------------------------------------------------------------------
# 1. Adjust Timing at Cursor
# ---------------------------------------------------------------------------

class QFAdjustTimingPlugin(FunscriptTransformationPlugin):
    """Snap closest selected point to playhead, remap others proportionally."""

    @property
    def name(self) -> str:
        return "Adjust Timing at Cursor"

    @property
    def description(self) -> str:
        return "Snap closest point to playhead time, remap selection proportionally (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def requires_cursor(self) -> bool:
        return True

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'current_time_ms': {
                'type': int,
                'required': True,
                'default': 0,
                'description': 'Current playhead position in milliseconds'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices to process'
            }
        }

    def transform(self, funscript, axis='both', **parameters) -> None:
        params = self.validate_parameters(parameters)
        for ax in (['primary', 'secondary'] if axis == 'both' else [axis]):
            self._apply(funscript, ax, params)

    def _apply(self, funscript, axis, params):
        actions = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        if not actions:
            return
        indices = _resolve(actions, params)
        if len(indices) < 2:
            return

        cursor_ms = params['current_time_ms']

        # Find closest selected action to cursor
        closest_idx = min(indices, key=lambda i: abs(actions[i]['at'] - cursor_ms))
        at_closest = actions[closest_idx]['at']
        at_first = actions[indices[0]]['at']
        at_last = actions[indices[-1]]['at']

        # Remap all selected actions proportionally
        for idx in indices:
            at = actions[idx]['at']
            if at == at_closest:
                actions[idx]['at'] = cursor_ms
            elif at < cursor_ms:
                # Before cursor: remap [at_first, at_closest] -> [at_first, cursor_ms]
                if at_closest != at_first:
                    actions[idx]['at'] = int(round(
                        at_first + (cursor_ms - at_first) / (at_closest - at_first) * (at - at_first)
                    ))
            else:
                # After cursor: remap [at_closest, at_last] -> [cursor_ms, at_last]
                if at_closest != at_last:
                    actions[idx]['at'] = int(round(
                        at_last + (cursor_ms - at_last) / (at_closest - at_last) * (at - at_last)
                    ))

        funscript._invalidate_cache(axis)
        self.logger.info(f"Adjust Timing at Cursor on {axis}: snapped to {cursor_ms}ms")


# ---------------------------------------------------------------------------
# 2. Center Points in Time
# ---------------------------------------------------------------------------

class QFCenterInTimePlugin(FunscriptTransformationPlugin):
    """Set each selected point's time to midpoint of its neighbors."""

    @property
    def name(self) -> str:
        return "Center Points in Time"

    @property
    def description(self) -> str:
        return "Move each selected point to the midpoint in time of its neighbors (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices to process'
            }
        }

    def transform(self, funscript, axis='both', **parameters) -> None:
        params = self.validate_parameters(parameters)
        for ax in (['primary', 'secondary'] if axis == 'both' else [axis]):
            self._apply(funscript, ax, params)

    def _apply(self, funscript, axis, params):
        actions = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        if not actions:
            return
        indices = _resolve(actions, params)
        if not indices:
            return

        # Compute new times first (don't mutate while iterating)
        new_times = {}
        for idx in indices:
            if idx <= 0 or idx >= len(actions) - 1:
                continue
            before_at = actions[idx - 1]['at']
            after_at = actions[idx + 1]['at']
            new_times[idx] = int(round(0.5 * (before_at + after_at)))

        for idx, new_t in new_times.items():
            actions[idx]['at'] = new_t

        funscript._invalidate_cache(axis)
        self.logger.info(f"Center Points in Time on {axis}: {len(new_times)} points centered")


# ---------------------------------------------------------------------------
# 3. Fill Gap at Cursor
# ---------------------------------------------------------------------------

class QFFillGapPlugin(FunscriptTransformationPlugin):
    """Fill a gap in the action by detecting period from surrounding points."""

    @property
    def name(self) -> str:
        return "Fill Gap at Cursor"

    @property
    def description(self) -> str:
        return "Auto-fill a gap by detecting period from surrounding actions (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def requires_cursor(self) -> bool:
        return True

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'current_time_ms': {
                'type': int,
                'required': True,
                'default': 0,
                'description': 'Current playhead position in milliseconds'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices (not used  - operates around cursor)'
            }
        }

    def transform(self, funscript, axis='both', **parameters) -> None:
        params = self.validate_parameters(parameters)
        for ax in (['primary', 'secondary'] if axis == 'both' else [axis]):
            self._apply(funscript, ax, params)

    def _apply(self, funscript, axis, params):
        actions = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        if not actions or len(actions) < 6:
            return

        cursor_ms = params['current_time_ms']

        # Find 3 actions before and 3 after cursor
        before = [a for a in actions if a['at'] < cursor_ms]
        after = [a for a in actions if a['at'] > cursor_ms]
        if len(before) < 3 or len(after) < 3:
            self.logger.warning("Need at least 3 actions on each side of cursor")
            return

        a1, a2, a3 = before[-3], before[-2], before[-1]
        a4, a5, a6 = after[0], after[1], after[2]

        dt0 = a3['at'] - a1['at']  # period before
        dt2 = a6['at'] - a4['at']  # period after
        dt1 = a4['at'] - a3['at']  # gap

        if dt0 + dt2 == 0:
            return
        nper = int(round(2 * dt1 / (dt2 + dt0) + 0.5))
        if nper < 1:
            return

        dtx = dt1 / nper

        # Generate peak positions (average of surrounding peaks)
        peak_pos = round((a1['pos'] + a3['pos'] + a4['pos'] + a6['pos']) / 4)
        valley_pos = round((a2['pos'] + a5['pos']) / 2)

        new_actions = []
        # Insert peaks
        for i in range(1, nper):
            new_actions.append({'at': int(round(a3['at'] + i * dtx)), 'pos': peak_pos})
        # Insert valleys
        for i in range(1, nper + 1):
            new_actions.append({'at': int(round(a2['at'] + i * dtx)), 'pos': valley_pos})

        for na in new_actions:
            actions.append(na)
        actions.sort(key=lambda a: a['at'])

        funscript._invalidate_cache(axis)
        self.logger.info(f"Fill Gap at Cursor on {axis}: {len(new_actions)} points added, "
                         f"{nper} cycles")


# ---------------------------------------------------------------------------
# 4. Clip Peaks at Cursor
# ---------------------------------------------------------------------------

class QFClipPeaksPlugin(FunscriptTransformationPlugin):
    """Clip selected peaks above the interpolated level at cursor."""

    @property
    def name(self) -> str:
        return "Clip Peaks at Cursor"

    @property
    def description(self) -> str:
        return "Clip selected peaks above the interpolated level at playhead (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def requires_cursor(self) -> bool:
        return True

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'current_time_ms': {
                'type': int,
                'required': True,
                'default': 0,
                'description': 'Current playhead position in milliseconds'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices to process'
            }
        }

    def transform(self, funscript, axis='both', **parameters) -> None:
        params = self.validate_parameters(parameters)
        for ax in (['primary', 'secondary'] if axis == 'both' else [axis]):
            self._apply(funscript, ax, params)

    def _apply(self, funscript, axis, params):
        actions = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        if not actions or len(actions) < 2:
            return
        indices = _resolve(actions, params)
        if not indices:
            return

        cursor_ms = params['current_time_ms']

        # Find the two actions bracketing the cursor
        before = [a for a in actions if a['at'] <= cursor_ms]
        after = [a for a in actions if a['at'] > cursor_ms]
        if not before or not after:
            return
        act1 = before[-1]
        act2 = after[0]

        dt = act2['at'] - act1['at']
        if dt == 0:
            clip_pos = act1['pos']
        else:
            clip_pos = act1['pos'] + (act2['pos'] - act1['pos']) / dt * (cursor_ms - act1['at'])

        # Process selected points: remove those above clip level, add intersection points
        sel_set = set(indices)
        to_remove = []
        new_actions = []

        for idx in indices:
            if actions[idx]['pos'] <= clip_pos:
                continue
            to_remove.append(idx)

            # Check if segment from previous point crosses clip level
            if idx > 0 and actions[idx - 1]['pos'] < clip_pos:
                a_prev = actions[idx - 1]
                a_cur = actions[idx]
                dp = a_cur['pos'] - a_prev['pos']
                if dp != 0:
                    cross_t = a_prev['at'] + (clip_pos - a_prev['pos']) / dp * (a_cur['at'] - a_prev['at'])
                    new_actions.append({'at': int(round(cross_t)), 'pos': round(clip_pos)})

            # Check if segment to next point crosses clip level
            if idx < len(actions) - 1 and actions[idx + 1]['pos'] < clip_pos:
                a_cur = actions[idx]
                a_next = actions[idx + 1]
                dp = a_cur['pos'] - a_next['pos']
                if dp != 0:
                    cross_t = a_cur['at'] + (a_cur['pos'] - clip_pos) / dp * (a_next['at'] - a_cur['at'])
                    new_actions.append({'at': int(round(cross_t)), 'pos': round(clip_pos)})

        # Remove clipped points (reverse order)
        for rm_idx in sorted(to_remove, reverse=True):
            if 0 <= rm_idx < len(actions):
                actions.pop(rm_idx)

        # Add intersection points
        for na in new_actions:
            actions.append(na)
        actions.sort(key=lambda a: a['at'])

        funscript._invalidate_cache(axis)
        self.logger.info(f"Clip Peaks at Cursor on {axis}: removed {len(to_remove)}, "
                         f"added {len(new_actions)} intersection points, clip={clip_pos:.0f}")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _resolve(actions, params):
    sel = params.get('selected_indices')
    if sel:
        return sorted(i for i in sel if 0 <= i < len(actions))
    return list(range(len(actions)))
