"""
Repeat Pattern plugin  - Quickfix Tools.

Finds the cycle immediately before the selection and repeats it across
the selection.  Two modes: Mode A removes valleys and inserts
time-scaled copies; Mode B keeps valleys and scales the pattern to fit.
Algorithm by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional, List

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class QFRepeatPatternPlugin(FunscriptTransformationPlugin):
    """Repeat the cycle before the selection across selected top points."""

    @property
    def name(self) -> str:
        return "Repeat Pattern"

    @property
    def description(self) -> str:
        return "Repeat the pattern before the selection across top-point cycles (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'keep_valleys': {
                'type': bool,
                'required': False,
                'default': False,
                'description': 'Keep valley positions and scale pattern to fit (Mode B)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices to process (must start at a top point)'
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
        sel = params.get('selected_indices')
        if not sel:
            return
        indices = sorted(i for i in sel if 0 <= i < len(actions))
        if not indices:
            return

        keep_valleys = params['keep_valleys']

        # Find the reference cycle: walk backwards from first selected point
        # until we find a point with the same position
        first_sel = indices[0]
        if first_sel < 2:
            self.logger.warning("Not enough points before selection for reference pattern")
            return

        start_pos = actions[first_sel]['pos']
        # Find pattern start
        pattern_start = first_sel - 1
        low_val = actions[pattern_start]['pos']
        low_idx = pattern_start
        while pattern_start > 0:
            pattern_start -= 1
            if actions[pattern_start]['pos'] < low_val:
                low_val = actions[pattern_start]['pos']
                low_idx = pattern_start
            if actions[pattern_start]['pos'] == start_pos:
                break
        else:
            self.logger.warning("Could not find matching reference pattern start")
            return

        # Reference pattern: pattern_start -> first_sel
        ref_start = pattern_start
        ref_low = low_idx
        ref_end = first_sel

        if ref_end - ref_start < 2:
            return

        # Build list of new actions and indices to remove
        to_remove = set()
        new_actions: List[Dict[str, Any]] = []

        sel_set = set(indices)
        for idx in indices:
            if idx + 4 >= len(actions):
                break
            a1 = actions[idx]
            if idx - 1 < 0:
                continue
            a0 = actions[idx - 1]
            a2 = actions[idx + 1] if idx + 1 < len(actions) else None
            a3 = actions[idx + 2] if idx + 2 < len(actions) else None
            a4 = actions[idx + 3] if idx + 3 < len(actions) else None

            if not (a2 and a3 and a4):
                continue

            # Check if this is a top point in a triangular pattern
            is_pattern = (a1['pos'] > a0['pos'] and a1['pos'] > a2['pos'] and
                          a3['pos'] > a2['pos'] and a3['pos'] > a4['pos'])
            if not is_pattern:
                continue

            if not keep_valleys:
                # Mode A: remove valley, insert time-scaled copies of reference
                to_remove.add(idx + 1)  # mark valley for removal
                ref_duration = actions[ref_end]['at'] - actions[ref_start]['at']
                if ref_duration == 0:
                    continue
                tscale = (a3['at'] - a1['at']) / ref_duration
                for ri in range(1, ref_end - ref_start):
                    ref_pt = actions[ref_start + ri]
                    new_t = a1['at'] + tscale * (ref_pt['at'] - actions[ref_start]['at'])
                    new_actions.append({'at': int(round(new_t)), 'pos': ref_pt['pos']})
            else:
                # Mode B: scale pattern to fit existing valley positions
                # First half: ref_start -> ref_low mapped to a1 -> a2
                ref_half1_dur = actions[ref_low]['at'] - actions[ref_start]['at']
                if ref_half1_dur > 0:
                    tscale1 = (a2['at'] - a1['at']) / ref_half1_dur
                    pscale1 = (a2['pos'] - a1['pos']) / max(1, actions[ref_low]['pos'] - actions[ref_start]['pos']) if actions[ref_low]['pos'] != actions[ref_start]['pos'] else 0
                    for ri in range(ref_start + 1, ref_low):
                        ref_pt = actions[ri]
                        new_t = a1['at'] + tscale1 * (ref_pt['at'] - actions[ref_start]['at'])
                        new_p = a1['pos'] + pscale1 * (ref_pt['pos'] - actions[ref_start]['pos'])
                        new_actions.append({'at': int(round(new_t)), 'pos': max(0, min(100, round(new_p)))})

                # Second half: ref_low -> ref_end mapped to a2 -> a3
                ref_half2_dur = actions[ref_end]['at'] - actions[ref_low]['at']
                if ref_half2_dur > 0:
                    tscale2 = (a3['at'] - a2['at']) / ref_half2_dur
                    pscale2 = (a3['pos'] - a2['pos']) / max(1, actions[ref_end]['pos'] - actions[ref_low]['pos']) if actions[ref_end]['pos'] != actions[ref_low]['pos'] else 0
                    for ri in range(ref_low + 1, ref_end):
                        ref_pt = actions[ri]
                        new_t = a2['at'] + tscale2 * (ref_pt['at'] - actions[ref_low]['at'])
                        new_p = a2['pos'] + pscale2 * (ref_pt['pos'] - actions[ref_low]['pos'])
                        new_actions.append({'at': int(round(new_t)), 'pos': max(0, min(100, round(new_p)))})

        # Remove marked actions (reverse order to preserve indices)
        for rm_idx in sorted(to_remove, reverse=True):
            if 0 <= rm_idx < len(actions):
                actions.pop(rm_idx)

        # Insert new actions
        for na in new_actions:
            actions.append(na)

        # Re-sort by time
        actions.sort(key=lambda a: a['at'])

        funscript._invalidate_cache(axis)
        self.logger.info(f"Repeat Pattern on {axis}: removed {len(to_remove)}, "
                         f"added {len(new_actions)} points")
