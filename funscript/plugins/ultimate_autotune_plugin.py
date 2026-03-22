"""
Ultimate Autotune Plugin v2 - Timing-preserving funscript enhancement.

Core principle: Peak and valley timing is sacred. This plugin enhances
amplitude and removes redundant intermediate points WITHOUT shifting when
direction changes occur.

Pipeline:
  1. Remove speed artifacts (impossibly fast intermediate points)
  2. Tag anchor points (peaks, valleys) — tagged on the dict, survives all mutations
  3. Amplify amplitude around center
  4. Remove jerk (small oscillations that aren't real direction changes)
  5. Simplify per-segment between anchors (RDP removes redundant intermediates)
  6. Remove collinear points (final cleanup)
  7. Clamp to 0-100 range
"""

from typing import Dict, Any, List, Optional
import copy
from funscript.plugins.base_plugin import FunscriptTransformationPlugin

_ANCHOR_KEY = '_anchor'


class UltimateAutotunePlugin(FunscriptTransformationPlugin):

    @property
    def name(self) -> str:
        return "Ultimate Autotune"

    @property
    def description(self) -> str:
        return "Timing-preserving enhancement: amplify, remove noise, simplify"

    @property
    def version(self) -> str:
        return "2.1.0"

    @property
    def category(self) -> str:
        return "Autotune"

    @property
    def author(self) -> str:
        return "FunGen Team"

    @property
    def ui_preference(self) -> str:
        return 'popup'

    @property
    def parameters_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "amplify_scale": {
                "type": float,
                "default": 1.2,
                "constraints": {"min": 0.5, "max": 3.0},
                "description": "Amplification scale factor (1.0 = no change)"
            },
            "amplify_center": {
                "type": int,
                "default": 50,
                "constraints": {"min": 0, "max": 100},
                "description": "Amplification center value"
            },
            "speed_threshold": {
                "type": float,
                "default": 800.0,
                "constraints": {"min": 100.0, "max": 5000.0},
                "description": "Remove points with both in/out speed above this (units/sec)"
            },
            "jerk_threshold": {
                "type": float,
                "default": 15.0,
                "constraints": {"min": 3.0, "max": 50.0},
                "description": "Remove direction changes smaller than this (units)"
            },
            "min_peak_amplitude": {
                "type": float,
                "default": 10.0,
                "constraints": {"min": 1.0, "max": 50.0},
                "description": "Minimum amplitude to qualify as a real peak/valley"
            },
            "simplify_tolerance": {
                "type": float,
                "default": 15.0,
                "constraints": {"min": 0.5, "max": 30.0},
                "description": "Remove intermediates within this distance of the line between peaks"
            },
            "selected_indices": {
                "type": list,
                "required": False,
                "default": None,
                "description": "Specific action indices to process"
            }
        }

    def transform(self, funscript_obj, axis: str = 'both', **parameters) -> Optional['MultiAxisFunscript']:
        try:
            params = self.validate_parameters(parameters)

            axes_to_process = []
            if axis == 'both':
                if funscript_obj.primary_actions:
                    axes_to_process.append('primary')
                if funscript_obj.secondary_actions:
                    axes_to_process.append('secondary')
            else:
                axes_to_process = [axis]

            for current_axis in axes_to_process:
                actions_list_ref = (funscript_obj.primary_actions if current_axis == 'primary'
                                    else funscript_obj.secondary_actions)

                if not actions_list_ref or len(actions_list_ref) < 3:
                    continue

                initial_count = len(actions_list_ref)

                selected_indices = params.get('selected_indices')
                if selected_indices and len(selected_indices) >= 3:
                    indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
                    if len(indices) < 3:
                        continue
                    working_actions = [copy.deepcopy(actions_list_ref[i]) for i in indices]
                else:
                    working_actions = copy.deepcopy(list(actions_list_ref))
                    indices = None

                result = self._process_actions(working_actions, params)

                # Clean up internal tags before writing back
                for a in result:
                    a.pop(_ANCHOR_KEY, None)

                if indices:
                    for idx in reversed(indices):
                        del actions_list_ref[idx]
                    for action in result:
                        insert_idx = 0
                        for i, existing in enumerate(actions_list_ref):
                            if existing['at'] > action['at']:
                                insert_idx = i
                                break
                            insert_idx = i + 1
                        actions_list_ref.insert(insert_idx, action)
                else:
                    actions_list_ref[:] = result

                funscript_obj._invalidate_cache(current_axis)
                self.logger.info(f"Ultimate Autotune on {current_axis}: {initial_count} -> {len(result)} points")

            return funscript_obj

        except Exception as e:
            self.logger.error(f"Ultimate Autotune failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _process_actions(self, actions: List[Dict], params: Dict) -> List[Dict]:
        if len(actions) < 3:
            return actions

        # 1. Remove speed artifacts (impossibly fast spikes)
        actions = self._remove_speed_artifacts(actions, params["speed_threshold"])
        if len(actions) < 3:
            return actions

        # 2. Tag anchors on ORIGINAL positions — real peaks before amplification
        self._tag_anchors(actions, params["min_peak_amplitude"])

        # 3. Amplify + soft clamp (anchor tags survive, positions scale)
        actions = self._amplify(actions, params["amplify_scale"], params["amplify_center"])
        for a in actions:
            a['pos'] = max(0.0, min(100.0, a['pos']))

        # 4. Remove jerk (small oscillations from non-anchor points)
        actions = self._remove_jerk(actions, params["jerk_threshold"])

        # 5. Simplify between anchors — RDP per segment, anchors always kept
        tol = params["simplify_tolerance"]
        if tol > 0:
            actions = self._simplify_between_anchors(actions, tol)

        # 6. Collinear cleanup — catches plateaus (e.g. clamped peaks at 100)
        #    and any remaining near-linear intermediates
        actions = self._remove_collinear(actions, max(tol * 0.4, 1.0))

        # 7. Keep only extrema — funscripts only need direction changes
        #    Remove any monotonic intermediate (same direction as predecessor)
        actions = self._keep_extrema(actions)

        # 8. Round to int (final step — float precision was needed for peak detection)
        for a in actions:
            a['pos'] = int(round(a['pos']))

        # 9. Remove consecutive same-position duplicates (e.g. two valleys at pos=0)
        actions = self._remove_position_duplicates(actions)

        return actions

    # ---- Anchor tagging (on-dict, survives list mutations) ----

    @staticmethod
    def _tag_anchors(actions: List[Dict], min_amplitude: float):
        """Tag peaks and valleys directly on action dicts. First/last always tagged.

        Two-pass approach:
        1. Find all candidate peaks/valleys meeting amplitude threshold
        2. Enforce alternating peak-valley-peak pattern — consecutive same-type
           anchors get merged (keep only the most extreme)
        """
        n = len(actions)
        actions[0][_ANCHOR_KEY] = True
        actions[-1][_ANCHOR_KEY] = True

        if n < 3:
            return

        # Pass 1: find all candidate peaks and valleys
        # Use >= to catch flat-topped peaks (e.g. 83, 89, 89, 84)
        # Use prominence (height relative to surrounding terrain) instead of
        # immediate-neighbor amplitude, since smooth 30fps signals have tiny
        # frame-to-frame differences that would miss real peaks/valleys
        positions = [a['pos'] for a in actions]
        # Window size for prominence: ~500ms at 30fps = 15 frames
        W = min(15, max(3, n // 6))

        candidates = []  # (index, type, position)
        for i in range(1, n - 1):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            next_pos = positions[i + 1]

            is_peak = (curr_pos >= prev_pos and curr_pos >= next_pos
                       and (curr_pos > prev_pos or curr_pos > next_pos))
            is_valley = (curr_pos <= prev_pos and curr_pos <= next_pos
                         and (curr_pos < prev_pos or curr_pos < next_pos))

            if is_peak or is_valley:
                # Prominence: how far this extremum stands out from its surroundings
                win_lo = max(0, i - W)
                win_hi = min(n, i + W + 1)
                if is_peak:
                    prominence = curr_pos - min(positions[win_lo:win_hi])
                else:
                    prominence = max(positions[win_lo:win_hi]) - curr_pos
                if prominence >= min_amplitude:
                    candidates.append((i, 'peak' if is_peak else 'valley', curr_pos))

        # Pass 2: enforce strict alternation — merge consecutive same-type candidates
        merged = []
        for cand in candidates:
            if merged and merged[-1][1] == cand[1]:
                if cand[1] == 'peak':
                    if cand[2] >= merged[-1][2]:
                        merged[-1] = cand
                else:
                    if cand[2] <= merged[-1][2]:
                        merged[-1] = cand
            else:
                merged.append(cand)

        # Pass 3: remove shallow reversals — if a valley between two peaks
        # doesn't dip far enough below the lower peak (or a peak between two
        # valleys doesn't rise far enough), collapse the triplet into one anchor.
        changed = True
        while changed and len(merged) >= 3:
            changed = False
            filtered = [merged[0]]
            i = 1
            while i < len(merged) - 1:
                prev = filtered[-1]
                curr = merged[i]
                nxt = merged[i + 1]

                shallow = False
                if curr[1] == 'valley':
                    # Valley between two peaks — must dip min_amplitude below the LOWER peak
                    depth = min(prev[2], nxt[2]) - curr[2]
                    shallow = depth < min_amplitude
                elif curr[1] == 'peak':
                    # Peak between two valleys — must rise min_amplitude above the HIGHER valley
                    height = curr[2] - max(prev[2], nxt[2])
                    shallow = height < min_amplitude

                if shallow:
                    # Remove the reversal + merge the two flanking same-type anchors
                    if nxt[1] == prev[1]:
                        if prev[1] == 'peak':
                            if nxt[2] >= prev[2]:
                                filtered[-1] = nxt
                        else:
                            if nxt[2] <= prev[2]:
                                filtered[-1] = nxt
                        i += 2  # skip both the shallow reversal and the merged neighbor
                    else:
                        i += 1  # shouldn't happen after alternation, but safe fallback
                    changed = True
                    continue

                filtered.append(curr)
                i += 1

            # Append remaining
            while i < len(merged):
                filtered.append(merged[i])
                i += 1
            merged = filtered

        for idx, _, _ in merged:
            actions[idx][_ANCHOR_KEY] = True

    @staticmethod
    def _is_anchor(action: Dict) -> bool:
        return action.get(_ANCHOR_KEY, False)

    # ---- Processing steps ----

    def _remove_speed_artifacts(self, actions: List[Dict], speed_threshold: float) -> List[Dict]:
        if len(actions) <= 2:
            return actions
        result = [actions[0]]
        for i in range(1, len(actions) - 1):
            p_prev, p_curr, p_next = actions[i - 1], actions[i], actions[i + 1]
            in_dt = p_curr['at'] - p_prev['at']
            in_speed = abs(p_curr['pos'] - p_prev['pos']) / (in_dt / 1000.0) if in_dt > 0 else float('inf')
            out_dt = p_next['at'] - p_curr['at']
            out_speed = abs(p_next['pos'] - p_curr['pos']) / (out_dt / 1000.0) if out_dt > 0 else float('inf')
            if not (in_speed > speed_threshold and out_speed > speed_threshold):
                result.append(p_curr)
        result.append(actions[-1])
        return result

    @staticmethod
    def _amplify(actions: List[Dict], scale: float, center: int) -> List[Dict]:
        for a in actions:
            a['pos'] = center + (a['pos'] - center) * scale
        return actions

    def _remove_jerk(self, actions: List[Dict], jerk_threshold: float) -> List[Dict]:
        """Remove non-anchor points that create small direction changes."""
        if len(actions) <= 3:
            return actions
        result = [actions[0]]
        i = 1
        while i < len(actions) - 1:
            if self._is_anchor(actions[i]):
                result.append(actions[i])
                i += 1
                continue
            prev_pos = result[-1]['pos']
            curr_pos = actions[i]['pos']
            next_pos = actions[i + 1]['pos']
            prev_to_curr = curr_pos - prev_pos
            curr_to_next = next_pos - curr_pos
            # Opposite directions = direction change — skip if small
            if prev_to_curr * curr_to_next < 0 and abs(prev_to_curr) < jerk_threshold:
                i += 1
                continue
            result.append(actions[i])
            i += 1
        result.append(actions[-1])
        return result

    def _simplify_between_anchors(self, actions: List[Dict], tolerance: float) -> List[Dict]:
        """RDP per segment between anchors. Anchors always survive."""
        if len(actions) <= 3:
            return actions

        # Find anchor indices
        anchor_idx = [i for i, a in enumerate(actions) if self._is_anchor(a)]
        if not anchor_idx or anchor_idx[0] != 0:
            anchor_idx.insert(0, 0)
        if anchor_idx[-1] != len(actions) - 1:
            anchor_idx.append(len(actions) - 1)

        result = []
        for s in range(len(anchor_idx) - 1):
            start = anchor_idx[s]
            end = anchor_idx[s + 1]

            if not result or result[-1]['at'] != actions[start]['at']:
                result.append(actions[start])

            if end - start > 1:
                segment = actions[start:end + 1]
                simplified = self._rdp_simplify(segment, tolerance)
                for pt in simplified[1:]:
                    result.append(pt)
            else:
                result.append(actions[end])

        if result[-1]['at'] != actions[-1]['at']:
            result.append(actions[-1])

        return result

    @staticmethod
    def _remove_collinear(actions: List[Dict], tolerance: float) -> List[Dict]:
        """Remove points within tolerance of line between their neighbors.
        Never removes anchors."""
        if len(actions) <= 2:
            return actions
        result = [actions[0]]
        for i in range(1, len(actions) - 1):
            curr = actions[i]
            # Always keep anchors
            if curr.get(_ANCHOR_KEY, False):
                result.append(curr)
                continue
            prev = result[-1]
            nxt = actions[i + 1]
            dt = nxt['at'] - prev['at']
            if dt > 0:
                t = (curr['at'] - prev['at']) / dt
                expected = prev['pos'] + t * (nxt['pos'] - prev['pos'])
                if abs(curr['pos'] - expected) > tolerance:
                    result.append(curr)
            else:
                result.append(curr)
        result.append(actions[-1])
        return result

    @staticmethod
    def _keep_extrema(actions: List[Dict]) -> List[Dict]:
        """Keep only first, last, and points where direction changes.
        A funscript device interpolates linearly — monotonic intermediates are redundant."""
        if len(actions) <= 2:
            return actions
        out = [actions[0]]
        for i in range(1, len(actions) - 1):
            d1 = actions[i]['pos'] - out[-1]['pos']
            d2 = actions[i + 1]['pos'] - actions[i]['pos']
            # Keep on direction reversal or plateau edge
            if d1 * d2 < 0 or (d1 == 0) != (d2 == 0):
                out.append(actions[i])
        out.append(actions[-1])
        return out

    @staticmethod
    def _remove_position_duplicates(actions: List[Dict]) -> List[Dict]:
        """Remove consecutive points with the same position value."""
        if len(actions) <= 1:
            return actions
        out = [actions[0]]
        for i in range(1, len(actions)):
            if actions[i]['pos'] != out[-1]['pos']:
                out.append(actions[i])
        return out

    @staticmethod
    def _rdp_simplify(points: List[Dict], epsilon: float) -> List[Dict]:
        if len(points) < 3:
            return points
        first = points[0]
        last = points[-1]
        dt = last['at'] - first['at']
        if dt == 0:
            return [first, last]
        max_dist = 0
        max_idx = 0
        for i in range(1, len(points) - 1):
            t = (points[i]['at'] - first['at']) / dt
            proj_pos = first['pos'] + t * (last['pos'] - first['pos'])
            dist = abs(points[i]['pos'] - proj_pos)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        if max_dist > epsilon:
            left = UltimateAutotunePlugin._rdp_simplify(points[:max_idx + 1], epsilon)
            right = UltimateAutotunePlugin._rdp_simplify(points[max_idx:], epsilon)
            return left + right[1:]
        else:
            return [first, last]


def register_plugin():
    from funscript.plugins.base_plugin import plugin_registry
    plugin_registry.register(UltimateAutotunePlugin())

register_plugin()
