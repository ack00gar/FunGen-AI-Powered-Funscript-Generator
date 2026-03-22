"""
Ultimate Autotune Plugin v2 - Timing-preserving funscript enhancement.

Core principle: Peak and valley timing is sacred. This plugin enhances
amplitude and removes noise WITHOUT shifting when direction changes occur.

Pipeline:
  1. Identify anchor points (peaks, valleys, inflection points)
  2. Assess signal noise level to determine smoothing intensity
  3. Remove speed artifacts (impossibly fast intermediate points)
  4. Amplify amplitude around center
  5. Smooth only between anchors (preserve anchor timestamps)
  6. Remove jerk (small oscillations within larger movements)
  7. Clamp to 0-100 range
"""

from typing import Dict, Any, List, Optional
import copy
import numpy as np
from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class UltimateAutotunePlugin(FunscriptTransformationPlugin):
    """
    Ultimate Autotune v2 - Timing-preserving enhancement pipeline.

    Identifies peaks, valleys, and inflection points as immutable timing anchors.
    All smoothing and simplification operates between anchors only.
    Noise level is assessed automatically to adapt processing intensity.
    """

    @property
    def name(self) -> str:
        return "Ultimate Autotune"

    @property
    def description(self) -> str:
        return "Timing-preserving enhancement: amplify, smooth between peaks, remove jerk"

    @property
    def version(self) -> str:
        return "2.0.0"

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
            "noise_smoothing": {
                "type": float,
                "default": 0.5,
                "constraints": {"min": 0.0, "max": 1.0},
                "description": "Smoothing intensity between anchors (0 = none, 1 = max)"
            },
            "speed_threshold": {
                "type": float,
                "default": 800.0,
                "constraints": {"min": 100.0, "max": 5000.0},
                "description": "Remove points with both in/out speed above this (units/sec)"
            },
            "jerk_threshold": {
                "type": float,
                "default": 20.0,
                "constraints": {"min": 3.0, "max": 50.0},
                "description": "Maximum oscillation size to consider as jerk"
            },
            "min_anchor_amplitude": {
                "type": float,
                "default": 10.0,
                "constraints": {"min": 1.0, "max": 30.0},
                "description": "Minimum position change to qualify as a peak/valley"
            },
            "simplify_tolerance": {
                "type": float,
                "default": 8.0,
                "constraints": {"min": 0.5, "max": 25.0},
                "description": "Point simplification tolerance (higher = fewer points)"
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

                # Handle selection
                selected_indices = params.get('selected_indices')
                if selected_indices and len(selected_indices) >= 3:
                    indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
                    if len(indices) < 3:
                        continue
                    working_actions = [copy.deepcopy(actions_list_ref[i]) for i in indices]
                else:
                    working_actions = copy.deepcopy(list(actions_list_ref))
                    indices = None

                # === PIPELINE ===
                result = self._process_actions(working_actions, params)

                # Write back
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
                self.logger.info(f"Ultimate Autotune v2 on {current_axis}: {initial_count} -> {len(result)} points")

            return funscript_obj

        except Exception as e:
            self.logger.error(f"Ultimate Autotune failed: {e}")
            return None

    def _process_actions(self, actions: List[Dict], params: Dict) -> List[Dict]:
        """Core processing pipeline, timing-preserving."""
        if len(actions) < 3:
            return actions

        # Step 1: Remove speed artifacts
        actions = self._remove_speed_artifacts(actions, params["speed_threshold"])
        if len(actions) < 3:
            return actions

        # Step 2: Identify anchors (peaks, valleys, inflection points)
        anchor_flags = self._identify_anchors(actions, params["min_anchor_amplitude"])

        # Step 3: Assess noise level
        noise_level = self._assess_noise(actions, anchor_flags)
        effective_smoothing = params["noise_smoothing"] * noise_level

        # Step 4: Amplify (operates on position values, preserves timestamps)
        actions = self._amplify(actions, params["amplify_scale"], params["amplify_center"])

        # Step 5: Smooth between anchors only
        if effective_smoothing > 0.05:
            actions = self._smooth_between_anchors(actions, anchor_flags, effective_smoothing)

        # Step 6: Remove jerk (small oscillations within larger movements)
        actions = self._remove_jerk(actions, anchor_flags, params["jerk_threshold"])

        # Step 7: Simplify (global RDP -- peaks/valleys are naturally preserved as most significant points)
        if params["simplify_tolerance"] > 0:
            actions = self._rdp_simplify(actions, params["simplify_tolerance"])

        # Step 8: Clamp
        for a in actions:
            a['pos'] = int(round(max(0, min(100, a['pos']))))

        return actions

    def _identify_anchors(self, actions: List[Dict], min_amplitude: float) -> List[bool]:
        """Identify peaks, valleys, and inflection points.

        Returns a boolean list where True = anchor (timing-locked point).
        """
        n = len(actions)
        anchors = [False] * n

        # First and last are always anchors
        anchors[0] = True
        anchors[-1] = True

        if n < 3:
            return anchors

        positions = [a['pos'] for a in actions]

        # Pass 1: Find peaks and valleys (direction reversals)
        for i in range(1, n - 1):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            next_pos = positions[i + 1]

            # Peak: higher than both neighbors
            if curr_pos >= prev_pos and curr_pos >= next_pos:
                if abs(curr_pos - prev_pos) >= min_amplitude or abs(curr_pos - next_pos) >= min_amplitude:
                    anchors[i] = True

            # Valley: lower than both neighbors
            if curr_pos <= prev_pos and curr_pos <= next_pos:
                if abs(prev_pos - curr_pos) >= min_amplitude or abs(next_pos - curr_pos) >= min_amplitude:
                    anchors[i] = True

        return anchors

    def _assess_noise(self, actions: List[Dict], anchor_flags: List[bool]) -> float:
        """Assess signal noise level (0.0 = clean, 1.0 = very noisy).

        Measures the ratio of non-anchor points to total points.
        A clean signal has mostly anchors; a noisy one has many intermediate jitter points.
        """
        if len(actions) < 5:
            return 0.0

        n_anchors = sum(anchor_flags)
        n_total = len(actions)

        # Ratio of non-anchor points
        non_anchor_ratio = 1.0 - (n_anchors / n_total)

        # Also check average inter-point jitter between anchors
        jitter_sum = 0.0
        jitter_count = 0
        positions = [a['pos'] for a in actions]

        for i in range(1, len(actions) - 1):
            if not anchor_flags[i]:
                # Expected position = linear interpolation between surrounding anchors
                prev_anchor_idx = i - 1
                while prev_anchor_idx > 0 and not anchor_flags[prev_anchor_idx]:
                    prev_anchor_idx -= 1
                next_anchor_idx = i + 1
                while next_anchor_idx < len(actions) - 1 and not anchor_flags[next_anchor_idx]:
                    next_anchor_idx += 1

                if prev_anchor_idx != next_anchor_idx:
                    t = (actions[i]['at'] - actions[prev_anchor_idx]['at']) / max(1, actions[next_anchor_idx]['at'] - actions[prev_anchor_idx]['at'])
                    expected = positions[prev_anchor_idx] + t * (positions[next_anchor_idx] - positions[prev_anchor_idx])
                    jitter_sum += abs(positions[i] - expected)
                    jitter_count += 1

        avg_jitter = jitter_sum / max(1, jitter_count)

        # Normalize: jitter of 5+ units = very noisy
        jitter_normalized = min(1.0, avg_jitter / 5.0)

        # Combine both signals
        noise = 0.5 * non_anchor_ratio + 0.5 * jitter_normalized
        return min(1.0, max(0.0, noise))

    def _remove_speed_artifacts(self, actions: List[Dict], speed_threshold: float) -> List[Dict]:
        """Remove points where both in-speed and out-speed exceed threshold."""
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

    def _amplify(self, actions: List[Dict], scale: float, center: int) -> List[Dict]:
        """Scale positions around center point. Preserves timestamps."""
        for a in actions:
            a['pos'] = center + (a['pos'] - center) * scale
        return actions

    def _smooth_between_anchors(self, actions: List[Dict], anchor_flags: List[bool],
                                 intensity: float) -> List[Dict]:
        """Smooth non-anchor points toward the linear path between surrounding anchors.

        Anchors are never moved. Non-anchors are blended toward the linear
        interpolation between their surrounding anchors, weighted by intensity.
        """
        if intensity <= 0 or len(actions) < 3:
            return actions

        positions = [a['pos'] for a in actions]
        smoothed = list(positions)

        for i in range(1, len(actions) - 1):
            if anchor_flags[i]:
                continue  # Never move anchors

            # Find surrounding anchors
            prev_anchor = i - 1
            while prev_anchor > 0 and not anchor_flags[prev_anchor]:
                prev_anchor -= 1
            next_anchor = i + 1
            while next_anchor < len(actions) - 1 and not anchor_flags[next_anchor]:
                next_anchor += 1

            if prev_anchor == next_anchor:
                continue

            # Linear interpolation between anchors
            t = (actions[i]['at'] - actions[prev_anchor]['at']) / max(1, actions[next_anchor]['at'] - actions[prev_anchor]['at'])
            linear_pos = positions[prev_anchor] + t * (positions[next_anchor] - positions[prev_anchor])

            # Blend current position toward linear path
            smoothed[i] = positions[i] * (1 - intensity) + linear_pos * intensity

        for i, a in enumerate(actions):
            a['pos'] = smoothed[i]

        return actions

    def _remove_jerk(self, actions: List[Dict], anchor_flags: List[bool],
                      jerk_threshold: float) -> List[Dict]:
        """Remove small oscillations between anchors.

        If a non-anchor point creates a direction change smaller than jerk_threshold
        within a larger movement, remove it.
        """
        if len(actions) <= 3:
            return actions

        result = [actions[0]]
        i = 1
        while i < len(actions) - 1:
            if anchor_flags[i]:
                result.append(actions[i])
                i += 1
                continue

            prev_pos = result[-1]['pos']
            curr_pos = actions[i]['pos']
            next_pos = actions[i + 1]['pos']

            # Check if this point creates a small direction change (jerk)
            prev_to_curr = curr_pos - prev_pos
            curr_to_next = next_pos - curr_pos

            # Opposite directions = direction change
            if prev_to_curr * curr_to_next < 0:
                oscillation_size = abs(prev_to_curr)
                if oscillation_size < jerk_threshold:
                    # Skip this jerk point
                    i += 1
                    continue

            result.append(actions[i])
            i += 1

        result.append(actions[-1])
        return result


    def _simplify_between_anchors(self, actions: List[Dict], anchor_flags: List[bool],
                                    tolerance: float) -> List[Dict]:
        """Apply RDP simplification per segment between anchors.

        Anchors are always kept. Non-anchor points within each segment are
        simplified using Ramer-Douglas-Peucker, which preserves segment
        endpoints (our anchors) and removes collinear intermediate points.
        """
        if len(actions) <= 3:
            return actions

        # Find anchor indices
        anchor_indices = [i for i, is_anchor in enumerate(anchor_flags) if is_anchor]

        # Always include first and last
        if 0 not in anchor_indices:
            anchor_indices.insert(0, 0)
        if len(actions) - 1 not in anchor_indices:
            anchor_indices.append(len(actions) - 1)

        result = []
        for seg_idx in range(len(anchor_indices) - 1):
            start = anchor_indices[seg_idx]
            end = anchor_indices[seg_idx + 1]

            # Always add the start anchor
            if not result or result[-1]['at'] != actions[start]['at']:
                result.append(actions[start])

            # If segment has intermediate points, simplify them
            if end - start > 1:
                segment = actions[start:end + 1]
                simplified = self._rdp_simplify(segment, tolerance)
                # Add simplified points (skip first which is the start anchor)
                for pt in simplified[1:]:
                    result.append(pt)
            else:
                # Direct connection, just add end point
                result.append(actions[end])

        # Ensure last point is included
        if result[-1]['at'] != actions[-1]['at']:
            result.append(actions[-1])

        return result

    @staticmethod
    def _rdp_simplify(points: List[Dict], epsilon: float) -> List[Dict]:
        """Ramer-Douglas-Peucker simplification on (time, position) pairs."""
        if len(points) < 3:
            return points

        # Find point with max distance from line between first and last
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


# Register the plugin
def register_plugin():
    """Register this plugin with the plugin system."""
    from funscript.plugins.base_plugin import plugin_registry
    plugin_registry.register(UltimateAutotunePlugin())


# Auto-register when imported
register_plugin()
