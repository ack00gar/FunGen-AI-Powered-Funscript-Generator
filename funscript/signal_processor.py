"""Signal-processing operations on MultiAxisFunscript action lists.

Accessed via `fs.signal.<method>()`. The flat `fs.<method>()` API is kept
as thin delegators in MultiAxisFunscript for backwards compatibility.
"""
from __future__ import annotations

import bisect
import copy
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

try:
    from scipy.signal import savgol_filter, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from funscript.multi_axis_funscript import MultiAxisFunscript


class SignalProcessor:
    """Heavy DSP on funscript action lists."""

    __slots__ = ("fs",)

    def __init__(self, fs: "MultiAxisFunscript") -> None:
        self.fs = fs

    # ---- SG auto-tune ----
    def auto_tune_sg_filter(self, axis: str,
                             saturation_low: int = 1,
                             saturation_high: int = 99,
                             max_window_size: int = 15,
                             polyorder: int = 2,
                             selected_indices: Optional[List[int]] = None) -> Optional[Dict]:
        fs = self.fs
        if not SCIPY_AVAILABLE:
            fs.logger.warning("scipy not installed. SG auto-tune cannot be applied.")
            return None

        actions_list_ref = fs.primary_actions if axis == 'primary' else fs.secondary_actions
        if not actions_list_ref:
            return None

        if selected_indices is not None and len(selected_indices) > 0:
            indices_to_filter = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
        else:
            indices_to_filter = list(range(len(actions_list_ref)))

        if len(indices_to_filter) < 3:
            fs.logger.warning("Not enough points for SG auto-tune.")
            return None

        positions = np.array([actions_list_ref[i]['pos'] for i in indices_to_filter])
        num_points_in_segment = len(positions)

        best_window_length = -1
        min_saturated_count = float('inf')

        for window_length in range(3, max_window_size + 1, 2):
            if num_points_in_segment < window_length:
                fs.logger.info(f"Auto-Tune: Segment size ({num_points_in_segment}) is smaller than window size ({window_length}). Stopping search.")
                break

            actual_polyorder = min(polyorder, window_length - 1)

            try:
                smoothed_positions = savgol_filter(positions, window_length, actual_polyorder)
            except ValueError as e:
                fs.logger.warning(f"Auto-Tune: SG filter failed for window {window_length}. Error: {e}. Stopping.")
                continue

            saturated_count = np.sum((smoothed_positions <= saturation_low) | (smoothed_positions >= saturation_high))
            fs.logger.debug(f"Auto-Tune trying W={window_length}, P={actual_polyorder}: Found {saturated_count} saturated points.")

            if saturated_count < min_saturated_count:
                min_saturated_count = saturated_count
                best_window_length = window_length

            if saturated_count == 0:
                break

        if best_window_length == -1:
            fs.logger.error("Auto-Tune: Could not determine a best window size. This should not happen if there are enough points.")
            return None

        fs.logger.info(f"Auto-Tune determined best window W={best_window_length} with {min_saturated_count} saturated points remaining.")
        final_polyorder = min(polyorder, best_window_length - 1)
        try:
            final_smoothed_positions = savgol_filter(positions, best_window_length, final_polyorder)
            for i, original_list_idx in enumerate(indices_to_filter):
                actions_list_ref[original_list_idx]['pos'] = int(round(np.clip(final_smoothed_positions[i], 0, 100)))

            result = {
                'window_length': best_window_length,
                'polyorder': final_polyorder,
                'points_affected': len(indices_to_filter),
            }
            fs.logger.info(f"Applied Auto-Tuned SG to {axis} axis with W={result['window_length']}, P={result['polyorder']}.")
            return result
        except Exception as e:
            fs.logger.error(f"Error applying final auto-tuned SG filter: {e}")
            return None

    # ---- Missing-stroke recovery ----
    def recover_missing_strokes(self, axis: str, original_actions: List[Dict],
                                threshold_factor: float = 1.8) -> None:
        """Re-insert significant missing strokes filtered out of the keyframes."""
        fs = self.fs
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        keyframes = getattr(fs, target_list_attr)

        if len(keyframes) < 2 or len(original_actions) < 3:
            return

        intervals = np.array([p2['at'] - p1['at']
                              for p1, p2 in zip(keyframes, keyframes[1:]) if p2['at'] > p1['at']])
        if len(intervals) < 2:
            return

        median_interval = np.median(intervals)
        gap_threshold = median_interval * threshold_factor

        points_to_add = []
        action_times = [a['at'] for a in original_actions]
        for i in range(len(keyframes) - 1):
            p1, p2 = keyframes[i], keyframes[i + 1]
            interval = p2['at'] - p1['at']
            if interval <= gap_threshold:
                continue

            s_idx = bisect.bisect_right(action_times, p1['at'])
            e_idx = bisect.bisect_left(action_times, p2['at'])
            if s_idx >= e_idx:
                continue

            candidates_in_gap = original_actions[s_idx:e_idx]
            if not candidates_in_gap:
                continue

            best_candidate = None
            max_significance = -1.0
            for p_cand in candidates_in_gap:
                progress = (p_cand['at'] - p1['at']) / float(interval)
                projected_pos = p1['pos'] + progress * (p2['pos'] - p1['pos'])
                significance = abs(p_cand['pos'] - projected_pos)
                if significance > max_significance:
                    max_significance = significance
                    best_candidate = p_cand

            if best_candidate:
                points_to_add.append(copy.deepcopy(best_candidate))

        if points_to_add:
            fs.logger.info(f"Ultimate Autotune: Recovered {len(points_to_add)} missing strokes.")
            batch_data = [{
                'timestamp_ms': p['at'],
                'primary_pos': p['pos'] if axis == 'primary' else None,
                'secondary_pos': p['pos'] if axis == 'secondary' else None,
            } for p in points_to_add]
            fs.add_actions_batch(batch_data)

    # ---- Peak / valley simplification ----
    def find_peaks_and_valleys(self, axis: str,
                               height: Optional[float] = None, threshold: Optional[float] = None,
                               distance: Optional[float] = None, prominence: Optional[float] = None,
                               width: Optional[float] = None,
                               selected_indices: Optional[List[int]] = None) -> None:
        fs = self.fs
        if not SCIPY_AVAILABLE:
            fs.logger.warning("scipy not installed. Peak finding cannot be applied.")
            return

        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(fs, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 3:
            fs.logger.warning(f"Not enough points on {axis} for peak finding.")
            return

        s_idx_orig, e_idx_orig = 0, len(actions_list_ref) - 1
        if selected_indices:
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 3:
                fs.logger.warning("Not enough valid selected indices for peak finding.")
                return
            s_idx_orig, e_idx_orig = valid_indices[0], valid_indices[-1]

        prefix_actions = actions_list_ref[:s_idx_orig]
        segment_to_process = actions_list_ref[s_idx_orig:e_idx_orig + 1]
        suffix_actions = actions_list_ref[e_idx_orig + 1:]

        if len(segment_to_process) < 3:
            actions_list_ref[:] = prefix_actions + segment_to_process + suffix_actions
            return

        positions = np.array([a['pos'] for a in segment_to_process])
        inverted_positions = 100 - positions

        kwargs = {
            'height': height if height else None,
            'threshold': threshold if threshold else None,
            'distance': distance if distance else None,
            'prominence': prominence if prominence else None,
            'width': width if width else None,
        }

        peak_indices, _ = find_peaks(positions, **kwargs)
        valley_indices, _ = find_peaks(inverted_positions, **kwargs)

        keyframe_indices = {0, len(segment_to_process) - 1}
        keyframe_indices.update(peak_indices)
        keyframe_indices.update(valley_indices)
        sorted_indices = sorted(list(keyframe_indices))

        new_segment_actions = [segment_to_process[i] for i in sorted_indices]
        actions_list_ref[:] = prefix_actions + new_segment_actions + suffix_actions

        last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
        if axis == 'primary':
            fs.last_timestamp_primary = last_ts
        else:
            fs.last_timestamp_secondary = last_ts

        fs._invalidate_cache(axis)
        fs.logger.info(
            f"Peak simplification applied to {axis} (indices {s_idx_orig}-{e_idx_orig}). "
            f"Points: {len(segment_to_process)} -> {len(new_segment_actions)}")

    # ---- Range scaling ----
    def scale_points_to_range(self, axis: str, output_min: int, output_max: int,
                              start_time_ms: Optional[int] = None,
                              end_time_ms: Optional[int] = None,
                              selected_indices: Optional[List[int]] = None) -> None:
        fs = self.fs
        actions_list_ref = fs.primary_actions if axis == 'primary' else fs.secondary_actions
        if not actions_list_ref or len(actions_list_ref) < 2:
            return

        if selected_indices is not None:
            indices_to_process = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
        elif start_time_ms is not None and end_time_ms is not None:
            s_idx, e_idx = fs._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            indices_to_process = list(range(s_idx, e_idx + 1)) if (s_idx is not None and e_idx is not None) else []
        else:
            indices_to_process = list(range(len(actions_list_ref)))

        if len(indices_to_process) < 2:
            fs.logger.info(f"Not enough points in selection for range scaling on {axis} axis.")
            return

        positions_in_segment = np.array([actions_list_ref[i]['pos'] for i in indices_to_process])
        effective_min = np.percentile(positions_in_segment, 10)
        effective_max = np.percentile(positions_in_segment, 90)

        current_effective_range = effective_max - effective_min
        target_range = output_max - output_min

        if current_effective_range <= 0:
            new_pos = int(round(output_min + target_range / 2.0))
            for idx in indices_to_process:
                actions_list_ref[idx]['pos'] = new_pos
            fs.logger.info(f"Scaled {len(indices_to_process)} flat points on {axis} axis to {new_pos}.")
            return

        for idx in indices_to_process:
            original_pos = actions_list_ref[idx]['pos']
            normalized_pos = (original_pos - effective_min) / current_effective_range
            clipped_normalized_pos = np.clip(normalized_pos, 0.0, 1.0)
            new_pos = int(round(output_min + clipped_normalized_pos * target_range))
            actions_list_ref[idx]['pos'] = np.clip(new_pos, 0, 100)

        fs.logger.info(
            f"Scaled {len(indices_to_process)} points on {axis} axis to new range [{output_min}-{output_max}].")

    # ---- Peak-preserving resample ----
    def apply_peak_preserving_resample(self, axis: str, resample_rate_ms: int = 50,
                                       selected_indices: Optional[List[int]] = None) -> None:
        fs = self.fs
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(fs, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 3:
            fs.logger.info("Not enough points for Peak-Preserving Resampling.")
            return

        s_idx, e_idx = 0, len(actions_list_ref) - 1
        if selected_indices:
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 3:
                fs.logger.info("Not enough selected points for resampling.")
                return
            s_idx, e_idx = valid_indices[0], valid_indices[-1]

        prefix_actions = actions_list_ref[:s_idx]
        segment_to_process = actions_list_ref[s_idx:e_idx + 1]
        suffix_actions = actions_list_ref[e_idx + 1:]

        anchors: List[Dict] = []
        if not segment_to_process:
            return
        anchors.append(segment_to_process[0])

        for i in range(1, len(segment_to_process) - 1):
            p_prev = segment_to_process[i - 1]['pos']
            p_curr = segment_to_process[i]['pos']
            p_next = segment_to_process[i + 1]['pos']

            if p_curr > p_prev and p_curr > p_next:
                anchors.append(segment_to_process[i])
            elif p_curr < p_prev and p_curr < p_next:
                anchors.append(segment_to_process[i])
            elif p_curr == p_next and p_curr != p_prev:
                j = i
                while j < len(segment_to_process) - 1 and segment_to_process[j]['pos'] == p_curr:
                    j += 1
                p_after_flat = segment_to_process[j]['pos']
                if (p_curr > p_prev and p_curr > p_after_flat) or \
                        (p_curr < p_prev and p_curr < p_after_flat):
                    anchor_candidate = segment_to_process[(i + j - 1) // 2]
                    if not anchors or anchors[-1] != anchor_candidate:
                        anchors.append(anchor_candidate)

        if not anchors or anchors[-1] != segment_to_process[-1]:
            anchors.append(segment_to_process[-1])

        new_actions: List[Dict] = [anchors[0]]
        for i in range(len(anchors) - 1):
            p1 = anchors[i]
            p2 = anchors[i + 1]
            t1, pos1 = p1['at'], p1['pos']
            t2, pos2 = p2['at'], p2['pos']
            duration = float(t2 - t1)
            pos_delta = float(pos2 - pos1)
            if duration <= 0:
                continue
            current_time = t1 + resample_rate_ms
            while current_time < t2:
                progress = (current_time - t1) / duration
                eased_progress = (1 - np.cos(progress * np.pi)) / 2.0
                new_pos = pos1 + eased_progress * pos_delta
                new_actions.append({
                    'at': int(current_time),
                    'pos': int(round(np.clip(new_pos, 0, 100))),
                })
                current_time += resample_rate_ms
            if not new_actions or new_actions[-1]['at'] < p2['at']:
                new_actions.append(p2)

        actions_list_ref[:] = prefix_actions + new_actions + suffix_actions
        fs.logger.info(
            f"Applied Peak-Preserving Resample to {axis}. "
            f"Points: {len(segment_to_process)} -> {len(new_actions)}")

    # ---- Internal: vectorized keyframe simplification ----
    def _simplify_keyframes_vectorized(self, extrema: List[Dict], position_tolerance: int) -> List[Dict]:
        """Vectorized keyframe simplification using numpy."""
        if len(extrema) <= 2:
            return extrema

        ext_positions = np.array([ext['pos'] for ext in extrema])
        ext_timestamps = np.array([ext['at'] for ext in extrema])

        while len(extrema) > 2:
            if len(ext_positions) <= 2:
                break

            prev_pos = ext_positions[:-2]
            curr_pos = ext_positions[1:-1]
            next_pos = ext_positions[2:]
            prev_time = ext_timestamps[:-2]
            curr_time = ext_timestamps[1:-1]
            next_time = ext_timestamps[2:]

            durations = next_time.astype(np.float64) - prev_time.astype(np.float64)
            time_deltas = curr_time.astype(np.float64) - prev_time.astype(np.float64)
            progress = np.divide(time_deltas, durations,
                                 out=np.zeros_like(time_deltas, dtype=np.float64),
                                 where=durations != 0)
            projected_pos = prev_pos + progress * (next_pos - prev_pos)
            significance_scores = np.abs(curr_pos - projected_pos)
            significance_scores[durations == 0] = np.inf

            min_idx = np.argmin(significance_scores)
            min_significance = significance_scores[min_idx]

            if min_significance < position_tolerance:
                remove_idx = min_idx + 1
                extrema.pop(remove_idx)
                ext_positions = np.delete(ext_positions, remove_idx)
                ext_timestamps = np.delete(ext_timestamps, remove_idx)
            else:
                break

        return extrema

    # ---- Internal: duplicate/min-interval filter ----
    def _filter_list_by_interval(self, axis: str) -> None:
        fs = self.fs
        actions_list = fs.primary_actions if axis == 'primary' else fs.secondary_actions
        if len(actions_list) < 2:
            return

        unique_actions = [actions_list[0]]
        for i in range(1, len(actions_list)):
            if actions_list[i]['at'] == unique_actions[-1]['at']:
                unique_actions[-1] = actions_list[i]
            else:
                unique_actions.append(actions_list[i])

        if fs.min_interval_ms > 0:
            final_actions = [unique_actions[0]]
            for i in range(1, len(unique_actions)):
                if unique_actions[i]['at'] - final_actions[-1]['at'] >= fs.min_interval_ms:
                    final_actions.append(unique_actions[i])
            actions_list[:] = final_actions
        else:
            actions_list[:] = unique_actions
