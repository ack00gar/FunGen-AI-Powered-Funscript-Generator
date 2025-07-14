import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
import logging
import bisect
import copy

# Attempt to import optional libraries for processing
try:
    from scipy.signal import savgol_filter, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from rdp import rdp

    RDP_AVAILABLE = True
except ImportError:
    RDP_AVAILABLE = False


class DualAxisFunscript:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.primary_actions: List[Dict] = []
        self.secondary_actions: List[Dict] = []
        self.min_interval_ms: int = 20
        self.last_timestamp_primary: int = 0
        self.last_timestamp_secondary: int = 0

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('DualAxisFunscript_fallback')
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())

    def _process_action_for_axis(self,
                                 actions_target_list: List[Dict],
                                 timestamp_ms: int,
                                 pos: int,
                                 min_interval_ms: int,
                                 is_from_live_tracker: bool
                                 ) -> int:
        """
        Processes and adds/updates a single action in the target list (in-place).
        Optimized for performance, especially with large action lists.
        Returns the timestamp of the last action in the list.
        """
        clamped_pos = max(0, min(100, pos))
        new_action = {"at": timestamp_ms, "pos": clamped_pos}

        original_length_before_add = len(actions_target_list)

        # Find insertion point or existing action with the same timestamp
        # This list comprehension for timestamps is O(N) but happens once per call.
        # For extremely frequent calls on huge lists, further optimization might involve
        # a data structure that keeps timestamps indexed, but that's a larger change.
        action_timestamps = [a["at"] for a in actions_target_list]
        idx = bisect.bisect_left(action_timestamps, timestamp_ms)

        action_inserted_or_updated = False
        if idx < len(actions_target_list) and actions_target_list[idx]["at"] == timestamp_ms:
            # Timestamp exists, update position if different
            if actions_target_list[idx]["pos"] != clamped_pos:
                actions_target_list[idx]["pos"] = clamped_pos
                action_inserted_or_updated = True  # Position updated
            # If pos is the same, no effective change from this new_action
        else:
            can_insert = True
            if idx > 0:  # Check interval with predecessor if not inserting at the beginning
                prev_action = actions_target_list[idx - 1]
                if timestamp_ms - prev_action["at"] < min_interval_ms:
                    can_insert = False

            if can_insert:
                actions_target_list.insert(idx, new_action)
                action_inserted_or_updated = True

        # Enforce min_interval_ms after an insert/update that might have caused violations
        # This pass is O(N) but ensures correctness similar to the original's filter pass.
        if action_inserted_or_updated and min_interval_ms > 0:  # Only re-filter if a change occurred
            if not actions_target_list:  # Should not happen if we just inserted/updated
                return 0

            # Efficient in-place filter for min_interval_ms
            current_valid_idx = 0
            # The first point is always valid if the list is not empty (which it isn't here).
            # Iterate from the second point.
            if len(actions_target_list) > 1:
                for i in range(1, len(actions_target_list)):
                    if actions_target_list[i]["at"] - actions_target_list[current_valid_idx]["at"] >= min_interval_ms:
                        current_valid_idx += 1
                        if i != current_valid_idx:  # if elements were skipped due to interval
                            actions_target_list[current_valid_idx] = actions_target_list[i]

            # Trim the list if elements were removed from the end by filtering
            if current_valid_idx + 1 < len(actions_target_list):
                del actions_target_list[current_valid_idx + 1:]

        return actions_target_list[-1]["at"] if actions_target_list else 0

    def add_action(self, timestamp_ms: int, primary_pos: Optional[int], secondary_pos: Optional[int] = None,
                   is_from_live_tracker: bool = True):
        """
        Adds an action for primary axis and optionally for secondary axis.
        :param timestamp_ms: The timestamp of the action in milliseconds.
        :param primary_pos: The position for the primary axis (0-100). Can be None.
        :param secondary_pos: Optional. The position for the secondary axis (0-100). Can be None.
        :param is_from_live_tracker: True if this action originates from live tracking,
                                     influencing max_history application. False for programmatic
                                     additions (e.g. file load, undo/redo) where max_history
                                     might not be desired for the loaded portion.
        """
        new_last_ts_primary = self.last_timestamp_primary
        if primary_pos is not None: # Only process if primary_pos is provided
            new_last_ts_primary = self._process_action_for_axis(
                actions_target_list=self.primary_actions,
                timestamp_ms=timestamp_ms,
                pos=primary_pos, # primary_pos is guaranteed not None here
                min_interval_ms=self.min_interval_ms,
                is_from_live_tracker=is_from_live_tracker
            )
        # Update last_timestamp_primary only if actions were actually processed or if list became empty
        self.last_timestamp_primary = new_last_ts_primary if self.primary_actions else 0


        new_last_ts_secondary = self.last_timestamp_secondary
        if secondary_pos is not None: # Only process if secondary_pos is provided
            new_last_ts_secondary = self._process_action_for_axis(
                actions_target_list=self.secondary_actions,
                timestamp_ms=timestamp_ms,
                pos=secondary_pos, # secondary_pos is guaranteed not None here
                min_interval_ms=self.min_interval_ms,
                is_from_live_tracker=is_from_live_tracker
            )
            self.last_timestamp_secondary = new_last_ts_secondary if self.secondary_actions else 0

    def reset_to_neutral(self, timestamp_ms: int):
        self.add_action(timestamp_ms, 100, 50, is_from_live_tracker=True)

    def get_value(self, time_ms: int, axis: str = 'primary') -> int:
        actions_to_search = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_to_search:
            return 50

        # Optimized with bisect_left for O(log N) find + O(N) for timestamp list (can be cached if list doesn't change often)
        # However, for get_value, the list of timestamps is small enough that direct creation is fine.
        action_timestamps = [a["at"] for a in actions_to_search]
        idx = bisect.bisect_left(action_timestamps, time_ms)

        if idx == 0:  # time_ms is before or at the first action
            return actions_to_search[0]["pos"]
        # Note: len(actions_to_search) because idx can be len(actions_to_search) if time_ms is after all actions
        if idx == len(actions_to_search):  # time_ms is after or at the last action
            return actions_to_search[-1]["pos"]

        # Interpolate between actions_to_search[idx-1] and actions_to_search[idx]
        p1 = actions_to_search[idx - 1]
        p2 = actions_to_search[idx]

        if time_ms == p1["at"]:  # Exact match with the point before insertion point
            return p1["pos"]
        # p2["at"] should be >= time_ms because of bisect_left. If p2["at"] == time_ms, it means p1["at"] < time_ms.

        # Denominator for interpolation
        time_diff = float(p2["at"] - p1["at"])
        if time_diff == 0:  # Should not happen if min_interval_ms > 0 and enforced
            return p1["pos"]

        t_ratio = (time_ms - p1["at"]) / time_diff
        val = p1["pos"] + t_ratio * (p2["pos"] - p1["pos"])
        return int(round(np.clip(val, 0, 100)))

    def get_latest_value(self, axis: str = 'primary') -> int:
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        if actions_list:
            return actions_list[-1]["pos"]
        return 50

    def clear(self):
        self.primary_actions = []
        self.secondary_actions = []
        self.last_timestamp_primary = 0
        self.last_timestamp_secondary = 0
        self.logger.info("Cleared all actions from DualAxisFunscript.")

    def find_next_jump_frame(self, current_frame: int, fps: float, axis: str = 'primary') -> Optional[int]:
        """
        Finds the frame index of the first action that occurs on a frame
        strictly after the current frame.
        """
        if not fps > 0: return None
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list: return None

        current_time_ms = current_frame * (1000.0 / fps)

        # Find the first action strictly after the current time
        for action in actions_list:
            if action['at'] > current_time_ms:
                target_frame = int(action['at'] * (fps / 1000.0))
                # Ensure we are actually moving to a new frame
                if target_frame > current_frame:
                    return target_frame
        return None

    def find_prev_jump_frame(self, current_frame: int, fps: float, axis: str = 'primary') -> Optional[int]:
        """
        Finds the frame index of the last action that occurs on a frame
        strictly before the current frame.
        """
        if not fps > 0: return None
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list: return None

        last_valid_frame = None
        # Find the last action that is on a frame strictly before the current one
        for action in actions_list:
            # We must use a strict < comparison on time to find previous points
            if action['at'] < (current_frame * (1000.0 / fps)):
                target_frame = int(action['at'] * (fps / 1000.0))
                # Ensure it's a different frame before we consider it
                if target_frame < current_frame:
                    last_valid_frame = target_frame
            else:
                # List is sorted, no more valid points past this
                break
        return last_valid_frame

    @property
    def actions(self) -> List[Dict]:
        return self.primary_actions

    @actions.setter
    def actions(self, value: List[Dict]):
        """
        Sets the primary actions list. Assumes 'value' is a list of action dictionaries.
        The list will be sorted by 'at'. This setter is typically used for loading
        scripts or undo/redo, where the input list is expected to be 'clean'
        (i.e., min_interval_ms and max_history are not re-applied here).
        """
        try:
            if not isinstance(value, list) or \
                    not all(isinstance(item, dict) and "at" in item and "pos" in item for item in value):
                self.logger.error(
                    "Invalid value for actions setter: Must be a list of action dicts {'at': ms, 'pos': val}.")
                self.primary_actions = []  # Clear to maintain a consistent state
            else:
                # Create a new list from sorted items to ensure we don't keep a reference to a mutable 'value'
                self.primary_actions = sorted(list(item for item in value), key=lambda x: x["at"])

            self.last_timestamp_primary = self.primary_actions[-1]["at"] if self.primary_actions else 0
            # self.logger.debug(f"Primary actions set externally. Count: {len(self.primary_actions)}")
        except Exception as e:
            self.logger.error(f"Error in actions.setter: {e}. Clearing primary actions as a precaution.")
            self.primary_actions = []
            self.last_timestamp_primary = 0

    def _get_default_stats_values(self) -> dict:
        return {
            "num_points": 0, "duration_scripted_s": 0.0, "avg_speed_pos_per_s": 0.0,
            "avg_intensity_percent": 0.0, "min_pos": -1, "max_pos": -1,
            "avg_interval_ms": 0.0, "min_interval_ms": -1, "max_interval_ms": -1,
            "total_travel_dist": 0, "num_strokes": 0
        }

    def get_actions_statistics(self, axis: str = 'primary') -> dict:
        # This method's logic is O(N). If called very frequently on large scripts,
        # consider caching its results or calling it less often from the UI.
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        stats = self._get_default_stats_values()
        if not actions_list: return stats
        stats["num_points"] = len(actions_list)
        stats["min_pos"] = min(act["pos"] for act in actions_list) if actions_list else -1
        stats["max_pos"] = max(act["pos"] for act in actions_list) if actions_list else -1
        if len(actions_list) < 2: return stats
        stats["duration_scripted_s"] = (actions_list[-1]["at"] - actions_list[0]["at"]) / 1000.0
        total_pos_change, total_time_ms_for_speed, intervals, num_strokes = 0, 0, [], 0
        last_direction = 0
        for i in range(len(actions_list) - 1):
            p1, p2 = actions_list[i], actions_list[i + 1]
            delta_pos, delta_t_ms = abs(p2["pos"] - p1["pos"]), p2["at"] - p1["at"]
            total_pos_change += delta_pos
            if delta_t_ms > 0:
                intervals.append(delta_t_ms)
                if delta_pos > 0: total_time_ms_for_speed += delta_t_ms
            current_direction = 1 if p2["pos"] > p1["pos"] else (-1 if p2["pos"] < p1["pos"] else 0)
            if current_direction != 0 and last_direction != 0 and current_direction != last_direction: num_strokes += 1
            if current_direction != 0: last_direction = current_direction
        stats["total_travel_dist"] = total_pos_change
        stats["num_strokes"] = num_strokes if num_strokes > 0 else (
            1 if total_pos_change > 0 and len(actions_list) >= 2 else 0)
        if total_time_ms_for_speed > 0: stats["avg_speed_pos_per_s"] = (total_pos_change / (total_time_ms_for_speed / 1000.0))
        num_segments = len(actions_list) - 1
        if num_segments > 0: stats["avg_intensity_percent"] = total_pos_change / float(num_segments)
        if intervals:
            stats["avg_interval_ms"] = sum(intervals) / float(len(intervals)) if intervals else 0
            stats["min_interval_ms"] = float(min(intervals)) if intervals else -1
            stats["max_interval_ms"] = float(max(intervals)) if intervals else -1
        return stats

    def _get_action_indices_in_time_range(self, actions_list: List[dict],
                                          start_time_ms: int, end_time_ms: int) -> Tuple[Optional[int], Optional[int]]:
        if not actions_list: return None, None
        action_timestamps = [a['at'] for a in actions_list]

        # Find the index of the first action >= start_time_ms
        s_idx = bisect.bisect_left(action_timestamps, start_time_ms)

        # Find the index of the first action > end_time_ms
        # The actions to include will be up to e_idx - 1
        e_idx = bisect.bisect_right(action_timestamps, end_time_ms)
        if s_idx >= e_idx: return None, None
        return s_idx, e_idx - 1

    def apply_savitzky_golay(self, axis: str, window_length: int, polyorder: int,
                             start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                             selected_indices: Optional[List[int]] = None):
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not installed. SG filter cannot be applied.")
            return
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref: return

        indices_to_filter: List[int] = []
        if selected_indices is not None and len(selected_indices) > 0:
            # Filter valid indices from the selection
            indices_to_filter = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if not indices_to_filter:
                self.logger.warning("No valid selected indices for SG.")
                return
        elif start_time_ms is not None and end_time_ms is not None:
            s_idx, e_idx = self._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            if s_idx is None or e_idx is None or s_idx > e_idx:
                self.logger.warning("No points in time range for SG.")
                return
            indices_to_filter = list(range(s_idx, e_idx + 1))
        else:
            indices_to_filter = list(range(len(actions_list_ref)))

        if not indices_to_filter:
            self.logger.warning("No points for SG filter.")
            return
        num_points_in_segment = len(indices_to_filter)

        # Validate window_length and polyorder against the number of points in the segment
        actual_window_length = int(window_length)
        if actual_window_length % 2 == 0: actual_window_length += 1
        actual_polyorder = int(polyorder)
        if actual_polyorder >= actual_window_length: actual_polyorder = actual_window_length - 1
        if actual_polyorder < 0: actual_polyorder = 0
        if num_points_in_segment < actual_window_length:
            self.logger.warning(
                f"Not enough points ({num_points_in_segment}) for SG (window: {actual_window_length}).")
            return

        # Extract positions from the identified segment of actions
        positions = np.array([actions_list_ref[i]['pos'] for i in indices_to_filter])

        try:
            smoothed_positions = savgol_filter(positions, actual_window_length, actual_polyorder)
            for i, original_list_idx in enumerate(indices_to_filter):
                actions_list_ref[original_list_idx]['pos'] = int(round(np.clip(smoothed_positions[i], 0, 100)))
            self.logger.info(f"Applied SG to {axis} axis, affecting {len(indices_to_filter)} points.")
        except Exception as e:
            self.logger.error(f"Error applying SG filter: {e}")

    def simplify_rdp(self, axis: str, epsilon: float,
                     start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                     selected_indices: Optional[List[int]] = None):
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(self, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 2:
            self.logger.warning(f"Not enough points on {axis} for RDP.")
            return

        # --- Segment Selection ---
        prefix_actions, suffix_actions, segment_to_simplify = [], [], []
        s_idx_orig, e_idx_orig = -1, -1

        if selected_indices is not None and len(selected_indices) > 0:
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 2:
                self.logger.warning("Not enough valid selected indices for RDP.")
                return
            s_idx_orig, e_idx_orig = valid_indices[0], valid_indices[-1]
            segment_to_simplify = actions_list_ref[s_idx_orig:e_idx_orig + 1]
            prefix_actions = actions_list_ref[:s_idx_orig]
            suffix_actions = actions_list_ref[e_idx_orig + 1:]
        elif start_time_ms is not None and end_time_ms is not None:
            res_s_idx, res_e_idx = self._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            if res_s_idx is None or res_e_idx is None or (res_e_idx - res_s_idx + 1) < 2:
                self.logger.warning("Not enough points in time range for RDP.")
                return
            s_idx_orig, e_idx_orig = res_s_idx, res_e_idx
            segment_to_simplify = actions_list_ref[s_idx_orig:e_idx_orig + 1]
            prefix_actions = actions_list_ref[:s_idx_orig]
            suffix_actions = actions_list_ref[e_idx_orig + 1:]
        else:
            if len(actions_list_ref) < 2:
                self.logger.warning("Not enough points in full script for RDP.")
                return
            s_idx_orig, e_idx_orig = 0, len(actions_list_ref) - 1
            segment_to_simplify = list(actions_list_ref)

        if len(segment_to_simplify) < 2:
            self.logger.info("Segment for RDP has < 2 points.")
            return

        # --- RDP Simplification ---
        points = np.array([[a['at'], a['pos']] for a in segment_to_simplify], dtype=np.float64)

        def rdp_numpy(points, epsilon):
            if len(points) < 3:
                return points
            d = np.abs(np.cross(points[-1] - points[0], points[0:-1] - points[0])) / np.linalg.norm(
                points[-1] - points[0])
            max_index = np.argmax(d)
            max_distance = d[max_index]
            if max_distance > epsilon:
                left = rdp_numpy(points[:max_index + 1], epsilon)
                right = rdp_numpy(points[max_index:], epsilon)
                return np.vstack((left[:-1], right))
            else:
                return np.vstack((points[0], points[-1]))

        try:
            simplified_points = rdp_numpy(points, epsilon)

            # --- Reconstruct Actions (Preserve Start/End Exactly) ---
            new_segment_actions = []
            for i, p in enumerate(simplified_points):
                if i == 0:
                    new_segment_actions.append(segment_to_simplify[0])  # Exact start
                elif i == len(simplified_points) - 1:
                    new_segment_actions.append(segment_to_simplify[-1])  # Exact end
                else:
                    new_segment_actions.append({
                        'at': int(p[0]),  # Round time
                        'pos': int(np.clip(round(p[1]), 0, 100))  # Round position
                    })

            # Remove accidental duplicates (e.g., if RDP collapses to 1 point)
            if len(new_segment_actions) >= 2 and new_segment_actions[0] == new_segment_actions[-1]:
                new_segment_actions = new_segment_actions[:-1]

            # Update the original list
            actions_list_ref[:] = prefix_actions + new_segment_actions + suffix_actions

            # Update last timestamp
            last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
            if axis == 'primary':
                self.last_timestamp_primary = last_ts
            else:
                self.last_timestamp_secondary = last_ts

            self.logger.info(
                f"RDP applied to {axis} (indices {s_idx_orig}-{e_idx_orig}). "
                f"Points: {len(segment_to_simplify)} -> {len(new_segment_actions)} (e={epsilon})")

        except Exception as e:
            self.logger.error(f"RDP failed: {str(e)}")

    def find_peaks_and_valleys(self, axis: str,
                               height: Optional[float] = None, threshold: Optional[float] = None,
                               distance: Optional[float] = None, prominence: Optional[float] = None,
                               width: Optional[float] = None,
                               selected_indices: Optional[List[int]] = None):
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not installed. Peak finding cannot be applied.")
            return

        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(self, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 3:
            self.logger.warning(f"Not enough points on {axis} for peak finding.")
            return

        # --- Segment Selection ---
        s_idx_orig, e_idx_orig = 0, len(actions_list_ref) - 1
        if selected_indices:
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 3:
                self.logger.warning("Not enough valid selected indices for peak finding.")
                return
            s_idx_orig, e_idx_orig = valid_indices[0], valid_indices[-1]

        prefix_actions = actions_list_ref[:s_idx_orig]
        segment_to_process = actions_list_ref[s_idx_orig:e_idx_orig + 1]
        suffix_actions = actions_list_ref[e_idx_orig + 1:]

        if len(segment_to_process) < 3:
            # Nothing to process, restore original and exit
            actions_list_ref[:] = prefix_actions + segment_to_process + suffix_actions
            return

        # --- Peak and Valley Finding ---
        positions = np.array([a['pos'] for a in segment_to_process])
        inverted_positions = 100 - positions

        # Scipy find_peaks can return empty arrays, which is fine.
        # Ensure None parameters are not passed if they are 0, as find_peaks expects None or a number.
        kwargs = {
            'height': height if height else None,
            'threshold': threshold if threshold else None,
            'distance': distance if distance else None,
            'prominence': prominence if prominence else None,
            'width': width if width else None
        }

        peak_indices, _ = find_peaks(positions, **kwargs)
        valley_indices, _ = find_peaks(inverted_positions, **kwargs)

        # Combine, sort, and unique the indices
        # Also include the first and last points of the segment
        keyframe_indices = {0, len(segment_to_process) - 1}
        keyframe_indices.update(peak_indices)
        keyframe_indices.update(valley_indices)

        sorted_indices = sorted(list(keyframe_indices))

        # --- Reconstruct Actions ---
        new_segment_actions = [segment_to_process[i] for i in sorted_indices]

        # Update the original list
        actions_list_ref[:] = prefix_actions + new_segment_actions + suffix_actions

        # Update last timestamp
        last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
        if axis == 'primary':
            self.last_timestamp_primary = last_ts
        else:
            self.last_timestamp_secondary = last_ts

        self.logger.info(
            f"Peak simplification applied to {axis} (indices {s_idx_orig}-{e_idx_orig}). "
            f"Points: {len(segment_to_process)} -> {len(new_segment_actions)}")

    def clamp_points_thresholded(self, axis: str, lower_thresh: int, upper_thresh: int,
                                 start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                                 selected_indices: Optional[List[int]] = None):
        """
        Clamps points on an axis: if pos < lower_thresh, pos becomes 0. If pos > upper_thresh, pos becomes 100.
        """
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref:
            return

        indices_to_process: List[int] = []
        if selected_indices is not None:
            indices_to_process = [i for i in selected_indices if 0 <= i < len(actions_list_ref)]
        elif start_time_ms is not None and end_time_ms is not None:
            s_idx, e_idx = self._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                indices_to_process = list(range(s_idx, e_idx + 1))
        else:  # Apply to all
            indices_to_process = list(range(len(actions_list_ref)))

        if not indices_to_process:
            self.logger.debug(f"No points for threshold clamping on {axis} axis.")
            return

        count_changed = 0
        for idx in indices_to_process:
            original_pos = actions_list_ref[idx]['pos']
            new_pos = original_pos
            if original_pos < lower_thresh:
                new_pos = 0
            elif original_pos > upper_thresh:
                new_pos = 100

            if new_pos != original_pos:
                actions_list_ref[idx]['pos'] = new_pos
                count_changed += 1

        if count_changed > 0:
            self.logger.info(
                f"Applied threshold clamping to {count_changed} points on {axis} axis (Lower: {lower_thresh} -> 0, Upper: {upper_thresh} -> 100).")

    def _apply_to_points(self, axis: str, operation_func: Callable[[int], int],
                         start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                         selected_indices: Optional[List[int]] = None):
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref: return

        indices_to_process: List[int] = []
        if selected_indices is not None:
            indices_to_process = [i for i in selected_indices if 0 <= i < len(actions_list_ref)]
        elif start_time_ms is not None and end_time_ms is not None:
            s_idx, e_idx = self._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                indices_to_process = list(range(s_idx, e_idx + 1))
        else:  # Apply to all
            indices_to_process = list(range(len(actions_list_ref)))

        if not indices_to_process:
            self.logger.warning("No points for operation.")
            return
        count_changed = 0
        for idx in indices_to_process:
            actions_list_ref[idx]['pos'] = max(0, min(100, operation_func(actions_list_ref[idx]['pos'])))
            count_changed += 1
        if count_changed > 0: self.logger.info(f"Applied operation to {count_changed} points on {axis} axis.")

    def clamp_points_values(self, axis: str, clamp_value: int,
                            start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                            selected_indices: Optional[List[int]] = None):
        if clamp_value not in [0, 100]:
            self.logger.warning("Clamp value must be 0 or 100.")
            return
        self._apply_to_points(axis, lambda pos: clamp_value, start_time_ms, end_time_ms, selected_indices)

    def invert_points_values(self, axis: str,
                             start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                             selected_indices: Optional[List[int]] = None):
        self._apply_to_points(axis, lambda pos: 100 - pos, start_time_ms, end_time_ms, selected_indices)

    def clear_points(self, axis: str = 'both',
                     start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                     selected_indices: Optional[List[int]] = None):
        if axis not in ['primary', 'secondary', 'both']:
            self.logger.warning("Axis for clear_points must be 'primary', 'secondary', or 'both'.")
            return

        affected_axes_names: List[str] = []
        if axis == 'primary' or axis == 'both': affected_axes_names.append('primary')
        if axis == 'secondary' or axis == 'both': affected_axes_names.append('secondary')

        total_cleared_count = 0

        for axis_name in affected_axes_names:
            target_actions_list = self.primary_actions if axis_name == 'primary' else self.secondary_actions
            initial_len = len(target_actions_list)

            if selected_indices is not None:
                valid_indices_to_remove_set = set(i for i in selected_indices if 0 <= i < len(target_actions_list))
                if not valid_indices_to_remove_set: continue
                target_actions_list[:] = [action for i, action in enumerate(target_actions_list) if
                                          i not in valid_indices_to_remove_set]
            elif start_time_ms is not None and end_time_ms is not None:
                s_idx, e_idx = self._get_action_indices_in_time_range(target_actions_list, start_time_ms, end_time_ms)
                if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                    del target_actions_list[s_idx: e_idx + 1]
            else:
                target_actions_list[:] = []

            num_cleared_on_this_axis = initial_len - len(target_actions_list)
            total_cleared_count += num_cleared_on_this_axis
            # self.logger.debug(f"Cleared {num_cleared_on_this_axis} points from {axis_name} axis.")

            # Update last timestamp
            if axis_name == 'primary':
                self.last_timestamp_primary = target_actions_list[-1]['at'] if target_actions_list else 0
            else:
                self.last_timestamp_secondary = target_actions_list[-1]['at'] if target_actions_list else 0

        if total_cleared_count > 0:
            self.logger.info(
                f"Cleared {total_cleared_count} points across affected axes ({', '.join(affected_axes_names)}).")

    def amplify_points_values(self, axis: str, scale_factor: float, center_value: int = 50,
                              start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                              selected_indices: Optional[List[int]] = None):
        def operation_func(pos):
            deviation = pos - center_value
            new_pos = center_value + deviation * scale_factor
            return int(round(np.clip(new_pos, 0, 100)))

        self._apply_to_points(axis, operation_func, start_time_ms, end_time_ms, selected_indices)
        # self.logger.info(f"Applied amplification (Factor: {scale_factor}, Center: {center_value}) to points on {axis} axis.")

    def clear_actions_in_time_range(self, start_time_ms: int, end_time_ms: int, axis: str = 'both'):
        """Clears actions within a specified millisecond time range for the given axis or both."""
        if axis not in ['primary', 'secondary', 'both']:
            self.logger.warning("Axis for clear_actions_in_time_range must be 'primary', 'secondary', or 'both'.")
            return

        axes_to_process: List[Tuple[str, List[Dict]]] = []
        if axis == 'primary' or axis == 'both':
            axes_to_process.append(('primary', self.primary_actions))
        if axis == 'secondary' or axis == 'both':
            axes_to_process.append(('secondary', self.secondary_actions))

        total_cleared_count = 0
        for axis_name, actions_list_ref in axes_to_process:
            if not actions_list_ref:
                continue

            s_idx, e_idx = self._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)

            if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                num_to_clear = e_idx - s_idx + 1
                del actions_list_ref[s_idx: e_idx + 1]
                total_cleared_count += num_to_clear
                self.logger.debug(
                    f"Cleared {num_to_clear} points from {axis_name} axis between {start_time_ms}ms and {end_time_ms}ms.")

                # Update last timestamp
                if axis_name == 'primary':
                    self.last_timestamp_primary = actions_list_ref[-1]['at'] if actions_list_ref else 0
                else:
                    self.last_timestamp_secondary = actions_list_ref[-1]['at'] if actions_list_ref else 0
            else:
                self.logger.debug(
                    f"No points found to clear in {axis_name} axis between {start_time_ms}ms and {end_time_ms}ms.")

        if total_cleared_count > 0:
            self.logger.info(
                f"Total {total_cleared_count} points cleared in time range [{start_time_ms}ms - {end_time_ms}ms].")


    def shift_points_time(self, axis: str, time_delta_ms: int):
        """
        Shifts the timestamp of all points by a given millisecond delta.
        Ensures that no timestamp becomes negative.
        """
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref:
            return

        # Check for negative shift that would make the first point's timestamp negative
        if time_delta_ms < 0 and actions_list_ref[0]['at'] + time_delta_ms < 0:
            actual_delta_ms = -actions_list_ref[0]['at']
            self.logger.warning(
                f"Original shift of {time_delta_ms}ms was too large. "
                f"Adjusted to {actual_delta_ms}ms to prevent negative timestamps."
            )
        else:
            actual_delta_ms = time_delta_ms

        if actual_delta_ms == 0 and time_delta_ms != 0:
            self.logger.info("No shift applied as it would result in negative timestamps.")
            return

        for action in actions_list_ref:
            action['at'] += actual_delta_ms

        # Re-sorting is good practice, though not strictly necessary if all points are shifted equally.
        actions_list_ref.sort(key=lambda x: x['at'])

        # Update last timestamp for the axis
        last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
        if axis == 'primary':
            self.last_timestamp_primary = last_ts
        else:
            self.last_timestamp_secondary = last_ts

        self.logger.info(f"Shifted {len(actions_list_ref)} points on {axis} axis by {actual_delta_ms}ms.")

    def add_actions_batch(self, actions_data: List[Dict], is_from_live_tracker: bool = False):
        """
        Adds a batch of actions efficiently. It temporarily disables min_interval checks
        during addition and re-applies it once at the end.
        """
        original_min_interval = self.min_interval_ms
        self.min_interval_ms = 0  # Temporarily disable for batch insertion

        for action in actions_data:
            self.add_action(
                timestamp_ms=action['timestamp_ms'],
                primary_pos=action.get('primary_pos'),
                secondary_pos=action.get('secondary_pos'),
                is_from_live_tracker=is_from_live_tracker
            )

        self.min_interval_ms = original_min_interval

        # Re-apply min_interval filtering to the entire list after batch addition
        if self.min_interval_ms > 0:
            for actions_list in [self.primary_actions, self.secondary_actions]:
                if len(actions_list) > 1:
                    valid_idx = 0
                    for i in range(1, len(actions_list)):
                        if actions_list[i]['at'] - actions_list[valid_idx]['at'] >= self.min_interval_ms:
                            valid_idx += 1
                            if i != valid_idx:
                                actions_list[valid_idx] = actions_list[i]
                    if valid_idx + 1 < len(actions_list):
                        del actions_list[valid_idx + 1:]

        # Update last timestamps
        self.last_timestamp_primary = self.primary_actions[-1]['at'] if self.primary_actions else 0
        self.last_timestamp_secondary = self.secondary_actions[-1]['at'] if self.secondary_actions else 0

    def scale_points_to_range(self, axis: str, output_min: int, output_max: int,
                              start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                              selected_indices: Optional[List[int]] = None):
        """
        Scales the position of points within a selection to a new output range,
        disregarding outliers when determining the signal's current range.
        """
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref or len(actions_list_ref) < 2:
            return

        # Determine which indices to process
        indices_to_process: List[int] = []
        if selected_indices is not None:
            indices_to_process = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
        elif start_time_ms is not None and end_time_ms is not None:
            s_idx, e_idx = self._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            if s_idx is not None and e_idx is not None:
                indices_to_process = list(range(s_idx, e_idx + 1))
        else:
            indices_to_process = list(range(len(actions_list_ref)))

        if len(indices_to_process) < 2:
            self.logger.info(f"Not enough points in selection for range scaling on {axis} axis.")
            return

        # --- Use percentiles to ignore outliers ---
        positions_in_segment = np.array([actions_list_ref[i]['pos'] for i in indices_to_process])

        # Use percentiles to find the effective min/max, ignoring the top and bottom 5% as outliers
        effective_min = np.percentile(positions_in_segment, 10)
        effective_max = np.percentile(positions_in_segment, 90)

        current_effective_range = effective_max - effective_min
        target_range = output_max - output_min

        if current_effective_range <= 0:  # If there's no variation in the main signal body
            # Set all points to the middle of the target range
            new_pos = int(round(output_min + target_range / 2.0))
            for idx in indices_to_process:
                actions_list_ref[idx]['pos'] = new_pos
            self.logger.info(f"Scaled {len(indices_to_process)} flat points on {axis} axis to {new_pos}.")
            return

        # Apply the scaling based on the effective range
        for idx in indices_to_process:
            original_pos = actions_list_ref[idx]['pos']
            # Normalize the position from 0-1 based on the effective range
            normalized_pos = (original_pos - effective_min) / current_effective_range
            # Clip the normalized value to handle outliers (points outside the 5-95 percentile range)
            clipped_normalized_pos = np.clip(normalized_pos, 0.0, 1.0)
            # Scale to the new target range
            new_pos = int(round(output_min + clipped_normalized_pos * target_range))
            actions_list_ref[idx]['pos'] = np.clip(new_pos, 0, 100)  # Final safety clip

        self.logger.info(
            f"Scaled {len(indices_to_process)} points on {axis} axis to new range [{output_min}-{output_max}].")

    # In dual_axis_funscript.py

    def calculate_filter_preview(self, axis: str, filter_type: str, filter_params: Dict,
                                 selected_indices: Optional[List[int]] = None) -> Optional[List[Dict]]:
        """
        Calculates the result of a filter without modifying the instance's data.
        Returns a new list of actions representing the full script with the filter applied
        to the specified segment, or None if the operation is not possible.
        """
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref or len(actions_list_ref) < 2:
            return None

        # --- 1. Determine the segment of actions to process ---
        s_idx_orig, e_idx_orig = -1, -1

        # Note: The Speed Limiter is designed to work on the full script, not a selection.
        # For other filters, we honor the selection.
        if selected_indices is not None and len(selected_indices) > 0 and filter_type != 'speed_limiter':
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 2: return None
            s_idx_orig, e_idx_orig = valid_indices[0], valid_indices[-1]
        else:
            s_idx_orig, e_idx_orig = 0, len(actions_list_ref) - 1

        # --- Create DEEP copies of the action dictionaries ---
        prefix_actions = [dict(a) for a in actions_list_ref[:s_idx_orig]]
        segment_to_process = [dict(a) for a in actions_list_ref[s_idx_orig:e_idx_orig + 1]]
        suffix_actions = [dict(a) for a in actions_list_ref[e_idx_orig + 1:]]

        if len(segment_to_process) < 2:
            return None

        # --- 2. Apply the selected filter logic ---
        new_segment_actions = []
        if filter_type == 'sg':
            if not SCIPY_AVAILABLE: return None

            window = filter_params.get('window_length', 7)
            poly = filter_params.get('polyorder', 3)

            if window % 2 == 0: window += 1
            if poly >= window: poly = window - 1
            if poly < 0: poly = 0
            if len(segment_to_process) < window: return None

            positions = np.array([a['pos'] for a in segment_to_process])
            smoothed_pos = savgol_filter(positions, window, poly)

            new_segment_actions = segment_to_process
            for i, action in enumerate(new_segment_actions):
                action['pos'] = int(round(np.clip(smoothed_pos[i], 0, 100)))

        elif filter_type == 'rdp':
            epsilon = filter_params.get('epsilon', 1.0)
            points = np.array([[a['at'], a['pos']] for a in segment_to_process], dtype=np.float64)

            def rdp_numpy(points, epsilon):
                if len(points) < 3: return points
                d = np.abs(np.cross(points[-1] - points[0], points[0:-1] - points[0])) / np.linalg.norm(
                    points[-1] - points[0])
                max_index = np.argmax(d)
                if d[max_index] > epsilon:
                    left = rdp_numpy(points[:max_index + 1], epsilon)
                    right = rdp_numpy(points[max_index:], epsilon)
                    return np.vstack((left[:-1], right))
                else:
                    return np.vstack((points[0], points[-1]))

            simplified_points = rdp_numpy(points, epsilon)

            for i, p in enumerate(simplified_points):
                if i == 0:
                    new_segment_actions.append(segment_to_process[0])
                elif i == len(simplified_points) - 1:
                    new_segment_actions.append(segment_to_process[-1])
                else:
                    new_segment_actions.append({'at': int(p[0]), 'pos': int(np.clip(round(p[1]), 0, 100))})
            if len(new_segment_actions) >= 2 and new_segment_actions[0] == new_segment_actions[-1]:
                new_segment_actions = new_segment_actions[:-1]

        elif filter_type == 'peaks':
            if not SCIPY_AVAILABLE: return None
            if len(segment_to_process) < 3: return None

            positions = np.array([a['pos'] for a in segment_to_process])
            inverted_positions = 100 - positions

            # Get params, defaulting to None if 0
            kwargs = {
                'height': filter_params.get('height') if filter_params.get('height') else None,
                'threshold': filter_params.get('threshold') if filter_params.get('threshold') else None,
                'distance': filter_params.get('distance') if filter_params.get('distance') else None,
                'prominence': filter_params.get('prominence') if filter_params.get('prominence') else None,
                'width': filter_params.get('width') if filter_params.get('width') else None,
            }

            peak_indices, _ = find_peaks(positions, **kwargs)
            valley_indices, _ = find_peaks(inverted_positions, **kwargs)

            keyframe_indices = {0, len(segment_to_process) - 1}
            keyframe_indices.update(peak_indices)
            keyframe_indices.update(valley_indices)

            new_segment_actions = [segment_to_process[i] for i in sorted(list(keyframe_indices))]

        elif filter_type == 'amp':
            scale_factor = filter_params.get('scale_factor', 1.0)
            center_value = filter_params.get('center_value', 50)

            def operation_func(pos):
                deviation = pos - center_value
                new_pos = center_value + deviation * scale_factor
                return int(round(np.clip(new_pos, 0, 100)))

            new_segment_actions = segment_to_process
            for action in new_segment_actions:
                action['pos'] = operation_func(action['pos'])

        elif filter_type == 'keyframe':
            position_tolerance = filter_params.get('position_tolerance', 10)
            time_tolerance_ms = filter_params.get('time_tolerance_ms', 50)

            if len(segment_to_process) < 3:
                new_segment_actions = segment_to_process
            else:
                extrema = [segment_to_process[0]]
                for i in range(1, len(segment_to_process) - 1):
                    p_prev, p_curr, p_next = segment_to_process[i - 1]['pos'], segment_to_process[i]['pos'], \
                        segment_to_process[i + 1]['pos']
                    if (p_curr > p_prev and p_curr >= p_next) or (p_curr < p_prev and p_curr <= p_next):
                        extrema.append(segment_to_process[i])
                extrema.append(segment_to_process[-1])

                while len(extrema) > 2:
                    min_significance = float('inf')
                    weakest_link_idx = -1
                    for i in range(1, len(extrema) - 1):
                        p_prev, p_curr, p_next = extrema[i - 1], extrema[i], extrema[i + 1]
                        duration = float(p_next['at'] - p_prev['at'])
                        if duration > 0:
                            progress = (p_curr['at'] - p_prev['at']) / duration
                            projected_pos = p_prev['pos'] + progress * (p_next['pos'] - p_prev['pos'])
                            significance = abs(p_curr['pos'] - projected_pos)
                        else:
                            significance = float('inf')
                        if significance < min_significance:
                            min_significance = significance
                            weakest_link_idx = i

                    if weakest_link_idx != -1 and min_significance < position_tolerance:
                        extrema.pop(weakest_link_idx)
                    else:
                        break

                if time_tolerance_ms > 0 and len(extrema) > 1:
                    final_keyframes = [extrema[0]]
                    for i in range(1, len(extrema)):
                        if (extrema[i]['at'] - final_keyframes[-1]['at']) >= time_tolerance_ms:
                            final_keyframes.append(extrema[i])
                        else:
                            if abs(extrema[i]['pos'] - 50) > abs(final_keyframes[-1]['pos'] - 50):
                                final_keyframes[-1] = extrema[i]
                else:
                    final_keyframes = extrema

                new_segment_actions = final_keyframes

        elif filter_type == 'speed_limiter':
            min_interval = filter_params.get('min_interval', 60)
            vibe_amount = filter_params.get('vibe_amount', 10)
            speed_threshold = filter_params.get('speed_threshold', 500.0)

            # Since this filter works on the whole script, we use the full reference
            actions = copy.deepcopy(actions_list_ref)

            # 1. Remove actions with short intervals
            if len(actions) > 1:
                reversed_actions = list(reversed(actions))
                last_action_at = reversed_actions[0]['at']
                actions_to_keep = [reversed_actions[0]]
                for i in range(1, len(reversed_actions)):
                    interval = abs(reversed_actions[i]['at'] - last_action_at)
                    if interval >= min_interval:
                        actions_to_keep.append(reversed_actions[i])
                        last_action_at = reversed_actions[i]['at']
                actions = sorted(actions_to_keep, key=lambda x: x['at'])

            # 2. Replace flat sections with vibrations
            if len(actions) > 2:
                last_action_at_vibe = actions[0]['at']
                last_vibe = ''
                unmod_last_action_height = 0
                for i in range(1, len(actions)):
                    current = actions[i]
                    last = actions[i - 1]
                    next_pos = actions[i + 1] if (i + 1) < len(actions) else None
                    travel = abs(current['pos'] - unmod_last_action_height)
                    unmod_last_direction = 'up' if current['pos'] > unmod_last_action_height else (
                        'down' if current['pos'] < unmod_last_action_height else '')
                    last_direction = 'up' if current['pos'] > last['pos'] else (
                        'down' if current['pos'] < last['pos'] else '')
                    next_direction = 'up' if next_pos and current['pos'] < next_pos['pos'] else (
                        'down' if next_pos and current['pos'] > next_pos['pos'] else '')
                    already_vibing = 1 if next_pos and not (unmod_last_direction == next_direction) and (
                                (abs(current['pos'] - unmod_last_action_height) > 8) or (
                                    abs(current['pos'] - next_pos['pos']) > 8)) else 0
                    next_travel = abs(next_pos['pos'] - current['pos']) if next_pos else float('inf')
                    interval = current['at'] - last_action_at_vibe
                    unmod_last_action_height = current['pos']

                    if (travel < 16) and (next_travel < 16) and (interval < 135) and (already_vibing == 0) and (
                            min_interval <= 134):
                        if not last_vibe:
                            if (last_direction == 'up') or (current['pos'] < 6):
                                last_vibe = 'down'
                            elif (last_direction == 'down') or (current['pos'] > 94):
                                last_vibe = 'up'
                            elif current['pos'] < 50:
                                last_vibe = 'down'
                            else:
                                last_vibe = 'up'

                        if last_vibe == 'down':
                            current['pos'] += vibe_amount
                            last_vibe = 'up'
                        elif last_vibe == 'up':
                            current['pos'] -= vibe_amount
                            last_vibe = 'down'
                    else:
                        last_vibe = ''
                    last_action_at_vibe = current['at']
                    current['pos'] = int(round(np.clip(current['pos'], 0, 100)))

            # 3. Limit speed
            if len(actions) > 1:
                for i in range(1, len(actions)):
                    current_action = actions[i]
                    prev_action = actions[i - 1]
                    time_diff_s = (current_action['at'] - prev_action['at']) / 1000.0
                    if time_diff_s == 0: continue
                    pos_diff = abs(current_action['pos'] - prev_action['pos'])
                    speed = pos_diff / time_diff_s
                    if speed > speed_threshold:
                        new_pos_diff = speed_threshold * time_diff_s
                        current_action['pos'] = int(prev_action['pos'] + new_pos_diff) if current_action['pos'] > \
                                                                                          prev_action['pos'] else int(
                            prev_action['pos'] - new_pos_diff)
                        current_action['pos'] = int(round(np.clip(current_action['pos'], 0, 100)))

            return actions

        else:
            return None

        # --- 3. Return the full, reconstructed list of actions ---
        return prefix_actions + new_segment_actions + suffix_actions


    def apply_peak_preserving_resample(self, axis: str, resample_rate_ms: int = 50,
                                       selected_indices: Optional[List[int]] = None):
        """
        Applies a custom resampling algorithm that preserves the timing of peaks and
        valleys while creating smooth, sinusoidal transitions between them.

        :param axis: The axis to process ('primary' or 'secondary').
        :param resample_rate_ms: The time interval for the newly generated points.
        :param selected_indices: Optional list of indices to apply the filter to.
        """
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(self, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 3:
            self.logger.info("Not enough points for Peak-Preserving Resampling.")
            return

        # --- 1. Determine the segment to process ---
        s_idx, e_idx = 0, len(actions_list_ref) - 1
        if selected_indices:
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 3:
                self.logger.info("Not enough selected points for resampling.")
                return
            s_idx, e_idx = valid_indices[0], valid_indices[-1]

        prefix_actions = actions_list_ref[:s_idx]
        segment_to_process = actions_list_ref[s_idx:e_idx + 1]
        suffix_actions = actions_list_ref[e_idx + 1:]

        # --- 2. Identify Peaks and Valleys (the Anchors) ---
        anchors = []
        if not segment_to_process: return

        # Always include the very first and last points of the segment as anchors
        anchors.append(segment_to_process[0])

        for i in range(1, len(segment_to_process) - 1):
            p_prev = segment_to_process[i - 1]['pos']
            p_curr = segment_to_process[i]['pos']
            p_next = segment_to_process[i + 1]['pos']

            # Check for local peak
            if p_curr > p_prev and p_curr > p_next:
                anchors.append(segment_to_process[i])
            # Check for local valley
            elif p_curr < p_prev and p_curr < p_next:
                anchors.append(segment_to_process[i])
            # Check for flat peak/valley (e.g., 80, 90, 90, 80)
            elif p_curr == p_next and p_curr != p_prev:
                # Look ahead to find the end of the flat section
                j = i
                while j < len(segment_to_process) - 1 and segment_to_process[j]['pos'] == p_curr:
                    j += 1
                p_after_flat = segment_to_process[j]['pos']

                # If it's a peak or valley, add the middle point of the flat section
                if (p_curr > p_prev and p_curr > p_after_flat) or \
                        (p_curr < p_prev and p_curr < p_after_flat):
                    anchor_candidate = segment_to_process[(i + j - 1) // 2]
                    if not anchors or anchors[-1] != anchor_candidate:
                        anchors.append(anchor_candidate)

        # Always include the last point, ensuring no duplicates
        if not anchors or anchors[-1] != segment_to_process[-1]:
            anchors.append(segment_to_process[-1])

        # --- 3. Generate new points with Cosine Easing between anchors ---
        new_actions = []
        if not anchors: return  # Should not happen

        new_actions.append(anchors[0])  # Start with the first anchor

        for i in range(len(anchors) - 1):
            p1 = anchors[i]
            p2 = anchors[i + 1]

            t1, pos1 = p1['at'], p1['pos']
            t2, pos2 = p2['at'], p2['pos']

            duration = float(t2 - t1)
            pos_delta = float(pos2 - pos1)

            if duration <= 0:
                continue

            # Start generating new points from the next time step after p1
            current_time = t1 + resample_rate_ms
            while current_time < t2:
                # Calculate progress and apply cosine easing
                progress = (current_time - t1) / duration
                eased_progress = (1 - np.cos(progress * np.pi)) / 2.0

                new_pos = pos1 + eased_progress * pos_delta

                new_actions.append({
                    'at': int(current_time),
                    'pos': int(round(np.clip(new_pos, 0, 100)))
                })
                current_time += resample_rate_ms

            # Add the next anchor, ensuring no duplicates
            if not new_actions or new_actions[-1]['at'] < p2['at']:
                new_actions.append(p2)

        # --- 4. Replace the old segment with the new resampled actions ---
        actions_list_ref[:] = prefix_actions + new_actions + suffix_actions

        self.logger.info(
            f"Applied Peak-Preserving Resample to {axis}. "
            f"Points: {len(segment_to_process)} -> {len(new_actions)}")


    def simplify_to_keyframes(self, axis: str, position_tolerance: int = 10, time_tolerance_ms: int = 50,
                              selected_indices: Optional[List[int]] = None):
        """
        FINAL CORRECTED ALGORITHM: Uses iterative global refinement to simplify the
        script to only the most significant peaks and valleys. This version contains
        the corrected significance calculation.
        """
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(self, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 3: return

        s_idx, e_idx = 0, len(actions_list_ref) - 1
        if selected_indices:
            valid_indices = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
            if len(valid_indices) < 3: return
            s_idx, e_idx = valid_indices[0], valid_indices[-1]

        prefix_actions = actions_list_ref[:s_idx]
        segment_to_process = actions_list_ref[s_idx:e_idx + 1]
        suffix_actions = actions_list_ref[e_idx + 1:]

        if len(segment_to_process) < 3:
            actions_list_ref[:] = prefix_actions + segment_to_process + suffix_actions
            return

        # Pass 1: Find all local extrema (peaks and valleys)
        extrema = [segment_to_process[0]]
        for i in range(1, len(segment_to_process) - 1):
            p_prev, p_curr, p_next = segment_to_process[i - 1]['pos'], segment_to_process[i]['pos'], segment_to_process[i + 1]['pos']
            if (p_curr > p_prev and p_curr >= p_next) or (p_curr < p_prev and p_curr <= p_next):
                extrema.append(segment_to_process[i])
        extrema.append(segment_to_process[-1])

        # Pass 2: Iteratively remove the least significant extremum
        while len(extrema) > 2:
            min_significance = float('inf')
            weakest_link_idx = -1

            for i in range(1, len(extrema) - 1):
                p_prev, p_curr, p_next = extrema[i - 1], extrema[i], extrema[i + 1]
                duration = float(p_next['at'] - p_prev['at'])

                if duration > 0:
                    progress = (p_curr['at'] - p_prev['at']) / duration
                    # CORRECTED a bug in the projection formula
                    projected_pos = p_prev['pos'] + progress * (p_next['pos'] - p_prev['pos'])
                    significance = abs(p_curr['pos'] - projected_pos)
                else:
                    significance = float('inf')

                if significance < min_significance:
                    min_significance = significance
                    weakest_link_idx = i

            if weakest_link_idx != -1 and min_significance < position_tolerance:
                extrema.pop(weakest_link_idx)
            else:
                break

        # Pass 3: Enforce time tolerance
        if time_tolerance_ms > 0 and len(extrema) > 1:
            final_keyframes = [extrema[0]]
            for i in range(1, len(extrema)):
                if (extrema[i]['at'] - final_keyframes[-1]['at']) >= time_tolerance_ms:
                    final_keyframes.append(extrema[i])
                else:
                    if abs(extrema[i]['pos'] - 50) > abs(final_keyframes[-1]['pos'] - 50):
                        final_keyframes[-1] = extrema[i]
        else:
            final_keyframes = extrema

        actions_list_ref[:] = prefix_actions + final_keyframes + suffix_actions

    def apply_speed_limiter(self, axis: str, min_interval: int, vibe_amount: int, speed_threshold: float):
        """
        Applies a series of filters to make a script more compatible with Handy in Bluetooth mode.
        This process involves removing rapid actions, replacing small movements with vibrations,
        and limiting the maximum speed.
        """
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        actions_list_ref = getattr(self, target_list_attr)

        if not actions_list_ref or len(actions_list_ref) < 2:
            self.logger.info(f"Not enough points on {axis} axis for Speed Limiter.")
            return

        # Operate on a deep copy of the actions
        actions = copy.deepcopy(actions_list_ref)

        # --- 1. Remove actions with short intervals ---
        action_points_removed = 0
        if len(actions) > 1:
            reversed_actions = list(reversed(actions))
            last_action_at = reversed_actions[0]['at']

            # Build a new list of actions to keep, avoiding modification during iteration
            actions_to_keep_after_interval_check = [reversed_actions[0]]

            for i in range(1, len(reversed_actions)):
                interval = abs(reversed_actions[i]['at'] - last_action_at)
                if interval < min_interval:
                    action_points_removed += 1
                else:
                    actions_to_keep_after_interval_check.append(reversed_actions[i])
                    last_action_at = reversed_actions[i]['at']

            # Restore the original order
            actions = sorted(actions_to_keep_after_interval_check, key=lambda x: x['at'])

        if action_points_removed > 0:
            self.logger.info(
                f"Speed Limiter: {action_points_removed} action points removed due to min interval ({min_interval}ms).")

        # --- 2. Replace flat sections with vibrations ---
        action_points_modified = 0
        if len(actions) > 2:
            last_action_at = actions[0]['at']
            already_vibing = 0
            last_vibe = ''
            unmod_last_action_height = 0

            # This loop modifies the 'actions' list in place
            for i in range(1, len(actions)):
                current = actions[i]
                last = actions[i - 1]
                next_pos = actions[i + 1] if (i + 1) < len(actions) else None

                travel = abs(current['pos'] - unmod_last_action_height)

                # Determine directions
                unmod_last_direction = 'up' if current['pos'] > unmod_last_action_height else (
                    'down' if current['pos'] < unmod_last_action_height else '')
                last_direction = 'up' if current['pos'] > last['pos'] else (
                    'down' if current['pos'] < last['pos'] else '')
                if next_pos:
                    next_direction = 'up' if current['pos'] < next_pos['pos'] else (
                        'down' if current['pos'] > next_pos['pos'] else '')
                else:
                    next_direction = ''

                # Detect if undulating movement is already happening
                if next_pos and not (unmod_last_direction == next_direction) and (
                        (abs(current['pos'] - unmod_last_action_height) > 8) or (
                        abs(current['pos'] - next_pos['pos']) > 8)):
                    already_vibing = 1
                else:
                    already_vibing = 0

                if next_pos:
                    next_travel = abs(next_pos['pos'] - current['pos'])
                else:
                    next_travel = float('inf')

                interval = current['at'] - last_action_at

                unmod_last_action_height = current['pos']

                # Main condition to insert a vibe
                # The original script disables vibe mods if min_interval is > 134
                if (travel < 16) and (next_travel < 16) and (interval < 135) and (already_vibing == 0) and (
                        min_interval <= 134):
                    # Determine starting direction of the vibration
                    if not last_vibe:
                        if (last_direction == 'up') or (current['pos'] < 6):
                            last_vibe = 'down'
                        elif (last_direction == 'down') or (current['pos'] > 94):
                            last_vibe = 'up'
                        # Default fallback
                        elif current['pos'] < 50:
                            last_vibe = 'down'
                        else:
                            last_vibe = 'up'

                    # Apply the vibration
                    if last_vibe == 'down':
                        current['pos'] += vibe_amount
                        last_vibe = 'up'
                    elif last_vibe == 'up':
                        current['pos'] -= vibe_amount
                        last_vibe = 'down'

                    action_points_modified += 1
                else:
                    last_vibe = ''

                last_action_at = current['at']

                # Sanity check position
                current['pos'] = int(round(np.clip(current['pos'], 0, 100)))

        if action_points_modified > 0:
            self.logger.info(f"Speed Limiter: {action_points_modified} action points modified to vibrate.")

        # --- 3. Limit speed ---
        speed_modified = 0
        if len(actions) > 1:
            for i in range(1, len(actions)):
                current_action = actions[i]
                prev_action = actions[i - 1]

                time_diff_s = (current_action['at'] - prev_action['at']) / 1000.0
                if time_diff_s == 0:
                    continue

                pos_diff = abs(current_action['pos'] - prev_action['pos'])
                speed = pos_diff / time_diff_s

                if speed > speed_threshold:
                    speed_modified += 1
                    new_pos_diff = speed_threshold * time_diff_s

                    if current_action['pos'] > prev_action['pos']:
                        current_action['pos'] = int(prev_action['pos'] + new_pos_diff)
                    else:
                        current_action['pos'] = int(prev_action['pos'] - new_pos_diff)

                    current_action['pos'] = int(round(np.clip(current_action['pos'], 0, 100)))

        if speed_modified > 0:
            self.logger.info(f"Speed Limiter: {speed_modified} action points modified to reduce speed.")

        # --- Finalize ---
        if action_points_removed > 0 or action_points_modified > 0 or speed_modified > 0:
            # Replace the original actions list with the modified one
            actions_list_ref[:] = actions

            # Update last timestamp for the axis
            last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
            if axis == 'primary':
                self.last_timestamp_primary = last_ts
            else:
                self.last_timestamp_secondary = last_ts

            self.logger.info(f"Speed Limiter applied successfully to {axis} axis.")
        else:
            self.logger.info(f"Speed Limiter: No changes were necessary for {axis} axis.")

