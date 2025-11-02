import numpy as np
from typing import Optional, Callable, List, Tuple, Dict, Any
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
        self.chapters: List[Dict] = []  # Funscript chapters/segments
        self.min_interval_ms: int = 20
        self.last_timestamp_primary: int = 0
        self.last_timestamp_secondary: int = 0

        # Timestamp caching mechanism
        self._primary_timestamps_cache: List[int] = []
        self._secondary_timestamps_cache: List[int] = []
        self._cache_dirty_primary: bool = True
        self._cache_dirty_secondary: bool = True

        # Point simplification settings
        self.enable_point_simplification: bool = True  # Enable by default

        # Point simplification statistics
        self._simplification_stats_primary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
        self._simplification_stats_secondary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
        self._last_simplification_log_time = 0
        self._simplification_log_interval_sec = 10  # Log every 10 seconds

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('DualAxisFunscript_fallback')
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())

    def _invalidate_cache(self, axis: str = 'both'):
        """Marks the timestamp cache(s) as dirty."""
        if axis == 'primary' or axis == 'both':
            self._cache_dirty_primary = True
        if axis == 'secondary' or axis == 'both':
            self._cache_dirty_secondary = True

    def _maybe_log_simplification_stats(self):
        """
        Periodically log point simplification statistics (every 10 seconds).
        Shows time window, frames considered, and reduction percentage.
        """
        import time
        current_time = time.time()

        # Only log every N seconds to avoid spam
        if current_time - self._last_simplification_log_time < self._simplification_log_interval_sec:
            return

        self._last_simplification_log_time = current_time
        self._log_simplification_stats_internal()

    def _log_simplification_stats_internal(self):
        """Internal helper to log stats (called by periodic logger and final summary)."""
        # Log stats for primary axis if active
        stats_p = self._simplification_stats_primary
        if stats_p['total_considered'] > 0:
            reduction_pct = (stats_p['total_removed'] / stats_p['total_considered']) * 100
            current_points = len(self.primary_actions)
            would_have_been = current_points + stats_p['total_removed']

            # Calculate time window
            if current_points > 0 and stats_p['start_time_ms'] > 0:
                time_window_ms = self.primary_actions[-1]['at'] - stats_p['start_time_ms']
                time_window_sec = time_window_ms / 1000.0

                self.logger.info(
                    f"📊 Point Simplification (Primary): {time_window_sec:.1f}s window, "
                    f"{stats_p['total_considered']:,} frames → {stats_p['total_removed']:,} points removed ({reduction_pct:.1f}% reduction), "
                    f"{would_have_been:,} → {current_points:,} points"
                )

        # Log stats for secondary axis if active
        stats_s = self._simplification_stats_secondary
        if stats_s['total_considered'] > 0:
            reduction_pct = (stats_s['total_removed'] / stats_s['total_considered']) * 100
            current_points = len(self.secondary_actions)
            would_have_been = current_points + stats_s['total_removed']

            # Calculate time window
            if current_points > 0 and stats_s['start_time_ms'] > 0:
                time_window_ms = self.secondary_actions[-1]['at'] - stats_s['start_time_ms']
                time_window_sec = time_window_ms / 1000.0

                self.logger.info(
                    f"📊 Point Simplification (Secondary): {time_window_sec:.1f}s window, "
                    f"{stats_s['total_considered']:,} frames → {stats_s['total_removed']:,} points removed ({reduction_pct:.1f}% reduction), "
                    f"{would_have_been:,} → {current_points:,} points"
                )

    def log_final_simplification_summary(self):
        """
        Log final point simplification summary (called when tracking stops).
        Forces a log regardless of time interval.
        """
        # Force log if any simplification happened
        if (self._simplification_stats_primary['total_considered'] > 0 or
            self._simplification_stats_secondary['total_considered'] > 0):
            self.logger.info("📊 Final Point Simplification Summary:")
            self._log_simplification_stats_internal()
            # Reset stats for next session
            self._simplification_stats_primary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
            self._simplification_stats_secondary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
            self._last_simplification_log_time = 0

    def _simplify_last_points(self, actions_list: List[Dict], axis: str = 'primary') -> None:
        """
        Ultra-lightweight point simplification that only checks the last 3 points.
        Removes middle point if all 3 have equal position OR are collinear.

        This runs on every frame so it must be EXTREMELY fast:
        - Only checks last 3 points (constant time O(1))
        - Simple integer arithmetic only
        - No loops, no numpy, no complex math
        - Early exits for common cases
        """
        # Track statistics for this axis
        stats = self._simplification_stats_primary if axis == 'primary' else self._simplification_stats_secondary

        # Need at least 3 points to simplify
        if len(actions_list) < 3:
            return

        # Initialize start time if first simplification
        if stats['start_time_ms'] == 0 and len(actions_list) >= 3:
            stats['start_time_ms'] = actions_list[-3]['at']

        stats['total_considered'] += 1

        # Get the last 3 points (direct list access is fastest)
        p1 = actions_list[-3]
        p2 = actions_list[-2]
        p3 = actions_list[-1]

        pos1, pos2, pos3 = p1['pos'], p2['pos'], p3['pos']

        # Fast check 1: All positions equal (most common redundant case)
        if pos1 == pos2 == pos3:
            actions_list.pop(-2)  # Remove middle point
            stats['total_removed'] += 1
            self._maybe_log_simplification_stats()
            return

        # Fast check 2: Collinear test using integer cross product
        # For points (t1,pos1), (t2,pos2), (t3,pos3) to be collinear:
        # (t2-t1)*(pos3-pos1) == (t3-t1)*(pos2-pos1)
        # We allow tolerance of 1 position unit for floating point errors

        t1, t2, t3 = p1['at'], p2['at'], p3['at']

        # Cross product calculation (all integer math)
        cross = (t2 - t1) * (pos3 - pos1) - (t3 - t1) * (pos2 - pos1)

        # Normalize by time range to make it position-based
        time_range = t3 - t1
        if time_range == 0:
            return  # Can't determine if timestamps are identical

        # If normalized cross product is ≤ time_range (equivalent to 1 pos unit tolerance)
        # then points are collinear within tolerance
        if abs(cross) <= time_range:
            actions_list.pop(-2)  # Remove redundant middle point
            stats['total_removed'] += 1
            self._maybe_log_simplification_stats()

    def _get_timestamps_for_axis(self, axis: str) -> List[int]:
        """
        Returns a cached list of timestamps for the specified axis,
        regenerating it from the actions list only if necessary.
        """
        if axis == 'primary':
            if self._cache_dirty_primary:
                self._primary_timestamps_cache = [a["at"] for a in self.primary_actions]
                self._cache_dirty_primary = False
            return self._primary_timestamps_cache
        else: # secondary
            if self._cache_dirty_secondary:
                self._secondary_timestamps_cache = [a["at"] for a in self.secondary_actions]
                self._cache_dirty_secondary = False
            return self._secondary_timestamps_cache

    def _process_action_for_axis(self,
                                 actions_target_list: List[Dict],
                                 timestamp_ms: int,
                                 pos: int,
                                 min_interval_ms: int,
                                 axis_name: str # 'primary' or 'secondary'
                                 ) -> int:
        """
        Processes and adds/updates a single action in the target list (in-place).
        Optimized with a timestamp cache.
        Returns the timestamp of the last action in the list.
        """
        clamped_pos = max(0, min(100, pos))
        new_action = {"at": timestamp_ms, "pos": clamped_pos}

        # Use the cached timestamps for performance
        action_timestamps = self._get_timestamps_for_axis(axis_name)
        idx = bisect.bisect_left(action_timestamps, timestamp_ms)

        action_inserted_or_updated = False
        if idx < len(actions_target_list) and actions_target_list[idx]["at"] == timestamp_ms:
            if actions_target_list[idx]["pos"] != clamped_pos:
                actions_target_list[idx]["pos"] = clamped_pos
                # No timestamp change, so cache is still valid
        else:
            can_insert = True
            if idx > 0 and len(actions_target_list) > 0:
                prev_action = actions_target_list[idx - 1]
                if timestamp_ms - prev_action["at"] < min_interval_ms:
                    can_insert = False

            if can_insert:
                actions_target_list.insert(idx, new_action)
                action_inserted_or_updated = True
                self._invalidate_cache(axis_name) # Cache is now dirty

                # Apply lightweight point simplification after insertion
                if self.enable_point_simplification:
                    self._simplify_last_points(actions_target_list, axis=axis_name)

        if action_inserted_or_updated and min_interval_ms > 0:
            if not actions_target_list:
                return 0

            original_len = len(actions_target_list)
            current_valid_idx = 0
            if len(actions_target_list) > 1:
                for i in range(1, len(actions_target_list)):
                    if actions_target_list[i]["at"] - actions_target_list[current_valid_idx]["at"] >= min_interval_ms:
                        current_valid_idx += 1
                        if i != current_valid_idx:
                            actions_target_list[current_valid_idx] = actions_target_list[i]

            if current_valid_idx + 1 < len(actions_target_list):
                del actions_target_list[current_valid_idx + 1:]

            # If filtering removed points, invalidate the cache again
            if len(actions_target_list) != original_len:
                self._invalidate_cache(axis_name)


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
        if primary_pos is not None:
            new_last_ts_primary = self._process_action_for_axis(
                actions_target_list=self.primary_actions,
                timestamp_ms=timestamp_ms,
                pos=primary_pos,
                min_interval_ms=self.min_interval_ms,
                axis_name='primary' # Pass axis name
            )
        # Update last_timestamp_primary only if actions were actually processed or if list became empty
        self.last_timestamp_primary = new_last_ts_primary if self.primary_actions else 0


        new_last_ts_secondary = self.last_timestamp_secondary
        if secondary_pos is not None:
            new_last_ts_secondary = self._process_action_for_axis(
                actions_target_list=self.secondary_actions,
                timestamp_ms=timestamp_ms,
                pos=secondary_pos,
                min_interval_ms=self.min_interval_ms,
                axis_name='secondary' # Pass axis name
            )
            self.last_timestamp_secondary = new_last_ts_secondary if self.secondary_actions else 0

    def reset_to_neutral(self, timestamp_ms: int):
        self.add_action(timestamp_ms, 100, 50, is_from_live_tracker=True)

    def get_value(self, time_ms: int, axis: str = 'primary') -> int:
        """
        [MODIFIED] Now thread-safe. Creates a local copy of the actions list
        to prevent race conditions during list clearing from other threads.
        """
        # Create a local, thread-safe copy of the actions list.
        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        actions_to_search = list(actions_list_ref) # A shallow copy is sufficient and fast.

        if not actions_to_search:
            return 50 # Default neutral position

        # All subsequent logic operates on the consistent 'actions_to_search' copy.
        # It's safer to derive timestamps directly from this copy rather than using the cache.
        action_timestamps = [a["at"] for a in actions_to_search]
        idx = bisect.bisect_left(action_timestamps, time_ms)

        # The rest of the logic is safe because 'actions_to_search' will not change.
        if idx == 0:
            return actions_to_search[0]["pos"]
        if idx == len(actions_to_search):
            return actions_to_search[-1]["pos"]

        p1 = actions_to_search[idx - 1]
        p2 = actions_to_search[idx]

        if time_ms == p1["at"]:
            return p1["pos"]

        # Denominator for interpolation
        time_diff = float(p2["at"] - p1["at"])
        if time_diff == 0:
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
        self._invalidate_cache('both') # Invalidate caches
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
                self.primary_actions = []
            else:
                # Create a new list from sorted items to ensure we don't keep a reference to a mutable 'value'
                self.primary_actions = sorted(list(item for item in value), key=lambda x: x["at"])

            self.last_timestamp_primary = self.primary_actions[-1]["at"] if self.primary_actions else 0
            self._invalidate_cache('primary') # Invalidate cache

        except Exception as e:
            self.logger.error(f"Error in actions.setter: {e}. Clearing primary actions as a precaution.")
            self.primary_actions = []
            self.last_timestamp_primary = 0
            self._invalidate_cache('primary')  # Invalidate cache

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

    def get_actions_in_range(self, start_time_ms: int, end_time_ms: int, axis: str = 'primary') -> List[Dict]:
        """
        Get all actions within a time range for streaming/query purposes.

        Args:
            start_time_ms: Start of time range (inclusive)
            end_time_ms: End of time range (inclusive)
            axis: 'primary' or 'secondary'

        Returns:
            List of action dictionaries [{'at': timestamp_ms, 'pos': position}, ...]
        """
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list:
            return []

        indices = self._get_action_indices_in_time_range(actions_list, start_time_ms, end_time_ms)
        if indices[0] is None or indices[1] is None:
            return []

        start_idx, end_idx = indices
        return actions_list[start_idx:end_idx + 1]

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

    def auto_tune_sg_filter(self, axis: str,
                             saturation_low: int = 1,
                             saturation_high: int = 99,
                             max_window_size: int = 15,
                             polyorder: int = 2,
                             selected_indices: Optional[List[int]] = None) -> Optional[Dict]:
        """
        Iteratively finds the best SG filter window size to minimize saturation and applies it.

        :param axis: The axis to process ('primary' or 'secondary').
        :param saturation_low: Position value at or below which is considered saturated.
        :param saturation_high: Position value at or above which is considered saturated.
        :param max_window_size: The largest window size to attempt.
        :param polyorder: The polynomial order for the SG filter.
        :param selected_indices: Optional list of indices to apply the filter to.
        :return: A dictionary with the applied parameters on success, None on failure.
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not installed. SG auto-tune cannot be applied.")
            return None

        actions_list_ref = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list_ref: return None

        # Determine the segment of actions to process
        indices_to_filter: List[int] = []
        if selected_indices is not None and len(selected_indices) > 0:
            indices_to_filter = sorted([i for i in selected_indices if 0 <= i < len(actions_list_ref)])
        else:
            indices_to_filter = list(range(len(actions_list_ref)))

        if len(indices_to_filter) < 3:
            self.logger.warning("Not enough points for SG auto-tune.")
            return None

        # Extract positions from the identified segment of actions
        positions = np.array([actions_list_ref[i]['pos'] for i in indices_to_filter])
        num_points_in_segment = len(positions)

        best_window_length = -1
        min_saturated_count = float('inf')

        # Iterate through window sizes to find the one that minimizes saturation
        for window_length in range(3, max_window_size + 1, 2):
            if num_points_in_segment < window_length:
                self.logger.info(f"Auto-Tune: Segment size ({num_points_in_segment}) is smaller than window size ({window_length}). Stopping search.")
                break  # Stop if the window becomes larger than the number of points

            actual_polyorder = min(polyorder, window_length - 1)

            try:
                # Apply filter to a temporary copy to check for saturation
                smoothed_positions = savgol_filter(positions, window_length, actual_polyorder)
            except ValueError as e:
                self.logger.warning(f"Auto-Tune: SG filter failed for window {window_length}. Error: {e}. Stopping.")
                continue

            # Count how many points are saturated after filtering
            saturated_count = np.sum((smoothed_positions <= saturation_low) | (smoothed_positions >= saturation_high))
            self.logger.debug(f"Auto-Tune trying W={window_length}, P={actual_polyorder}: Found {saturated_count} saturated points.")

            # If this window size is better than the previous best, update it.
            if saturated_count < min_saturated_count:
                min_saturated_count = saturated_count
                best_window_length = window_length

            # If we find a perfect solution, we can stop early.
            if saturated_count == 0:
                break

        if best_window_length == -1:
            self.logger.error("Auto-Tune: Could not determine a best window size. This should not happen if there are enough points.")
            return None

        # Apply the best found filter, even if it's not perfect
        self.logger.info(f"Auto-Tune determined best window W={best_window_length} with {min_saturated_count} saturated points remaining.")
        final_polyorder = min(polyorder, best_window_length - 1)
        try:
            final_smoothed_positions = savgol_filter(positions, best_window_length, final_polyorder)
            for i, original_list_idx in enumerate(indices_to_filter):
                actions_list_ref[original_list_idx]['pos'] = int(round(np.clip(final_smoothed_positions[i], 0, 100)))

            result = {
                'window_length': best_window_length,
                'polyorder': final_polyorder,
                'points_affected': len(indices_to_filter)
            }
            self.logger.info(f"Applied Auto-Tuned SG to {axis} axis with W={result['window_length']}, P={result['polyorder']}.")
            return result
        except Exception as e:
            self.logger.error(f"Error applying final auto-tuned SG filter: {e}")
            return None

    def recover_missing_strokes(self, axis: str, original_actions: List[Dict], threshold_factor: float = 1.8):
        """
        Analyzes the rhythm of keyframes to find and re-insert significant strokes
        that were filtered out from the original script. This method is destructive.
        """
        target_list_attr = 'primary_actions' if axis == 'primary' else 'secondary_actions'
        keyframes = getattr(self, target_list_attr)

        if len(keyframes) < 2 or len(original_actions) < 3:
            return  # Not enough data to analyze

        # 1. Establish the rhythmic baseline from the current keyframes
        intervals = np.array([p2['at'] - p1['at'] for p1, p2 in zip(keyframes, keyframes[1:]) if p2['at'] > p1['at']])
        if len(intervals) < 2: return

        median_interval = np.median(intervals)
        gap_threshold = median_interval * threshold_factor

        # 2. Find gaps and search for the most significant missing stroke in each
        points_to_add = []
        for i in range(len(keyframes) - 1):
            p1, p2 = keyframes[i], keyframes[i + 1]
            interval = p2['at'] - p1['at']

            if interval > gap_threshold:
                best_candidate = None
                max_significance = -1

                # Find original points within this time range using bisect for efficiency
                action_times = [a['at'] for a in original_actions]
                s_idx = bisect.bisect_right(action_times, p1['at'])
                e_idx = bisect.bisect_left(action_times, p2['at'])
                if s_idx >= e_idx: continue

                candidates_in_gap = original_actions[s_idx:e_idx]
                if not candidates_in_gap: continue

                # Determine the most significant point by its distance from the connecting line
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
            self.logger.info(f"Ultimate Autotune: Recovered {len(points_to_add)} missing strokes.")
            # Use add_actions_batch for efficient, sorted, and non-overlapping insertion
            batch_data = [{
                'timestamp_ms': p['at'],
                'primary_pos': p['pos'] if axis == 'primary' else None,
                'secondary_pos': p['pos'] if axis == 'secondary' else None
            } for p in points_to_add]

            self.add_actions_batch(batch_data)


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

        self._invalidate_cache(axis)
        self.logger.info(
            f"Peak simplification applied to {axis} (indices {s_idx_orig}-{e_idx_orig}). "
            f"Points: {len(segment_to_process)} -> {len(new_segment_actions)}")

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

        # 1. Extract only the positions to a NumPy array
        positions = np.array([actions_list_ref[i]['pos'] for i in indices_to_process], dtype=np.float64)

        # 2. Apply the vectorized operation function to the entire array at once
        new_positions = operation_func(positions)

        # 3. Clip the results and convert to int
        new_positions = np.clip(new_positions, 0, 100).round().astype(int)

        # 4. Update the original list with the new values
        for i, original_list_idx in enumerate(indices_to_process):
            actions_list_ref[original_list_idx]['pos'] = new_positions[i]

        self.logger.info(f"Applied vectorized operation to {len(indices_to_process)} points on {axis} axis.")

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
                self._invalidate_cache(axis_name)
            elif start_time_ms is not None and end_time_ms is not None:
                s_idx, e_idx = self._get_action_indices_in_time_range(target_actions_list, start_time_ms, end_time_ms)
                if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                    del target_actions_list[s_idx: e_idx + 1]
                    self._invalidate_cache(axis_name)
            else:
                target_actions_list[:] = []
                self._invalidate_cache(axis_name)

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
        self._invalidate_cache(axis)

        # Update last timestamp for the axis
        last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
        if axis == 'primary':
            self.last_timestamp_primary = last_ts
        else:
            self.last_timestamp_secondary = last_ts

        self.logger.info(f"Shifted {len(actions_list_ref)} points on {axis} axis by {actual_delta_ms}ms.")

    def add_actions_batch(self, actions_data: List[Dict], is_from_live_tracker: bool = False):
        """
        Adds a batch of actions efficiently by extending and sorting once.
        """
        primary_to_add = []
        secondary_to_add = []
        for action in actions_data:
            if action.get('primary_pos') is not None:
                primary_to_add.append({'at': action['timestamp_ms'], 'pos': int(action['primary_pos'])})
            if action.get('secondary_pos') is not None:
                secondary_to_add.append({'at': action['timestamp_ms'], 'pos': int(action['secondary_pos'])})

        # Process Primary Axis
        if primary_to_add:
            self.primary_actions.extend(primary_to_add)
            self.primary_actions.sort(key=lambda x: x['at'])
            self._filter_list_by_interval('primary')

        # Process Secondary Axis
        if secondary_to_add:
            self.secondary_actions.extend(secondary_to_add)
            self.secondary_actions.sort(key=lambda x: x['at'])
            self._filter_list_by_interval('secondary')

        self._invalidate_cache('both')
        self.last_timestamp_primary = self.primary_actions[-1]['at'] if self.primary_actions else 0
        self.last_timestamp_secondary = self.secondary_actions[-1]['at'] if self.secondary_actions else 0

    def _filter_list_by_interval(self, axis: str):
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        if len(actions_list) < 2:
            return

        unique_actions = [actions_list[0]]
        for i in range(1, len(actions_list)):
            # Keep only the last point at a given timestamp to remove duplicates
            if actions_list[i]['at'] == unique_actions[-1]['at']:
                unique_actions[-1] = actions_list[i]
            else:
                unique_actions.append(actions_list[i])

        # Now apply the min_interval filter
        if self.min_interval_ms > 0:
            final_actions = [unique_actions[0]]
            for i in range(1, len(unique_actions)):
                if unique_actions[i]['at'] - final_actions[-1]['at'] >= self.min_interval_ms:
                    final_actions.append(unique_actions[i])
            actions_list[:] = final_actions
        else:
            actions_list[:] = unique_actions

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


    def _simplify_keyframes_vectorized(self, extrema: List[Dict], position_tolerance: int) -> List[Dict]:
        """OPTIMIZED: Vectorized keyframe simplification using numpy for massive speedup."""
        if len(extrema) <= 2:
            return extrema

        # Convert to numpy arrays for vectorized operations
        ext_positions = np.array([ext['pos'] for ext in extrema])
        ext_timestamps = np.array([ext['at'] for ext in extrema])

        # Iteratively remove weakest points using vectorized calculations
        while len(extrema) > 2:
            if len(ext_positions) <= 2:
                break

            # Vectorized significance calculation for all internal points at once
            prev_pos = ext_positions[:-2]
            curr_pos = ext_positions[1:-1]
            next_pos = ext_positions[2:]
            prev_time = ext_timestamps[:-2]
            curr_time = ext_timestamps[1:-1]
            next_time = ext_timestamps[2:]

            # Calculate projection-based significance vectorized
            durations = next_time.astype(np.float64) - prev_time.astype(np.float64)
            time_deltas = curr_time.astype(np.float64) - prev_time.astype(np.float64)

            # Avoid division by zero
            progress = np.divide(time_deltas, durations,
                               out=np.zeros_like(time_deltas, dtype=np.float64), where=durations!=0)

            projected_pos = prev_pos + progress * (next_pos - prev_pos)
            significance_scores = np.abs(curr_pos - projected_pos)

            # Set significance to infinity where duration is zero
            significance_scores[durations == 0] = np.inf

            # Find weakest point
            min_idx = np.argmin(significance_scores)
            min_significance = significance_scores[min_idx]

            if min_significance < position_tolerance:
                # Remove the weakest point (add 1 because we're looking at internal points)
                remove_idx = min_idx + 1
                extrema.pop(remove_idx)
                ext_positions = np.delete(ext_positions, remove_idx)
                ext_timestamps = np.delete(ext_timestamps, remove_idx)
            else:
                break

        return extrema


    def list_available_plugins(self) -> List[Dict]:
        """Return a list of available plugins with their metadata."""
        from funscript.plugins.base_plugin import plugin_registry
        from funscript.plugins.plugin_loader import plugin_loader
        
        # Ensure plugins are loaded if they haven't been already
        if not plugin_registry.is_global_plugins_loaded():
            # Load built-in plugins
            builtin_results = plugin_loader.load_builtin_plugins()
            self.logger.debug(f"Loaded {len(builtin_results)} built-in plugins")
            
            # Load user plugins
            user_results = plugin_loader.load_user_plugins()
            self.logger.debug(f"Loaded {len(user_results)} user plugins")
            
            plugin_registry.set_global_plugins_loaded(True)
        
        # Get all registered plugins
        return plugin_registry.list_plugins()

    def apply_plugin(self, plugin_name: str, axis: str = 'both', **parameters) -> bool:
        """
        Apply a plugin to the funscript.
        
        Args:
            plugin_name: Name of the plugin to apply
            axis: Which axis to apply to ('primary', 'secondary', 'both')
            **parameters: Plugin-specific parameters
            
        Returns:
            True if plugin was applied successfully, False otherwise
        """
        from funscript.plugins.base_plugin import plugin_registry
        from funscript.plugins.plugin_loader import plugin_loader
        
        # Ensure plugins are loaded
        if not plugin_registry.is_global_plugins_loaded():
            plugin_loader.load_builtin_plugins()
            plugin_loader.load_user_plugins()
            plugin_registry.set_global_plugins_loaded(True)
        
        # Get the plugin
        plugin = plugin_registry.get_plugin(plugin_name)
        if not plugin:
            self.logger.error(f"Plugin '{plugin_name}' not found")
            return False
        
        try:
            # Apply the plugin
            result = plugin.transform(self, axis=axis, **parameters)
            
            # Plugin might return None (for in-place modification) or a new funscript
            if result is not None:
                # Plugin returns a new funscript - replace our data
                if axis in ['primary', 'both']:
                    self.primary_actions = result.primary_actions
                if axis in ['secondary', 'both']:
                    self.secondary_actions = result.secondary_actions
                self._invalidate_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying plugin '{plugin_name}': {e}")
            return False

    def get_plugin_preview(self, plugin_name: str, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """
        Get a preview of what a plugin would do without applying it.
        
        Args:
            plugin_name: Name of the plugin to preview
            axis: Which axis to preview ('primary', 'secondary', 'both') 
            **parameters: Plugin-specific parameters
            
        Returns:
            Dictionary with preview information
        """
        from funscript.plugins.base_plugin import plugin_registry
        from funscript.plugins.plugin_loader import plugin_loader
        
        # Ensure plugins are loaded
        if not plugin_registry.is_global_plugins_loaded():
            plugin_loader.load_builtin_plugins()
            plugin_loader.load_user_plugins()
            plugin_registry.set_global_plugins_loaded(True)
        
        # Get the plugin
        plugin = plugin_registry.get_plugin(plugin_name)
        if not plugin:
            return {"error": f"Plugin '{plugin_name}' not found"}
        
        try:
            # Get plugin preview
            return plugin.get_preview(self, axis=axis, **parameters)
            
        except Exception as e:
            return {"error": f"Error generating preview for '{plugin_name}': {e}"}

    def set_chapters_from_segments(self, video_segments: List, video_fps: float):
        """
        Set funscript chapters from video segments.
        
        Args:
            video_segments: List of VideoSegment objects or dictionaries
            video_fps: Video frames per second for timestamp conversion
        """
        self.chapters = []
        
        for segment in video_segments:
            # Handle both VideoSegment objects and dictionaries
            if hasattr(segment, 'start_frame_id'):
                # VideoSegment object
                start_frame_id = segment.start_frame_id
                end_frame_id = segment.end_frame_id
                position_short = segment.position_short_name
                position_long = segment.position_long_name
            elif isinstance(segment, dict):
                # Dictionary representation
                start_frame_id = segment.get('start_frame_id', 0)
                end_frame_id = segment.get('end_frame_id', 0)
                position_short = segment.get('position_short_name', segment.get('major_position', 'Unknown'))
                position_long = segment.get('position_long_name', segment.get('major_position', 'Unknown'))
            else:
                self.logger.warning(f"Unknown segment type: {type(segment)}, skipping")
                continue
            
            start_time_ms = int((start_frame_id / video_fps) * 1000)
            end_time_ms = int((end_frame_id / video_fps) * 1000)
            
            chapter = {
                "name": position_short,  # Use short name for UI display
                "start": start_time_ms,
                "end": end_time_ms,
                "startTime": start_time_ms,  # Keep both for compatibility
                "endTime": end_time_ms,
                "position_short": position_short,
                "position_long": position_long
            }
            
            self.chapters.append(chapter)
        
        self.logger.debug(f"Set {len(self.chapters)} chapters from video segments")

    def clear_chapters(self):
        """Clear all chapters from the funscript."""
        self.chapters = []
        self.logger.debug("Cleared all chapters")

    def add_chapter(self, start_time_ms: int, end_time_ms: int, name: str = "Chapter", 
                   position_short: str = "", position_long: str = "", **kwargs):
        """
        Add a chapter to the funscript.
        
        Args:
            start_time_ms: Chapter start time in milliseconds
            end_time_ms: Chapter end time in milliseconds  
            name: Chapter name/title
            position_short: Short position name
            position_long: Long position name
            **kwargs: Additional chapter properties
        """
        chapter = {
            "name": name,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "position_short": position_short,
            "position_long": position_long,
            **kwargs
        }
        self.chapters.append(chapter)
        self.logger.debug(f"Added chapter '{name}' ({start_time_ms}-{end_time_ms}ms)")

