"""Action-list editing operations on MultiAxisFunscript.

Accessed via `fs.editor.<method>()`. Flat API preserved as delegators.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from funscript.multi_axis_funscript import MultiAxisFunscript


class ActionEditor:
    """Point / range editing ops."""

    __slots__ = ("fs",)

    def __init__(self, fs: "MultiAxisFunscript") -> None:
        self.fs = fs

    def _apply_to_points(self, axis: str, operation_func: Callable[[int], int],
                         start_time_ms: Optional[int] = None,
                         end_time_ms: Optional[int] = None,
                         selected_indices: Optional[List[int]] = None) -> None:
        fs = self.fs
        actions_list_ref = fs.primary_actions if axis == 'primary' else fs.secondary_actions
        if not actions_list_ref:
            return

        if selected_indices is not None:
            indices_to_process = [i for i in selected_indices if 0 <= i < len(actions_list_ref)]
        elif start_time_ms is not None and end_time_ms is not None:
            s_idx, e_idx = fs._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)
            indices_to_process = list(range(s_idx, e_idx + 1)) if (s_idx is not None and e_idx is not None and s_idx <= e_idx) else []
        else:
            indices_to_process = list(range(len(actions_list_ref)))

        if not indices_to_process:
            fs.logger.warning("No points for operation.")
            return

        positions = np.array([actions_list_ref[i]['pos'] for i in indices_to_process], dtype=np.float64)
        new_positions = operation_func(positions)
        new_positions = np.clip(new_positions, 0, 100).round().astype(int)

        for i, original_list_idx in enumerate(indices_to_process):
            actions_list_ref[original_list_idx]['pos'] = new_positions[i]

        fs.logger.info(f"Applied vectorized operation to {len(indices_to_process)} points on {axis} axis.")

    def clear_points(self, axis: str = 'both',
                     start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None,
                     selected_indices: Optional[List[int]] = None) -> None:
        fs = self.fs
        valid_axes = {'primary', 'secondary', 'both'} | set(fs.additional_axes.keys())
        if axis not in valid_axes:
            fs.logger.warning(f"Axis '{axis}' not recognized for clear_points.")
            return

        if axis == 'both':
            affected_axes_names = ['primary', 'secondary']
        else:
            affected_axes_names = [axis]

        total_cleared_count = 0
        for axis_name in affected_axes_names:
            target_actions_list = fs.get_axis_actions(axis_name)
            initial_len = len(target_actions_list)

            if selected_indices is not None:
                valid_indices_to_remove_set = {i for i in selected_indices if 0 <= i < len(target_actions_list)}
                if not valid_indices_to_remove_set:
                    continue
                target_actions_list[:] = [a for i, a in enumerate(target_actions_list)
                                          if i not in valid_indices_to_remove_set]
                fs._invalidate_cache(axis_name)
            elif start_time_ms is not None and end_time_ms is not None:
                s_idx, e_idx = fs._get_action_indices_in_time_range(target_actions_list, start_time_ms, end_time_ms)
                if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                    del target_actions_list[s_idx: e_idx + 1]
                    fs._invalidate_cache(axis_name)
            else:
                target_actions_list[:] = []
                fs._invalidate_cache(axis_name)

            total_cleared_count += initial_len - len(target_actions_list)

            if axis_name == 'primary':
                fs.last_timestamp_primary = target_actions_list[-1]['at'] if target_actions_list else 0
            else:
                fs.last_timestamp_secondary = target_actions_list[-1]['at'] if target_actions_list else 0

        if total_cleared_count > 0:
            fs.logger.info(
                f"Cleared {total_cleared_count} points across affected axes ({', '.join(affected_axes_names)}).")

    def clear_actions_in_time_range(self, start_time_ms: int, end_time_ms: int,
                                    axis: str = 'both') -> None:
        fs = self.fs
        valid_axes = {'primary', 'secondary', 'both'} | set(fs.additional_axes.keys())
        if axis not in valid_axes:
            fs.logger.warning(f"Axis '{axis}' not recognized for clear_actions_in_time_range.")
            return

        axes_to_process: List[Tuple[str, List[Dict]]] = []
        if axis == 'both':
            axes_to_process.append(('primary', fs.primary_actions))
            axes_to_process.append(('secondary', fs.secondary_actions))
        else:
            axes_to_process.append((axis, fs.get_axis_actions(axis)))

        total_cleared_count = 0
        for axis_name, actions_list_ref in axes_to_process:
            if not actions_list_ref:
                continue

            s_idx, e_idx = fs._get_action_indices_in_time_range(actions_list_ref, start_time_ms, end_time_ms)

            if s_idx is not None and e_idx is not None and s_idx <= e_idx:
                num_to_clear = e_idx - s_idx + 1
                del actions_list_ref[s_idx: e_idx + 1]
                total_cleared_count += num_to_clear
                fs._invalidate_cache(axis_name)
                fs.logger.debug(
                    f"Cleared {num_to_clear} points from {axis_name} axis between {start_time_ms}ms and {end_time_ms}ms.")

                if axis_name == 'primary':
                    fs.last_timestamp_primary = actions_list_ref[-1]['at'] if actions_list_ref else 0
                else:
                    fs.last_timestamp_secondary = actions_list_ref[-1]['at'] if actions_list_ref else 0
            else:
                fs.logger.debug(
                    f"No points found to clear in {axis_name} axis between {start_time_ms}ms and {end_time_ms}ms.")

        if total_cleared_count > 0:
            fs.logger.info(
                f"Total {total_cleared_count} points cleared in time range [{start_time_ms}ms - {end_time_ms}ms].")

    def shift_points_time(self, axis: str, time_delta_ms: int) -> None:
        fs = self.fs
        actions_list_ref = fs.primary_actions if axis == 'primary' else fs.secondary_actions
        if not actions_list_ref:
            return

        if time_delta_ms < 0 and actions_list_ref[0]['at'] + time_delta_ms < 0:
            actual_delta_ms = -actions_list_ref[0]['at']
            fs.logger.warning(
                f"Original shift of {time_delta_ms}ms was too large. "
                f"Adjusted to {actual_delta_ms}ms to prevent negative timestamps."
            )
        else:
            actual_delta_ms = time_delta_ms

        if actual_delta_ms == 0 and time_delta_ms != 0:
            fs.logger.info("No shift applied as it would result in negative timestamps.")
            return

        for action in actions_list_ref:
            action['at'] += actual_delta_ms

        actions_list_ref.sort(key=lambda x: x['at'])
        fs._invalidate_cache(axis)

        last_ts = actions_list_ref[-1]['at'] if actions_list_ref else 0
        if axis == 'primary':
            fs.last_timestamp_primary = last_ts
        else:
            fs.last_timestamp_secondary = last_ts

        fs.logger.info(f"Shifted {len(actions_list_ref)} points on {axis} axis by {actual_delta_ms}ms.")

    def add_actions_batch(self, actions_data: List[Dict], is_from_live_tracker: bool = False) -> None:
        fs = self.fs
        primary_to_add: List[Dict] = []
        secondary_to_add: List[Dict] = []
        for action in actions_data:
            ts = fs.snap_to_frame(action['timestamp_ms'])
            if action.get('primary_pos') is not None:
                primary_to_add.append({'at': ts, 'pos': int(action['primary_pos'])})
            if action.get('secondary_pos') is not None:
                secondary_to_add.append({'at': ts, 'pos': int(action['secondary_pos'])})

        if primary_to_add:
            fs.primary_actions.extend(primary_to_add)
            fs.primary_actions.sort(key=lambda x: x['at'])
            fs.signal._filter_list_by_interval('primary')

        if secondary_to_add:
            fs.secondary_actions.extend(secondary_to_add)
            fs.secondary_actions.sort(key=lambda x: x['at'])
            fs.signal._filter_list_by_interval('secondary')

        fs._invalidate_cache('both')
        fs.last_timestamp_primary = fs.primary_actions[-1]['at'] if fs.primary_actions else 0
        fs.last_timestamp_secondary = fs.secondary_actions[-1]['at'] if fs.secondary_actions else 0
