"""
Dynamic amplification plugin for funscript transformations.

This plugin dynamically amplifies the signal by normalizing each point based on the
min/max position within a sliding time window around it, creating a more uniform
intensity throughout the script while preserving relative motion patterns.
"""

import numpy as np
from bisect import bisect_left, bisect_right
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class DynamicAmplifyPlugin(FunscriptTransformationPlugin):
    """
    Dynamic amplification plugin.

    Dynamically amplifies the signal by normalizing each point based on the
    min/max position within a sliding time window around it. This creates more
    uniform intensity throughout the script while preserving relative motion patterns.
    """

    @property
    def name(self) -> str:
        return "Dynamic Amplify"

    @property
    def description(self) -> str:
        return "Dynamically amplifies based on local min/max in sliding window"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'window_ms': {
                'type': int,
                'required': False,
                'default': 4000,
                'description': 'Size of the sliding time window in milliseconds',
                'constraints': {'min': 500, 'max': 10000}
            },
            'min_range_threshold': {
                'type': int,
                'required': False,
                'default': 5,
                'description': 'Minimum local range to apply amplification (prevents amplifying noise)',
                'constraints': {'min': 0, 'max': 50}
            },
            'start_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Start time for amplification range (None for full range)'
            },
            'end_time_ms': {
                'type': int,
                'required': False,
                'default': None,
                'description': 'End time for amplification range (None for full range)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to amplify (overrides time range)'
            }
        }

    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply dynamic amplification to the specified axis."""
        # Validate parameters
        validated_params = self.validate_parameters(parameters)

        # Validate axis
        if axis not in self.supported_axes:
            raise ValueError(f"Unsupported axis '{axis}'. Must be one of {self.supported_axes}")

        # Determine which axes to process
        axes_to_process = []
        if axis == 'both':
            axes_to_process = ['primary', 'secondary']
        else:
            axes_to_process = [axis]

        for current_axis in axes_to_process:
            self._apply_dynamic_amplification_to_axis(funscript, current_axis, validated_params)

        return None  # Modifies in-place

    def _apply_dynamic_amplification_to_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Apply dynamic amplification to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions

        if not actions_list or len(actions_list) < 3:
            self.logger.warning(f"Not enough points for dynamic amplification on {axis} axis (need at least 3)")
            return

        # Make a copy to read from while modifying the original
        actions_copy = list(actions_list)

        # Determine which indices to process
        indices_to_process = self._get_indices_to_process(actions_list, params)

        if not indices_to_process:
            self.logger.warning(f"No points to process for {axis} axis")
            return

        window_ms = params['window_ms']
        min_range_threshold = params['min_range_threshold']

        # Create a list of timestamps for efficient searching
        action_timestamps = [a['at'] for a in actions_copy]

        modified_count = 0

        for i in indices_to_process:
            current_action = actions_copy[i]
            current_time = current_action['at']

            # Define the local window for analysis
            start_window = current_time - (window_ms // 2)
            end_window = current_time + (window_ms // 2)

            # Find the indices of the actions within this window using binary search
            start_idx = bisect_left(action_timestamps, start_window)
            end_idx = bisect_right(action_timestamps, end_window)

            local_actions = actions_copy[start_idx:end_idx]
            if not local_actions:
                continue

            # Find the min/max position within the local window
            local_positions = [a['pos'] for a in local_actions]
            local_min = min(local_positions)
            local_max = max(local_positions)
            local_range = local_max - local_min

            # Don't amplify if local motion is negligible
            if local_range < min_range_threshold:
                continue

            # Normalize the current point's position within its local range
            normalized_pos = (current_action['pos'] - local_min) / local_range

            # Scale the normalized position to the full 0-100 range
            new_pos = int(round(np.clip(normalized_pos * 100, 0, 100)))

            # Only update if the value changed
            if actions_list[i]['pos'] != new_pos:
                actions_list[i]['pos'] = new_pos
                modified_count += 1

        # Invalidate cache
        funscript._invalidate_cache(axis)

        self.logger.info(
            f"Applied dynamic amplification to {axis} axis: "
            f"{modified_count}/{len(indices_to_process)} points modified "
            f"(window={window_ms}ms, min_range={min_range_threshold})"
        )

    def _get_indices_to_process(self, actions_list: List[Dict], params: Dict[str, Any]) -> List[int]:
        """Determine which action indices should be processed."""
        selected_indices = params.get('selected_indices')
        start_time_ms = params.get('start_time_ms')
        end_time_ms = params.get('end_time_ms')

        if selected_indices is not None and len(selected_indices) > 0:
            # Use selected indices
            indices_to_process = sorted([
                i for i in selected_indices
                if 0 <= i < len(actions_list)
            ])
            return indices_to_process

        elif start_time_ms is not None and end_time_ms is not None:
            # Use time range filtering
            if len(actions_list) > 10000:
                # Optimized path for large datasets
                timestamps = np.array([action['at'] for action in actions_list])
                time_mask = (timestamps >= start_time_ms) & (timestamps <= end_time_ms)
                indices_to_process = np.where(time_mask)[0].tolist()
            else:
                # Original method for smaller datasets
                indices_to_process = []
                for i, action in enumerate(actions_list):
                    if start_time_ms <= action['at'] <= end_time_ms:
                        indices_to_process.append(i)
            return indices_to_process

        else:
            # Process entire list
            return list(range(len(actions_list)))

    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the dynamic amplification effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}

        preview_info = {
            "filter_type": "Dynamic Amplification",
            "parameters": validated_params
        }

        window_ms = validated_params['window_ms']
        min_range_threshold = validated_params['min_range_threshold']

        # Determine which axes would be affected
        if axis == 'both':
            axes_to_check = ['primary', 'secondary']
        else:
            axes_to_check = [axis]

        for current_axis in axes_to_check:
            actions_list = funscript.primary_actions if current_axis == 'primary' else funscript.secondary_actions
            if not actions_list or len(actions_list) < 3:
                preview_info[f"{current_axis}_axis"] = {
                    "error": "Not enough points (need at least 3)"
                }
                continue

            indices_to_process = self._get_indices_to_process(actions_list, validated_params)

            if indices_to_process:
                # Make a copy for preview calculation
                actions_copy = list(actions_list)
                action_timestamps = [a['at'] for a in actions_copy]

                # Calculate preview statistics
                positions_before = np.array([actions_list[i]['pos'] for i in indices_to_process])
                positions_after = []
                points_amplified = 0

                for i in indices_to_process:
                    current_action = actions_copy[i]
                    current_time = current_action['at']

                    start_window = current_time - (window_ms // 2)
                    end_window = current_time + (window_ms // 2)

                    start_idx = bisect_left(action_timestamps, start_window)
                    end_idx = bisect_right(action_timestamps, end_window)

                    local_actions = actions_copy[start_idx:end_idx]
                    if not local_actions:
                        positions_after.append(current_action['pos'])
                        continue

                    local_positions = [a['pos'] for a in local_actions]
                    local_min = min(local_positions)
                    local_max = max(local_positions)
                    local_range = local_max - local_min

                    if local_range < min_range_threshold:
                        positions_after.append(current_action['pos'])
                        continue

                    normalized_pos = (current_action['pos'] - local_min) / local_range
                    new_pos = int(round(np.clip(normalized_pos * 100, 0, 100)))
                    positions_after.append(new_pos)

                    if new_pos != current_action['pos']:
                        points_amplified += 1

                positions_after = np.array(positions_after)

                # Calculate statistics
                original_range = np.max(positions_before) - np.min(positions_before)
                new_range = np.max(positions_after) - np.min(positions_after)
                avg_change = np.mean(np.abs(positions_after - positions_before))

                axis_info = {
                    "total_points": len(actions_list),
                    "points_in_range": len(indices_to_process),
                    "points_modified": points_amplified,
                    "original_range": int(original_range),
                    "new_range": int(new_range),
                    "avg_position_change": round(float(avg_change), 1),
                    "window_ms": window_ms,
                    "min_range_threshold": min_range_threshold,
                    "effect": "Normalizes intensity within local windows"
                }

                preview_info[f"{current_axis}_axis"] = axis_info
            else:
                preview_info[f"{current_axis}_axis"] = {
                    "total_points": len(actions_list),
                    "points_in_range": 0,
                    "can_apply": False
                }

        return preview_info
