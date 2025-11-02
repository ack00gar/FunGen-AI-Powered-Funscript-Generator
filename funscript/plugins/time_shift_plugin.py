"""
Time shift plugin for funscript transformations.

This plugin shifts all action points forward or backward in time.
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class TimeShiftPlugin(FunscriptTransformationPlugin):
    """
    Time shift plugin.

    Shifts all action points forward or backward in time by a specified delta.
    Useful for syncing scripts with video or adjusting timing.
    """

    @property
    def name(self) -> str:
        return "Time Shift"

    @property
    def description(self) -> str:
        return "Shift all actions forward or backward in time"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'time_delta_ms': {
                'type': int,
                'required': True,
                'default': 0,
                'description': 'Time delta in milliseconds (positive = forward, negative = backward)',
                'constraints': {'min': -60000, 'max': 60000}
            }
        }

    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """Apply time shift to the specified axis."""
        # Validate parameters
        validated_params = self.validate_parameters(parameters)

        # Validate axis
        if axis not in self.supported_axes:
            raise ValueError(f"Unsupported axis '{axis}'. Must be one of {self.supported_axes}")

        time_delta_ms = validated_params['time_delta_ms']

        if time_delta_ms == 0:
            self.logger.info("Time delta is 0, no shift applied")
            return None

        # Determine which axes to process
        axes_to_process = []
        if axis == 'both':
            axes_to_process = ['primary', 'secondary']
        else:
            axes_to_process = [axis]

        for current_axis in axes_to_process:
            self._apply_time_shift_to_axis(funscript, current_axis, time_delta_ms)

        return None  # Modifies in-place

    def _apply_time_shift_to_axis(self, funscript, axis: str, time_delta_ms: int):
        """Apply time shift to a single axis."""
        actions_list = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions

        if not actions_list:
            self.logger.debug(f"No actions found for {axis} axis")
            return

        # Shift all timestamps
        for action in actions_list:
            action['at'] += time_delta_ms

        # Sort by time (in case shifting made them out of order)
        actions_list.sort(key=lambda x: x['at'])

        # Remove any actions that went negative
        original_count = len(actions_list)
        if axis == 'primary':
            funscript.primary_actions = [a for a in actions_list if a['at'] >= 0]
        else:
            funscript.secondary_actions = [a for a in actions_list if a['at'] >= 0]

        removed_count = original_count - len(actions_list)

        # Invalidate cache
        funscript._invalidate_cache(axis)

        if removed_count > 0:
            self.logger.info(
                f"Applied time shift of {time_delta_ms}ms to {axis} axis. "
                f"Removed {removed_count} actions with negative timestamps."
            )
        else:
            self.logger.info(
                f"Applied time shift of {time_delta_ms}ms to {axis} axis. "
                f"{len(actions_list)} actions shifted."
            )

    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """Generate a preview of the time shift effect."""
        try:
            validated_params = self.validate_parameters(parameters)
        except ValueError as e:
            return {"error": str(e)}

        time_delta_ms = validated_params['time_delta_ms']

        preview_info = {
            "filter_type": "Time Shift",
            "parameters": validated_params,
            "description": f"Shift all actions by {time_delta_ms}ms"
        }

        # Determine which axes would be affected
        if axis == 'both':
            axes_to_check = ['primary', 'secondary']
        else:
            axes_to_check = [axis]

        for current_axis in axes_to_check:
            actions_list = funscript.primary_actions if current_axis == 'primary' else funscript.secondary_actions
            if not actions_list:
                continue

            # Calculate how many actions would be removed (negative timestamps)
            actions_that_would_be_removed = sum(1 for a in actions_list if (a['at'] + time_delta_ms) < 0)

            # Calculate new time range
            first_time = actions_list[0]['at'] + time_delta_ms
            last_time = actions_list[-1]['at'] + time_delta_ms

            axis_info = {
                "total_actions": len(actions_list),
                "actions_removed": actions_that_would_be_removed,
                "actions_kept": len(actions_list) - actions_that_would_be_removed,
                "original_time_range": f"{actions_list[0]['at']}ms - {actions_list[-1]['at']}ms",
                "new_time_range": f"{max(0, first_time)}ms - {last_time}ms",
                "shift_direction": "forward" if time_delta_ms > 0 else "backward",
                "shift_amount": f"{abs(time_delta_ms)}ms"
            }

            if actions_that_would_be_removed > 0:
                axis_info["warning"] = f"{actions_that_would_be_removed} actions would have negative timestamps and will be removed"

            preview_info[f"{current_axis}_axis"] = axis_info

        return preview_info
