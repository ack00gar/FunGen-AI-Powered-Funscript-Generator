"""Dynamic injection plugin for funscript transformations.

Adds intermediate points between existing action pairs based on configurable
interpolation methods. Useful for adding smoothness to sparse scripts or
converting step-function patterns to smooth curves.
"""
import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class DynamicInjectionPlugin(FunscriptTransformationPlugin):
    """Injects intermediate points between existing actions.

    For each action pair, calculates the segment speed, determines
    injection count based on target interval, and interpolates
    positions using the selected method.
    """

    @property
    def name(self) -> str:
        return "Dynamic Injection"

    @property
    def description(self) -> str:
        return "Inject intermediate points between actions for smoother motion"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'target_interval_ms': {
                'type': int,
                'required': False,
                'default': 100,
                'label': 'Target Interval (ms)',
                'description': 'Target time interval between injected points',
                'constraints': {'min': 20, 'max': 500}
            },
            'speed_adaptive': {
                'type': bool,
                'required': False,
                'default': True,
                'label': 'Speed Adaptive',
                'description': 'Inject more points in faster segments',
            },
            'interpolation': {
                'type': str,
                'required': False,
                'default': 'cosine',
                'label': 'Interpolation',
                'description': 'Interpolation method between points',
                'constraints': {'choices': ['linear', 'cosine', 'cubic']}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Specific action indices to process (None for all)',
            },
        }

    def transform(self, funscript, axis: str = 'both', **parameters) -> None:
        """Apply dynamic injection to the specified axis."""
        validated = self.validate_parameters(parameters)

        axes_to_process = ['primary', 'secondary'] if axis == 'both' else [axis]

        for current_axis in axes_to_process:
            self._inject_axis(funscript, current_axis, validated)

        return None  # Modifies in-place

    def _inject_axis(self, funscript, axis: str, params: Dict[str, Any]):
        """Inject intermediate points into a single axis."""
        actions_list = funscript.get_axis_actions(axis)
        if not actions_list or len(actions_list) < 2:
            return

        target_interval = params['target_interval_ms']
        speed_adaptive = params['speed_adaptive']
        interp_method = params['interpolation']
        selected_indices = params.get('selected_indices')

        new_actions = [actions_list[0]]

        for i in range(len(actions_list) - 1):
            a0 = actions_list[i]
            a1 = actions_list[i + 1]

            # Skip if not in selection
            if selected_indices is not None and i not in selected_indices:
                new_actions.append(a1)
                continue

            dt = a1['at'] - a0['at']
            dp = abs(a1['pos'] - a0['pos'])

            if dt <= target_interval:
                # Segment already shorter than target, keep as-is
                new_actions.append(a1)
                continue

            # Determine number of injections
            if speed_adaptive and dt > 0:
                speed = (dp / dt) * 1000.0  # units/sec
                # More points for faster segments
                adaptive_factor = max(0.5, min(2.0, speed / 200.0))
                effective_interval = target_interval / adaptive_factor
            else:
                effective_interval = target_interval

            num_injections = max(1, int(dt / effective_interval)) - 1

            if num_injections == 0:
                new_actions.append(a1)
                continue

            # Inject intermediate points
            for j in range(1, num_injections + 1):
                t_frac = j / (num_injections + 1)
                t_ms = a0['at'] + dt * t_frac

                # Interpolate position
                pos = self._interpolate(a0['pos'], a1['pos'], t_frac, interp_method)

                new_actions.append({
                    'at': int(round(t_ms)),
                    'pos': max(0, min(100, int(round(pos)))),
                })

            new_actions.append(a1)

        # Sort by time (should already be sorted but ensure)
        new_actions.sort(key=lambda a: a['at'])

        # Update the funscript
        funscript.set_axis_actions(axis, new_actions)

    @staticmethod
    def _interpolate(p0: float, p1: float, t: float, method: str) -> float:
        """Interpolate between two position values.

        Args:
            p0, p1: Start and end positions
            t: Fraction 0-1
            method: 'linear', 'cosine', or 'cubic'
        """
        if method == 'linear':
            return p0 + (p1 - p0) * t
        elif method == 'cosine':
            # Cosine interpolation for smooth easing
            t2 = (1.0 - np.cos(t * np.pi)) / 2.0
            return p0 + (p1 - p0) * t2
        elif method == 'cubic':
            # Cubic Hermite (ease in-out)
            t2 = t * t * (3.0 - 2.0 * t)
            return p0 + (p1 - p0) * t2
        else:
            return p0 + (p1 - p0) * t
