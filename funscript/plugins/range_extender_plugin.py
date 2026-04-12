"""
Range Extender plugin.

Expands or compresses the amplitude range of selected points by pushing
them away from (or toward) the center of the current range.  Unlike
Amplify which scales around a fixed center value, Range Extender finds
the actual center of the selected points and extends outward from there.

Example: points at 30-70 with extend_amount=10 become ~20-80.
"""

import numpy as np
from typing import Any, Dict, List, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class RangeExtenderPlugin(FunscriptTransformationPlugin):

    @property
    def name(self) -> str:
        return "Range Extender"

    @property
    def description(self) -> str:
        return "Expand or compress the amplitude range of selected points"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'extend_amount': {
                'type': int,
                'required': False,
                'default': 10,
                'description': 'Amount to extend range (positive expands, negative compresses)',
                'constraints': {'min': -50, 'max': 50}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Indices of selected actions to process (None for all)',
            },
        }

    def transform(self, funscript, axis='both', **parameters):
        validated = self.validate_parameters(parameters)
        extend = validated['extend_amount']
        selected_indices = validated.get('selected_indices')

        if extend == 0:
            return None

        axes = []
        if axis == 'both':
            axes = ['primary', 'secondary']
        else:
            axes = [axis]

        for current_axis in axes:
            actions = (funscript.primary_actions if current_axis == 'primary'
                       else funscript.secondary_actions)
            if not actions:
                continue

            indices = selected_indices
            if indices is None:
                indices = list(range(len(actions)))
            else:
                indices = [i for i in indices if 0 <= i < len(actions)]

            if len(indices) < 2:
                continue

            positions = np.array([actions[i]['pos'] for i in indices], dtype=np.float64)
            center = (positions.min() + positions.max()) / 2.0

            # Push each point away from center proportionally
            distances = positions - center
            max_dist = np.abs(distances).max()
            if max_dist < 0.5:
                continue  # All points at same position

            # Scale: extend_amount adds that many units to the extremes
            scale = (max_dist + extend) / max_dist
            new_positions = center + distances * scale
            new_positions = np.clip(np.round(new_positions), 0, 100).astype(int)

            for i, idx in enumerate(indices):
                actions[idx]['pos'] = int(new_positions[i])

            funscript._invalidate_cache(current_axis)

        return None  # In-place modification
