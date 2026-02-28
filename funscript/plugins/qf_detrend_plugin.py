"""
Detrend (Drift Removal) plugin  - Quickfix Tools.

Removes linear camera drift from a selection by analyzing the
min/max at the start and end of the selection and interpolating a
correction curve.
Algorithm by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class QFDetrendPlugin(FunscriptTransformationPlugin):
    """Remove linear position / range drift across a selection."""

    @property
    def name(self) -> str:
        return "Detrend (Drift Removal)"

    @property
    def description(self) -> str:
        return "Remove linear camera drift from a selection (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'sample_points': {
                'type': int,
                'required': False,
                'default': 4,
                'description': 'Number of points at start/end to analyze for drift',
                'constraints': {'min': 1, 'max': 20}
            },
            'adjust_range': {
                'type': bool,
                'required': False,
                'default': True,
                'description': 'Also correct range drift (not just center drift)'
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices to process'
            }
        }

    def transform(self, funscript, axis='both', **parameters) -> None:
        params = self.validate_parameters(parameters)
        for ax in (['primary', 'secondary'] if axis == 'both' else [axis]):
            self._apply(funscript, ax, params)

    def _apply(self, funscript, axis, params):
        actions = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        if not actions:
            return
        indices = self._resolve(actions, params)
        if len(indices) < 4:
            return

        n = min(params['sample_points'], len(indices) // 2)
        if n < 1:
            return

        # Analyze start region
        start_slice = indices[:n + 1]
        start_positions = [actions[i]['pos'] for i in start_slice]
        min0, max0 = min(start_positions), max(start_positions)

        # Analyze end region
        end_slice = indices[-(n + 1):]
        end_positions = [actions[i]['pos'] for i in end_slice]
        min1, max1 = min(end_positions), max(end_positions)

        # Center and range at start/end
        center0 = 0.5 * (min0 + max0)
        center1 = 0.5 * (min1 + max1)
        range0 = max0 - min0
        range1 = max1 - min1

        # Time boundaries
        at0 = actions[indices[0]]['at']
        at1 = actions[indices[-1]]['at']
        dt = at1 - at0
        if dt == 0:
            return

        adjust_range = params['adjust_range']

        for idx in indices:
            a = actions[idx]
            t_frac = (a['at'] - at0) / dt
            # Interpolated center correction
            center = center0 + (center1 - center0) * t_frac
            if adjust_range:
                rng = range0 + (range1 - range0) * t_frac
                rng = max(rng, 1)  # avoid division by zero
            else:
                rng = 50  # fixed denominator like Lua original
            new_pos = 50 * (a['pos'] - center) / rng + 50
            a['pos'] = max(0, min(100, round(new_pos)))

        funscript._invalidate_cache(axis)
        self.logger.info(f"Detrend on {axis}: center {center0:.0f}->{center1:.0f}, "
                         f"range {range0:.0f}->{range1:.0f}, {len(indices)} pts")

    @staticmethod
    def _resolve(actions, params):
        sel = params.get('selected_indices')
        if sel:
            return sorted(i for i in sel if 0 <= i < len(actions))
        return list(range(len(actions)))
