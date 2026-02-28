"""
Fit to Range plugin  - Quickfix Tools.

Linearly rescales selected actions to fill a target position range.
Algorithm by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class QFfitRangePlugin(FunscriptTransformationPlugin):
    """Rescale selection so min/max map to target_min/target_max."""

    @property
    def name(self) -> str:
        return "Fit to Range"

    @property
    def description(self) -> str:
        return "Linearly rescale selection to fill a target range (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'target_min': {
                'type': int,
                'required': False,
                'default': 0,
                'description': 'Target minimum position value',
                'constraints': {'min': 0, 'max': 100}
            },
            'target_max': {
                'type': int,
                'required': False,
                'default': 100,
                'description': 'Target maximum position value',
                'constraints': {'min': 0, 'max': 100}
            },
            'selected_indices': {
                'type': list,
                'required': False,
                'default': None,
                'description': 'Action indices to process'
            }
        }

    def transform(self, funscript, axis: str = 'both', **parameters) -> None:
        validated = self.validate_parameters(parameters)
        axes = ['primary', 'secondary'] if axis == 'both' else [axis]
        for ax in axes:
            self._apply(funscript, ax, validated)

    def _apply(self, funscript, axis, params):
        actions = funscript.primary_actions if axis == 'primary' else funscript.secondary_actions
        if not actions:
            return
        indices = self._resolve_indices(actions, params)
        if len(indices) < 2:
            return

        positions = [actions[i]['pos'] for i in indices]
        cur_min = min(positions)
        cur_max = max(positions)
        span = cur_max - cur_min
        if span == 0:
            return

        t_min = params['target_min']
        t_max = params['target_max']

        for i in indices:
            old = actions[i]['pos']
            actions[i]['pos'] = max(0, min(100, round(t_min + (old - cur_min) * (t_max - t_min) / span)))

        funscript._invalidate_cache(axis)
        self.logger.info(f"Fit to Range on {axis}: {len(indices)} pts, [{cur_min}-{cur_max}] -> [{t_min}-{t_max}]")

    @staticmethod
    def _resolve_indices(actions, params):
        sel = params.get('selected_indices')
        if sel:
            return sorted(i for i in sel if 0 <= i < len(actions))
        return list(range(len(actions)))
