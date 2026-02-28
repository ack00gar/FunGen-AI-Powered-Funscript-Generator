"""
Continue Min/Max plugin  - Quickfix Tools.

Sets top/bottom peaks in the selection to match the min/max found in
the N points immediately before the selection.
Algorithm by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class QFContinueMinMaxPlugin(FunscriptTransformationPlugin):
    """Align peaks in selection to the min/max of preceding points."""

    @property
    def name(self) -> str:
        return "Continue Min/Max"

    @property
    def description(self) -> str:
        return "Align peaks to the min/max of points before the selection (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'lookback_points': {
                'type': int,
                'required': False,
                'default': 4,
                'description': 'Number of points before selection to sample min/max from',
                'constraints': {'min': 2, 'max': 10}
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
        sel = params.get('selected_indices')
        if not sel or len(sel) < 2:
            return
        indices = sorted(i for i in sel if 0 <= i < len(actions))
        if not indices:
            return

        first_sel = indices[0]
        n = params['lookback_points']

        # Sample min/max from N points before selection
        lookback_start = max(0, first_sel - n)
        if lookback_start >= first_sel:
            self.logger.warning("No points before selection to sample from")
            return

        ref_min = 100
        ref_max = 0
        for i in range(lookback_start, first_sel):
            p = actions[i]['pos']
            ref_min = min(ref_min, p)
            ref_max = max(ref_max, p)

        # Detect top/bottom peaks and set them
        changed = 0
        for idx in indices:
            if self._is_top_point(actions, idx):
                actions[idx]['pos'] = ref_max
                changed += 1
            elif self._is_bottom_point(actions, idx):
                actions[idx]['pos'] = ref_min
                changed += 1

        funscript._invalidate_cache(axis)
        self.logger.info(f"Continue Min/Max on {axis}: {changed} peaks adjusted to [{ref_min}-{ref_max}]")

    @staticmethod
    def _is_top_point(actions, i):
        if i <= 0 or i >= len(actions) - 1:
            # Edge: compare with available neighbor only
            if i <= 0:
                return i < len(actions) - 1 and actions[i]['pos'] > actions[i + 1]['pos']
            return actions[i]['pos'] > actions[i - 1]['pos']
        return actions[i]['pos'] > actions[i - 1]['pos'] and actions[i]['pos'] > actions[i + 1]['pos']

    @staticmethod
    def _is_bottom_point(actions, i):
        if i <= 0 or i >= len(actions) - 1:
            if i <= 0:
                return i < len(actions) - 1 and actions[i]['pos'] < actions[i + 1]['pos']
            return actions[i]['pos'] < actions[i - 1]['pos']
        return actions[i]['pos'] < actions[i - 1]['pos'] and actions[i]['pos'] < actions[i + 1]['pos']
