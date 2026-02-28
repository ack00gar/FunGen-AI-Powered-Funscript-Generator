"""
Normalize (Moving Average) plugin  - Quickfix Tools.

Two-pass moving-average normalization: computes a trapezoid-weighted
moving average and absolute deviation, then normalizes each point's
deviation to a target range.  Quickfix's signature tool for camera
movement artifacts.
Algorithm by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


class QFNormalizePlugin(FunscriptTransformationPlugin):
    """Moving-average normalization  - centers action and normalizes range."""

    @property
    def name(self) -> str:
        return "Normalize (Moving Average)"

    @property
    def description(self) -> str:
        return "Two-pass moving-average normalization for camera drift (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'window_size': {
                'type': int,
                'required': False,
                'default': 7,
                'description': 'Moving average window (odd number recommended)',
                'constraints': {'min': 3, 'max': 31}
            },
            'target_range': {
                'type': int,
                'required': False,
                'default': 50,
                'description': 'Target deviation range (higher = more spread)',
                'constraints': {'min': 10, 'max': 100}
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
        if len(indices) < 3:
            return

        window = params['window_size']
        target_range = params['target_range']
        hwindow = window // 2

        # Build a contiguous array of positions for the selected indices
        idx_set = set(indices)
        idx_lo = indices[0]
        idx_hi = indices[-1]

        # --- Pass 1: compute moving average and absolute deviation ---
        mav = {}
        dev = {}
        for idx in indices:
            mavsum = 0.0
            count = 0
            for ii in range(-hwindow, hwindow + 1):
                j = idx + ii
                if idx_lo <= j <= idx_hi and 0 <= j < len(actions):
                    mavsum += actions[j]['pos']
                    count += 1
            # Subtract half of the boundary values (trapezoid weighting)
            lo_bound = max(idx - hwindow, idx_lo)
            hi_bound = min(idx + hwindow, idx_hi)
            if 0 <= lo_bound < len(actions) and 0 <= hi_bound < len(actions):
                mavsum -= 0.5 * (actions[lo_bound]['pos'] + actions[hi_bound]['pos'])
            mav[idx] = mavsum / max(window - 1, 1)
            dev[idx] = abs(actions[idx]['pos'] - mav[idx])

        # --- Pass 2: normalize using moving average of deviation ---
        for idx in indices:
            devsum = 0.0
            lo_count = min(idx - idx_lo, hwindow)
            hi_count = min(idx_hi - idx, hwindow)
            denom = lo_count + hi_count
            if denom == 0:
                continue
            for ii in range(-hwindow, hwindow):
                j = idx + ii
                if idx_lo <= j <= idx_hi and j in dev:
                    devsum += dev[j]
            devmav = devsum / denom
            if devmav == 0:
                continue
            new_pos = 50 + 0.005 * target_range * (actions[idx]['pos'] - mav[idx]) / devmav * 100
            actions[idx]['pos'] = max(0, min(100, round(new_pos)))

        funscript._invalidate_cache(axis)
        self.logger.info(f"Normalize on {axis}: window={window}, target_range={target_range}, "
                         f"{len(indices)} pts")

    @staticmethod
    def _resolve(actions, params):
        sel = params.get('selected_indices')
        if sel:
            return sorted(i for i in sel if 0 <= i < len(actions))
        return list(range(len(actions)))
