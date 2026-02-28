"""
Directional Speed Limiter plugin  - Quickfix Tools.

Adjusts existing point positions to stay within a speed limit, with
optional directional bias (top/bottom preference for BJ/deep scenes).
Complementary to the existing Speed Limiter which adds intermediate points.
Algorithm by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin

import math


class QFDirectionalSpeedLimiterPlugin(FunscriptTransformationPlugin):
    """Adjust point positions to stay within a speed limit."""

    @property
    def name(self) -> str:
        return "Directional Speed Limiter"

    @property
    def description(self) -> str:
        return "Limit speed by adjusting positions, with optional directional bias (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'max_speed': {
                'type': float,
                'required': False,
                'default': 600.0,
                'description': 'Maximum speed in position-units per second',
                'constraints': {'min': 50.0, 'max': 2000.0}
            },
            'direction_bias': {
                'type': str,
                'required': False,
                'default': 'symmetric',
                'description': 'Bias for which direction to limit',
                'constraints': {'choices': ['symmetric', 'top', 'bottom']}
            },
            'prepare_mode': {
                'type': bool,
                'required': False,
                'default': False,
                'description': 'Nudge symmetric points to break ties before main pass'
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
        if not actions or len(actions) < 3:
            return
        indices = self._resolve(actions, params)
        if len(indices) < 2:
            return
        sel_set = set(indices)

        speed = params['max_speed'] / 1000.0  # convert to pos-units per ms
        bias = params['direction_bias']
        lim_top = bias in ('symmetric', 'top')
        lim_bot = bias in ('symmetric', 'bottom')

        if params['prepare_mode']:
            self._prepare(actions, indices, sel_set, lim_top, lim_bot)
        else:
            self._limit(actions, indices, sel_set, speed, lim_top, lim_bot)

        funscript._invalidate_cache(axis)
        self.logger.info(f"Directional Speed Limiter on {axis}: speed={params['max_speed']}, "
                         f"bias={bias}, {len(indices)} pts")

    def _limit(self, actions, indices, sel_set, speed, lim_top, lim_bot):
        """Main speed limiting pass  - adjust positions to stay within speed."""
        for idx in indices:
            p1 = actions[idx]['pos']
            t1 = actions[idx]['at']

            # Find neighbors
            if idx <= 0 or idx >= len(actions) - 1:
                continue
            a0 = actions[idx - 1]
            a2 = actions[idx + 1]
            p0, t0 = a0['pos'], a0['at']
            p2, t2 = a2['pos'], a2['at']

            dt2 = t2 - t1
            if dt2 <= 0:
                continue
            s2 = (p2 - p1) / dt2
            av2 = 0.5 * (p1 + p2)

            # Check speed towards next point
            if s2 < -speed and ((av2 < 50 or idx + 1 not in sel_set) or (av2 == 50 and lim_top)):
                actions[idx]['pos'] = min(p1, math.floor(p2 + speed * dt2))
                p1 = actions[idx]['pos']
            if s2 > speed and ((av2 > 50 or idx + 1 not in sel_set) or (av2 == 50 and lim_bot)):
                actions[idx]['pos'] = max(p1, math.ceil(p2 - speed * dt2))
                p1 = actions[idx]['pos']

            # Check speed from previous point
            dt0 = t1 - t0
            if dt0 <= 0:
                continue
            s0 = (p1 - p0) / dt0
            av0 = 0.5 * (p0 + p1)

            if s0 < -speed and ((av0 > 50 or idx - 1 not in sel_set) or (av0 == 50 and lim_bot)):
                actions[idx]['pos'] = max(p1, math.ceil(p0 - speed * dt0))
            if s0 > speed and ((av0 < 50 or idx - 1 not in sel_set) or (av0 == 50 and lim_top)):
                actions[idx]['pos'] = min(p1, math.floor(p0 + speed * dt0))

    def _prepare(self, actions, indices, sel_set, lim_top, lim_bot):
        """Prepare pass  - nudge symmetric points to break ties."""
        for idx in indices:
            if idx <= 0 or idx >= len(actions) - 1:
                continue
            p1 = actions[idx]['pos']
            t1 = actions[idx]['at']
            a2 = actions[idx + 1]
            p2, t2 = a2['pos'], a2['at']
            dt2 = t2 - t1
            if dt2 <= 0:
                continue
            s2 = (p2 - p1) / dt2
            av2 = 0.5 * (p1 + p2)

            if av2 == 50 and lim_top and s2 < 0:
                actions[idx]['pos'] = p1 - 1
            if av2 == 50 and lim_bot and s2 > 0:
                actions[idx]['pos'] = p1 + 1

    @staticmethod
    def _resolve(actions, params):
        sel = params.get('selected_indices')
        if sel:
            return sorted(i for i in sel if 0 <= i < len(actions))
        return list(range(len(actions)))
