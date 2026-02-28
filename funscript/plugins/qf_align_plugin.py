"""
Alignment plugins  - Quickfix Tools.

Three alignment strategies for selected actions:
  1. Align Peaks  - snap top/bottom peaks to selection's global max/min
  2. Align Top/Bottom  - average extreme points and shift entire sections
  3. Align Sections  - shift sections between selected points so midpoints align

Algorithms by Quickfix (EroScripts community).
"""

from typing import Dict, Any, Optional

try:
    from .base_plugin import FunscriptTransformationPlugin
except ImportError:
    from funscript.plugins.base_plugin import FunscriptTransformationPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_top(actions, i):
    if i <= 0 or i >= len(actions) - 1:
        return False
    return actions[i]['pos'] > actions[i - 1]['pos'] and actions[i]['pos'] > actions[i + 1]['pos']


def _is_bot(actions, i):
    if i <= 0 or i >= len(actions) - 1:
        return False
    return actions[i]['pos'] < actions[i - 1]['pos'] and actions[i]['pos'] < actions[i + 1]['pos']


def _clamp(v):
    return max(0, min(100, round(v)))


# ---------------------------------------------------------------------------
# 1. Align Peaks
# ---------------------------------------------------------------------------

class QFAlignPeaksPlugin(FunscriptTransformationPlugin):
    """Snap all top peaks to selection max, bottom peaks to selection min."""

    @property
    def name(self) -> str:
        return "Align Peaks"

    @property
    def description(self) -> str:
        return "Set all top peaks to selection max, bottoms to min (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'align_tops': {
                'type': bool, 'required': False, 'default': True,
                'description': 'Align top peaks to selection maximum'
            },
            'align_bottoms': {
                'type': bool, 'required': False, 'default': True,
                'description': 'Align bottom peaks to selection minimum'
            },
            'selected_indices': {
                'type': list, 'required': False, 'default': None,
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
        indices = _resolve(actions, params)
        if len(indices) < 3:
            return

        positions = [actions[i]['pos'] for i in indices]
        sel_max = max(positions)
        sel_min = min(positions)

        changed = 0
        for idx in indices:
            if params['align_tops'] and _is_top(actions, idx):
                actions[idx]['pos'] = sel_max
                changed += 1
            elif params['align_bottoms'] and _is_bot(actions, idx):
                actions[idx]['pos'] = sel_min
                changed += 1

        funscript._invalidate_cache(axis)
        self.logger.info(f"Align Peaks on {axis}: {changed} peaks aligned to [{sel_min}-{sel_max}]")


# ---------------------------------------------------------------------------
# 2. Align Top/Bottom
# ---------------------------------------------------------------------------

class QFAlignTopBottomPlugin(FunscriptTransformationPlugin):
    """Average extreme points and shift entire sections so peaks align."""

    @property
    def name(self) -> str:
        return "Align Top/Bottom"

    @property
    def description(self) -> str:
        return "Shift sections so top or bottom peaks align to their mean (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'align_mode': {
                'type': str, 'required': False, 'default': 'top',
                'description': 'Which peaks to align',
                'constraints': {'choices': ['top', 'bottom']}
            },
            'selected_indices': {
                'type': list, 'required': False, 'default': None,
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
        sel_indices = _resolve(actions, params)
        if len(sel_indices) < 3:
            return

        top_mode = params['align_mode'] == 'top'
        check_fn = _is_top if top_mode else _is_bot

        # Find extreme-point indices within selection
        extreme_indices = [i for i in sel_indices if check_fn(actions, i)]
        if len(extreme_indices) < 2:
            return

        # Target position = mean of extreme positions
        target = sum(actions[i]['pos'] for i in extreme_indices) / len(extreme_indices)

        # Shift each section between consecutive extreme points
        for s in range(len(extreme_indices) - 1):
            start = extreme_indices[s]
            end = extreme_indices[s + 1]
            avg_pos = 0.5 * (actions[start]['pos'] + actions[end]['pos'])
            shift = target - avg_pos
            for j in range(start, end):
                actions[j]['pos'] = _clamp(actions[j]['pos'] + shift)

        funscript._invalidate_cache(axis)
        self.logger.info(f"Align Top/Bottom ({params['align_mode']}) on {axis}: "
                         f"{len(extreme_indices)} peaks -> target {target:.0f}")


# ---------------------------------------------------------------------------
# 3. Align Sections
# ---------------------------------------------------------------------------

class QFAlignSectionsPlugin(FunscriptTransformationPlugin):
    """Shift sections between selected points so midpoints align to average."""

    @property
    def name(self) -> str:
        return "Align Sections"

    @property
    def description(self) -> str:
        return "Shift sections between selected points so midpoints align (by Quickfix)"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> str:
        return "Quickfix Tools"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'selected_indices': {
                'type': list, 'required': False, 'default': None,
                'description': 'Action indices to process (at least 2 needed)'
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
        sel = _resolve(actions, params)
        if len(sel) < 2:
            return

        # Overall average position of selected points
        target = sum(actions[i]['pos'] for i in sel) / len(sel)

        # Shift each section between consecutive selected points
        for s in range(len(sel) - 1):
            start = sel[s]
            end = sel[s + 1]
            avg_pos = 0.5 * (actions[start]['pos'] + actions[end]['pos'])
            shift = target - avg_pos
            for j in range(start, end):
                actions[j]['pos'] = _clamp(actions[j]['pos'] + shift)

        funscript._invalidate_cache(axis)
        self.logger.info(f"Align Sections on {axis}: {len(sel)} anchor points, target {target:.0f}")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _resolve(actions, params):
    sel = params.get('selected_indices')
    if sel:
        return sorted(i for i in sel if 0 <= i < len(actions))
    return list(range(len(actions)))
