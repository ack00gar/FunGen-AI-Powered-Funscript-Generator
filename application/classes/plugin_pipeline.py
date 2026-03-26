"""
Plugin Pipeline — ordered chain of plugins applied sequentially.

Manages an ordered list of pipeline steps (plugin_name, params, enabled),
preset save/load, and execution against a funscript object.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Tuple


class PipelineStep:
    """A single step in a plugin pipeline."""

    __slots__ = ('plugin_name', 'params', 'enabled')

    def __init__(self, plugin_name: str, params: Optional[Dict[str, Any]] = None, enabled: bool = True):
        self.plugin_name = plugin_name
        self.params = params or {}
        self.enabled = enabled

    def to_dict(self) -> dict:
        return {
            'plugin_name': self.plugin_name,
            'params': copy.deepcopy(self.params),
            'enabled': self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PipelineStep':
        return cls(
            plugin_name=d['plugin_name'],
            params=d.get('params', {}),
            enabled=d.get('enabled', True),
        )


# Default presets shipped with the app
_DEFAULT_PRESETS = {
    "Ultimate Autotune": [
        {"plugin_name": "Ultimate Autotune", "params": {}, "enabled": True},
    ],
    "Light Polish": [
        {"plugin_name": "Smooth (SG)", "params": {"window_length": 5, "polyorder": 2}, "enabled": True},
        {"plugin_name": "Simplify (RDP)", "params": {"epsilon": 3.0}, "enabled": True},
    ],
    "Full Enhancement": [
        {"plugin_name": "Ultimate Autotune", "params": {}, "enabled": True},
        {"plugin_name": "Simplify (RDP)", "params": {"epsilon": 5.0}, "enabled": True},
        {"plugin_name": "Amplify", "params": {"scale_factor": 1.1, "center_value": 50}, "enabled": True},
    ],
}


# Map timeline labels to funscript axis names
AXIS_LABEL_TO_NAME = {"T1": "primary", "T2": "secondary"}


def timeline_label_to_axis(label: str, funscript_obj=None) -> str:
    """Convert a UI label like 'T1', 'T3' to the funscript axis name."""
    if label in AXIS_LABEL_TO_NAME:
        return AXIS_LABEL_TO_NAME[label]
    # T3+ use funscript axis assignment
    if label.startswith("T") and label[1:].isdigit():
        tnum = int(label[1:])
        if funscript_obj and hasattr(funscript_obj, 'get_axis_for_timeline'):
            return funscript_obj.get_axis_for_timeline(tnum)
        return f"axis_{tnum}"
    return 'primary'


class PluginPipeline:
    """Ordered pipeline of plugin steps with preset management."""

    def __init__(self, app_instance, logger: Optional[logging.Logger] = None):
        self.app = app_instance
        self.logger = logger or logging.getLogger('PluginPipeline')
        self.steps: List[PipelineStep] = []
        self.target_axis: str = "T1"  # "T1", "T2", "T3"..., or "All"

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, plugin_name: str, params: Optional[Dict[str, Any]] = None, index: Optional[int] = None):
        """Add a step. If index is None, append to end."""
        step = PipelineStep(plugin_name, params)
        if index is not None:
            self.steps.insert(index, step)
        else:
            self.steps.append(step)

    def remove_step(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps.pop(index)

    def move_step(self, from_idx: int, to_idx: int):
        """Move a step from one position to another."""
        if from_idx == to_idx:
            return
        if not (0 <= from_idx < len(self.steps) and 0 <= to_idx < len(self.steps)):
            return
        step = self.steps.pop(from_idx)
        self.steps.insert(to_idx, step)

    def clear(self):
        self.steps.clear()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, funscript_obj, axis: str = 'primary',
            selected_indices: Optional[List[int]] = None) -> Tuple[bool, List[str]]:
        """
        Execute all enabled steps in order on the funscript object.

        Returns:
            (success, errors) -- success is True if all steps succeeded.
        """
        errors = []
        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue
            params = copy.deepcopy(step.params)
            if selected_indices is not None:
                params['selected_indices'] = selected_indices
            try:
                ok = funscript_obj.apply_plugin(step.plugin_name, axis=axis, **params)
                if not ok:
                    errors.append(f"Step {i + 1} ({step.plugin_name}): plugin returned failure")
            except Exception as e:
                errors.append(f"Step {i + 1} ({step.plugin_name}): {e}")
                self.logger.warning(f"Pipeline step failed: {step.plugin_name}: {e}")

        return (len(errors) == 0, errors)

    def run_with_target(self, funscript_obj,
                        selected_indices: Optional[List[int]] = None) -> Tuple[bool, List[str]]:
        """Run pipeline using self.target_axis setting. 'All' runs on every axis."""
        if self.target_axis == "All":
            all_axes = funscript_obj.get_all_axis_names() if hasattr(funscript_obj, 'get_all_axis_names') else ['primary', 'secondary']
            all_ok = True
            all_errors = []
            for axis_name in all_axes:
                ok, errs = self.run(funscript_obj, axis=axis_name, selected_indices=selected_indices)
                if not ok:
                    all_ok = False
                all_errors.extend(errs)
            return (all_ok, all_errors)
        axis_name = timeline_label_to_axis(self.target_axis, funscript_obj)
        return self.run(funscript_obj, axis=axis_name, selected_indices=selected_indices)

    # ------------------------------------------------------------------
    # Serialization / presets
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize pipeline including target axis."""
        return {
            'steps': [s.to_dict() for s in self.steps],
            'target_axis': self.target_axis,
        }

    def to_list(self) -> List[dict]:
        """Legacy: serialize steps only (for backward-compatible presets)."""
        return [s.to_dict() for s in self.steps]

    def load_from_dict(self, data: dict):
        """Load pipeline from dict (new format with target_axis)."""
        self.steps = [PipelineStep.from_dict(d) for d in data.get('steps', [])]
        self.target_axis = data.get('target_axis', 'T1')

    def load_from_list(self, data: List[dict]):
        """Legacy: load steps from list (backward compat, target_axis defaults to T1)."""
        self.steps = [PipelineStep.from_dict(d) for d in data]

    def get_available_presets(self) -> Dict[str, List[dict]]:
        """Return merged dict of default + user presets."""
        user_presets = self.app.app_settings.get("plugin_pipeline_presets", {})
        merged = {}
        merged.update(_DEFAULT_PRESETS)
        merged.update(user_presets)
        return merged

    def load_preset(self, name: str) -> bool:
        """Load a preset by name. Returns True if found.

        Handles both old format (list of steps) and new format (dict with target_axis).
        """
        presets = self.get_available_presets()
        if name not in presets:
            return False
        data = presets[name]
        if isinstance(data, dict) and 'steps' in data:
            self.load_from_dict(data)
        else:
            self.load_from_list(data)
        return True

    def save_preset(self, name: str):
        """Save current pipeline as a user preset (new format with target_axis)."""
        user_presets = self.app.app_settings.get("plugin_pipeline_presets", {})
        user_presets[name] = self.to_dict()
        self.app.app_settings.set("plugin_pipeline_presets", user_presets)

    def delete_preset(self, name: str) -> bool:
        """Delete a user preset. Cannot delete built-in presets."""
        if name in _DEFAULT_PRESETS:
            return False
        user_presets = self.app.app_settings.get("plugin_pipeline_presets", {})
        if name in user_presets:
            del user_presets[name]
            self.app.app_settings.set("plugin_pipeline_presets", user_presets)
            return True
        return False

    def is_builtin_preset(self, name: str) -> bool:
        return name in _DEFAULT_PRESETS
