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


class PluginPipeline:
    """Ordered pipeline of plugin steps with preset management."""

    def __init__(self, app_instance, logger: Optional[logging.Logger] = None):
        self.app = app_instance
        self.logger = logger or logging.getLogger('PluginPipeline')
        self.steps: List[PipelineStep] = []

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
            (success, errors) — success is True if all steps succeeded.
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

    # ------------------------------------------------------------------
    # Serialization / presets
    # ------------------------------------------------------------------

    def to_list(self) -> List[dict]:
        return [s.to_dict() for s in self.steps]

    def load_from_list(self, data: List[dict]):
        self.steps = [PipelineStep.from_dict(d) for d in data]

    def get_available_presets(self) -> Dict[str, List[dict]]:
        """Return merged dict of default + user presets."""
        user_presets = self.app.app_settings.get("plugin_pipeline_presets", {})
        merged = {}
        merged.update(_DEFAULT_PRESETS)
        merged.update(user_presets)
        return merged

    def load_preset(self, name: str) -> bool:
        """Load a preset by name. Returns True if found."""
        presets = self.get_available_presets()
        if name not in presets:
            return False
        self.load_from_list(presets[name])
        return True

    def save_preset(self, name: str):
        """Save current pipeline as a user preset."""
        user_presets = self.app.app_settings.get("plugin_pipeline_presets", {})
        user_presets[name] = self.to_list()
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
