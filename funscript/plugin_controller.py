"""Plugin registry facade for MultiAxisFunscript.

Accessed via `fs.plugins.<method>()`. Flat API preserved as delegators.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from funscript.multi_axis_funscript import MultiAxisFunscript


def _ensure_plugins_loaded(logger) -> None:
    from funscript.plugins.base_plugin import plugin_registry
    from funscript.plugins.plugin_loader import plugin_loader
    if not plugin_registry.is_global_plugins_loaded():
        builtin_results = plugin_loader.load_builtin_plugins()
        if logger is not None:
            logger.debug(f"Loaded {len(builtin_results)} built-in plugins")
        user_results = plugin_loader.load_user_plugins()
        if logger is not None:
            logger.debug(f"Loaded {len(user_results)} user plugins")
        plugin_registry.set_global_plugins_loaded(True)


class PluginController:
    """Thin facade over `funscript.plugins.plugin_registry`."""

    __slots__ = ("fs",)

    def __init__(self, fs: "MultiAxisFunscript") -> None:
        self.fs = fs

    def list_available_plugins(self) -> List[Dict]:
        from funscript.plugins.base_plugin import plugin_registry
        _ensure_plugins_loaded(self.fs.logger)
        return plugin_registry.list_plugins()

    def apply_plugin(self, plugin_name: str, axis: str = 'both', **parameters) -> bool:
        from funscript.plugins.base_plugin import plugin_registry
        fs = self.fs
        _ensure_plugins_loaded(fs.logger)

        plugin = plugin_registry.get_plugin(plugin_name)
        if not plugin:
            fs.logger.error(f"Plugin '{plugin_name}' not found")
            return False

        try:
            result = plugin.transform(fs, axis=axis, **parameters)
            if result is not None:
                if axis in ('primary', 'both'):
                    fs.primary_actions = result.primary_actions
                if axis in ('secondary', 'both'):
                    fs.secondary_actions = result.secondary_actions
                fs._invalidate_cache()
            return True
        except Exception as e:
            fs.logger.error(f"Error applying plugin '{plugin_name}': {e}")
            return False

    def get_plugin_preview(self, plugin_name: str, axis: str = 'both', **parameters) -> Dict[str, Any]:
        from funscript.plugins.base_plugin import plugin_registry
        fs = self.fs
        _ensure_plugins_loaded(fs.logger)

        plugin = plugin_registry.get_plugin(plugin_name)
        if not plugin:
            return {"error": f"Plugin '{plugin_name}' not found"}

        try:
            return plugin.get_preview(fs, axis=axis, **parameters)
        except Exception as e:
            return {"error": f"Error generating preview for '{plugin_name}': {e}"}
