"""
Funscript package initialization.
"""

from .multi_axis_funscript import MultiAxisFunscript

# Backward-compat alias
DualAxisFunscript = MultiAxisFunscript

# Import plugin system components
try:
    from .plugins.base_plugin import (
        FunscriptTransformationPlugin,
        PluginRegistry,
        plugin_registry
    )
    from .plugins.plugin_loader import PluginLoader, plugin_loader

    # Export plugin system
    __all__ = [
        'MultiAxisFunscript',
        'DualAxisFunscript',
        'FunscriptTransformationPlugin',
        'PluginRegistry',
        'plugin_registry',
        'PluginLoader',
        'plugin_loader'
    ]
except ImportError:
    # Plugin system not available
    __all__ = ['MultiAxisFunscript', 'DualAxisFunscript']
