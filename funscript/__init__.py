"""
Funscript package initialization.
"""

from .dual_axis_funscript import DualAxisFunscript

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
        'DualAxisFunscript',
        'FunscriptTransformationPlugin',
        'PluginRegistry', 
        'plugin_registry',
        'PluginLoader',
        'plugin_loader'
    ]
except ImportError:
    # Plugin system not available
    __all__ = ['DualAxisFunscript']
