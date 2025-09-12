"""
Base plugin interface for funscript transformations.

This module defines the standardized interface that all funscript transformation
plugins must implement. This allows for modular, extensible funscript processing
with user-defined add-ins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging


class FunscriptTransformationPlugin(ABC):
    """
    Abstract base class for all funscript transformation plugins.
    
    Each plugin represents a single transformation or filter that can be applied
    to a funscript. Plugins receive a funscript instance, apply their transformation,
    and return the modified funscript.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the plugin with optional logger."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this plugin."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what this plugin does."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of this plugin."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        Return the schema for parameters this plugin accepts.
        
        Schema format:
        {
            'parameter_name': {
                'type': str|int|float|bool|list,
                'required': bool,
                'default': Any,
                'description': str,
                'constraints': Dict (optional - min/max values, choices, etc.)
            }
        }
        """
        pass
    
    @property
    def supported_axes(self) -> List[str]:
        """Return list of supported axes. Default: ['primary', 'secondary', 'both']"""
        return ['primary', 'secondary', 'both']
    
    @property
    def requires_scipy(self) -> bool:
        """Return True if this plugin requires scipy."""
        return False
    
    @property
    def requires_rdp(self) -> bool:
        """Return True if this plugin requires rdp library."""
        return False
    
    @property
    def modifies_inplace(self) -> bool:
        """Return True if this plugin modifies the funscript in-place, False if it returns a copy."""
        return True
    
    @property
    def ui_preference(self) -> str:
        """
        Return the preferred UI behavior for this plugin.
        
        Returns:
            'direct' - Apply immediately with default parameters (no popup)
            'popup' - Show parameter configuration window (default)
        """
        return 'popup'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the provided parameters against the schema.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Validated and normalized parameters with defaults applied
            
        Raises:
            ValueError: If parameters don't match the schema
        """
        schema = self.parameters_schema
        validated = {}
        
        # Check required parameters
        for param_name, param_info in schema.items():
            if param_info.get('required', False) and param_name not in parameters:
                raise ValueError(f"Required parameter '{param_name}' is missing")
        
        # Validate and apply defaults
        for param_name, param_info in schema.items():
            if param_name in parameters:
                value = parameters[param_name]
                expected_type = param_info['type']
                
                # Type checking - allow None values when explicitly set as default
                if value is not None and not isinstance(value, expected_type):
                    try:
                        value = expected_type(value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Parameter '{param_name}' must be of type {expected_type.__name__}")
                
                # Constraint checking - skip for None values
                if value is not None and 'constraints' in param_info:
                    constraints = param_info['constraints']
                    if 'min' in constraints and value < constraints['min']:
                        raise ValueError(f"Parameter '{param_name}' must be >= {constraints['min']}")
                    if 'max' in constraints and value > constraints['max']:
                        raise ValueError(f"Parameter '{param_name}' must be <= {constraints['max']}")
                    if 'choices' in constraints and value not in constraints['choices']:
                        raise ValueError(f"Parameter '{param_name}' must be one of {constraints['choices']}")
                
                validated[param_name] = value
            elif 'default' in param_info:
                validated[param_name] = param_info['default']
        
        return validated
    
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        if self.requires_scipy:
            try:
                import scipy.signal
            except ImportError:
                self.logger.error(f"Plugin '{self.name}' requires scipy but it's not available")
                return False
        
        if self.requires_rdp:
            try:
                import rdp
            except ImportError:
                self.logger.error(f"Plugin '{self.name}' requires rdp but it's not available")
                return False
        
        return True
    
    @abstractmethod
    def transform(self, funscript, axis: str = 'both', **parameters) -> Optional['DualAxisFunscript']:
        """
        Apply the transformation to the funscript.
        
        Args:
            funscript: The DualAxisFunscript instance to transform
            axis: Which axis to transform ('primary', 'secondary', 'both')
            **parameters: Plugin-specific parameters
        
        Returns:
            Modified funscript if modifies_inplace is False, otherwise None
            
        Raises:
            ValueError: If parameters are invalid or axis is unsupported
        """
        pass
    
    def get_preview(self, funscript, axis: str = 'both', **parameters) -> Dict[str, Any]:
        """
        Generate a preview of what the transformation would do without applying it.
        
        Args:
            funscript: The DualAxisFunscript instance
            axis: Which axis to preview ('primary', 'secondary', 'both')
            **parameters: Plugin-specific parameters
        
        Returns:
            Dictionary with preview information (implementation-specific)
        """
        return {"preview": "Not implemented"}


class PluginRegistry:
    """Registry for managing funscript transformation plugins."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('PluginRegistry')
        self._plugins: Dict[str, FunscriptTransformationPlugin] = {}
        self._global_plugins_loaded = False
    
    def register(self, plugin: FunscriptTransformationPlugin) -> bool:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if successful, False if failed (e.g., dependencies missing)
        """
        if not plugin.check_dependencies():
            return False
        
        #if plugin.name in self._plugins:
        #    self.logger.warning(f"Plugin '{plugin.name}' is already registered, replacing")
        
        self._plugins[plugin.name] = plugin
        self.logger.debug(f"Registered plugin '{plugin.name}' v{plugin.version}")
        return True
    
    def unregister(self, plugin_name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if successful, False if plugin not found
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            self.logger.info(f"Unregistered plugin '{plugin_name}'")
            return True
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[FunscriptTransformationPlugin]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins with their metadata.
        
        Returns:
            List of dictionaries with plugin information
        """
        # Hidden plugins that should not appear in UI
        hidden_plugin_names = {'template_plugin', 'advanced_template', 'simple_scale'}
        
        return [
            {
                'name': plugin.name,
                'description': plugin.description,
                'version': plugin.version,
                'supported_axes': plugin.supported_axes,
                'parameters_schema': plugin.parameters_schema,
                'requires_scipy': plugin.requires_scipy,
                'requires_rdp': plugin.requires_rdp,
                'modifies_inplace': plugin.modifies_inplace,
                'ui_preference': plugin.ui_preference
            }
            for plugin in self._plugins.values()
            if plugin.name not in hidden_plugin_names 
            and 'template' not in plugin.name.lower() 
            and 'example' not in plugin.name.lower()
        ]
    
    def get_plugins_by_capability(self, requires_scipy: Optional[bool] = None, 
                                  requires_rdp: Optional[bool] = None,
                                  supports_axis: Optional[str] = None) -> List[str]:
        """
        Get plugin names filtered by capabilities.
        
        Args:
            requires_scipy: Filter by scipy requirement
            requires_rdp: Filter by rdp requirement  
            supports_axis: Filter by supported axis
            
        Returns:
            List of plugin names matching criteria
        """
        results = []
        for name, plugin in self._plugins.items():
            if requires_scipy is not None and plugin.requires_scipy != requires_scipy:
                continue
            if requires_rdp is not None and plugin.requires_rdp != requires_rdp:
                continue
            if supports_axis is not None and supports_axis not in plugin.supported_axes:
                continue
            results.append(name)
        return results
    
    def is_global_plugins_loaded(self) -> bool:
        """Check if plugins have been loaded globally."""
        return self._global_plugins_loaded
    
    def set_global_plugins_loaded(self, loaded: bool = True):
        """Set the global plugins loaded flag."""
        self._global_plugins_loaded = loaded


# Global plugin registry instance
plugin_registry = PluginRegistry()