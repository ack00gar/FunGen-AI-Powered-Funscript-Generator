"""
Plugin UI Manager - Dynamic plugin-driven user interface system.

This module provides a completely generic UI system that automatically discovers
plugins and generates UI elements based on plugin metadata, eliminating the need
for hardcoded filter handling.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class PluginUIState(Enum):
    """States for plugin UI elements."""
    CLOSED = "closed"
    OPEN = "open"
    PREVIEWING = "previewing"


@dataclass
class PluginUIContext:
    """Context information for plugin UI rendering."""
    plugin_name: str
    plugin_instance: Any
    state: PluginUIState = PluginUIState.CLOSED
    parameters: Dict[str, Any] = None
    apply_to_selection: bool = False
    preview_actions: Optional[List[Dict]] = None
    error_message: Optional[str] = None
    apply_requested: bool = False

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class PluginUIManager:
    """
    Manages plugin-driven UI generation and interaction.
    
    This class provides a completely generic interface for:
    - Auto-discovering available plugins
    - Generating UI elements based on plugin schemas
    - Handling plugin previews and execution
    - Managing plugin state and parameters
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('PluginUIManager')
        self.plugin_contexts: Dict[str, PluginUIContext] = {}
        self.active_preview_plugin: Optional[str] = None
        self._plugin_registry = None
    
    def initialize(self):
        """Initialize the plugin system and discover available plugins."""
        try:
            from funscript.plugins.base_plugin import plugin_registry
            from funscript.plugins.plugin_loader import plugin_loader
            
            self._plugin_registry = plugin_registry
            
            # Load all available plugins
            builtin_results = plugin_loader.load_builtin_plugins()
            user_results = plugin_loader.load_user_plugins()
            
            # Create UI contexts for all loaded plugins
            self._create_plugin_contexts()
            
            self.logger.info(f"Initialized plugin UI manager with {len(self.plugin_contexts)} plugins")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin UI manager: {e}")
    
    def _create_plugin_contexts(self):
        """Create UI contexts for all available plugins."""
        if not self._plugin_registry:
            return
            
        # Get all registered plugins
        all_plugins = self._plugin_registry.list_plugins()
        
        for plugin_info in all_plugins:
            plugin_name = plugin_info['name']
            plugin_instance = self._plugin_registry.get_plugin(plugin_name)
            
            if plugin_instance:
                context = PluginUIContext(
                    plugin_name=plugin_name,
                    plugin_instance=plugin_instance,
                    parameters=self._get_default_parameters(plugin_instance)
                )
                self.plugin_contexts[plugin_name] = context
    
    def _get_default_parameters(self, plugin_instance) -> Dict[str, Any]:
        """Extract default parameters from plugin schema."""
        try:
            schema = plugin_instance.parameters_schema
            defaults = {}
            
            for param_name, param_info in schema.items():
                if 'default' in param_info:
                    defaults[param_name] = param_info['default']
            
            return defaults
        except Exception as e:
            self.logger.warning(f"Failed to get default parameters for plugin: {e}")
            return {}
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names."""
        return list(self.plugin_contexts.keys())
    
    def get_plugin_display_name(self, plugin_name: str) -> str:
        """Get display name for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        if context and context.plugin_instance:
            return context.plugin_instance.name
        return plugin_name
    
    def get_plugin_description(self, plugin_name: str) -> str:
        """Get description for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        if context and context.plugin_instance:
            return context.plugin_instance.description
        return ""
    
    def is_plugin_available(self, plugin_name: str) -> bool:
        """Check if a plugin is available and has all dependencies."""
        context = self.plugin_contexts.get(plugin_name)
        if not context or not context.plugin_instance:
            return False
        
        try:
            return context.plugin_instance.check_dependencies()
        except Exception:
            return False
    
    def get_plugin_parameters_schema(self, plugin_name: str) -> Dict[str, Any]:
        """Get parameter schema for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        if context and context.plugin_instance:
            try:
                return context.plugin_instance.parameters_schema
            except Exception as e:
                self.logger.warning(f"Failed to get parameter schema for {plugin_name}: {e}")
        return {}
    
    def update_plugin_parameter(self, plugin_name: str, param_name: str, value: Any):
        """Update a parameter value for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        if context:
            context.parameters[param_name] = value
    
    def get_plugin_parameter(self, plugin_name: str, param_name: str, default: Any = None) -> Any:
        """Get a parameter value for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        if context:
            return context.parameters.get(param_name, default)
        return default
    
    def set_plugin_state(self, plugin_name: str, state: PluginUIState):
        """Set the UI state for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        if context:
            context.state = state
    
    def get_plugin_state(self, plugin_name: str) -> PluginUIState:
        """Get the UI state for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        return context.state if context else PluginUIState.CLOSED
    
    def generate_preview(self, plugin_name: str, funscript_obj, axis: str = 'primary') -> bool:
        """
        Generate a preview for the specified plugin.
        
        Returns:
            True if preview was generated successfully, False otherwise
        """
        context = self.plugin_contexts.get(plugin_name)
        if not context or not context.plugin_instance:
            self.logger.warning(f"Plugin '{plugin_name}' not available for preview")
            return False
        
        try:
            # Clear any previous error
            context.error_message = None
            
            # Create a copy for preview
            temp_funscript = copy.deepcopy(funscript_obj)
            
            # Validate parameters
            validated_params = context.plugin_instance.validate_parameters(context.parameters)
            
            # Apply transformation
            result = context.plugin_instance.transform(temp_funscript, axis, **validated_params)
            
            if result:
                # Extract the relevant actions
                if axis == 'primary':
                    context.preview_actions = result.primary_actions
                elif axis == 'secondary':
                    context.preview_actions = result.secondary_actions
                else:  # both
                    # For 'both', we'll preview the primary axis
                    context.preview_actions = result.primary_actions
                
                context.state = PluginUIState.PREVIEWING
                self.active_preview_plugin = plugin_name
                
                self.logger.debug(f"Generated preview for plugin '{plugin_name}' with {len(context.preview_actions or [])} actions")
                return True
            else:
                context.error_message = "Plugin failed to generate result"
                self.logger.warning(f"Plugin '{plugin_name}' failed to generate preview")
                return False
                
        except Exception as e:
            context.error_message = str(e)
            self.logger.error(f"Error generating preview for plugin '{plugin_name}': {e}")
            return False
    
    def apply_plugin(self, plugin_name: str, funscript_obj, axis: str = 'primary') -> bool:
        """
        Apply the specified plugin to the funscript.
        
        Returns:
            True if plugin was applied successfully, False otherwise
        """
        context = self.plugin_contexts.get(plugin_name)
        if not context or not context.plugin_instance:
            self.logger.warning(f"Plugin '{plugin_name}' not available for application")
            return False
        
        try:
            # Clear any previous error
            context.error_message = None
            
            # Validate parameters
            validated_params = context.plugin_instance.validate_parameters(context.parameters)
            
            # Apply transformation
            result = context.plugin_instance.transform(funscript_obj, axis, **validated_params)
            
            # Some plugins return the funscript object, others return None (modify in-place)
            # Both are considered successful - the transform() call completing without
            # exception indicates success
            self.logger.info(f"Successfully applied plugin '{plugin_name}' to {axis} axis")
            return True
                
        except Exception as e:
            context.error_message = str(e)
            self.logger.error(f"Error applying plugin '{plugin_name}': {e}")
            return False
    
    def clear_preview(self, plugin_name: Optional[str] = None):
        """Clear preview for a specific plugin or all plugins."""
        if plugin_name:
            context = self.plugin_contexts.get(plugin_name)
            if context:
                context.preview_actions = None
                if context.state == PluginUIState.PREVIEWING:
                    context.state = PluginUIState.OPEN
                if self.active_preview_plugin == plugin_name:
                    self.active_preview_plugin = None
        else:
            # Clear all previews
            for context in self.plugin_contexts.values():
                context.preview_actions = None
                if context.state == PluginUIState.PREVIEWING:
                    context.state = PluginUIState.CLOSED
            self.active_preview_plugin = None
    
    def get_preview_actions(self, plugin_name: str) -> Optional[List[Dict]]:
        """Get preview actions for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        return context.preview_actions if context else None
    
    def get_plugin_error(self, plugin_name: str) -> Optional[str]:
        """Get error message for a plugin."""
        context = self.plugin_contexts.get(plugin_name)
        return context.error_message if context else None
    
    def close_all_plugins(self):
        """Close all plugin UIs and clear previews."""
        for context in self.plugin_contexts.values():
            context.state = PluginUIState.CLOSED
            context.preview_actions = None
            context.error_message = None
        self.active_preview_plugin = None
    
    def has_any_open_windows(self) -> bool:
        """Check if any plugin windows are currently open."""
        return any(context.state != PluginUIState.CLOSED 
                  for context in self.plugin_contexts.values())
    
    def has_any_active_previews(self) -> bool:
        """Check if any plugins have active previews."""
        return any(context.preview_actions is not None 
                  for context in self.plugin_contexts.values())
    
    def should_clear_all_previews(self) -> bool:
        """
        Determine if all previews should be cleared.
        Returns True if no plugins are open and no previews are active.
        """
        return not self.has_any_open_windows() and not self.has_any_active_previews()
    
    def check_and_handle_apply_requests(self) -> List[str]:
        """
        Check for plugins that have been requested to apply and return their names.
        Clears the apply_requested flag after checking.
        """
        apply_requests = []
        for plugin_name, context in self.plugin_contexts.items():
            if context.apply_requested:
                apply_requests.append(plugin_name)
                context.apply_requested = False  # Clear the flag
        return apply_requests
    
    def get_plugin_ui_data(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get all UI data for a plugin in a single call.
        
        Returns:
            Dictionary with all plugin UI information, or None if plugin not found
        """
        context = self.plugin_contexts.get(plugin_name)
        if not context:
            return None
        
        return {
            'name': plugin_name,
            'display_name': self.get_plugin_display_name(plugin_name),
            'description': self.get_plugin_description(plugin_name),
            'available': self.is_plugin_available(plugin_name),
            'state': context.state,
            'parameters': context.parameters,
            'schema': self.get_plugin_parameters_schema(plugin_name),
            'apply_to_selection': context.apply_to_selection,
            'has_preview': context.preview_actions is not None,
            'error': context.error_message
        }