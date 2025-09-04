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
    
    # Class-level flag to track global plugin initialization
    _plugins_loaded = False
    _global_plugin_count = 0
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('PluginUIManager')
        self.plugin_contexts: Dict[str, PluginUIContext] = {}
        self.active_preview_plugin: Optional[str] = None
        self._plugin_registry = None
        self.preview_renderer = None  # Will be set by timeline
    
    def initialize(self):
        """Initialize the plugin system and discover available plugins."""
        try:
            from funscript.plugins.base_plugin import plugin_registry
            from funscript.plugins.plugin_loader import plugin_loader
            
            self._plugin_registry = plugin_registry
            
            # Check if plugins are already loaded globally
            if not PluginUIManager._plugins_loaded:
                # Load all available plugins only once
                builtin_results = plugin_loader.load_builtin_plugins()
                user_results = plugin_loader.load_user_plugins()
                
                # Count successful loads
                builtin_success = sum(builtin_results.values()) if builtin_results else 0
                user_success = sum(user_results.values()) if user_results else 0
                total_attempted = len(builtin_results) + len(user_results)
                total_success = builtin_success + user_success
                
                # Store global count and mark as loaded
                PluginUIManager._global_plugin_count = total_success
                PluginUIManager._plugins_loaded = True
                
                # Log concise summary only on first load
                if total_success > 0:
                    summary_parts = []
                    if builtin_success > 0:
                        summary_parts.append(f"{builtin_success} built-in")
                    if user_success > 0:
                        summary_parts.append(f"{user_success} user")
                    
                    summary = " + ".join(summary_parts)
                    self.logger.info(f"Plugin system ready: {total_success} plugins loaded ({summary})")
                else:
                    self.logger.info("Plugin system ready: No plugins loaded")
                    
                if total_attempted > total_success:
                    failed_count = total_attempted - total_success
                    self.logger.warning(f"{failed_count} plugins failed to load")
            else:
                # Plugins already loaded, just log a brief message
                self.logger.debug(f"Plugin system already initialized ({PluginUIManager._global_plugin_count} plugins)")
            
            # Create UI contexts for all loaded plugins (each instance needs its own contexts)
            self._create_plugin_contexts()
            
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
    
    def generate_preview(self, plugin_name: str, funscript_obj, axis: str = 'primary', selected_indices: Optional[List[int]] = None) -> bool:
        """
        Generate a preview for the specified plugin.
        
        Args:
            plugin_name: Name of the plugin
            funscript_obj: Funscript object to preview on
            axis: Axis to preview ('primary', 'secondary', or 'both')
            selected_indices: Optional list of selected action indices for "apply to selection"
        
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
            
            # Validate parameters and add selection info if applicable
            validated_params = context.plugin_instance.validate_parameters(context.parameters)
            if selected_indices:
                validated_params['selected_indices'] = selected_indices
            
            # Check if plugin supports new preview generation method
            if hasattr(context.plugin_instance, 'generate_preview'):
                # Use the new preview generation method
                preview_data = context.plugin_instance.generate_preview(funscript_obj, axis, **validated_params)
                
                if preview_data and not preview_data.get('error'):
                    # Send preview data to renderer if available
                    if self.preview_renderer:
                        self.preview_renderer.set_preview_data(plugin_name, preview_data)
                    
                    context.state = PluginUIState.PREVIEWING
                    self.active_preview_plugin = plugin_name
                    
                    self.logger.debug(f"Generated preview for plugin '{plugin_name}'")
                    return True
                else:
                    error_msg = preview_data.get('error', 'Failed to generate preview') if preview_data else 'Failed to generate preview'
                    context.error_message = error_msg
                    self.logger.warning(f"Plugin '{plugin_name}' preview generation failed: {error_msg}")
                    return False
            
            else:
                # Fallback to old preview method (creating a copy and transforming)
                # Create a copy for preview
                temp_funscript = copy.deepcopy(funscript_obj)
                
                # Store original actions for comparison
                if axis == 'primary':
                    original_actions = list(temp_funscript.primary_actions)
                elif axis == 'secondary':
                    original_actions = list(temp_funscript.secondary_actions)
                else:  # both - preview primary
                    original_actions = list(temp_funscript.primary_actions)
                
                # Apply transformation
                result = context.plugin_instance.transform(temp_funscript, axis, **validated_params)
                
                # Check if transform modified in-place (result is None) or returned new funscript
                final_funscript = result if result else temp_funscript
                
                # Extract the relevant actions after transformation
                if axis == 'primary':
                    transformed_actions = final_funscript.primary_actions
                elif axis == 'secondary':
                    transformed_actions = final_funscript.secondary_actions
                else:  # both
                    transformed_actions = final_funscript.primary_actions
                
                # Create preview data showing the changes
                preview_data = self._create_fallback_preview_data(
                    original_actions, transformed_actions, plugin_name, selected_indices
                )
                
                if preview_data:
                    # Send to preview renderer
                    if self.preview_renderer:
                        self.preview_renderer.set_preview_data(plugin_name, preview_data)
                    
                    context.state = PluginUIState.PREVIEWING
                    self.active_preview_plugin = plugin_name
                    
                    self.logger.debug(f"Generated fallback preview for plugin '{plugin_name}'")
                    return True
                else:
                    context.error_message = "Plugin failed to generate result"
                    self.logger.warning(f"Plugin '{plugin_name}' failed to generate preview")
                    return False
                
        except Exception as e:
            context.error_message = str(e)
            self.logger.error(f"Error generating preview for plugin '{plugin_name}': {e}")
            return False
    
    def apply_plugin(self, plugin_name: str, funscript_obj, axis: str = 'primary', selected_indices: Optional[List[int]] = None) -> bool:
        """
        Apply the specified plugin to the funscript.
        
        Args:
            plugin_name: Name of the plugin
            funscript_obj: Funscript object to apply plugin to
            axis: Axis to apply to ('primary', 'secondary', or 'both')
            selected_indices: Optional list of selected action indices for "apply to selection"
        
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
            
            # Validate parameters and add selection info if applicable
            validated_params = context.plugin_instance.validate_parameters(context.parameters)
            if selected_indices:
                validated_params['selected_indices'] = selected_indices
            
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
                # Clear preview in renderer if available
                if self.preview_renderer:
                    self.preview_renderer.clear_preview(plugin_name)
        else:
            # Clear all previews
            for context in self.plugin_contexts.values():
                context.preview_actions = None
                if context.state == PluginUIState.PREVIEWING:
                    context.state = PluginUIState.CLOSED
            self.active_preview_plugin = None
            # Clear all previews in renderer if available
            if self.preview_renderer:
                self.preview_renderer.clear_all_previews()
    
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
    
    def _create_fallback_preview_data(self, original_actions: List[Dict], 
                                    transformed_actions: List[Dict], 
                                    plugin_name: str, 
                                    selected_indices: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """Create preview data by comparing original and transformed actions."""
        try:
            # Create lookup for original actions by timestamp
            original_by_time = {action['at']: action for action in original_actions}
            transformed_by_time = {action['at']: action for action in transformed_actions}
            
            preview_points = []
            
            # If we have selected indices, create a set of selected timestamps for filtering
            selected_timestamps = None
            if selected_indices is not None and len(selected_indices) > 0:
                selected_timestamps = set()
                for idx in selected_indices:
                    if 0 <= idx < len(original_actions):
                        selected_timestamps.add(original_actions[idx]['at'])
            
            # Check all transformed actions for new/modified points
            for action in transformed_actions:
                timestamp = action['at']
                new_pos = action['pos']
                
                # Determine if this point is in the selection
                is_selected = selected_timestamps is None or timestamp in selected_timestamps
                
                if timestamp in original_by_time:
                    # Existing point - check if modified
                    original_pos = original_by_time[timestamp]['pos']
                    if new_pos != original_pos and is_selected:
                        # Modified point (only if selected)
                        preview_points.append({
                            'at': timestamp,
                            'pos': new_pos,
                            'is_modified': True,
                            'original_pos': original_pos,
                            'is_selected': is_selected
                        })
                    else:
                        # Unchanged point or unselected point - show for context
                        preview_points.append({
                            'at': timestamp,
                            'pos': new_pos,
                            'is_modified': False,
                            'is_selected': is_selected
                        })
                else:
                    # New point (only if selected)
                    if is_selected:
                        preview_points.append({
                            'at': timestamp,
                            'pos': new_pos,
                            'is_new': True,
                            'is_selected': is_selected
                        })
            
            # For plugins that remove many points, show all remaining points
            # This gives visual feedback about what will remain after the filter
            if len(transformed_actions) < len(original_actions) * 0.8:  # Significant reduction
                self.logger.debug(f"Plugin {plugin_name} reduced points significantly: {len(original_actions)} -> {len(transformed_actions)}")
            
            # Always return preview data if we have any transformation
            if preview_points or len(transformed_actions) != len(original_actions):
                # Use consistent 'default' style for all plugins - orange color
                return {
                    'preview_points': preview_points if preview_points else [
                        {'at': action['at'], 'pos': action['pos'], 'is_modified': False}
                        for action in transformed_actions
                    ],
                    'style': 'default'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create fallback preview data: {e}")
            return None
    
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