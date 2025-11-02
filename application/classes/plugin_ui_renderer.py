"""
Plugin UI Renderer - Dynamic ImGui interface generation from plugin metadata.

This module provides automatic UI generation for plugins based on their parameter
schemas, eliminating the need for hardcoded UI elements.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from .plugin_ui_manager import PluginUIManager, PluginUIState

try:
    import imgui
except ImportError:
    imgui = None


class PluginUIRenderer:
    """
    Renders plugin UIs dynamically based on plugin metadata.
    
    This class automatically generates ImGui interfaces for any plugin
    based on its parameter schema, without requiring hardcoded UI code.
    """
    
    def __init__(self, plugin_manager: PluginUIManager, logger: Optional[logging.Logger] = None):
        self.plugin_manager = plugin_manager
        self.logger = logger or logging.getLogger('PluginUIRenderer')
        self.timeline_reference = None  # Will be set by timeline
        
        if not imgui:
            self.logger.error("ImGui not available - UI rendering will not work")
    
    def render_plugin_buttons(self, timeline_num: int, view_mode: str = 'expert') -> bool:
        """
        Render plugin buttons dynamically on the same line.
        
        Returns:
            True if any plugin button was clicked, False otherwise
        """
        if not imgui:
            return False
        
        any_button_clicked = False
        available_plugins = self.plugin_manager.get_available_plugins()
        
        # Sort plugins alphabetically for consistent order
        sorted_plugins = sorted(available_plugins)
        
        for i, plugin_name in enumerate(sorted_plugins):
            ui_data = self.plugin_manager.get_plugin_ui_data(plugin_name)
            if not ui_data or not ui_data['available']:
                continue
            
            # Add same_line() for all buttons except the first one
            if i > 0:
                imgui.same_line()
            
            # Create unique button ID
            button_id = f"{ui_data['display_name']}##Plugin{plugin_name}T{timeline_num}"
            
            # Render button
            if imgui.button(button_id):
                # Check if plugin should apply directly or open configuration window
                if self._should_apply_directly(plugin_name, ui_data):
                    # Apply directly - this will be handled by the timeline's callback system
                    context = self.plugin_manager.plugin_contexts.get(plugin_name)
                    if context:
                        context.apply_requested = True
                else:
                    # Open configuration window
                    self.plugin_manager.set_plugin_state(plugin_name, PluginUIState.OPEN)

                    # Auto-tick "apply to selection" if points are selected
                    if self.timeline_reference and hasattr(self.timeline_reference, 'multi_selected_action_indices'):
                        if self.timeline_reference.multi_selected_action_indices:
                            context = self.plugin_manager.plugin_contexts.get(plugin_name)
                            if context:
                                context.apply_to_selection = True
                any_button_clicked = True
            
            # Add tooltip
            if imgui.is_item_hovered() and ui_data['description']:
                imgui.set_tooltip(ui_data['description'])
        
        return any_button_clicked
    
    def render_plugin_windows(self, timeline_num: int, window_id_suffix: str) -> bool:
        """
        Render all open plugin configuration windows.
        
        Returns:
            True if any plugin is currently open, False otherwise
        """
        if not imgui:
            return False
        
        any_window_open = False
        
        for plugin_name in self.plugin_manager.get_available_plugins():
            ui_data = self.plugin_manager.get_plugin_ui_data(plugin_name)
            if not ui_data or ui_data['state'] == PluginUIState.CLOSED:
                continue
            
            any_window_open = True
            
            # Render the plugin window
            self._render_plugin_window(plugin_name, ui_data, timeline_num, window_id_suffix)
        
        return any_window_open
    
    def _render_plugin_window(self, plugin_name: str, ui_data: Dict[str, Any], 
                            timeline_num: int, window_id_suffix: str):
        """Render a single plugin configuration window."""
        window_title = f"{ui_data['display_name']} (Timeline {timeline_num})##Plugin{plugin_name}Window{window_id_suffix}"
        
        # Set window properties - center on screen
        if imgui.APPEARING:
            main_viewport = imgui.get_main_viewport()
            if main_viewport:
                # Center the window
                window_width = 480
                center_x = main_viewport.pos[0] + (main_viewport.size[0] - window_width) * 0.5
                center_y = main_viewport.pos[1] + main_viewport.size[1] * 0.3  # Slightly above center
                imgui.set_next_window_position(center_x, center_y, condition=imgui.APPEARING)
        
        imgui.set_next_window_size(480, 0, condition=imgui.APPEARING)
        
        window_expanded, window_open = imgui.begin(
            window_title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        
        if not window_open:
            self.plugin_manager.set_plugin_state(plugin_name, PluginUIState.CLOSED)
            self.plugin_manager.clear_preview(plugin_name)
            imgui.end()
            return
        
        if window_expanded:
            # Render plugin description
            if ui_data['description']:
                imgui.text_wrapped(ui_data['description'])
                imgui.separator()
            
            # Render parameter controls
            parameters_changed = self._render_parameter_controls(plugin_name, ui_data)
            
            # Auto-generate preview when parameters change or window first opens
            if parameters_changed or ui_data['state'] == PluginUIState.OPEN:
                self._start_plugin_preview(plugin_name, timeline_num)
            
            imgui.separator()
            
            # Render apply to selection checkbox if applicable
            if self._has_selection_support(ui_data):
                changed, apply_to_sel = imgui.checkbox(
                    f"Apply to selection##Plugin{plugin_name}ApplyToSel",
                    ui_data['apply_to_selection']
                )
                if changed:
                    context = self.plugin_manager.plugin_contexts.get(plugin_name)
                    if context:
                        context.apply_to_selection = apply_to_sel
                        # Refresh preview when apply_to_selection changes
                        self._start_plugin_preview(plugin_name, timeline_num)
            
            # Render action buttons
            self._render_action_buttons(plugin_name, ui_data, timeline_num, window_id_suffix)
            
            # Show error message if any
            if ui_data['error']:
                imgui.text_colored(ui_data['error'], 1.0, 0.3, 0.3, 1.0)
        
        imgui.end()
    
    def _render_parameter_controls(self, plugin_name: str, ui_data: Dict[str, Any]) -> bool:
        """
        Render parameter controls based on plugin schema.
        
        Returns:
            True if any parameter was changed, False otherwise
        """
        schema = ui_data['schema']
        parameters = ui_data['parameters']
        any_changed = False
        
        for param_name, param_info in schema.items():
            control_id = f"{param_name}##Plugin{plugin_name}Param"
            current_value = parameters.get(param_name, param_info.get('default'))
            
            # Render control based on parameter type
            changed, new_value = self._render_parameter_control(
                control_id, param_name, param_info, current_value
            )
            
            if changed:
                self.plugin_manager.update_plugin_parameter(plugin_name, param_name, new_value)
                any_changed = True
            
            # Add tooltip if description exists
            if imgui.is_item_hovered() and param_info.get('description'):
                imgui.set_tooltip(param_info['description'])
        
        return any_changed
    
    def _render_parameter_control(self, control_id: str, param_name: str, 
                                param_info: Dict[str, Any], current_value: Any) -> Tuple[bool, Any]:
        """Render a single parameter control based on its type and constraints."""
        param_type = param_info['type']
        constraints = param_info.get('constraints', {})
        
        # Format parameter name for display
        display_name = param_name.replace('_', ' ').title()
        
        if param_type == int:
            return self._render_int_control(control_id, display_name, current_value, constraints)
        elif param_type == float:
            return self._render_float_control(control_id, display_name, current_value, constraints)
        elif param_type == bool:
            return self._render_bool_control(control_id, display_name, current_value)
        elif param_type == str:
            return self._render_string_control(control_id, display_name, current_value, constraints)
        else:
            # Fallback for unsupported types
            imgui.text(f"{display_name}: {current_value} (unsupported type)")
            return False, current_value
    
    def _render_int_control(self, control_id: str, display_name: str, 
                          current_value: int, constraints: Dict[str, Any]) -> Tuple[bool, int]:
        """Render an integer parameter control."""
        min_val = constraints.get('min', 0)
        max_val = constraints.get('max', 100)
        
        # Ensure current_value is an integer
        if not isinstance(current_value, int):
            current_value = int(current_value) if current_value is not None else min_val
        
        if 'choices' in constraints:
            # Render as combo box for discrete choices
            choices = constraints['choices']
            current_index = choices.index(current_value) if current_value in choices else 0
            
            changed, new_index = imgui.combo(
                f"{display_name}##{control_id}", current_index, [str(choice) for choice in choices]
            )
            return changed, choices[new_index] if changed else current_value
        else:
            # Render as slider
            changed, new_value = imgui.slider_int(f"{display_name}##{control_id}", current_value, min_val, max_val)
            return changed, new_value
    
    def _render_float_control(self, control_id: str, display_name: str, 
                            current_value: float, constraints: Dict[str, Any]) -> Tuple[bool, float]:
        """Render a float parameter control."""
        min_val = constraints.get('min', 0.0)
        max_val = constraints.get('max', 1.0)
        
        # Ensure current_value is a float
        if not isinstance(current_value, (int, float)):
            current_value = float(current_value) if current_value is not None else min_val
        
        changed, new_value = imgui.slider_float(f"{display_name}##{control_id}", float(current_value), min_val, max_val)
        return changed, new_value
    
    def _render_bool_control(self, control_id: str, display_name: str, 
                           current_value: bool) -> Tuple[bool, bool]:
        """Render a boolean parameter control."""
        # Ensure current_value is a bool
        if not isinstance(current_value, bool):
            current_value = bool(current_value) if current_value is not None else False
        
        changed, new_value = imgui.checkbox(f"{display_name}##{control_id}", current_value)
        return changed, new_value
    
    def _render_string_control(self, control_id: str, display_name: str, 
                             current_value: str, constraints: Dict[str, Any]) -> Tuple[bool, str]:
        """Render a string parameter control."""
        # Ensure current_value is a string
        if not isinstance(current_value, str):
            current_value = str(current_value) if current_value is not None else ""
        
        if 'choices' in constraints:
            # Render as combo box for discrete choices
            choices = constraints['choices']
            current_index = choices.index(current_value) if current_value in choices else 0
            
            changed, new_index = imgui.combo(f"{display_name}##{control_id}", current_index, choices)
            return changed, choices[new_index] if changed else current_value
        else:
            # Render as text input
            changed, new_value = imgui.input_text(f"{display_name}##{control_id}", current_value, 256)
            return changed, new_value
    
    def _render_action_buttons(self, plugin_name: str, ui_data: Dict[str, Any],
                             timeline_num: int, window_id_suffix: str):
        """Render apply and cancel buttons for a plugin."""
        button_width = 100
        
        # Apply button
        apply_id = f"Apply##Plugin{plugin_name}Apply{window_id_suffix}"
        if imgui.button(apply_id, width=button_width):
            self._apply_plugin(plugin_name, timeline_num)
        
        imgui.same_line()
        
        # Cancel button
        cancel_id = f"Cancel##Plugin{plugin_name}Cancel{window_id_suffix}"
        if imgui.button(cancel_id, width=button_width):
            self.plugin_manager.set_plugin_state(plugin_name, PluginUIState.CLOSED)
            self.plugin_manager.clear_preview(plugin_name)
    
    def _has_selection_support(self, ui_data: Dict[str, Any]) -> bool:
        """Check if a plugin supports apply-to-selection functionality."""
        # For now, assume all plugins support selection
        # This could be determined from plugin metadata in the future
        return True
    
    def _start_plugin_preview(self, plugin_name: str, timeline_num: int):
        """Start/update preview for a plugin."""
        # Get the timeline instance to access funscript and axis
        if not self.timeline_reference:
            self.logger.debug("No timeline reference available for preview")
            return
        
        # Get funscript and axis information
        funscript_instance, axis_name = self.timeline_reference._get_target_funscript_details()
        if not funscript_instance:
            self.logger.debug("No funscript available for preview")
            return
        
        # Get selection information from the plugin context
        context = self.plugin_manager.plugin_contexts.get(plugin_name)
        selected_indices = None
        if context and context.apply_to_selection and hasattr(self.timeline_reference, 'multi_selected_action_indices'):
            selected_indices = list(self.timeline_reference.multi_selected_action_indices) if self.timeline_reference.multi_selected_action_indices else None
        
        # Generate preview through the plugin manager
        success = self.plugin_manager.generate_preview(plugin_name, funscript_instance, axis_name, selected_indices)
        if success:
            self.logger.debug(f"Preview generated successfully for {plugin_name}")
        else:
            self.logger.debug(f"Failed to generate preview for {plugin_name}")
    
    def _apply_plugin(self, plugin_name: str, timeline_num: int):
        """Apply a plugin through the timeline's callback system."""
        # Signal to the timeline that a plugin should be applied
        # The timeline will handle the actual application and undo management
        self.logger.info(f"Apply requested for plugin: {plugin_name} on timeline {timeline_num}")
        
        # Set a flag that the timeline can check
        context = self.plugin_manager.plugin_contexts.get(plugin_name)
        if context:
            context.apply_requested = True
    
    def set_timeline_reference(self, timeline):
        """Set reference to the timeline that owns this renderer."""
        self.timeline_reference = timeline
    
    def render_preview_line(self, plugin_name: str, draw_list, points_to_draw, color: int):
        """Render preview line for a plugin."""
        if not imgui or len(points_to_draw) < 2:
            return
        
        # This would render the preview line on the timeline
        # Implementation depends on timeline rendering system
        pass
    
    def should_clear_previews(self) -> bool:
        """Check if all plugin windows are closed and previews should be cleared."""
        for plugin_name in self.plugin_manager.get_available_plugins():
            state = self.plugin_manager.get_plugin_state(plugin_name)
            if state != PluginUIState.CLOSED:
                return False
        return True
    
    def _should_apply_directly(self, plugin_name: str, ui_data: Dict[str, Any]) -> bool:
        """
        Determine if a plugin should apply directly or open a configuration window.
        
        Returns True if the plugin should apply directly (no popup needed).
        Uses the plugin's ui_preference property to respect plugin author's intent.
        """
        # Check if plugin declares its UI preference
        plugin_instance = self.plugin_manager.plugin_contexts.get(plugin_name)
        if plugin_instance and hasattr(plugin_instance.plugin_instance, 'ui_preference'):
            ui_pref = plugin_instance.plugin_instance.ui_preference
            return ui_pref == 'direct'
        
        # Fallback: check for required parameters (conservative approach)
        schema = ui_data.get('schema', {})
        has_required_params = any(param.get('required', False) for param in schema.values())
        
        # Only apply directly if no required parameters AND no meaningful optional parameters
        if has_required_params:
            return False
        
        # Check if plugin has only selection/timing parameters
        non_selection_params = [p for p in schema.keys() 
                              if p not in ['start_time_ms', 'end_time_ms', 'selected_indices']]
        
        # If only selection/timing params, safe to apply directly
        return len(non_selection_params) == 0