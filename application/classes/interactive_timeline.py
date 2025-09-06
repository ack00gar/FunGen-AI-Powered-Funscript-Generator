import imgui
import os
import numpy as np
import math
import time
import glfw
import copy
from typing import Optional, List, Dict, Tuple
from bisect import bisect_left, bisect_right
from .plugin_ui_manager import PluginUIManager, PluginUIState
from .plugin_ui_renderer import PluginUIRenderer
from .plugin_preview_renderer import PluginPreviewRenderer

from application.utils import _format_time
from config.element_group_colors import TimelineColors


class InteractiveFunscriptTimeline:
    def __init__(self, app_instance, timeline_num: int):
        self.app = app_instance
        self.timeline_num = timeline_num

        # GPU Integration (if available)
        self.gpu_integration = getattr(app_instance, 'gpu_integration', None)

        self.selected_action_idx = -1
        self.dragging_action_idx = -1
        self.drag_start_action_state: Optional[Dict] = None
        self.drag_undo_recorded = False
        self.context_mouse_pos_screen = (0, 0)
        self.new_point_candidate_at = 0
        self.new_point_candidate_pos = 50

        self.show_sg_settings_popup = False
        self.sg_apply_to_selection = False
        self.show_rdp_settings_popup = False
        self.rdp_apply_to_selection = False

        # --- AMP STATE ---
        self.show_amp_settings_popup = False
        self.amp_apply_to_selection = False
        self.amp_scale_factor = self.app.app_settings.get(f"timeline{self.timeline_num}_amp_default_scale", 1.5)
        self.amp_center_value = self.app.app_settings.get(f"timeline{self.timeline_num}_amp_default_center", 50)

        # PERFORMANCE OPTIMIZATION: Cache pre-computed arrays to avoid recreating them every frame
        self._cached_actions_data = None  # Stores pre-computed arrays
        self._cached_actions_hash = None  # Hash of actions_list to detect changes
        self._cached_arrays_initialized = False

        # RADICAL OPTIMIZATION: Motion-based temporal culling
        self._last_pan_offset = 0
        self._last_frame_time = time.time()
        self._pan_velocity = 0
        
        # New Plugin System
        self.plugin_manager = PluginUIManager(logger=self.app.logger)
        self.plugin_renderer = PluginUIRenderer(self.plugin_manager, logger=self.app.logger)
        self.plugin_preview_renderer = PluginPreviewRenderer(logger=self.app.logger)
        
        # Connect preview renderer to plugin manager
        self.plugin_manager.preview_renderer = self.plugin_preview_renderer
        
        # Connect timeline reference to plugin renderer
        self.plugin_renderer.set_timeline_reference(self)
        
        # Initialize plugin system
        self.plugin_manager.initialize()
        
        # Legacy plugin system state (will be removed)
        self._available_plugins = None
        self._load_plugins()
        self.plugin_popups = {}  # Store plugin popup states

        # --- KEYFRAME STATE ---
        self.show_keyframe_settings_popup = False
        self.keyframe_apply_to_selection = False
        self.keyframe_position_tolerance = self.app.app_settings.get(
            f"timeline{self.timeline_num}_keyframe_default_pos_tol", 10)
        self.keyframe_time_tolerance = self.app.app_settings.get(
            f"timeline{self.timeline_num}_keyframe_default_time_tol", 50)

        # --- SPEED LIMITER STATE ---
        self.show_speed_limiter_popup = False
        self.min_interval = self.app.app_settings.get(f"timeline{self.timeline_num}_min_interval", 60)
        self.vibe_amount = self.app.app_settings.get(f"timeline{self.timeline_num}_vibe_amount", 0)
        self.speed_threshold = self.app.app_settings.get(f"timeline{self.timeline_num}_speed_threshold", 500.0)

        # Get initial defaults from app_settings (AppLogic holds app_settings directly)
        self.sg_window_length = self.app.app_settings.get(f"timeline{self.timeline_num}_sg_default_window", 5)
        self.sg_poly_order = self.app.app_settings.get(f"timeline{self.timeline_num}_sg_default_polyorder", 2)
        self.rdp_epsilon = self.app.app_settings.get(f"timeline{self.timeline_num}_rdp_default_epsilon", 8.0)

        # --- PEAKS STATE ---
        self.show_peaks_settings_popup = False
        self.peaks_apply_to_selection = False
        self.peaks_height = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_height", 0)
        self.peaks_threshold = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_threshold", 0)
        self.peaks_distance = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_distance", 1)
        self.peaks_prominence = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_prominence", 0)
        self.peaks_width = self.app.app_settings.get(f"timeline{self.timeline_num}_peaks_default_width", 0)

        # --- AUTO-TUNE STATE ---
        self.show_autotune_popup = False
        self.autotune_apply_to_selection = False
        self.autotune_sat_low = self.app.app_settings.get(f"timeline{self.timeline_num}_autotune_default_sat_low", 1)
        self.autotune_sat_high = self.app.app_settings.get(f"timeline{self.timeline_num}_autotune_default_sat_high", 99)
        self.autotune_max_window = self.app.app_settings.get(f"timeline{self.timeline_num}_autotune_default_max_window", 15)
        self.autotune_polyorder = self.app.app_settings.get(f"timeline{self.timeline_num}_autotune_default_polyorder", 2)

        # --- ULTIMATE AUTOTUNE STATE ---
        self.show_ultimate_autotune_popup = False

        self.ultimate_presmoothing_enabled = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_presmoothing_enabled", True)
        self.ultimate_presmoothing_max_window = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_presmoothing_max_window", 15)
        self.ultimate_presmoothing_polyorder = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_presmoothing_polyorder", 2)

        self.ultimate_peaks_enabled = self.app.app_settings.get(f"timeline{self.timeline_num}_ultimate_peaks_enabled", True)
        self.ultimate_peaks_prominence = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_peaks_prominence", 10)

        self.ultimate_recovery_enabled = self.app.app_settings.get(f"timeline{self.timeline_num}_ultimate_recovery_enabled", True)
        self.ultimate_recovery_threshold = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_recovery_threshold", 1.8)

        self.ultimate_normalization_enabled = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_normalization_enabled", True)

        # self.ultimate_regeneration_enabled = self.app.app_settings.get(f"timeline{self.timeline_num}_ultimate_regeneration_enabled", True)
        # self.ultimate_resample_rate = self.app.app_settings.get(f"timeline{self.timeline_num}_ultimate_resample_rate", 40)

        self.ultimate_speed_limit_enabled = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_speed_limit_enabled", True)
        self.ultimate_speed_threshold = self.app.app_settings.get(
            f"timeline{self.timeline_num}_ultimate_speed_threshold", 500.0)

        self.multi_selected_action_indices = set()
        self.is_marqueeing = False
        self.marquee_start_screen_pos = None
        self.marquee_end_screen_pos = None

        self.shift_frames_amount = 1

        # Unified interaction flag, replacing is_interacting_with_pan_zoom and app_state.timeline_interaction_active
        self.is_interacting: bool = False
        self.is_panning_active: bool = False
        self.is_zooming_active: bool = False

        self.zoom_action_request: Optional[float] = None

        self.is_previewing: bool = False
        self.preview_actions: Optional[List[Dict]] = None

        # For range selection
        self.selection_anchor_idx: int = -1

        # Stores the index of the point that was right-clicked to open the context menu
        self.context_menu_point_idx: int = -1
        
        # ALT+Drag range selection state
        self.range_selecting = False
        self.range_selection_start_time = 0
        self.range_selection_end_time = 0

        # --- Persistent Ultimate Autotune Preview ---
        # In simple mode, always enable ultimate autotune preview
        is_simple_mode = getattr(self.app.app_state_ui, 'ui_view_mode', 'expert') == 'simple'
        if is_simple_mode:
            self.show_ultimate_autotune_preview = True
        else:
            self.show_ultimate_autotune_preview = self.app.app_settings.get(
                f"timeline{self.timeline_num}_show_ultimate_preview", True)
        self.ultimate_autotune_preview_actions: Optional[List[Dict]] = None
        self._ultimate_preview_dirty: bool = True

    def invalidate_cache(self):
        """
        Forces the timeline to re-compute its cached data on the next frame.
        Call this after any external modification to the actions list.
        """
        self._cached_actions_hash = None
        self._cached_arrays_initialized = False
        self.app.logger.debug(f"T{self.timeline_num}: Cache invalidated.")

    def invalidate_ultimate_preview(self):
        """Marks the ultimate autotune preview as dirty, forcing recalculation."""
        self._ultimate_preview_dirty = True

    def _update_ultimate_autotune_preview(self):
        """
        Recalculates the ultimate autotune preview if it's enabled and dirty.
        This is designed to be called once per frame during rendering.
        """
        if not self.show_ultimate_autotune_preview:
            if self.ultimate_autotune_preview_actions is not None:
                self.ultimate_autotune_preview_actions = None
            return

        funscript_instance, axis_name = self._get_target_funscript_details()
        actions_list = self._get_actions_list_ref()

        if not funscript_instance or not axis_name or not actions_list:
            if self.ultimate_autotune_preview_actions is not None:
                self.ultimate_autotune_preview_actions = None
            self._ultimate_preview_dirty = False  # Nothing to process
            return

        if not self._ultimate_preview_dirty:
            return

        # It's dirty and needs recalculation
        try:
            # Use the Ultimate Autotune plugin for preview
            from funscript.plugins.base_plugin import plugin_registry
            ultimate_plugin = plugin_registry.get_plugin('Ultimate Autotune')
            if ultimate_plugin:
                # Create a copy for preview
                import copy
                temp_funscript = copy.deepcopy(funscript_instance)
                result = ultimate_plugin.transform(temp_funscript, axis_name)
                if result:
                    self.ultimate_autotune_preview_actions = (result.primary_actions if axis_name == 'primary' else result.secondary_actions)
                else:
                    self.ultimate_autotune_preview_actions = None
            else:
                self.logger.warning("Ultimate Autotune plugin not available")
                self.ultimate_autotune_preview_actions = None
        except Exception as e:
            self.app.logger.error(f"T{self.timeline_num}: Failed to generate ultimate autotune preview: {e}", exc_info=True)
            self.ultimate_autotune_preview_actions = None
        finally:
            self._ultimate_preview_dirty = False

    def _get_target_funscript_details(self) -> Tuple[Optional[object], Optional[str]]:
        if self.app.funscript_processor:
            return self.app.funscript_processor._get_target_funscript_object_and_axis(self.timeline_num)
        return None, None

    def _get_actions_list_ref(self) -> Optional[List[Dict]]:
        funscript_instance, axis_name = self._get_target_funscript_details()
        if funscript_instance and axis_name:
            return getattr(funscript_instance, f"{axis_name}_actions", None)
        return None

    def _get_or_compute_cached_arrays(self, actions_list: List[Dict]) -> Dict:
        """
        PERFORMANCE OPTIMIZATION: Get or compute cached arrays to avoid recreating them every frame.
        Returns pre-computed arrays for timestamps, positions for efficient timeline rendering.
        """
        if not actions_list:
            return {"ats": np.array([]), "poss": np.array([])}

        # Create a simple hash of the actions list to detect changes
        # Using length + first/last elements as a fast approximation
        if len(actions_list) >= 2:
            actions_hash = (
                len(actions_list), actions_list[0]["at"],
                actions_list[0]["pos"],
                actions_list[-1]["at"],
                actions_list[-1]["pos"]
            )
        else:
            actions_hash = (
                len(actions_list),
                actions_list[0]["at"] if actions_list else 0,
                actions_list[0]["pos"] if actions_list else 0
            )

        # Check if we need to recompute the cache
        if (
            self._cached_actions_hash != actions_hash or
            self._cached_actions_data is None or
            not self._cached_arrays_initialized
        ):
            # Recompute cached arrays
            self._cached_actions_data = {
                "ats": np.array([action["at"] for action in actions_list], dtype=float),
                "poss": np.array([action["pos"] for action in actions_list], dtype=float),
            }
            self._cached_actions_hash = actions_hash
            self._cached_arrays_initialized = True

        return self._cached_actions_data

    def _render_dense_envelope(
            self,
            draw_list,
            canvas_abs_pos: Tuple[float, float],
            canvas_size: Tuple[float, float],
            app_state,
            cached_data: Dict,
            step_px: int = 2,
        ) -> None:
        """
        Dense-mode rendering for very large action counts or high zoom-out levels.
        Renders a min/max envelope per screen x-column using vectorized binning.
        """
        ats_all: np.ndarray = cached_data.get("ats", np.array([], dtype=float))
        poss_all: np.ndarray = cached_data.get("poss", np.array([], dtype=float))

        if ats_all.size < 2:
            return

        canvas_x0 = int(canvas_abs_pos[0])
        canvas_w = int(canvas_size[0])
        if canvas_w <= 0:
            return

        zoom_ms_per_px: float = getattr(app_state, 'timeline_zoom_factor_ms_per_px', 1.0) or 1.0
        pan_ms: float = float(getattr(app_state, 'timeline_pan_offset_ms', 0))

        # Vectorized transform to screen coords
        x_screen = np.round((ats_all - pan_ms) / zoom_ms_per_px).astype(np.int32) + canvas_x0
        y_screen = (canvas_abs_pos[1] + canvas_size[1] - (poss_all / 100.0 * canvas_size[1])).astype(np.int32)

        # Visible mask within current canvas
        x_min = canvas_x0
        x_max = canvas_x0 + canvas_w - 1
        visible_mask = (x_screen >= x_min) & (x_screen <= x_max)
        if not np.any(visible_mask):
            return

        x_vis = x_screen[visible_mask]
        y_vis = y_screen[visible_mask]

        # Bin y values per x column using numpy indexed reductions
        x_idx = x_vis - canvas_x0  # 0..canvas_w-1
        min_vals = np.full(canvas_w, np.iinfo(np.int32).max, dtype=np.int32)
        max_vals = np.full(canvas_w, np.iinfo(np.int32).min, dtype=np.int32)
        np.minimum.at(min_vals, x_idx, y_vis)
        np.maximum.at(max_vals, x_idx, y_vis)

        # Draw vertical filled bars for bins that received data
        alpha = 0.35 if getattr(self, 'is_previewing', False) else 0.8
        rgba = (*TimelineColors.AUDIO_WAVEFORM[:3], TimelineColors.AUDIO_WAVEFORM[3] * alpha)
        color_u32 = imgui.get_color_u32_rgba(rgba[0], rgba[1], rgba[2], rgba[3])

        # Step rendering to limit draw calls when very wide
        if step_px < 1: step_px = 1

        for xi in range(0, canvas_w, step_px):
            y0 = int(min_vals[xi])
            y1 = int(max_vals[xi])
            if y1 < y0:
                continue
            x0_screen = canvas_x0 + xi
            # Filled thin rect for the envelope column
            draw_list.add_rect_filled(x0_screen, y0, x0_screen + step_px, y1, color_u32)

    def _load_plugins(self):
        """Load plugins once at startup."""
        target_funscript, _ = self._get_target_funscript_details()
        if target_funscript and hasattr(target_funscript, 'list_available_plugins'):
            try:
                self._available_plugins = target_funscript.list_available_plugins()
            except Exception as e:
                if hasattr(self.app, 'logger'):
                    self.app.logger.warning(f"Error loading plugins: {e}")
                self._available_plugins = []
        else:
            self._available_plugins = []
    
    def _reload_plugins(self):
        """Manually reload plugins (callable from menu)."""
        self._load_plugins()
    
    def _get_available_plugins(self):
        """Get the loaded plugins list."""
        return self._available_plugins or []

    def _get_plugin_display_name(self, plugin_info):
        """Generate a user-friendly display name for a plugin."""
        name = plugin_info.get('name', 'unknown')
        
        # Create readable names from plugin names
        display_names = {
            'savgol_filter': 'Smooth (SG)',
            'rdp_simplify': 'Simplify (RDP)', 
            'peak_valley': 'Peaks',
            'speed_limiter': 'Speed Limiter',
            'autotune': 'Auto-Tune',
            'amplify': 'Amplify',
            'clamp': 'Clamp',
            'invert': 'Invert',
            'keyframe': 'Keyframes',
            'resample': 'Resample'
        }
        
        return display_names.get(name, name.replace('_', ' ').title())

    def _requires_parameters(self, plugin_info):
        """Check if plugin requires parameters (has non-optional parameters)."""
        schema = plugin_info.get('parameters_schema', {})
        # Generic logic: return True if any parameters are required
        # This ensures consistency with the new plugin renderer system
        return any(param.get('required', False) for param in schema.values())

    def _apply_plugin_directly(self, plugin_name, target_funscript, axis_name, fs_proc, selected_indices=None):
        """Apply a plugin directly with default parameters."""
        try:
            # Get default parameters from schema
            plugin_info = None
            for plugin in self._get_available_plugins():
                if plugin['name'] == plugin_name:
                    plugin_info = plugin
                    break

            if not plugin_info:
                return False

            # Build default parameters
            default_params = {}
            schema = plugin_info.get('parameters_schema', {})
            for param_name, param_info in schema.items():
                if 'default' in param_info and param_info['default'] is not None:
                    # Honor explicit defaults only
                    default_params[param_name] = param_info['default']
                elif param_info.get('required', False):
                    # Can't apply directly if required params have no defaults
                    return False
                # For optional params with no default, DO NOT inject synthetic values;
                # omitting them ensures full-script behavior by default (e.g., no time range).

            # Only pass selected_indices when 2+ points are selected; else omit to apply to full script
            if selected_indices and len(selected_indices) >= 2:
                default_params['selected_indices'] = selected_indices

            # Apply plugin using the appropriate method based on plugin name
            success = self._apply_plugin_by_name(target_funscript, plugin_name, axis_name, default_params)
            return success

        except Exception as e:
            self.app.logger.error(f"Error applying plugin {plugin_name}: {e}")
            return False

    def _apply_plugin_by_name(self, target_funscript, plugin_name, axis_name, params):
        """Apply a plugin by name using the standardized plugin system."""
        try:
            # Use the standardized plugin system instead of hardcoded implementations
            return target_funscript.apply_plugin(plugin_name, axis=axis_name, **params)
            
        except Exception as e:
            self.app.logger.error(f"Error applying plugin '{plugin_name}': {e}")
            return False

    def _render_dynamic_plugin_buttons(
            self,
            window_id_suffix,
            allow_editing_timeline,
            has_actions,
            target_funscript_instance_for_render,
            axis_name_for_render,
            fs_proc,
        ):
        """Render buttons for all available plugins dynamically."""
        if not target_funscript_instance_for_render:
            return
            
        available_plugins = self._get_available_plugins()
        if not available_plugins:
            # Fallback text if no plugins available
            imgui.text("No plugins available")
            return
        
        # Render buttons for each plugin
        for plugin_info in available_plugins:
            plugin_name = plugin_info.get('name', 'unknown')
            display_name = self._get_plugin_display_name(plugin_info)
            
            # Determine if plugin is disabled
            plugin_disabled = not allow_editing_timeline or not has_actions
            
            # Set disabled styling
            if plugin_disabled:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
            
            # Create button
            button_id = f"{display_name}##{plugin_name}{window_id_suffix}"
            if imgui.button(button_id):
                if not plugin_disabled:
                    # Check if plugin requires parameters
                    if self._requires_parameters(plugin_info):
                        # Show popup for parameter input
                        popup_key = f"{plugin_name}_popup"
                        if popup_key not in self.plugin_popups:
                            # Initialize with proper default parameters
                            default_params = {}
                            schema = plugin_info.get('parameters_schema', {})
                            # Precompute type->default mapping for performance
                            type_defaults = {
                                int: lambda constraints: constraints.get('min', 0),
                                float: lambda constraints: constraints.get('min', 0.0),
                                bool: lambda constraints: False,
                                str: lambda constraints: "",
                            }
                            for param_name, param_info in schema.items():
                                default = param_info.get('default')
                                if default is not None:
                                    default_params[param_name] = default
                                    continue
                                param_type = param_info.get('type', str)
                                constraints = param_info.get('constraints', {})
                                default_params[param_name] = type_defaults.get(param_type, lambda constraints: None)(constraints)
                            
                            self.plugin_popups[popup_key] = {
                                'show': False,
                                'apply_to_selection': bool(self.multi_selected_action_indices),
                                'params': default_params
                            }
                        self.plugin_popups[popup_key]['show'] = True
                        self.plugin_popups[popup_key]['apply_to_selection'] = bool(
                            self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2)
                    else:
                        # Apply plugin directly with default parameters
                        selected_indices = list(self.multi_selected_action_indices) if self.multi_selected_action_indices else None
                        
                        suffix = " to selection" if selected_indices else ""
                        op_desc = f"Applied {display_name}{suffix}"
                        
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._apply_plugin_directly(
                            plugin_name,
                            target_funscript_instance_for_render,
                            axis_name_for_render,
                            fs_proc,
                            selected_indices,
                        ):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.logger.info(
                                f"{op_desc} on T{self.timeline_num}.",
                                extra={"status_message": True},
                            )
            
            # Remove disabled styling
            if plugin_disabled:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
            
            # Add same line for horizontal layout
            imgui.same_line()
        
        # Render plugin popups
        self._render_plugin_popups(
            window_id_suffix,
            target_funscript_instance_for_render,
            axis_name_for_render,
            fs_proc,
        )

    def _update_plugin_preview(self, plugin_name, params, apply_to_selection):
        """Update preview for a specific plugin with given parameters."""
        target_funscript, axis_name = self._get_target_funscript_details()
        if not target_funscript:
            self.clear_preview()
            return
        
        try:
            # Prepare parameters
            preview_params = params.copy()
            if apply_to_selection and self.multi_selected_action_indices:
                preview_params['selected_indices'] = list(self.multi_selected_action_indices)

            # Try to get plugin-provided preview metadata (optional)
            preview_data = target_funscript.get_plugin_preview(plugin_name, axis=axis_name, **preview_params)

            # Always attempt to build a visual preview by applying to a temp copy,
            # even if the plugin's get_preview returns an error or minimal data.
            temp_funscript = target_funscript.__class__()

            # Copy current actions
            source_actions = getattr(target_funscript, f"{axis_name}_actions", [])
            temp_actions = copy.deepcopy(source_actions)
            setattr(temp_funscript, f"{axis_name}_actions", temp_actions)

            # Apply the plugin to the temporary copy
            success = temp_funscript.apply_plugin(plugin_name, axis=axis_name, **preview_params)

            if success:
                preview_actions = getattr(temp_funscript, f"{axis_name}_actions", [])
                self.set_preview_actions(preview_actions)
            else:
                # Fall back to clearing preview if application failed
                self.clear_preview()
                
        except Exception as e:
            self.app.logger.warning(f"Error generating preview for {plugin_name}: {e}")
            self.clear_preview()

    def _render_plugin_popups(self, window_id_suffix, target_funscript, axis_name, fs_proc):
        """Render parameter input popups for plugins that require them."""
        for popup_key, popup_state in self.plugin_popups.items():
            if not popup_state['show']:
                continue
                
            plugin_name = popup_key.replace('_popup', '')
            
            # Find plugin info
            plugin_info = None
            for plugin in self._get_available_plugins():
                if plugin['name'] == plugin_name:
                    plugin_info = plugin
                    break
            
            if not plugin_info:
                popup_state['show'] = False
                continue
            
            display_name = self._get_plugin_display_name(plugin_info)
            popup_title = f"{display_name} Settings (Timeline {self.timeline_num})##{plugin_name}Settings{window_id_suffix}"
            
            # Initialize preview state if not exists
            if 'show_preview' not in popup_state:
                popup_state['show_preview'] = True
                popup_state['preview_initialized'] = False
            
            # Initialize preview immediately when popup opens
            if not popup_state['preview_initialized'] and popup_state['show_preview']:
                self._update_plugin_preview(plugin_name, popup_state['params'], popup_state.get('apply_to_selection', False))
                popup_state['preview_initialized'] = True
            
            # Create popup window
            main_viewport = imgui.get_main_viewport()
            popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 450) * 0.5
            popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 350) * 0.5
            
            imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
            imgui.set_next_window_size(450, 350, condition=imgui.APPEARING)
            
            is_open, _ = imgui.begin(popup_title, closable=True)
            if not is_open:
                popup_state['show'] = False
                self.clear_preview()  # Clear preview when popup closes
                imgui.end()
                continue
            
            # Track parameter changes to update preview
            params_changed = False
            
            # Render parameter inputs based on schema
            schema = plugin_info.get('parameters_schema', {})
            for param_name, param_info in schema.items():
                param_type = param_info.get('type', str)
                param_desc = param_info.get('description', param_name)
                default_value = param_info.get('default', 0)
                current_value = popup_state['params'].get(param_name, default_value)
                
                imgui.text(f"{param_desc}:")
                
                old_value = current_value
                try:
                    if param_type == float:
                        constraints = param_info.get('constraints', {})
                        min_val = float(constraints.get('min', 0.0))
                        max_val = float(constraints.get('max', 100.0))
                        # Ensure current_value is float
                        current_value = float(current_value) if current_value is not None else min_val
                        _, new_value = imgui.slider_float(f"##{param_name}", current_value, min_val, max_val)
                        popup_state['params'][param_name] = new_value
                    elif param_type == int:
                        constraints = param_info.get('constraints', {})
                        min_val = int(constraints.get('min', 0))
                        max_val = int(constraints.get('max', 100))
                        # Ensure current_value is int
                        current_value = int(current_value) if current_value is not None else min_val
                        _, new_value = imgui.slider_int(f"##{param_name}", current_value, min_val, max_val)
                        popup_state['params'][param_name] = new_value
                    elif param_type == bool:
                        # Ensure current_value is bool
                        current_value = bool(current_value) if current_value is not None else False
                        _, new_value = imgui.checkbox(f"##{param_name}", current_value)
                        popup_state['params'][param_name] = new_value
                    else:
                        # For string or other types, use text input
                        current_value = str(current_value) if current_value is not None else ""
                        _, new_value = imgui.input_text(f"##{param_name}", current_value, 256)
                        popup_state['params'][param_name] = new_value
                        
                except (ValueError, TypeError) as e:
                    # Fallback for type conversion errors
                    self.app.logger.warning(f"Parameter type error for {param_name}: {e}")
                    default_map = {float: 0.0, int: 0, bool: False, str: ""}
                    new_value = default_map.get(param_type, "")
                    popup_state['params'][param_name] = new_value
                    imgui.text_colored(f"Error: Invalid {param_type.__name__} value", 1.0, 0.4, 0.4, 1.0)
                
                if new_value != old_value:
                    params_changed = True
                    # Force preview update when parameters change
                    if popup_state.get('show_preview', False):
                        self._update_plugin_preview(
                            plugin_name,
                            popup_state["params"],
                            popup_state.get("apply_to_selection", False),
                        )
            imgui.separator()
            
            # Apply to selection checkbox
            old_selection = popup_state['apply_to_selection']
            has_selection = self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2
            
            if has_selection:
                _, popup_state["apply_to_selection"] = imgui.checkbox(
                    f"Apply to {len(self.multi_selected_action_indices)} selected points only",
                    popup_state["apply_to_selection"],
                )
            else:
                imgui.text_disabled("Apply to: Full Timeline (no selection)")
                popup_state['apply_to_selection'] = False
            
            if popup_state['apply_to_selection'] != old_selection:
                params_changed = True
                # Force preview update when selection changes
                if popup_state['show_preview']:
                    self._update_plugin_preview(plugin_name, popup_state['params'], popup_state['apply_to_selection'])
            
            # Preview checkbox
            preview_changed, popup_state['show_preview'] = imgui.checkbox(
                "Show Preview", popup_state['show_preview'])
            if preview_changed:
                params_changed = True
                # Force preview update when preview is toggled
                if popup_state['show_preview']:
                    self._update_plugin_preview(plugin_name, popup_state['params'], popup_state['apply_to_selection'])
                    popup_state['preview_initialized'] = True
            
            # Update preview if parameters changed or first time
            if (params_changed or not popup_state['preview_initialized']) and popup_state['show_preview']:
                self._update_plugin_preview(plugin_name, popup_state['params'], popup_state['apply_to_selection'])
                popup_state['preview_initialized'] = True
            elif not popup_state['show_preview']:
                self.clear_preview()
            imgui.separator()
            
            # Apply and Cancel buttons
            if imgui.button("Apply"):
                selected_indices = (
                    list(self.multi_selected_action_indices)
                    if popup_state["apply_to_selection"] and self.multi_selected_action_indices
                    else None
                )

                params = popup_state['params'].copy()
                if selected_indices:
                    params['selected_indices'] = selected_indices
                
                suffix = " to selection" if selected_indices else ""
                op_desc = f"Applied {display_name}{suffix}"
                
                fs_proc._record_timeline_action(self.timeline_num, op_desc)
                try:
                    success = target_funscript.apply_plugin(plugin_name, axis=axis_name, **params)
                    if success:
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(
                            f"{op_desc} on T{self.timeline_num}.",
                            extra={"status_message": True},
                        )
                        popup_state['show'] = False
                        self.clear_preview()
                    else:
                        self.app.logger.error(f"Failed to apply {display_name}")
                except Exception as e:
                    self.app.logger.error(f"Error applying {display_name}: {e}")
            
            imgui.same_line()
            if imgui.button("Cancel"):
                popup_state['show'] = False
                self.clear_preview()
            imgui.end()
        
        # Clear preview if no popups are showing
        any_popup_showing = any(state['show'] for state in self.plugin_popups.values())
        if not any_popup_showing and self.is_previewing:
            self.clear_preview()

    def _perform_time_shift(self, frame_delta: int):
        fs_proc = self.app.funscript_processor
        video_fps_for_calc = self.app.processor.fps if self.app.processor and self.app.processor.fps and self.app.processor.fps > 0 else 0
        if video_fps_for_calc <= 0:
            self.app.logger.warning(
                f"T{self.timeline_num}: Cannot shift time. Video FPS is not available.",
                extra={"status_message": True},
            )
            return

        time_delta_ms = int(round((frame_delta / video_fps_for_calc) * 1000.0))
        op_desc = f"Shifted All Points by {frame_delta} frames"

        fs_proc._record_timeline_action(self.timeline_num, op_desc)
        if self._call_funscript_method('shift_points_time', 'time shift', time_delta_ms=time_delta_ms):
            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
            self.app.logger.info(
                f"{op_desc} on T{self.timeline_num}.",
                extra={"status_message": True},
            )

    # --- COPY/PASTE HELPER METHODS ---
    def _get_selected_actions_for_copy(self) -> List[Dict]:
        actions_list_ref = self._get_actions_list_ref()
        if not actions_list_ref: return []
        actions_to_copy_refs = []
        if self.multi_selected_action_indices:
            # Ensure indices are valid and sorted before accessing
            valid_indices = sorted(
                [idx for idx in self.multi_selected_action_indices if 0 <= idx < len(actions_list_ref)],
            )
            for idx in valid_indices: actions_to_copy_refs.append(actions_list_ref[idx])
        elif self.selected_action_idx != -1 and 0 <= self.selected_action_idx < len(actions_list_ref):
            actions_to_copy_refs.append(actions_list_ref[self.selected_action_idx])
        if not actions_to_copy_refs: return []
        actions_to_copy_refs.sort(key=lambda x: x['at'])
        earliest_at = actions_to_copy_refs[0]['at']
        return [{"relative_at": action['at'] - earliest_at, "pos": action['pos']} for action in actions_to_copy_refs]

    def _handle_copy_selection(self):
        actions_to_copy = self._get_selected_actions_for_copy()
        self.app.funscript_processor.set_clipboard_actions(actions_to_copy)

    def _handle_paste_actions(self, paste_at_time_ms: int):
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance or not axis_name: return

        clipboard_data = self.app.funscript_processor.get_clipboard_actions()
        if not clipboard_data:
            self.app.logger.info(f"T{self.timeline_num}: Clipboard empty.")
            return

        op_desc = f"Pasted {len(clipboard_data)} Point(s)"
        self.app.funscript_processor._record_timeline_action(self.timeline_num, op_desc)

        actions_to_add = []
        newly_pasted_timestamps = []
        for action_data in clipboard_data:
            new_at = max(0, int(paste_at_time_ms + action_data['relative_at']))
            new_pos = int(action_data['pos'])
            actions_to_add.append({
                'timestamp_ms': new_at,
                'primary_pos': new_pos if axis_name == 'primary' else None,
                'secondary_pos': new_pos if axis_name == 'secondary' else None
            })
            newly_pasted_timestamps.append(new_at)

        if not actions_to_add: return

        # Use the batch add method for efficiency and correctness
        funscript_instance.add_actions_batch(actions_to_add, is_from_live_tracker=False)

        updated_actions_list_ref = self._get_actions_list_ref()
        if not updated_actions_list_ref: return
        self.multi_selected_action_indices.clear()
        new_selected_indices = set()

        # A simple way to find pasted actions is by their timestamp, though this can be ambiguous.
        # A more robust method would be needed if timestamps could easily collide.
        for ts_pasted in newly_pasted_timestamps:
            for i, act in enumerate(updated_actions_list_ref):
                if act['at'] == ts_pasted:
                    new_selected_indices.add(i)

        self.multi_selected_action_indices = new_selected_indices
        self.selected_action_idx = min(self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

        self.app.funscript_processor._finalize_action_and_update_ui(self.timeline_num, op_desc)
        self.app.logger.info(f"Pasted {len(actions_to_add)} point(s).", extra={'status_message': True})

    def _handle_copy_to_other_timeline(self):
        source_actions_ref = self._get_actions_list_ref()
        if not source_actions_ref or not self.multi_selected_action_indices:
            self.app.logger.warning(
                f"T{self.timeline_num}: No points selected to copy.",
                extra={"status_message": True},
            )
            return

        destination_timeline_num = 2 if self.timeline_num == 1 else 1
        dest_funscript_instance, dest_axis_name = self.app.funscript_processor._get_target_funscript_object_and_axis(destination_timeline_num)

        if not dest_funscript_instance or not dest_axis_name:
            self.app.logger.error(f"T{self.timeline_num}: Could not find destination timeline to copy points.", extra={'status_message': True})
            return

        actions_to_copy = [source_actions_ref[idx] for idx in sorted(list(self.multi_selected_action_indices))]
        if not actions_to_copy:
            return

        # Determine the time range of the actions being copied
        min_timestamp = min(action['at'] for action in actions_to_copy)
        max_timestamp = max(action['at'] for action in actions_to_copy)

        actions_to_add = []
        for action in actions_to_copy:
            new_action = {'timestamp_ms': action['at']}
            if dest_axis_name == 'primary':
                new_action['primary_pos'] = action['pos']
                new_action['secondary_pos'] = None
            else:  # secondary
                new_action['primary_pos'] = None
                new_action['secondary_pos'] = action['pos']
            actions_to_add.append(new_action)

        if not actions_to_add:
            return

        op_desc = f"Pasted {len(actions_to_add)} points from T{self.timeline_num} (overwrite)"
        # Record a single undo action for the entire operation on the DESTINATION timeline
        self.app.funscript_processor._record_timeline_action(destination_timeline_num, op_desc)

        # --- Delete existing points in the target range ---
        dest_actions_ref = getattr(dest_funscript_instance, f"{dest_axis_name}_actions", [])
        if dest_actions_ref:
            indices_to_delete = [
                i for i, action in enumerate(dest_actions_ref)
                if min_timestamp <= action['at'] <= max_timestamp
            ]
            if indices_to_delete:
                # This call is now part of the recorded undo action
                dest_funscript_instance.clear_points(axis=dest_axis_name, selected_indices=indices_to_delete)

        # Add the new points
        dest_funscript_instance.add_actions_batch(actions_to_add, is_from_live_tracker=False)

        # Finalize the single undoable action on the DESTINATION timeline
        self.app.funscript_processor._finalize_action_and_update_ui(destination_timeline_num, op_desc)
        self.app.logger.info(
            f"Pasted to T{destination_timeline_num}, overwriting points in range [{min_timestamp}, {max_timestamp}].",
            extra={'status_message': True})

    # --- Selection Filtering Methods ---
    def _select_top_points(self, selection: List[Dict]) -> List[Dict]:
        """
        Returns a list of points that are local maxima (top points) in the selection.
        This version correctly handles endpoints and plateaus.
        A point is a top point if it is greater than its left neighbor and greater than or equal to its right neighbor.
        """
        if len(selection) < 2:
            return selection  # A single point is a peak; empty list is empty

        top_points = []
        for i, point in enumerate(selection):
            pos = point['pos']

            # Get previous and next position. Use -1 as a proxy for -infinity since position is always >= 0.
            prev_pos = selection[i - 1]['pos'] if i > 0 else -1
            next_pos = selection[i + 1]['pos'] if i < len(selection) - 1 else -1

            # To be a peak, it must be strictly greater than what came before (or be the first point)
            # and greater than or equal to what comes after (to select the start of a plateau).
            if pos > prev_pos and pos >= next_pos:
                top_points.append(point)
        return top_points

    def _select_bottom_points(self, selection: List[Dict]) -> List[Dict]:
        """
        Returns a list of points that are local minima (bottom points) in the selection.
        This version correctly handles endpoints and plateaus.
        A point is a bottom point if it is less than its left neighbor and less than or equal to its right neighbor.
        """
        if len(selection) < 2:
            # For a 2-point selection, the lower point is the bottom point.
            if len(selection) == 2:
                return [selection[0]] if selection[0]['pos'] <= selection[1]['pos'] else [selection[1]]
            return selection

        bottom_points = []
        for i, point in enumerate(selection):
            pos = point['pos']

            # Use 101 as a proxy for +infinity since position is always <= 100.
            prev_pos = selection[i - 1]['pos'] if i > 0 else 101
            next_pos = selection[i + 1]['pos'] if i < len(selection) - 1 else 101

            # To be a valley, it must be strictly less than what came before (or be the first point)
            # and less than or equal to what comes after (to select the start of a plateau).
            if pos < prev_pos and pos <= next_pos:
                bottom_points.append(point)
        return bottom_points

    def _select_mid_points(self, selection: List[Dict]) -> List[Dict]:
        """
        Returns a list of points that are neither local maxima nor local minima (mid points).
        """
        if len(selection) < 3:
            return []  # Mid-points are only meaningful in selections of 3 or more.

        # Get top and bottom points using the corrected functions
        top_points = self._select_top_points(selection)
        bottom_points = self._select_bottom_points(selection)

        # Create a set of the unique object IDs of all "extreme" points for efficient lookup
        extreme_ids = {id(p) for p in top_points} | {id(p) for p in bottom_points}

        # A point is a "mid point" if it's not in the set of extreme points.
        mid_points = [p for p in selection if id(p) not in extreme_ids]
        return mid_points

    def _handle_selection_filtering(self, filter_func):
        """Generic handler to apply a selection filter and update the UI state."""
        actions_list_ref = self._get_actions_list_ref()
        if not actions_list_ref or len(self.multi_selected_action_indices) < 3:
            return

        # Get the selected actions, sorted by time, which is crucial for neighbor comparison
        selected_indices_sorted = sorted(list(self.multi_selected_action_indices))
        selection_of_actions = [actions_list_ref[i] for i in selected_indices_sorted]

        # Apply the provided filter function (e.g., _select_top_points)
        filtered_actions = filter_func(selection_of_actions)

        if not filtered_actions:
            self.multi_selected_action_indices.clear()
            self.selected_action_idx = -1
            return

        # Create a map of object IDs to original indices for efficient lookup
        action_id_to_index = {id(action): i for i, action in enumerate(actions_list_ref)}

        # Find the new indices based on the filtered action objects
        new_indices = {action_id_to_index[id(pt)] for pt in filtered_actions if id(pt) in action_id_to_index}

        self.multi_selected_action_indices = new_indices
        self.selected_action_idx = min(new_indices) if new_indices else -1

    # --- Bulk Operations Direct Calls to Funscript Object ---
    def _call_funscript_method(self, method_name: str, error_context: str, **kwargs) -> bool:
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance or not axis_name:
            self.app.logger.warning(
                f"T{self.timeline_num}: Cannot {error_context}. Funscript object not found.",
                extra={"status_message": True},
            )
            return False
        try:
            method_to_call = getattr(funscript_instance, method_name)
            method_to_call(axis=axis_name, **kwargs)
            return True
        except Exception as e:
            self.app.logger.error(
                f"T{self.timeline_num} Error in {error_context} ({method_name}): {str(e)}",
                exc_info=True,
                extra={"status_message": True},
            )
            return False

    def _call_funscript_method_with_result(self, method_name: str, error_context: str, **kwargs) -> Optional[Dict]:
        """ Helper to call a method on the funscript object that is expected to return a result dictionary. """
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance or not axis_name:
            self.app.logger.warning(
                f"T{self.timeline_num}: Cannot {error_context}. Funscript object not found.",
                extra={"status_message": True},
            )
            return None
        try:
            method_to_call = getattr(funscript_instance, method_name)
            result = method_to_call(axis=axis_name, **kwargs)
            return result
        except Exception as e:
            self.app.logger.error(
                f"T{self.timeline_num} Error in {error_context} ({method_name}): {str(e)}",
                exc_info=True,
                extra={"status_message": True},
            )
            return None

    def _perform_autotune_sg(
            self,
            sat_low: int,
            sat_high: int,
            max_window: int,
            polyorder: int,
            selected_indices: Optional[List[int]],
        ):
            """Wrapper to call the auto_tune_sg_filter method on the funscript object."""
            return self._call_funscript_method_with_result(
                "auto_tune_sg_filter",
                "auto-tune",
                saturation_low=sat_low,
                saturation_high=sat_high,
                max_window_size=max_window,
                polyorder=polyorder,
                selected_indices=selected_indices,
            )

    def _perform_ultimate_autotune(self) -> bool:
        """
        Runs the ultimate autotune pipeline and applies the result using the undo system.
        Returns True on success, False on failure.
        """
        try:
            from funscript.plugins.base_plugin import plugin_registry
            ultimate_plugin = plugin_registry.get_plugin("Ultimate Autotune")
            if ultimate_plugin:
                axis_name = "primary" if self.active_axis == 0 else "secondary"
                funscript_instance = self.app.app_logic.app_funscript_processor.get_funscript_obj()
                if funscript_instance:
                    result = ultimate_plugin.transform(funscript_instance, axis_name)
                    if result:
                        self.app.logger.info(
                            f"T{self.timeline_num}: Ultimate Autotune applied successfully",
                            extra={"status_message": True},
                        )
                        return True
                    else:
                        self.app.logger.warning(
                            f"T{self.timeline_num}: Ultimate Autotune failed",
                            extra={"status_message": True},
                        )
                        return False
            else:
                self.app.logger.error("Ultimate Autotune plugin not available")
                return False
        except Exception as e:
            self.app.logger.error(
                f"T{self.timeline_num}: Error applying Ultimate Autotune: {e}",
                extra={"status_message": True},
            )
            return False

    def _perform_sg_filter(self, window_length: int, polyorder: int, selected_indices: Optional[List[int]]):
        return self._call_funscript_method(
            "apply_savitzky_golay",
            "SG filter",
            window_length=window_length,
            polyorder=polyorder,
            selected_indices=selected_indices,
        )

    def _perform_speed_limiter(self, min_interval: int, vibe_amount: int, speed_threshold: float):
        return self._call_funscript_method(
            "apply_speed_limiter",
            "Speed Limiter",
            min_interval=min_interval,
            vibe_amount=vibe_amount,
            speed_threshold=speed_threshold,
        )

    def _perform_rdp_simplification(
            self,
            epsilon: float,
            selected_indices: Optional[List[int]],
        ):
            return self._call_funscript_method(
                "simplify_rdp",
                "RDP simplification",
                epsilon=epsilon,
                selected_indices=selected_indices,
            )

    def _perform_peaks_simplification(
            self,
            height: float,
            threshold: float,
            distance: float,
            prominence: float,
            width: float,
            selected_indices: Optional[List[int]],
        ):
            return self._call_funscript_method(
                "find_peaks_and_valleys",
                "Peak finding",
                height=height,
                threshold=threshold,
                distance=distance,
                prominence=prominence,
                width=width,
                selected_indices=selected_indices,
            )

    def _perform_inversion(self, selected_indices: Optional[List[int]]):
        return self._call_funscript_method(
            "invert_points_values",
            "inversion",
            selected_indices=selected_indices,
        )

    def _perform_clamp(self, clamp_value: int, selected_indices: Optional[List[int]]):
        return self._call_funscript_method(
            "clamp_points_values",
            f"clamp to {clamp_value}",
            clamp_value=clamp_value,
            selected_indices=selected_indices,
        )

    def _perform_amplify(self, scale_factor: float, center_value: int, selected_indices: Optional[List[int]]):
        return self._call_funscript_method(
            "amplify_points_values",
            "Amplify",
            scale_factor=scale_factor,
            center_value=center_value,
            selected_indices=selected_indices,
        )

    def _perform_resample(self, selected_indices: Optional[List[int]]):
        # We can hardcode the resample rate for now, or add a popup later.
        # 50ms is a good starting point.
        resample_rate = 50
        return self._call_funscript_method(
            "apply_peak_preserving_resample",
            "Resample",
            resample_rate_ms=resample_rate,
            selected_indices=selected_indices,
        )

    def _perform_keyframe_simplification(
            self,
            position_tolerance: int,
            time_tolerance_ms: int,
            selected_indices: Optional[List[int]],
        ):
            return self._call_funscript_method(
                "simplify_to_keyframes",
                "Keyframe Extraction",
                position_tolerance=position_tolerance,
                time_tolerance_ms=time_tolerance_ms,
                selected_indices=selected_indices,
            )

    def _get_ultimate_autotune_params(self) -> Dict:
        """Helper to gather all Ultimate Autotune settings from the UI state."""
        return {
            "presmoothing": {
                "enabled": self.ultimate_presmoothing_enabled,
                "max_window_size": self.ultimate_presmoothing_max_window,
            },
            "peaks": {
                "enabled": self.ultimate_peaks_enabled,
                "prominence": self.ultimate_peaks_prominence,
                "distance": 1,
            },
            "recovery": {
                "enabled": self.ultimate_recovery_enabled,
                "threshold_factor": self.ultimate_recovery_threshold,
            },
            "normalization": {"enabled": self.ultimate_normalization_enabled},
            # "regeneration": {
            #     "enabled": self.ultimate_regeneration_enabled,
            #     "resample_rate_ms": self.ultimate_resample_rate,
            # },
            "speed_limiter": {
                "enabled": self.ultimate_speed_limit_enabled,
                "speed_threshold": self.ultimate_speed_threshold,
            },
        }

    def set_preview_actions(self, preview_actions: Optional[List[Dict]]):
        self.preview_actions = preview_actions
        self.is_previewing = preview_actions is not None
        # Force a redraw to show the preview immediately
        if hasattr(self.app, "request_redraw"):
            self.app.request_redraw()

    def clear_preview(self):
        if self.is_previewing:
            self.is_previewing = False
            self.preview_actions = None
            # Force a redraw to hide the preview immediately
            if hasattr(self.app, 'request_redraw'):
                self.app.request_redraw()
        
        # Clear plugin previews
        if hasattr(self, 'plugin_manager'):
            self.plugin_manager.clear_preview()
    
    def _should_clear_all_previews(self) -> bool:
        """
        Generic method to determine if all previews should be cleared.
        Completely plugin-driven - no hardcoded checks.
        """
        if hasattr(self, 'plugin_manager'):
            return self.plugin_manager.should_clear_all_previews()
        return True
    
    def _handle_plugin_apply_request(self, plugin_name: str):
        """Handle a plugin apply request with proper undo system integration."""
        try:
            # Get the funscript and axis details
            funscript_instance, axis_name = self._get_target_funscript_details()
            if not funscript_instance:
                self.app.logger.warning(f"No funscript available to apply plugin {plugin_name}")
                return
            
            # Get the funscript processor for undo management
            fs_proc = self.app.funscript_processor
            if not fs_proc:
                self.app.logger.warning(f"No funscript processor available for plugin {plugin_name}")
                return
            
            # Record state for undo
            op_desc = f"Applied {plugin_name}"
            fs_proc._record_timeline_action(self.timeline_num, op_desc)
            
            # Get selection information for apply to selection
            context = self.plugin_manager.plugin_contexts.get(plugin_name)
            selected_indices = None
            if context and context.apply_to_selection and self.multi_selected_action_indices:
                selected_indices = list(self.multi_selected_action_indices)
            
            # Apply the plugin
            result = self.plugin_manager.apply_plugin(plugin_name, funscript_instance, axis_name, selected_indices)
            
            if result:
                # Finalize the action and update UI
                fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                self.app.logger.info(
                    f"✨ {plugin_name} applied successfully to timeline {self.timeline_num}",
                    extra={"status_message": True, "duration": 3.0},
                )
                
                # Close the plugin window
                self.plugin_manager.set_plugin_state(plugin_name, PluginUIState.CLOSED)
                self.plugin_manager.clear_preview(plugin_name)
            else:
                # On failure, the recorded action is automatically discarded by the undo manager
                self.app.logger.warning(
                    f"Failed to apply {plugin_name}",
                    extra={"status_message": True},
                )

        except Exception as e:
            self.app.logger.error(
                f"Error applying plugin {plugin_name}: {e}",
                extra={"status_message": True},
            )

    def _update_preview(self, filter_type: str):
        """Generates and sets the preview data using the new plugin system."""
        funscript_instance, axis_name = self._get_target_funscript_details()
        if not funscript_instance:
            self.clear_preview()
            return

        # Try new plugin system first
        if self._try_plugin_preview(filter_type, funscript_instance, axis_name):
            return
        
        # Fallback to legacy system for hardcoded filters
        self._legacy_update_preview(filter_type, funscript_instance, axis_name)

    def _try_plugin_preview(self, filter_type: str, funscript_instance, axis_name: str) -> bool:
        """Try to generate preview using the new plugin system."""
        # Map filter types to plugin names
        plugin_mapping = {
            "ultimate": "Ultimate Autotune",
            "autotune": "Ultimate Autotune",  # Fixed: autotune filter maps to same plugin
            # Add other plugins as they're converted
        }

        plugin_name = plugin_mapping.get(filter_type)
        if not plugin_name:
            return False
        
        try:
            success = self.plugin_manager.generate_preview(plugin_name, funscript_instance, axis_name)
            if success:
                preview_actions = self.plugin_manager.get_preview_actions(plugin_name)
                self.set_preview_actions(preview_actions)
                return True
        except Exception as e:
            self.app.logger.error(f"Plugin preview failed for {plugin_name}: {e}")
        
        return False

    def _legacy_update_preview(self, filter_type: str, funscript_instance, axis_name: str):
        """Legacy preview system for hardcoded filters."""
        # Map filter to selection flag attribute name
        selection_flag_attr = {
            "sg": "sg_apply_to_selection",
            "rdp": "rdp_apply_to_selection",
            "amp": "amp_apply_to_selection",
            "keyframe": "keyframe_apply_to_selection",
            "peaks": "peaks_apply_to_selection",
            "autotune": "autotune_apply_to_selection",
            # Intentionally no dedicated flag for "speed_limiter" (defaults to full timeline)
        }

        apply_to_selection = bool(getattr(self, selection_flag_attr.get(filter_type, ""), False))
        indices_to_process = list(self.multi_selected_action_indices) if apply_to_selection else None

        # Parameter builders per filter for clarity and minimal branching
        s = self
        param_builders = {
            "sg": lambda: {"window_length": s.sg_window_length, "polyorder": s.sg_poly_order},
            "rdp": lambda: {"epsilon": s.rdp_epsilon},
            "amp": lambda: {"scale_factor": s.amp_scale_factor, "center_value": s.amp_center_value},
            "keyframe": lambda: {
                "position_tolerance": s.keyframe_position_tolerance,
                "time_tolerance_ms": s.keyframe_time_tolerance,
            },
            "peaks": lambda: {
                "height": s.peaks_height,
                "threshold": s.peaks_threshold,
                "distance": s.peaks_distance,
                "prominence": s.peaks_prominence,
                "width": s.peaks_width,
            },
            "autotune": lambda: {
                "saturation_low": s.autotune_sat_low,
                "saturation_high": s.autotune_sat_high,
                "max_window_size": s.autotune_max_window,
                "polyorder": s.autotune_polyorder,
            },
            "speed_limiter": lambda: {
                "min_interval": s.min_interval,
                "vibe_amount": s.vibe_amount,
                "speed_threshold": s.speed_threshold,
            },
        }

        params_builder = param_builders.get(filter_type)
        params = params_builder() if params_builder else {}

        generated_preview_actions = funscript_instance.calculate_filter_preview(
            axis=axis_name,
            filter_type=filter_type,
            filter_params=params,
            selected_indices=indices_to_process,
        )
        self.set_preview_actions(generated_preview_actions)

    def _handle_copy_full_timeline_to_other(self):
        fs_proc = self.app.funscript_processor
        source_actions_ref = self._get_actions_list_ref()

        # We can copy even if the source is empty, which effectively clears the destination.

        destination_timeline_num = 2 if self.timeline_num == 1 else 1
        dest_funscript_instance, dest_axis_name = fs_proc._get_target_funscript_object_and_axis(
            destination_timeline_num)

        if not dest_funscript_instance or not dest_axis_name:
            self.app.logger.error(
                f"T{self.timeline_num}: Could not find destination timeline to copy to.",
                extra={"status_message": True},
            )
            return

        # Deep copy the actions to avoid reference issues
        actions_to_copy = copy.deepcopy(source_actions_ref) if source_actions_ref else []

        op_desc = f"Replaced T{destination_timeline_num} with T{self.timeline_num}"

        # Record the action on the DESTINATION timeline for undo
        fs_proc._record_timeline_action(destination_timeline_num, op_desc)

        # Directly set the actions list on the destination object
        dest_actions_attr = f"{dest_axis_name}_actions"
        setattr(dest_funscript_instance, dest_actions_attr, actions_to_copy)

        # Invalidate the cache for the destination axis
        dest_funscript_instance._invalidate_cache(dest_axis_name)

        # Finalize the action and update UI state
        fs_proc._finalize_action_and_update_ui(destination_timeline_num, op_desc)
        self.app.logger.info(
            f"Copied all points from T{self.timeline_num} to T{destination_timeline_num}.",
            extra={"status_message": True},
        )

    def _handle_swap_with_other_timeline(self):
        fs_proc = self.app.funscript_processor

        # Get details for both timelines
        other_timeline_num = 2 if self.timeline_num == 1 else 1
        this_fs_obj, this_axis = fs_proc._get_target_funscript_object_and_axis(self.timeline_num)
        other_fs_obj, other_axis = fs_proc._get_target_funscript_object_and_axis(other_timeline_num)

        if not this_fs_obj or not other_fs_obj:
            self.app.logger.error(
                "Could not find both funscript objects to perform swap.",
                extra={"status_message": True},
            )
            return

        # Get the action lists
        this_actions_attr = f"{this_axis}_actions"
        other_actions_attr = f"{other_axis}_actions"
        this_actions = getattr(this_fs_obj, this_actions_attr, [])
        other_actions = getattr(other_fs_obj, other_actions_attr, [])

        # Record undo actions on BOTH timelines.
        op_desc = f"Swap T{self.timeline_num} and T{other_timeline_num}"
        fs_proc._record_timeline_action(self.timeline_num, op_desc)
        fs_proc._record_timeline_action(other_timeline_num, op_desc)

        # Perform the swap using deep copies to be safe
        actions_copy_this = copy.deepcopy(this_actions)
        actions_copy_other = copy.deepcopy(other_actions)

        setattr(this_fs_obj, this_actions_attr, actions_copy_other)
        setattr(other_fs_obj, other_actions_attr, actions_copy_this)

        # Invalidate caches for both
        this_fs_obj._invalidate_cache(this_axis)
        other_fs_obj._invalidate_cache(other_axis)

        # Finalize both actions
        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
        fs_proc._finalize_action_and_update_ui(other_timeline_num, op_desc)

        self.app.logger.info(op_desc, extra={"status_message": True})

    def render(self, timeline_y_start_coord: float = 0, timeline_render_height: float = 0, view_mode: str = 'expert'):
        app_state = self.app.app_state_ui
        is_floating = app_state.ui_layout_mode == "floating"

        visibility_flag_name = f"show_funscript_interactive_timeline{'' if self.timeline_num == 1 else '2'}"
        is_visible = getattr(app_state, visibility_flag_name, False)

        if not is_visible:
            return

        # --- Window Creation (Begin) ---
        should_render_content = True
        if is_floating:
            window_title = f"Interactive Timeline {self.timeline_num}"
            imgui.set_next_window_size(app_state.window_width, 180, condition=imgui.APPEARING)
            is_open, new_visibility = imgui.begin(
                window_title,
                closable=True,
                flags=(
                    imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE
                ),
            )
            if new_visibility != is_visible:
                setattr(app_state, visibility_flag_name, new_visibility)
                self.app.project_manager.project_dirty = True
            if not is_open:
                should_render_content = False
        else:  # Fixed mode
            if timeline_y_start_coord < 0 or timeline_render_height <= 0:
                return  # Cannot render fixed without coordinates
            imgui.set_next_window_position(0, timeline_y_start_coord)
            imgui.set_next_window_size(app_state.window_width, timeline_render_height)
            imgui.begin(
                f"Funscript Editor Timeline##Interactive{self.timeline_num}",
                flags=(
                    imgui.WINDOW_NO_TITLE_BAR
                    | imgui.WINDOW_NO_RESIZE
                    | imgui.WINDOW_NO_MOVE
                    | imgui.WINDOW_NO_SCROLLBAR
                    | imgui.WINDOW_NO_SCROLL_WITH_MOUSE
                ),
            )

        # --- Content Rendering ---
        if should_render_content:
            # Update the persistent preview first, so it's ready for rendering
            self._update_ultimate_autotune_preview()

            mouse_pos = imgui.get_mouse_pos()
            io = imgui.get_io()

            fs_proc = self.app.funscript_processor
            target_funscript_instance_for_render, axis_name_for_render = self._get_target_funscript_details()
            actions_list = []
            if target_funscript_instance_for_render and axis_name_for_render:
                actions_list = getattr(target_funscript_instance_for_render, f"{axis_name_for_render}_actions", [])

            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})

            window_id_suffix = f"Timeline{self.timeline_num}"
            allow_editing_timeline = True
            has_actions = bool(actions_list)
            context_popup_id = f"TimelineActionPopup##{window_id_suffix}"

            script_info_text = f"Timeline {self.timeline_num} - "
            if self.timeline_num == 1:
                loaded_path = self.app.file_manager.loaded_funscript_path
                if loaded_path and os.path.exists(loaded_path):
                    script_info_text += f"{os.path.basename(loaded_path)}"
                elif has_actions:
                    script_info_text += "(Edited/Generated)"
                else:
                    script_info_text += "(Empty - Drag .funscript here or Load/Generate)"
            elif self.timeline_num == 2:
                stats_t2 = fs_proc.funscript_stats_t2
                if stats_t2["path"] != "N/A" and stats_t2["source_type"] == "File":
                    script_info_text += f"{os.path.basename(stats_t2['path'])}"
                elif has_actions:
                    script_info_text += "(Secondary Axis - Edited/Generated)"
                else:
                    script_info_text += "(Secondary Axis - Empty)"

            if view_mode == 'expert':
                # --- Buttons ---
                # Clear Button
                num_selected_for_clear = len(self.multi_selected_action_indices)
                clear_button_text = f"Clear Sel. ({num_selected_for_clear})" if num_selected_for_clear > 0 and has_actions else "Clear All"
                clear_op_disabled = not (
                        allow_editing_timeline and has_actions and target_funscript_instance_for_render and axis_name_for_render)
                if clear_op_disabled:
                    imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                if imgui.button(f"{clear_button_text}##ClearButton{window_id_suffix}"):
                    if not clear_op_disabled:
                        indices_to_clear = list(
                            self.multi_selected_action_indices) if num_selected_for_clear > 0 else None
                        op_desc = f"Cleared {len(indices_to_clear) if indices_to_clear else 'All'} Point(s)"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        target_funscript_instance_for_render.clear_points(axis=axis_name_for_render, selected_indices=indices_to_clear)
                        if indices_to_clear: self.multi_selected_action_indices.clear()
                        self.selected_action_idx = -1
                        self.dragging_action_idx = -1
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})
                if clear_op_disabled: imgui.pop_style_var(); imgui.internal.pop_item_flag()
                imgui.same_line()

                if imgui.button(f"Unload##Unload{window_id_suffix}"):
                    if allow_editing_timeline and has_actions and target_funscript_instance_for_render:
                        op_desc = "Funscript Unloaded via Timeline"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        target_funscript_instance_for_render.clear_points(axis=axis_name_for_render)
                        self.selected_action_idx = -1
                        self.multi_selected_action_indices.clear()
                        self.dragging_action_idx = -1
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        self.app.logger.info(f"T{self.timeline_num} Unloaded.", extra={'status_message': True})
                imgui.same_line()

                # --- Dynamic Plugin Buttons ---
                if self.app.app_state_ui.show_timeline_editor_buttons:
                    plugin_disabled = not allow_editing_timeline or not has_actions
                    
                    if plugin_disabled:
                        imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                        imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                    
                    # Continue on the same line as previous buttons
                    imgui.same_line()
                    
                    # Render plugin buttons using the new system
                    button_clicked = self.plugin_renderer.render_plugin_buttons(
                        self.timeline_num, 
                        getattr(self.app.app_state_ui, 'ui_view_mode', 'expert')
                    )
                    
                    if plugin_disabled:
                        imgui.pop_style_var()
                        imgui.internal.pop_item_flag()

                # --- Time Shift controls ---
                imgui.same_line()
                time_shift_disabled_bool = not allow_editing_timeline or not has_actions or not (
                        self.app.processor and self.app.processor.fps and self.app.processor.fps > 0)
                if time_shift_disabled_bool:
                    imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

                if imgui.button(f"<<##ShiftLeft{window_id_suffix}"):
                    if not time_shift_disabled_bool and self.shift_frames_amount > 0: self._perform_time_shift(
                        -self.shift_frames_amount)
                imgui.same_line()

                imgui.push_item_width(80 * self.app.app_settings.get("global_font_scale", 1.0))
                _, self.shift_frames_amount = imgui.input_int(
                    f"Frames##ShiftAmount{window_id_suffix}",
                    self.shift_frames_amount,
                    1,
                    10,
                )
                if self.shift_frames_amount < 0: self.shift_frames_amount = 0
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button(f">>##ShiftRight{window_id_suffix}"):
                    if not time_shift_disabled_bool and self.shift_frames_amount > 0: self._perform_time_shift(self.shift_frames_amount)
                if time_shift_disabled_bool: imgui.pop_style_var(); imgui.internal.pop_item_flag()
                imgui.same_line()

                # --- Zoom Buttons ---
                video_loaded = self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0
                # Allow scrubbing if not actively processing, or if paused (is_processing True and pause_event is set)
                processor = self.app.processor
                can_manual_pan_zoom = (
                    video_loaded
                    and (
                        not processor.is_processing
                        or (processor.is_processing and hasattr(processor, "pause_event") and processor.pause_event.is_set())
                    )
                ) or not video_loaded
                zoom_disabled_bool = False  # not allow_editing_timeline or not can_manual_pan_zoom
                if zoom_disabled_bool:
                    imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

                imgui.text("Zoom:")
                imgui.same_line()
                if imgui.button(f"+##ZoomIn{window_id_suffix}"):
                    self.zoom_action_request = 0.85  # Set zoom-in request
                imgui.same_line()
                if imgui.button(f"-##ZoomOut{window_id_suffix}"):
                    self.zoom_action_request = 1.15  # Set zoom-out request

                if zoom_disabled_bool:
                    imgui.pop_style_var()
                    imgui.internal.pop_item_flag()
                imgui.same_line()

                # --- Ultimate Autotune Preview Checkbox (hidden in Simple Mode) ---
                view_mode = getattr(self.app.app_state_ui, "ui_view_mode", "expert")
                if view_mode != "simple":  # Only show in Expert mode
                    imgui.same_line()
                    changed, self.show_ultimate_autotune_preview = imgui.checkbox(
                        f"Show Ultimate Preview##UltimatePreviewCheckbox{window_id_suffix}",
                        self.show_ultimate_autotune_preview)
                    if changed:
                        self.app.app_settings.set(
                            f"timeline{self.timeline_num}_show_ultimate_preview",
                            self.show_ultimate_autotune_preview,
                        )
                        if self.show_ultimate_autotune_preview:
                            self.invalidate_ultimate_preview()  # Make it re-render on next frame
                        else:
                            self.ultimate_autotune_preview_actions = None  # Clear immediately
                    imgui.same_line()

                # endregion

            # --- Popup Windows ---
            # region Popups
            # --- Speed Limiter Settings Window ---
            speed_window_title = f"Speed Limiter Settings (Timeline {self.timeline_num})##SpeedLimiterSettingsWindow{window_id_suffix}"
            if self.show_speed_limiter_popup:
                # On first open, generate an initial preview
                if not self.is_previewing:
                    self._update_preview("speed_limiter")

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 400) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 220) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(400, 0, condition=imgui.APPEARING)
                window_expanded, self.show_speed_limiter_popup = imgui.begin(
                    speed_window_title,
                    closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
                )

                if window_expanded:
                    imgui.text(f"Speed Limiter (Timeline {self.timeline_num})")
                    imgui.separator()

                    interval_changed, self.min_interval = imgui.slider_int(
                        "Min Interval (ms)##HandyMinIntervalPopup",
                        self.min_interval,
                        0,
                        200,
                    )
                    if interval_changed: self._update_preview("speed_limiter")
                    if imgui.is_item_hovered(): imgui.set_tooltip("Action points closer than this (in ms) are removed.")

                    # vibe_changed, self.vibe_amount = imgui.slider_int("Vibration Amplitude##HandyVibeAmountPopup",
                    #                                                        self.vibe_amount, 0, 100)
                    # if vibe_changed: self._update_preview('speed_limiter')
                    # if imgui.is_item_hovered(): imgui.set_tooltip("Vibration starting amplitude (1-100).")

                    speed_changed, self.speed_threshold = imgui.slider_float(
                        "Speed Threshold##HandySpeedThresholdPopup",
                        self.speed_threshold,
                        100.0,
                        1000.0,
                        "%.1f",
                    )
                    if speed_changed: self._update_preview("speed_limiter")
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Speed limit for the script. 500 is the default 'Handy' limit.")

                    imgui.separator()

                    if imgui.button(f"Apply##HandyBTApplyPop{window_id_suffix}"):
                        op_desc = f"Applied Speed Limiter (Interval:{self.min_interval}, Vibe:{self.vibe_amount}, Speed:{self.speed_threshold:.1f})"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        if self._perform_speed_limiter(self.min_interval, self.vibe_amount, self.speed_threshold):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_min_interval", self.min_interval)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_vibe_amount", self.vibe_amount)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_speed_threshold", self.speed_threshold)
                            self.app.logger.info(
                                f"{op_desc} on T{self.timeline_num}.",
                                extra={"status_message": True},
                            )

                        self.clear_preview()
                        self.show_speed_limiter_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##HandyBTCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_speed_limiter_popup = False

                if not self.show_speed_limiter_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- SG Settings Window ---
            sg_window_title = f"Savitzky-Golay Filter Settings (Timeline {self.timeline_num})##SGSettingsWindow{window_id_suffix}"
            if self.show_sg_settings_popup:
                # On first open, generate an initial preview
                if not self.is_previewing:
                    self._update_preview("sg")

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 200) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)
                window_expanded, self.show_sg_settings_popup = imgui.begin(
                    sg_window_title,
                    closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
                )  # Use True for closable to handle 'X'

                if window_expanded:
                    imgui.text(f"Savitzky-Golay Filter (Timeline {self.timeline_num})")
                    imgui.separator()

                    # --- Window Length Slider ---
                    wl_changed, current_wl = imgui.slider_int("Window Length##SGWinPopup", self.sg_window_length, 3, 99)
                    if wl_changed:
                        self.sg_window_length = current_wl if current_wl % 2 != 0 else current_wl + 1
                        if self.sg_window_length < 3:
                            self.sg_window_length = 3
                        self._update_preview("sg")  # CORRECT: Only update preview

                    # --- Polyorder Slider ---
                    max_po = max(1, self.sg_window_length - 1)
                    po_val = min(self.sg_poly_order, max_po)
                    if po_val < 1:
                        po_val = 1
                    po_changed, current_po = imgui.slider_int("Polyorder##SGPolyPopup", po_val, 1, max_po)
                    if po_changed:
                        self.sg_poly_order = current_po
                        self._update_preview("sg")  # CORRECT: Only update preview

                    # --- Apply to Selection Checkbox ---
                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        sel_changed, self.sg_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##SGApplyToSel",
                            self.sg_apply_to_selection,
                        )
                        if sel_changed:
                            self._update_preview("sg")  # CORRECT: Only update preview
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.sg_apply_to_selection = False

                    # --- Buttons ---
                    if imgui.button(f"Apply##SGApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.sg_apply_to_selection else None
                        op_desc = f"Applied SG (W:{self.sg_window_length}, P:{self.sg_poly_order})" + (" to selection" if indices_to_use else "")

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)  # Record for Undo

                        # This calls the original, DESTRUCTIVE method
                        if self._perform_sg_filter(
                            self.sg_window_length,
                            self.sg_poly_order,
                            selected_indices=indices_to_use,
                        ):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_sg_default_window", self.sg_window_length)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_sg_default_polyorder", self.sg_poly_order)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_sg_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##SGCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_sg_settings_popup = False

                # This handles closing the popup with the 'X' button
                if not self.show_sg_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- Auto-Tune SG Settings Window ---
            autotune_window_title = f"Auto-Tune SG Filter Settings (Timeline {self.timeline_num})##AutoTuneSettingsWindow{window_id_suffix}"
            if self.show_autotune_popup:
                # On first open, generate an initial preview
                if not self.is_previewing:
                    self._update_preview('autotune')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 400) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 250) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(400, 0, condition=imgui.APPEARING)
                window_expanded, self.show_autotune_popup = imgui.begin(autotune_window_title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text(f"Auto-Tune SG Filter (Timeline {self.timeline_num})")
                    imgui.text_wrapped("Finds the smallest SG window that removes harsh edges (0/100 clipping).")
                    imgui.separator()

                    # MODIFIED: Sliders now update the preview
                    sl_changed, self.autotune_sat_low = imgui.slider_int("Saturation Low##AutoTuneSatLow", self.autotune_sat_low, 0, 10)
                    if sl_changed:
                        self._update_preview("autotune")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Points at or below this value are 'saturated'.")

                    sh_changed, self.autotune_sat_high = imgui.slider_int("Saturation High##AutoTuneSatHigh", self.autotune_sat_high, 90, 100)
                    if sh_changed:
                        self._update_preview("autotune")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Points at or above this value are 'saturated'.")

                    mw_changed, self.autotune_max_window = imgui.slider_int("Max Window Size##AutoTuneMaxWin", self.autotune_max_window, 5, 25)
                    if mw_changed:
                        if self.autotune_max_window % 2 == 0:
                            self.autotune_max_window += 1
                        self._update_preview("autotune")
                    if imgui.is_item_hovered(): imgui.set_tooltip("The largest window size to try during the search.")

                    imgui.text_disabled(f"Polynomial Order: {self.autotune_polyorder} (fixed)")
                    imgui.separator()

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3:
                        sel_changed, self.autotune_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##AutoTuneApplyToSel",
                            self.autotune_apply_to_selection)
                        if sel_changed: self._update_preview('autotune')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.autotune_apply_to_selection = False
                    imgui.separator()

                    if imgui.button(f"Apply##AutoTuneApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.autotune_apply_to_selection else None
                        op_desc = f"Applied Auto-Tune SG" + (" to selection" if indices_to_use else "")
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        result = self._perform_autotune_sg(
                            self.autotune_sat_low, self.autotune_sat_high,
                            self.autotune_max_window, self.autotune_polyorder,
                            selected_indices=indices_to_use)

                        if result:
                            # Success! Finalize the undo action with a more descriptive message.
                            final_op_desc = f"Auto-Tune SG (W:{result['window_length']}, P:{result['polyorder']})" + (
                                " to sel." if indices_to_use else "")
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, final_op_desc)

                            # Save settings for next time
                            self.app.app_settings.set(f"timeline{self.timeline_num}_autotune_default_sat_low", self.autotune_sat_low)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_autotune_default_sat_high", self.autotune_sat_high)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_autotune_default_max_window", self.autotune_max_window)
                            self.app.logger.info(f"{final_op_desc} on T{self.timeline_num}.", extra={'status_message': True})
                        else:
                            # Failure or no solution found. The recorded action will be discarded by the undo manager.
                            self.app.logger.warning(
                                f"Auto-Tune SG on T{self.timeline_num} did not find a solution or failed.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_autotune_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##AutoTuneCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_autotune_popup = False

                # Handle closing with 'X'
                if not self.show_autotune_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- Ultimate Autotune Settings Window ---
            ultimate_window_title = f"Ultimate Autotune (Timeline {self.timeline_num})##UltimateSettingsWindow{window_id_suffix}"
            if self.show_ultimate_autotune_popup:
                if not self.is_previewing:
                    self._update_preview('ultimate')

                imgui.set_next_window_size(480, 0, condition=imgui.APPEARING)
                window_expanded, self.show_ultimate_autotune_popup = imgui.begin(
                    ultimate_window_title, closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

                if window_expanded:
                    imgui.text_wrapped("The Ultimate Autotune will process the script to enhance and clean it. The result is previewed on the timeline with an orange line.")
                    imgui.separator()
                    imgui.text_wrapped("Do you want to replace the current script with this previewed version?")
                    imgui.dummy(0, 10)

                    button_width = (imgui.get_content_region_available_width() - imgui.get_style().item_spacing[0]) / 2.0

                    if imgui.button("Apply##UltimateApply", width=button_width):
                        op_desc = "Applied Ultimate Autotune"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_ultimate_autotune():
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.logger.info("✨ Ultimate Autotune applied successfully.", extra={'status_message': True})
                        # On failure, the recorded action is automatically discarded by the undo manager.
                        self.clear_preview()
                        self.show_ultimate_autotune_popup = False

                    imgui.same_line()
                    if imgui.button("Cancel##UltimateCancel", width=button_width):
                        self.clear_preview()
                        self.show_ultimate_autotune_popup = False

                if not self.show_ultimate_autotune_popup:
                    self.clear_preview()
                if window_expanded:
                    imgui.end()

            # --- New Plugin System Windows ---
            plugin_windows_open = self.plugin_renderer.render_plugin_windows(
                self.timeline_num, window_id_suffix
            )
            
            # --- Handle Plugin Apply Requests ---
            apply_requests = self.plugin_manager.check_and_handle_apply_requests()
            for plugin_name in apply_requests:
                self._handle_plugin_apply_request(plugin_name)

            # --- RDP Settings Window ---
            rdp_window_title = f"RDP Simplification Settings (Timeline {self.timeline_num})##RDPSettingsWindow{window_id_suffix}"
            if self.show_rdp_settings_popup:
                # --- RDP Settings Window ---
                if self.show_rdp_settings_popup:
                    if not self.is_previewing:
                        self._update_preview('rdp')

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 180) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)
                window_expanded, self.show_rdp_settings_popup = imgui.begin(
                    rdp_window_title,
                    closable=self.show_rdp_settings_popup,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
                )

                if window_expanded:
                    imgui.text(f"RDP Simplification (Timeline {self.timeline_num})")
                    imgui.separator()
                    epsilon_changed, self.rdp_epsilon = imgui.slider_float("Epsilon##RDPEpsPopup", self.rdp_epsilon, 0.1, 20.0, "%.1f")
                    if epsilon_changed:
                        self._update_preview("rdp")

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        sel_changed, self.rdp_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##RDPApplyToSel",
                            self.rdp_apply_to_selection,
                        )
                        if sel_changed:
                            self._update_preview("rdp")
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.rdp_apply_to_selection = False
                    if imgui.button(f"Apply##RDPApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.rdp_apply_to_selection else None
                        op_desc = (
                            f"Applied RDP (Epsilon:{self.rdp_epsilon:.1f})"
                            + (" to selection" if indices_to_use else "")
                        )
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_rdp_simplification(self.rdp_epsilon, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(
                                f"timeline{self.timeline_num}_rdp_default_epsilon",
                                self.rdp_epsilon,
                            )
                            self.app.logger.info(
                                f"{op_desc} on T{self.timeline_num}.",
                                extra={"status_message": True},
                            )
                        self.clear_preview()
                        self.show_rdp_settings_popup = False
                    imgui.same_line()
                    if imgui.button(f"Cancel##RDPCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_rdp_settings_popup = False
                imgui.end()

            # --- Peaks Settings Window ---
            peaks_window_title = f"Peak Extraction Settings (Timeline {self.timeline_num})##PeaksSettingsWindow{window_id_suffix}"
            if self.show_peaks_settings_popup:
                if not self.is_previewing:
                    self._update_preview("peaks")

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 450) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 300) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(450, 0, condition=imgui.APPEARING)

                window_expanded, self.show_peaks_settings_popup = imgui.begin(
                    peaks_window_title,
                    closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
                )

                if window_expanded:
                    imgui.text("Peak & Valley Extraction (scipy.find_peaks)")
                    imgui.text_wrapped("Filters to only include peaks and valleys. Parameters with 0 are disabled.")
                    imgui.separator()

                    h_changed, self.peaks_height = imgui.slider_int("Height##PeaksHeight", self.peaks_height, 0, 100)
                    if h_changed:
                        self._update_preview("peaks")
                    if imgui.is_item_hovered(): imgui.set_tooltip("Required height of peaks.")

                    t_changed, self.peaks_threshold = imgui.slider_int("Threshold##PeaksThreshold", self.peaks_threshold, 0, 2,)
                    if t_changed:
                        self._update_preview("peaks")
                    if imgui.is_item_hovered(): imgui.set_tooltip("Vertical distance between a peak and its neighbors.")

                    d_changed, self.peaks_distance = imgui.slider_int("Distance##PeaksDistance", self.peaks_distance, 1, 50)
                    if d_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Minimum horizontal distance (in # of points) between peaks.")

                    p_changed, self.peaks_prominence = imgui.slider_int("Prominence##PeaksProminence", self.peaks_prominence, 0, 50)
                    if p_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "The vertical distance between the peak and its lowest contour line.")

                    w_changed, self.peaks_width = imgui.slider_int("Width##PeaksWidth", self.peaks_width, 0, 5)
                    if w_changed:
                        self._update_preview('peaks')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Required width of peaks in samples.")
                    imgui.separator()

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3:
                        sel_changed, self.peaks_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##PeaksApplyToSel",
                            self.peaks_apply_to_selection)
                        if sel_changed: self._update_preview('peaks')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.peaks_apply_to_selection = False

                    if imgui.button(f"Apply##PeaksApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.peaks_apply_to_selection else None
                        op_desc = f"Applied Peaks Filter" + (" to selection" if indices_to_use else "")

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_peaks_simplification(
                                height=self.peaks_height,
                                threshold=self.peaks_threshold,
                                distance=self.peaks_distance,
                                prominence=self.peaks_prominence,
                                width=self.peaks_width,
                                selected_indices=indices_to_use
                        ):
                                fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                                settings = self.app.app_settings
                                tnum = self.timeline_num
                                peaks_defaults = {
                                    "height": self.peaks_height,
                                    "threshold": self.peaks_threshold,
                                    "distance": self.peaks_distance,
                                    "prominence": self.peaks_prominence,
                                    "width": self.peaks_width,
                                }
                                for key, value in peaks_defaults.items():
                                    settings.set(f"timeline{tnum}_peaks_default_{key}", value)
                                self.app.logger.info(f"{op_desc} on T{tnum}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_peaks_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##PeaksCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_peaks_settings_popup = False

                if not self.show_peaks_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- Amplify Settings Window ---
            amp_window_title = f"Amplify Settings (Timeline {self.timeline_num})##AmpSettingsWindow{window_id_suffix}"
            if self.show_amp_settings_popup:
                if not self.is_previewing:
                    self._update_preview("amp")

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 200) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)

                window_expanded, self.show_amp_settings_popup = imgui.begin(
                    amp_window_title,
                    closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
                )

                if window_expanded:
                    imgui.text(f"Amplify Values (Timeline {self.timeline_num})")
                    imgui.separator()

                    sf_changed, self.amp_scale_factor = imgui.slider_float("Scale Factor##AmpScalePopup", self.amp_scale_factor, 0.1, 5.0, "%.2f")
                    if sf_changed:
                        self._update_preview("amp")

                    cv_changed, self.amp_center_value = imgui.slider_int("Center Value##AmpCenterPopup", self.amp_center_value, 0, 100)
                    if cv_changed:
                        self._update_preview('amp')

                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 2:
                        sel_changed, self.amp_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##AmpApplyToSel",
                            self.amp_apply_to_selection)
                        if sel_changed:
                            self._update_preview('amp')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.amp_apply_to_selection = False

                    if imgui.button(f"Apply##AmpApplyPop{window_id_suffix}"):
                        indices_to_use = list(
                            self.multi_selected_action_indices) if self.amp_apply_to_selection else None
                        op_desc = (f"Amplified (S:{self.amp_scale_factor:.2f}, C:{self.amp_center_value})" + (
                            " to selection" if indices_to_use else ""))

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_amplify(self.amp_scale_factor, self.amp_center_value, selected_indices=indices_to_use):
                            fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_amp_default_scale", self.amp_scale_factor)
                            self.app.app_settings.set(f"timeline{self.timeline_num}_amp_default_center", self.amp_center_value)
                            self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_amp_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##AmpCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_amp_settings_popup = False

                if not self.show_amp_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # --- Keyframe Extraction Settings Window ---
            keyframe_window_title = f"Keyframe Extraction Settings (Timeline {self.timeline_num})##KeyframeSettingsWindow{window_id_suffix}"
            if self.show_keyframe_settings_popup:
                if not self.is_previewing:
                    self._update_preview("keyframe")

                main_viewport = imgui.get_main_viewport()
                popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 350) * 0.5
                popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 220) * 0.5
                imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
                imgui.set_next_window_size(350, 0, condition=imgui.APPEARING)

                window_expanded, self.show_keyframe_settings_popup = imgui.begin(
                    keyframe_window_title,
                    closable=True,
                    flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
                )

                if window_expanded:
                    imgui.text(f"Dominant Keyframe Extraction")
                    imgui.separator()

                    imgui.text_wrapped("Filters out minor peaks/valleys based on position and time.")

                    # Slider 1: Position Tolerance
                    pos_tol_changed, self.keyframe_position_tolerance = imgui.slider_int("Position Tolerance##KeyframePosTol", self.keyframe_position_tolerance, 0, 15)
                    if pos_tol_changed: self._update_preview('keyframe')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Minimum change in position to create a new keyframe.")

                    # Slider 2: Time Tolerance
                    time_tol_changed, self.keyframe_time_tolerance = imgui.slider_int(
                        "Time Tolerance (ms)##KeyframeTimeTol", self.keyframe_time_tolerance, 0, 100)
                    if time_tol_changed: self._update_preview('keyframe')
                    if imgui.is_item_hovered(): imgui.set_tooltip("Minimum time between two keyframes.")

                    # ... (The rest of the popup: selection checkbox and Apply/Cancel buttons) ...
                    if self.multi_selected_action_indices and len(self.multi_selected_action_indices) >= 3:
                        sel_changed, self.keyframe_apply_to_selection = imgui.checkbox(
                            f"Apply to {len(self.multi_selected_action_indices)} selected##KeyframeApplyToSel",
                            self.keyframe_apply_to_selection)
                        if sel_changed:
                            self._update_preview('keyframe')
                    else:
                        imgui.text_disabled("Apply to: Full Script")
                        self.keyframe_apply_to_selection = False

                    if imgui.button(f"Apply##KeyframeApplyPop{window_id_suffix}"):
                        indices_to_use = list(self.multi_selected_action_indices) if self.keyframe_apply_to_selection else None
                        op_desc = (f"Simplified to Keyframes (P-Tol:{self.keyframe_position_tolerance}, T-Tol:{self.keyframe_time_tolerance}")

                        fs_proc._record_timeline_action(self.timeline_num, op_desc)
                        if self._perform_keyframe_simplification(
                            self.keyframe_position_tolerance,
                            self.keyframe_time_tolerance,
                            selected_indices=indices_to_use):
                                fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                                self.app.app_settings.set(f"timeline{self.timeline_num}_keyframe_default_pos_tol",
                                self.keyframe_position_tolerance)
                                self.app.app_settings.set(f"timeline{self.timeline_num}_keyframe_default_time_tol", self.keyframe_time_tolerance)
                                self.app.logger.info(f"{op_desc} on T{self.timeline_num}.", extra={'status_message': True})

                        self.clear_preview()
                        self.show_keyframe_settings_popup = False

                    imgui.same_line()
                    if imgui.button(f"Cancel##KeyframeCancelPop{window_id_suffix}"):
                        self.clear_preview()
                        self.show_keyframe_settings_popup = False

                if not self.show_keyframe_settings_popup:
                    self.clear_preview()

                if window_expanded:
                    imgui.end()

            # Generic preview clearing - no hardcoded popup checks
            if self._should_clear_all_previews():
                self.clear_preview()

            imgui.text_colored(script_info_text, 0.75, 0.75, 0.75, 0.95)  # TODO: change to theme color
            # --- (Drag and drop target for T2 remains the same) ---
            if self.timeline_num == 2:
                if imgui.begin_drag_drop_target():
                    payload = imgui.accept_drag_drop_payload("FILES")
                    if payload is not None and self.app.file_manager.last_dropped_files:
                        if self.app.file_manager.last_dropped_files[0].lower().endswith(".funscript"):
                            self.app.file_manager.load_funscript_to_timeline(
                                self.app.file_manager.last_dropped_files[0], timeline_num=2)
                        self.app.file_manager.last_dropped_files = None
                    imgui.end_drag_drop_target()

            # --- (Canvas setup, grid, points, lines drawing remains the same) ---
            draw_list = imgui.get_window_draw_list()
            canvas_abs_pos = imgui.get_cursor_screen_pos()
            canvas_size = imgui.get_content_region_available()

            center_x_marker = canvas_abs_pos[0] + canvas_size[0] / 2.0

            if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                imgui.end()
                return

            draw_list.add_rect_filled(canvas_abs_pos[0], canvas_abs_pos[1], canvas_abs_pos[0] + canvas_size[0], canvas_abs_pos[1] + canvas_size[1], imgui.get_color_u32_rgba(*TimelineColors.CANVAS_BACKGROUND))

            video_loaded = self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0

            processor = self.app.processor
            can_manual_pan_zoom = (
                video_loaded and (
                    not processor.is_processing or (
                        processor.is_processing and
                        hasattr(processor, "pause_event") and
                        processor.pause_event.is_set()
                    )
                )
            ) or not video_loaded

            video_fps_for_calc = self.app.processor.fps if video_loaded and self.app.processor.fps > 0 else 30.0
            effective_total_duration_s, _, _ = self.app.get_effective_video_duration_params()
            effective_total_duration_ms = effective_total_duration_s * 1000.0

            # --- Coordinate Transformation Helpers (Unchanged) ---
            def time_to_x(time_ms: float) -> float:
                if app_state.timeline_zoom_factor_ms_per_px == 0: return canvas_abs_pos[0]
                return canvas_abs_pos[0] + (time_ms - app_state.timeline_pan_offset_ms) / app_state.timeline_zoom_factor_ms_per_px

            def x_to_time(x_pos: float) -> float:
                return (x_pos - canvas_abs_pos[0]) * app_state.timeline_zoom_factor_ms_per_px + app_state.timeline_pan_offset_ms

            def ms_to_frame(time_ms: float) -> int:
                return int(round((time_ms / 1000.0) * video_fps_for_calc))

            def frame_to_ms(frame_index: int) -> float:
                return (frame_index / video_fps_for_calc) * 1000.0 if video_fps_for_calc > 0 else 0.0

            def apply_zoom_with_center(scale_factor: float) -> None:
                # Zoom around the timeline center marker to keep the visible dot fixed
                anchor_time_ms_local = x_to_time(center_x_marker)
                # Clamp zoom factor to sensible bounds
                min_ms_per_px, max_ms_per_px = 0.01, 2000.0
                app_state.timeline_zoom_factor_ms_per_px = max(
                    min_ms_per_px,
                    min(app_state.timeline_zoom_factor_ms_per_px * scale_factor, max_ms_per_px),
                )
                app_state.timeline_pan_offset_ms = anchor_time_ms_local - (
                    canvas_size[0] / 2.0
                ) * app_state.timeline_zoom_factor_ms_per_px

            def pos_to_y(val: int) -> float:
                if canvas_size[1] == 0:
                    return canvas_abs_pos[1] + canvas_size[1] / 2.0
                return canvas_abs_pos[1] + canvas_size[1] * (1.0 - (val / 100.0))

            def y_to_pos(y_pos: float) -> int:
                if canvas_size[1] == 0: return 50
                val = (1.0 - (y_pos - canvas_abs_pos[1]) / canvas_size[1]) * 100.0
                return min(100, max(0, int(round(val))))

            def time_to_x_vec(time_ms_arr: np.ndarray) -> np.ndarray:
                if app_state.timeline_zoom_factor_ms_per_px == 0:
                    return np.full_like(time_ms_arr, canvas_abs_pos[0], dtype=float)
                return canvas_abs_pos[0] + (time_ms_arr - app_state.timeline_pan_offset_ms) / app_state.timeline_zoom_factor_ms_per_px

            def pos_to_y_vec(val_arr: np.ndarray) -> np.ndarray:
                if canvas_size[1] == 0: return np.full_like(val_arr, canvas_abs_pos[1] + canvas_size[1] / 2.0, dtype=float)
                return canvas_abs_pos[1] + canvas_size[1] * (1.0 - (val_arr / 100.0))

            # --- Deferred Zoom Logic Execution ---
            if self.zoom_action_request is not None:
                # Deferred zoom request (e.g., buttons/shortcuts)
                apply_zoom_with_center(self.zoom_action_request)
                app_state.timeline_interaction_active = True
                self.is_zooming_active = True

                self.zoom_action_request = None 

            # Pan/Zoom boundaries
            center_marker_offset_ms = (canvas_size[0] / 2.0) * app_state.timeline_zoom_factor_ms_per_px
            min_pan_allowed = -center_marker_offset_ms
            max_pan_allowed = effective_total_duration_ms - center_marker_offset_ms
            if max_pan_allowed < min_pan_allowed: max_pan_allowed = min_pan_allowed

            # --- Unified Input and State Machine ---
            was_interacting = app_state.timeline_interaction_active
            is_interacting_this_frame = False
            # Reset pan and zoom flags at the start of each frame
            was_panning_active = self.is_panning_active
            was_zooming_active = self.is_zooming_active
            self.is_panning_active = False
            self.is_zooming_active = False

            # Hover checks for timeline interaction and marquee selection
            strict_bounds = (
                canvas_abs_pos[0],
                canvas_abs_pos[0] + canvas_size[0],
                canvas_abs_pos[1],
                canvas_abs_pos[1] + canvas_size[1],
            )
            is_timeline_hovered = (
                imgui.is_window_hovered()
                and strict_bounds[0] <= mouse_pos[0] < strict_bounds[1]
                and strict_bounds[2] <= mouse_pos[1] < strict_bounds[3]
            )

            marquee_padding = 5.0 * self.app.app_settings.get("global_font_scale", 1.0)
            relaxed_bounds = (
                strict_bounds[0] - marquee_padding,
                strict_bounds[1] + marquee_padding,
                strict_bounds[2] - marquee_padding,
                strict_bounds[3] + marquee_padding,
            )
            is_timeline_hovered_for_marquee_start = (
                imgui.is_window_hovered()
                and relaxed_bounds[0] <= mouse_pos[0] < relaxed_bounds[1]
                and relaxed_bounds[2] <= mouse_pos[1] < relaxed_bounds[3]
            )

            # Check all inputs that count as interaction
            if is_timeline_hovered:
                # Mouse Pan (still restricted)
                if can_manual_pan_zoom:
                    is_mouse_panning = (
                        imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_MIDDLE) or (
                            io.key_shift and
                            imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT) and
                            self.dragging_action_idx == -1 and not self.is_marqueeing
                        ))

                    if is_mouse_panning:
                        app_state.timeline_pan_offset_ms -= io.mouse_delta[0] * app_state.timeline_zoom_factor_ms_per_px
                        is_interacting_this_frame = True
                        self.is_panning_active = True

                # Mouse Wheel Zoom (always allowed)
                if io.mouse_wheel != 0.0:
                    # Mouse wheel zoom around center; avoid timeline drift
                    scale_factor = 0.85 if io.mouse_wheel > 0 else 1.15
                    apply_zoom_with_center(scale_factor)
                    is_interacting_this_frame = True
                    self.is_zooming_active = True

            # Also count point dragging and marqueeing as interaction
            if self.dragging_action_idx != -1 and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT):
                is_interacting_this_frame = True
            if self.is_marqueeing and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT):
                is_interacting_this_frame = True

            # --- Keyboard Shortcuts ---
            if imgui.is_window_focused(imgui.FOCUS_ROOT_AND_CHILD_WINDOWS) and allow_editing_timeline:
                # Keyboard Panning
                pan_multiplier = self.app.app_settings.get("timeline_pan_speed_multiplier", 5)
                pan_key_speed_ms = pan_multiplier * app_state.timeline_zoom_factor_ms_per_px
                pan_left_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("pan_timeline_left", "ALT+LEFT_ARROW"))

                if pan_left_tuple and (
                    pan_left_tuple[1]["alt"] == io.key_alt
                    and pan_left_tuple[1]["ctrl"] == io.key_ctrl
                    and pan_left_tuple[1]["shift"] == io.key_shift
                    and pan_left_tuple[1]["super"] == io.key_super
                ) and imgui.is_key_down(pan_left_tuple[0]):
                    app_state.timeline_pan_offset_ms -= pan_key_speed_ms
                    is_interacting_this_frame = True
                    self.is_panning_active = True

                pan_right_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("pan_timeline_right", "ALT+RIGHT_ARROW"))

                if pan_right_tuple and (
                    pan_right_tuple[1]["alt"] == io.key_alt
                    and pan_right_tuple[1]["ctrl"] == io.key_ctrl
                    and pan_right_tuple[1]["shift"] == io.key_shift
                    and pan_right_tuple[1]["super"] == io.key_super
                ) and imgui.is_key_down(pan_right_tuple[0]):
                    app_state.timeline_pan_offset_ms += pan_key_speed_ms
                    is_interacting_this_frame = True
                    self.is_panning_active = True

                # Select All
                select_all_tuple = self.app._map_shortcut_to_glfw_key(shortcuts.get("select_all_points", "CTRL+A"))
                if select_all_tuple and actions_list and (
                    imgui.is_key_pressed(select_all_tuple[0]) and
                    all(m == io_m for m, io_m in
                        zip(select_all_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
                ):
                    self.multi_selected_action_indices = set(range(len(actions_list)))
                    self.selected_action_idx = min(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

                # Nudge Position
                pos_nudge_delta = 0
                nudge_up_tuple = self.app._map_shortcut_to_glfw_key(shortcuts.get("nudge_selection_pos_up", "UP_ARROW"))
                if nudge_up_tuple and (
                    imgui.is_key_pressed(nudge_up_tuple[0]) and
                    all(m == io_m for m, io_m in
                        zip(nudge_up_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
                ):
                    pos_nudge_delta = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                nudge_down_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("nudge_selection_pos_down", "DOWN_ARROW"))
                if pos_nudge_delta == 0 and nudge_down_tuple and (
                    imgui.is_key_pressed(nudge_down_tuple[0]) and
                    all(m == io_m for m, io_m in
                        zip(nudge_down_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
                ):
                    pos_nudge_delta = -(app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1)

                if pos_nudge_delta != 0 and self.multi_selected_action_indices:
                    op_desc = f"Nudged Point(s) Pos by {pos_nudge_delta}"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    for idx in self.multi_selected_action_indices:
                        if 0 <= idx < len(actions_list): actions_list[idx]["pos"] = min(100, max(0, actions_list[idx][
                            "pos"] + pos_nudge_delta))
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                # Determine paused state
                is_paused = False
                if video_loaded and self.app.processor and self.app.processor.is_processing:
                    is_paused = hasattr(self.app.processor, "pause_event") and self.app.processor.pause_event.is_set()

                # Nudge Time
                time_nudge_delta_ms = 0
                snap_grid_time_ms_for_nudge = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else (
                    int(1000 / video_fps_for_calc) if video_fps_for_calc > 0 else 20)

                nudge_prev_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("nudge_selection_time_prev", "SHIFT+LEFT_ARROW"))
                def _modifiers_match(tuple_mods, io):
                    return (
                        tuple_mods["shift"] == io.key_shift and
                        tuple_mods["ctrl"] == io.key_ctrl and
                        tuple_mods["alt"] == io.key_alt and
                        tuple_mods["super"] == io.key_super
                    )

                if (
                    nudge_prev_tuple
                    and imgui.is_key_pressed(nudge_prev_tuple[0])
                    and _modifiers_match(nudge_prev_tuple[1], io)
                ):
                    time_nudge_delta_ms = -snap_grid_time_ms_for_nudge

                nudge_next_tuple = self.app._map_shortcut_to_glfw_key(
                    shortcuts.get("nudge_selection_time_next", "SHIFT+RIGHT_ARROW")
                )
                if (
                    time_nudge_delta_ms == 0
                    and nudge_next_tuple
                    and imgui.is_key_pressed(nudge_next_tuple[0])
                    and _modifiers_match(nudge_next_tuple[1], io)
                ):
                    time_nudge_delta_ms = snap_grid_time_ms_for_nudge

                if time_nudge_delta_ms != 0 and self.multi_selected_action_indices:
                    op_desc = f"Nudged Point(s) Time by {time_nudge_delta_ms}ms"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    indices_to_affect = sorted(
                        list(self.multi_selected_action_indices),
                        reverse=(time_nudge_delta_ms < 0)
                    )

                    objects_to_move = [actions_list[idx] for idx in indices_to_affect]
                    for i, action_obj in enumerate(objects_to_move):
                        current_idx_in_list = actions_list.index(action_obj)
                        new_at = action_obj["at"] + time_nudge_delta_ms

                        prev_at_limit = -1
                        # Find true previous point NOT in the current selection being moved
                        for k in range(current_idx_in_list - 1, -1, -1):
                            if actions_list[k] not in objects_to_move: prev_at_limit = actions_list[k]["at"] + 1; break
                        if prev_at_limit == -1: prev_at_limit = 0

                        next_at_limit = float('inf')
                        # Find true next point NOT in the current selection being moved
                        for k in range(current_idx_in_list + 1, len(actions_list)):
                            if actions_list[k] not in objects_to_move: next_at_limit = actions_list[k]["at"] - 1; break
                        if video_loaded: next_at_limit = min(next_at_limit, effective_total_duration_ms)

                        action_obj["at"] = int(
                            round(np.clip(float(new_at), float(prev_at_limit), float(next_at_limit))))

                    actions_list.sort(key=lambda a: a["at"])
                    # Re-select moved points robustly
                    self.multi_selected_action_indices = {actions_list.index(obj) for obj in objects_to_move if obj in actions_list}
                    self.selected_action_idx = min(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                    # After nudging, seek the video to the new position of the primary selected point
                    if video_loaded and (not self.app.processor.is_processing or is_paused) and self.selected_action_idx != -1:
                        try:
                            # Ensure the selected index is still valid
                            if 0 <= self.selected_action_idx < len(actions_list):
                                new_time_ms = actions_list[self.selected_action_idx]['at']
                                target_frame = int(round((new_time_ms / 1000.0) * video_fps_for_calc))
                                clamped_frame = np.clip(target_frame, 0, self.app.processor.total_frames - 1)

                                # Seek the video and force the timeline to pan to the new frame
                                self.app.processor.seek_video(clamped_frame)
                                app_state.force_timeline_pan_to_current_frame = True
                        except (IndexError, ValueError) as e:
                            self.app.logger.warning(f"Could not sync video after nudge due to an error: {e}")

                # Delete Selected
                del_sc_str = shortcuts.get("delete_selected_point", "DELETE")
                del_alt_sc_str = shortcuts.get("delete_selected_point_alt", "BACKSPACE")
                del_key_tuple = self.app._map_shortcut_to_glfw_key(del_sc_str)
                bck_key_tuple = self.app._map_shortcut_to_glfw_key(del_alt_sc_str)
                delete_pressed = False

                if del_key_tuple and (
                    imgui.is_key_pressed(del_key_tuple[0]) and
                    all(m == io_m for m, io_m in
                        zip(del_key_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
                ):
                    delete_pressed = True

                if (not delete_pressed and bck_key_tuple and (
                    imgui.is_key_pressed(bck_key_tuple[0]) and
                    all(m == io_m for m, io_m in
                        zip(bck_key_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
                )):
                        delete_pressed = True

                if delete_pressed and self.multi_selected_action_indices:
                    op_desc = f"Deleted {len(self.multi_selected_action_indices)} Selected Point(s) (Key)"
                    fs_proc._record_timeline_action(self.timeline_num, op_desc)
                    target_funscript_instance_for_render.clear_points(axis=axis_name_for_render, selected_indices=list(
                        self.multi_selected_action_indices))
                    self.multi_selected_action_indices.clear()
                    self.selected_action_idx = -1
                    fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)

                # Add Point with Number Keys
                time_at_center_add = x_to_time(center_x_marker)
                snap_time_add_key = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                snapped_time_add_key = max(0, int(round(time_at_center_add / snap_time_add_key)) * snap_time_add_key)

                for i in range(10):
                    bound_key_str_num = shortcuts.get(f"add_point_{i * 10}", str(i))
                    if not bound_key_str_num: continue
                    glfw_key_num_tuple = self.app._map_shortcut_to_glfw_key(bound_key_str_num)
                    if (glfw_key_num_tuple and (
                        imgui.is_key_pressed(glfw_key_num_tuple[0]) and
                        all(m == io_m for m, io_m in
                            zip(glfw_key_num_tuple[1].values(), [io.key_shift, io.key_ctrl, io.key_alt, io.key_super]))
                    )):
                        target_pos_val = i * 10
                        op_desc = "Added Point (Key)"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        target_funscript_instance_for_render.add_action(
                            timestamp_ms=snapped_time_add_key,
                            primary_pos=target_pos_val if axis_name_for_render == 'primary' else None,
                            secondary_pos=target_pos_val if axis_name_for_render == 'secondary' else None,
                            is_from_live_tracker=False
                        )

                        # Re-fetch actions and select the new point
                        actions_list = getattr(target_funscript_instance_for_render, f"{axis_name_for_render}_actions", [])
                        new_idx = next((idx for idx, act in enumerate(actions_list) if act['at'] == snapped_time_add_key and act['pos'] == target_pos_val), -1)
                        if new_idx != -1:
                            # Set the new point as the only selected point
                            self.selected_action_idx = new_idx
                            self.multi_selected_action_indices = {new_idx}

                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        break

            # Update the shared state flag for this frame
            app_state.timeline_interaction_active = is_interacting_this_frame

            # Clip pan offset if any interaction occurred
            if app_state.timeline_interaction_active:
                app_state.timeline_pan_offset_ms = np.clip(app_state.timeline_pan_offset_ms, min_pan_allowed, max_pan_allowed)

            # Determine paused state
            is_paused = False
            if video_loaded and self.app.processor and self.app.processor.is_processing:
                is_paused = hasattr(self.app.processor, "pause_event") and self.app.processor.pause_event.is_set()

            # Detect end of panning interaction to seek video (do not seek after pure zoom)
            if was_panning_active and not self.is_panning_active and not self.is_zooming_active:
                # Allow seeking if stopped or paused (not actively playing)
                if video_loaded and (not self.app.processor.is_processing or is_paused):
                    center_x_marker = canvas_abs_pos[0] + canvas_size[0] / 2.0
                    time_at_center_ms = x_to_time(center_x_marker)
                    clamped_time_ms = np.clip(time_at_center_ms, 0, effective_total_duration_ms)
                    target_frame = ms_to_frame(clamped_time_ms)
                    clamped_frame = np.clip(target_frame, 0, self.app.processor.total_frames - 1)
                    # Only seek if the target frame is different from current
                    if abs(clamped_frame - self.app.processor.current_frame_index) > 0:
                        self.app.processor.seek_video(clamped_frame)
                        self.app.project_manager.project_dirty = True
                        app_state.force_timeline_pan_to_current_frame = True  # This will cause it to pan to the frame if needed

            # Auto-pan logic (for playback or forced sync)
            is_playing = video_loaded and self.app.processor.is_processing and not is_paused
            pan_to_current_frame = video_loaded and not is_playing and app_state.force_timeline_pan_to_current_frame
            if (is_playing or pan_to_current_frame) and not app_state.timeline_interaction_active:
                # No manual interaction right now
                current_video_time_ms = (self.app.processor.current_frame_index / video_fps_for_calc) * 1000.0
                # Using effective center of screen
                target_pan_offset = current_video_time_ms - center_marker_offset_ms
                app_state.timeline_pan_offset_ms = np.clip(
                    target_pan_offset, min_pan_allowed,
                    max_pan_allowed
                )

                if pan_to_current_frame:
                    app_state.force_timeline_pan_to_current_frame = False

            marker_color_fixed = imgui.get_color_u32_rgba(*TimelineColors.CENTER_MARKER)
            draw_list.add_line(center_x_marker, canvas_abs_pos[1], center_x_marker, canvas_abs_pos[1] + canvas_size[1], marker_color_fixed, 1.5)
            tri_half_base, tri_height = 5.0, 10.0

            draw_list.add_triangle_filled(
                center_x_marker,
                canvas_abs_pos[1] + tri_height, center_x_marker - tri_half_base, canvas_abs_pos[1],
                center_x_marker + tri_half_base, canvas_abs_pos[1],
                marker_color_fixed
            )

            time_at_center_ms_display = x_to_time(center_x_marker)
            time_str_display_main = _format_time(self.app, time_at_center_ms_display / 1000.0)
            frame_str_display = ""

            if video_loaded and video_fps_for_calc > 0:
                frame_at_center = int(round((time_at_center_ms_display / 1000.0) * video_fps_for_calc))
                frame_str_display = f" (F: {frame_at_center})"

            full_time_display_str = f"{time_str_display_main}{frame_str_display}"

            draw_list.add_text(
                center_x_marker + 5,
                canvas_abs_pos[1] + tri_height + 5,
                imgui.get_color_u32_rgba(*TimelineColors.TIME_DISPLAY_TEXT),
                full_time_display_str
            )

            # Grid drawing (Horizontal - Position)
            for i in range(5):  # 0, 25, 50, 75, 100
                y_grid_h = canvas_abs_pos[1] + (i / 4.0) * canvas_size[1]
                grid_col = imgui.get_color_u32_rgba(*TimelineColors.GRID_LINES if i != 2 else TimelineColors.GRID_MAJOR_LINES)  # Center line darker
                line_thickness = 1.0 if i != 2 else 1.5

                draw_list.add_line(
                    canvas_abs_pos[0],
                    y_grid_h,
                    canvas_abs_pos[0] + canvas_size[0],
                    y_grid_h,
                    grid_col,
                    line_thickness
                )

                pos_val = 100 - int((i / 4.0) * 100)
                text_y_offset = -imgui.get_text_line_height() - 2 if i == 4 else (
                    2 if i == 0 else -imgui.get_text_line_height() / 2)

                draw_list.add_text(
                    canvas_abs_pos[0] + 3, y_grid_h + text_y_offset,
                    imgui.get_color_u32_rgba(*TimelineColors.GRID_LABELS),
                    str(pos_val)
                )

            # Grid drawing (Vertical - Time)
            time_per_screen_ms_grid = canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px
            px_per_100ms_grid = 100.0 / app_state.timeline_zoom_factor_ms_per_px if app_state.timeline_zoom_factor_ms_per_px > 0 else 0

            try:
                grid_steps = [
                    (80, 100),
                    (40, 200),
                    (20, 500),
                    (8, 1000),
                    (4, 2000),
                    (2, 5000),
                    (0.5, 10000),
                ]
                time_step_ms_grid = 60000
                for threshold, step in grid_steps:
                    if px_per_100ms_grid > threshold:
                        time_step_ms_grid = step
                        break
            except Exception as e:
                time_step_ms_grid = 100
                self.app.logger.error("Error calculating time step for grid: {e}")

            if time_step_ms_grid > 0:
                start_time = app_state.timeline_pan_offset_ms
                end_time = start_time + time_per_screen_ms_grid

                # Find the first grid line within the visible range
                first_grid_time = math.ceil(start_time / time_step_ms_grid) * time_step_ms_grid

                # Loop over grid lines within the visible range
                for t_ms in np.arange(first_grid_time, end_time + time_step_ms_grid, time_step_ms_grid):
                    x_pos = time_to_x(t_ms)

                    # Skip if x position is outside the canvas
                    if not (canvas_abs_pos[0] <= x_pos <= canvas_abs_pos[0] + canvas_size[0]):
                        continue

                    is_major_tick = (t_ms % (time_step_ms_grid * 5)) == 0
                    tick_color = imgui.get_color_u32_rgba(
                        *TimelineColors.GRID_MAJOR_LINES if is_major_tick else TimelineColors.GRID_LINES)
                    tick_thickness = 1.5 if is_major_tick else 1.0

                    # Draw vertical grid line
                    draw_list.add_line(
                        x_pos, canvas_abs_pos[1],
                        x_pos, canvas_abs_pos[1] + canvas_size[1],
                        tick_color, tick_thickness)

                    # Optionally draw time label
                    if not video_loaded or (0 <= t_ms <= effective_total_duration_ms + 1e-4):
                        label = f"{t_ms / 1000.0:.1f}s"
                        label_color = imgui.get_color_u32_rgba(*TimelineColors.GRID_LABELS)
                        draw_list.add_text(x_pos + 3, canvas_abs_pos[1] + 3, label_color, label)

            # --- Draw Audio Waveform ---
            if self.app.app_state_ui.show_audio_waveform and self.app.audio_waveform_data is not None:
                waveform_data = self.app.audio_waveform_data
                num_samples = len(waveform_data)

                if num_samples > 1 and effective_total_duration_ms > 0:
                    # Compute visible time range on timeline
                    visible_start_ms = app_state.timeline_pan_offset_ms
                    visible_end_ms = visible_start_ms + (canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px)

                    # Convert visible time range to sample indices
                    start_idx = int(max(0, (visible_start_ms / effective_total_duration_ms) * num_samples))
                    end_idx = int(min(num_samples, (visible_end_ms / effective_total_duration_ms) * num_samples + 2))

                    if end_idx > start_idx:
                        # Slice visible waveform segment
                        visible_indices = np.arange(start_idx, end_idx)
                        visible_times_ms = (visible_indices / (num_samples - 1)) * effective_total_duration_ms
                        visible_amplitudes = waveform_data[start_idx:end_idx]

                        # Calculate coordinates
                        x_coords = time_to_x_vec(visible_times_ms)
                        canvas_center_y = canvas_abs_pos[1] + (canvas_size[1] / 2.0)
                        y_offsets = visible_amplitudes * (canvas_size[1] / 2.0)
                        y_coords_top = canvas_center_y - y_offsets
                        y_coords_bottom = canvas_center_y + y_offsets

                    waveform_color = imgui.get_color_u32_rgba(*TimelineColors.AUDIO_WAVEFORM)
                    points_top = list(zip(x_coords, y_coords_top))
                    points_bottom = list(zip(x_coords, y_coords_bottom))
                    draw_list.add_polyline(points_top, waveform_color, False, 1.0)
                    draw_list.add_polyline(points_bottom, waveform_color, False, 1.0)

            # --- Draw Actions (Points and Lines) with GPU Acceleration ---
            hovered_action_idx_current_timeline = -1

            # Try GPU rendering first (if available and enabled)
            gpu_render_success = False
            if (self.gpu_integration and
                    self.app.app_settings.get("timeline_gpu_enabled", False) and
                    actions_list):

                # Gather selected/hovered indices for GPU rendering
                selected_indices = list(self.multi_selected_action_indices)
                if self.selected_action_idx >= 0:
                    selected_indices.append(self.selected_action_idx)

                # Attempt GPU rendering
                gpu_render_success = self.gpu_integration.render_timeline_optimized(
                    actions_list=actions_list,
                    canvas_abs_pos=canvas_abs_pos,
                    canvas_size=canvas_size,
                    app_state=app_state,
                    draw_list=draw_list,
                    mouse_pos=mouse_pos,
                    selected_indices=selected_indices,
                    hovered_index=self.selected_action_idx
                )

                if gpu_render_success:
                    # Ensure GPU actually drew content; otherwise fall back to CPU drawing
                    perf_summary = self.gpu_integration.get_performance_summary()
                    backend_used = str(perf_summary.get('current_backend', 'cpu'))
                    gpu_stats = perf_summary.get('gpu_details', {})
                    drew_anything = (gpu_stats.get('points_rendered', 0) + gpu_stats.get('lines_rendered', 0)) > 0
                    if backend_used != 'gpu' or not drew_anything:
                        gpu_render_success = False
                    else:
                        # GPU rendering succeeded - get hovered index from GPU system
                        hovered_action_idx_current_timeline = getattr(self.gpu_integration, '_last_hovered_index', -1)

                        # Show GPU indicator if enabled
                        if self.app.app_settings.get("show_timeline_optimization_indicator", False):
                            gpu_text = f"GPU Mode ({gpu_stats.get('points_rendered', 0):,} pts, {gpu_stats.get('render_time_ms', 0):.1f}ms)"
                            draw_list.add_text(canvas_abs_pos[0] + 10, canvas_abs_pos[1] + 10, imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 0.8), gpu_text)

            # Fallback to optimized CPU rendering if GPU failed or unavailable
            if not gpu_render_success:
                visible_actions_indices_range = None
                if actions_list:
                    action_times = [action["at"] for action in actions_list]
                    margin_ms_act = 2000
                    search_start_time = app_state.timeline_pan_offset_ms - margin_ms_act
                    search_end_time = app_state.timeline_pan_offset_ms + canvas_size[
                        0] * app_state.timeline_zoom_factor_ms_per_px + margin_ms_act
                    start_idx = bisect_left(action_times, search_start_time)
                    end_idx = bisect_right(action_times, search_end_time)
                    if start_idx < end_idx:
                        visible_actions_indices_range = (start_idx, end_idx)

                # --- START: LOD OPTIMIZATION ---
                actions_to_render = self.preview_actions if self.is_previewing else actions_list
                indices_to_draw = []
                avg_interval_ms = 0
                points_per_pixel = 1.0
                s_idx, e_idx = 0, 0
                if actions_to_render:
                    # Calculate avg interval (pre-computed in stats)
                    fs_proc = self.app.funscript_processor
                    stats = fs_proc.funscript_stats_t1 if self.timeline_num == 1 else fs_proc.funscript_stats_t2
                    avg_interval_ms = stats.get('avg_interval_ms', 20)

                    points_per_pixel = (app_state.timeline_zoom_factor_ms_per_px / avg_interval_ms) if avg_interval_ms > 0 else 1.0

                    # Visible range
                    s_idx, e_idx = 0, len(actions_to_render)
                    if visible_actions_indices_range:
                        s_idx, e_idx = visible_actions_indices_range

                    # Decimate for line drawing
                    draw_step = max(1, int(points_per_pixel / 4.0))
                    indices_to_draw = range(s_idx, e_idx, draw_step)
                    last_visible_idx = e_idx - 1
                    if last_visible_idx >= s_idx and last_visible_idx not in indices_to_draw:
                        indices_to_draw = list(indices_to_draw) + [last_visible_idx]
                # --- END: LOD OPTIMIZATION ---

                # --- Draw Lines OR Dense Envelope (Vectorized) ---
                if len(actions_list) > 1:
                    cached_data = self._get_or_compute_cached_arrays(actions_list)
                    use_envelope = False
                    if s_idx < e_idx:
                        visible_count = (e_idx - s_idx)
                        use_envelope = (points_per_pixel >= 4.0 and visible_count >= 2000)

                    if use_envelope:
                        # Dense waveform-like envelope to drastically cut draw calls
                        step_px = max(1, int(points_per_pixel / 2.0))
                        self._render_dense_envelope(draw_list, canvas_abs_pos, canvas_size, app_state, cached_data, step_px)
                    else:
                        if len(indices_to_draw) > 1:
                            all_ats, all_poss = cached_data["ats"], cached_data["poss"]

                            indices_array = np.array(list(indices_to_draw))
                            p1_indices = indices_array[:-1]
                            p2_indices = indices_array[1:]

                            # Use array indexing on cached arrays instead of list comprehensions
                            p1_ats = all_ats[p1_indices]
                            p1_poss = all_poss[p1_indices]
                            p2_ats = all_ats[p2_indices]
                            p2_poss = all_poss[p2_indices]

                            x1s = time_to_x_vec(p1_ats)
                            y1s = pos_to_y_vec(p1_poss)
                            x2s = time_to_x_vec(p2_ats)
                            y2s = pos_to_y_vec(p2_poss)

                            delta_t_ms_vec = p2_ats - p1_ats
                            delta_pos_vec = np.abs(p2_poss - p1_poss)
                            speeds_vec = np.divide(
                                delta_pos_vec,
                                delta_t_ms_vec / 1000.0,
                                out=np.zeros_like(delta_pos_vec),
                                where=delta_t_ms_vec > 1e-5
                            )

                            colors_rgba = self.app.utility.get_speed_colors_vectorized(speeds_vec)
                            alpha = 0.2 if self.is_previewing else 1.0
                            thickness = 1.0 if self.is_previewing else 2.0

                            canvas_x, canvas_w = canvas_abs_pos[0], canvas_size[0]
                            canvas_x_end = canvas_x + canvas_w

                            for i in range(len(x1s)):
                                x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                                if (x1 < canvas_x and x2 < canvas_x) or (x1 > canvas_x_end and x2 > canvas_x_end):
                                    continue
                                color = colors_rgba[i]
                                final_color = imgui.get_color_u32_rgba(color[0], color[1], color[2], color[3] * alpha)
                                draw_list.add_line(x1, y1, x2, y2, final_color, thickness)

                # --- RADICAL OPTIMIZATION: Motion-based temporal culling ---
                current_time = time.time()
                current_pan_offset = app_state.timeline_pan_offset_ms
                time_delta = current_time - self._last_frame_time

                if time_delta > 0:
                    pan_delta = abs(current_pan_offset - self._last_pan_offset)
                    self._pan_velocity = pan_delta / (time_delta * 1000)  # pixels per second

                self._last_pan_offset = current_pan_offset
                self._last_frame_time = current_time

                # Detect fast scrolling/panning
                FAST_SCROLL_THRESHOLD = 500
                is_fast_scrolling = self._pan_velocity > FAST_SCROLL_THRESHOLD

                points_per_pixel = (
                    app_state.timeline_zoom_factor_ms_per_px / avg_interval_ms
                    if avg_interval_ms > 0 else 1.0
                )
                visible_count = (
                    (e_idx - s_idx)
                    if (s_idx is not None and e_idx is not None)
                    else len(actions_list)
                )

                # Render points unless fast scrolling
                # except for key interaction states or low density
                should_render_points = (
                    not is_fast_scrolling and (
                        points_per_pixel < 2.0 or
                        visible_count <= 4 or
                        self.selected_action_idx >= 0 or
                        len(self.multi_selected_action_indices) > 0 or
                        self.dragging_action_idx >= 0
                    )
                )

                # Visual indicator for optimization modes (optional debug info, optimized)
                show_opt = self.app.app_settings.get("show_timeline_optimization_indicator", False)
                if not show_opt:
                    # Early exit if indicator is not enabled
                    pass
                else:
                    opt_text = (
                        "Fast Scroll Mode" if is_fast_scrolling else
                        "Envelope Mode" if points_per_pixel >= 4.0 and (e_idx - s_idx) >= 2000 else
                        "Lines-Only Mode" if points_per_pixel >= 2.0 else
                        f"LOD Active ({len(indices_to_draw)}/{len(actions_list)})" if len(indices_to_draw) != len(actions_list) else
                        ""
                    )
                    if opt_text:
                        draw_list.add_text(
                            canvas_abs_pos[0] + 5, canvas_abs_pos[1] + 15,
                            imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8),
                            opt_text
                        )

                if actions_list and should_render_points:
                    if indices_to_draw:
                        cached_data = self._get_or_compute_cached_arrays(actions_list)
                        all_ats, all_poss = cached_data["ats"], cached_data["poss"]

                        point_indices = np.array(list(indices_to_draw))
                        point_ats = all_ats[point_indices]
                        point_poss = all_poss[point_indices]
                        pxs = time_to_x_vec(point_ats)
                        pys = pos_to_y_vec(point_poss)

                        hover_radius_sq = (app_state.timeline_point_radius + 4) ** 2
                        distances_sq = (mouse_pos[0] - pxs) ** 2 + (mouse_pos[1] - pys) ** 2
                        hovered_mask = distances_sq < hover_radius_sq

                        for i_loop, original_list_idx in enumerate(point_indices):
                            px, py = pxs[i_loop], pys[i_loop]
                            is_hovered_pt = hovered_mask[i_loop]
                            is_primary_selected = (original_list_idx == self.selected_action_idx)
                            is_in_multi_selection = (original_list_idx in self.multi_selected_action_indices)
                            is_being_dragged = (original_list_idx == self.dragging_action_idx)
                            point_radius_draw = app_state.timeline_point_radius
                            pt_color_tuple = TimelineColors.POINT_DEFAULT

                            if is_being_dragged:
                                pt_color_tuple = TimelineColors.POINT_DRAGGING
                                point_radius_draw += 1
                            elif is_primary_selected or is_in_multi_selection:
                                pt_color_tuple = TimelineColors.POINT_SELECTED
                                if is_in_multi_selection and not is_primary_selected: point_radius_draw += 0.5
                            elif is_hovered_pt and imgui.is_window_hovered() and not self.is_marqueeing:
                                pt_color_tuple = TimelineColors.POINT_HOVER
                                if self.dragging_action_idx == -1:
                                    hovered_action_idx_current_timeline = original_list_idx

                            point_alpha = 0.3 if self.is_previewing else 1.0
                            final_pt_color = imgui.get_color_u32_rgba(
                                pt_color_tuple[0],
                                pt_color_tuple[1],
                                pt_color_tuple[2],
                                pt_color_tuple[3] * point_alpha
                            )

                            draw_list.add_circle_filled(px, py, point_radius_draw, final_pt_color)

                            if is_primary_selected and not is_being_dragged:
                                draw_list.add_circle(
                                    px, py, point_radius_draw + 1,
                                    imgui.get_color_u32_rgba(*TimelineColors.SELECTED_POINT_BORDER),
                                    thickness=1.0
                                )

            # --- Draw Ultimate Autotune Preview (if enabled) ---
            if self.ultimate_autotune_preview_actions:
                preview_points_to_draw = self.ultimate_autotune_preview_actions
                if len(preview_points_to_draw) > 1:
                    preview_at_np = np.array([p['at'] for p in preview_points_to_draw], dtype=np.int64)
                    preview_pos_np = np.array([p['pos'] for p in preview_points_to_draw], dtype=np.int32)

                    view_start_time_ms = app_state.timeline_pan_offset_ms
                    view_end_time_ms = view_start_time_ms + canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px

                    visible_mask = (preview_at_np >= view_start_time_ms) & (preview_at_np <= view_end_time_ms)
                    visible_indices = np.where(visible_mask)[0]

                    if visible_indices.size > 0:
                        start_idx = max(0, visible_indices[0] - 1)
                        end_idx = min(len(preview_at_np) - 1, visible_indices[-1] + 1)

                        if end_idx > start_idx:
                            x_coords = time_to_x_vec(preview_at_np[start_idx:end_idx + 1])
                            y_coords = pos_to_y_vec(preview_pos_np[start_idx:end_idx + 1])

                            preview_color = imgui.get_color_u32_rgba(*TimelineColors.ULTIMATE_AUTOTUNE_PREVIEW)
                            for i in range(len(x_coords) - 1):
                                draw_list.add_line(x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1], preview_color, 1.5)

            # --- Draw Preview Actions on Top ---
            if self.is_previewing and self.preview_actions:
                preview_line_color = imgui.get_color_u32_rgba(*TimelineColors.PREVIEW_LINES)  # Bright Orange
                preview_point_color = imgui.get_color_u32_rgba(*TimelineColors.PREVIEW_POINTS)
                preview_point_radius = app_state.timeline_point_radius

                # Draw preview lines and points using the self.preview_actions list
                if len(self.preview_actions) > 1:
                    p_ats = np.array([a['at'] for a in self.preview_actions])
                    p_poss = np.array([a['pos'] for a in self.preview_actions])
                    pxs = time_to_x_vec(p_ats)
                    pys = pos_to_y_vec(p_poss)
                    for i in range(len(pxs) - 1):
                        draw_list.add_line(pxs[i], pys[i], pxs[i + 1], pys[i + 1], preview_line_color, 2.0)

                for action in self.preview_actions:
                    px = time_to_x(action['at'])
                    py = pos_to_y(action['pos'])
                    draw_list.add_circle_filled(px, py, preview_point_radius, preview_point_color)
            
            # --- Draw Plugin Preview Overlays ---
            # Render preview overlays from the new plugin system
            if self.plugin_preview_renderer:
                # Calculate visible time range in milliseconds
                visible_start_ms = int(app_state.timeline_pan_offset_ms)
                visible_end_ms = int(app_state.timeline_pan_offset_ms + canvas_size[0] * app_state.timeline_zoom_factor_ms_per_px)
                
                # Render all active plugin previews
                self.plugin_preview_renderer.render_preview_overlay(
                    draw_list,
                    canvas_abs_pos[0],  # timeline_x
                    canvas_abs_pos[1],  # timeline_y
                    canvas_size[0],     # timeline_width
                    canvas_size[1],     # timeline_height
                    visible_start_ms,
                    visible_end_ms,
                    None  # Render all active previews
                )

            # --- Draw Marquee Selection ---
            if self.is_marqueeing and self.marquee_start_screen_pos and self.marquee_end_screen_pos:
                min_x, max_x = min(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0]), max(
                    self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                min_y, max_y = min(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1]), max(
                    self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])
                draw_list.add_rect_filled(min_x, min_y, max_x, max_y, imgui.get_color_u32_rgba(*TimelineColors.MARQUEE_SELECTION_FILL))
                draw_list.add_rect(min_x, min_y, max_x, max_y, imgui.get_color_u32_rgba(*TimelineColors.MARQUEE_SELECTION_BORDER))
                
            # --- Draw ALT+Drag Range Selection ---
            if self.range_selecting:
                start_x = time_to_x(self.range_selection_start_time)
                end_x = time_to_x(self.range_selection_end_time)
                min_x, max_x = min(start_x, end_x), max(start_x, end_x)
                
                # Draw a vertical band across the full height of the timeline
                top_y = canvas_abs_pos[1]
                bottom_y = canvas_abs_pos[1] + canvas_size[1]
                
                # Use a different color for range selection (more blue/cyan tint)
                range_fill_color = imgui.get_color_u32_rgba(0.0, 0.7, 1.0, 0.2)  # Cyan with transparency
                range_border_color = imgui.get_color_u32_rgba(0.0, 0.7, 1.0, 0.8)  # Cyan border
                
                draw_list.add_rect_filled(min_x, top_y, max_x, bottom_y, range_fill_color)
                draw_list.add_rect(min_x, top_y, max_x, bottom_y, range_border_color, thickness=2.0)

            # --- Mouse Interactions ---
            if imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and not io.key_shift:
                if hovered_action_idx_current_timeline != -1:
                    # ====== START: CALIBRATION HOOK ======
                    if self.app.calibration and self.app.calibration.is_calibration_mode_active:
                        # 1. Notify the calibration manager with the timestamp
                        clicked_action_time_ms = actions_list[hovered_action_idx_current_timeline]['at']
                        self.app.calibration.handle_calibration_point_selection(clicked_action_time_ms)

                        # 2. Update this timeline's UI to select the point
                        self.selected_action_idx = hovered_action_idx_current_timeline
                        self.multi_selected_action_indices = {hovered_action_idx_current_timeline}

                        # 3. Seek the video and focus the timeline on the selected point
                        if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                            target_frame_on_click = int(
                                round((clicked_action_time_ms / 1000.0) * video_fps_for_calc))
                            self.app.processor.seek_video(np.clip(target_frame_on_click, 0, self.app.processor.total_frames - 1))
                            app_state.force_timeline_pan_to_current_frame = True

                        # Prevent any other click logic from running for this event
                        imgui.end()
                        return
                    # ====== END: CALIBRATION HOOK ======

                    self.is_marqueeing = False
                    if not io.key_ctrl: self.multi_selected_action_indices.clear()
                    if hovered_action_idx_current_timeline in self.multi_selected_action_indices and io.key_ctrl:
                        self.multi_selected_action_indices.remove(hovered_action_idx_current_timeline)
                    else:
                        self.multi_selected_action_indices.add(hovered_action_idx_current_timeline)
                    self.selected_action_idx = min(
                        self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

                    self.dragging_action_idx = hovered_action_idx_current_timeline
                    if not self.drag_undo_recorded and allow_editing_timeline:
                        fs_proc._record_timeline_action(self.timeline_num, "Start Point Drag")
                        self.drag_undo_recorded = True

                    if video_loaded and not self.app.processor.is_processing and video_fps_for_calc > 0:
                        target_frame_on_click = int(round(
                            (actions_list[hovered_action_idx_current_timeline]["at"] / 1000.0) * video_fps_for_calc))
                        self.app.processor.seek_video(
                            np.clip(target_frame_on_click, 0, self.app.processor.total_frames - 1))
                        app_state.force_timeline_pan_to_current_frame = True
                elif is_timeline_hovered_for_marquee_start:  # Use the relaxed hover check here for marquee
                    if io.key_alt:
                        # Start ALT+drag range selection by time
                        self.range_selecting = True
                        self.range_selection_start_time = x_to_time(mouse_pos[0])
                        self.range_selection_end_time = self.range_selection_start_time
                        if not io.key_ctrl:
                            self.multi_selected_action_indices.clear()
                            self.selected_action_idx = -1
                    else:
                        # Start marquee
                        self.is_marqueeing = True
                        self.marquee_start_screen_pos = mouse_pos
                        self.marquee_end_screen_pos = mouse_pos
                        if not io.key_ctrl:
                            self.multi_selected_action_indices.clear()
                            self.selected_action_idx = -1

            # Context Menu (only when strictly hovered over the canvas, not the padding)
            if imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_RIGHT) and is_timeline_hovered:
                self.context_mouse_pos_screen = mouse_pos
                time_at_click = x_to_time(mouse_pos[0])
                pos_at_click = y_to_pos(mouse_pos[1])
                snap_time = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                snap_pos = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                self.new_point_candidate_at = max(0, int(round(time_at_click / snap_time)) * snap_time)
                self.new_point_candidate_pos = min(100, max(0, int(round(pos_at_click / snap_pos)) * snap_pos))

                self.context_menu_point_idx = hovered_action_idx_current_timeline

                if self.context_menu_point_idx != -1 and not (
                        self.context_menu_point_idx in self.multi_selected_action_indices):
                    if not io.key_ctrl: self.multi_selected_action_indices.clear()
                    self.multi_selected_action_indices.add(self.context_menu_point_idx)
                    self.selected_action_idx = self.context_menu_point_idx
                imgui.open_popup(context_popup_id)

            # Point Drag
            if self.dragging_action_idx != -1 and imgui.is_mouse_dragging(
                    glfw.MOUSE_BUTTON_LEFT) and allow_editing_timeline and not io.key_shift:
                if 0 <= self.dragging_action_idx < len(actions_list):
                    action_to_drag = actions_list[self.dragging_action_idx]
                    new_time_cand_ms = x_to_time(mouse_pos[0])
                    new_pos_cand = y_to_pos(mouse_pos[1])
                    snap_time = app_state.snap_to_grid_time_ms if app_state.snap_to_grid_time_ms > 0 else 1
                    snap_pos = app_state.snap_to_grid_pos if app_state.snap_to_grid_pos > 0 else 1
                    snapped_new_at = max(0, int(round(new_time_cand_ms / snap_time)) * snap_time)
                    snapped_new_pos = min(100, max(0, int(round(new_pos_cand / snap_pos)) * snap_pos))

                    effective_prev_at_lim = actions_list[self.dragging_action_idx - 1]["at"] + 1 if self.dragging_action_idx > 0 else 0
                    effective_next_at_lim = actions_list[self.dragging_action_idx + 1]["at"] - 1 if self.dragging_action_idx < len(actions_list) - 1 else float('inf')
                    action_to_drag["at"] = int(np.clip(float(snapped_new_at), float(effective_prev_at_lim), float(effective_next_at_lim)))
                    action_to_drag["pos"] = snapped_new_pos

                    self.app.project_manager.project_dirty = True
                    if self.timeline_num == 1:
                        app_state.heatmap_dirty = True
                        app_state.funscript_preview_dirty = True

            # Marquee Drag
            if self.is_marqueeing and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT) and not io.key_shift:
                self.marquee_end_screen_pos = mouse_pos
                
            # ALT+Drag Range Selection
            if self.range_selecting and imgui.is_mouse_dragging(glfw.MOUSE_BUTTON_LEFT) and io.key_alt:
                self.range_selection_end_time = x_to_time(mouse_pos[0])

            # Handle Mouse Release for Marquee Selection
            if imgui.is_mouse_released(glfw.MOUSE_BUTTON_LEFT) and self.is_marqueeing:
                self.is_marqueeing = False
                if self.marquee_start_screen_pos and self.marquee_end_screen_pos and actions_list:
                    # 1. Get marquee rectangle in screen coordinates
                    min_x, max_x = min(self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0]), max(
                        self.marquee_start_screen_pos[0], self.marquee_end_screen_pos[0])
                    min_y, max_y = min(self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1]), max(
                        self.marquee_start_screen_pos[1], self.marquee_end_screen_pos[1])

                    # 2. Find potential points in the time range to optimize
                    time_start_ms = x_to_time(min_x)
                    time_end_ms = x_to_time(max_x)
                    action_times = [a['at'] for a in actions_list]
                    s_idx_time = bisect_left(action_times, time_start_ms)
                    e_idx_time = bisect_right(action_times, time_end_ms)

                    newly_selected = set()
                    # 3. Iterate over the time-culled slice and check screen coordinates
                    for i in range(s_idx_time, e_idx_time):
                        action = actions_list[i]
                        # Convert point to screen space to check against marquee
                        px = time_to_x(action['at'])
                        py = pos_to_y(action['pos'])

                        if min_x <= px <= max_x and min_y <= py <= max_y:
                            newly_selected.add(i)

                    # 4. Update selection state
                    if io.key_ctrl:
                        self.multi_selected_action_indices.symmetric_difference_update(newly_selected)
                    else:
                        self.multi_selected_action_indices = newly_selected

                    self.selected_action_idx = min(self.multi_selected_action_indices) if self.multi_selected_action_indices else -1

                # Reset marquee state
                self.marquee_start_screen_pos = None
                self.marquee_end_screen_pos = None

            # Handle Mouse Release for ALT+Drag Range Selection
            if imgui.is_mouse_released(glfw.MOUSE_BUTTON_LEFT) and self.range_selecting:
                self.range_selecting = False
                if actions_list:
                    # Get time range
                    start_time = min(self.range_selection_start_time, self.range_selection_end_time)
                    end_time = max(self.range_selection_start_time, self.range_selection_end_time)
                    
                    # Find all points within the time range
                    newly_selected = set()
                    for i, action in enumerate(actions_list):
                        if start_time <= action['at'] <= end_time:
                            newly_selected.add(i)
                    
                    # Update selection
                    if io.key_ctrl:
                        self.multi_selected_action_indices.symmetric_difference_update(newly_selected)
                    else:
                        self.multi_selected_action_indices = newly_selected
                    
                    self.selected_action_idx = min(self.multi_selected_action_indices) if self.multi_selected_action_indices else -1
                    
                    # Log the selection
                    num_selected = len(newly_selected)
                    if num_selected > 0:
                        self.app.logger.info(f"T{self.timeline_num}: ALT+Drag selected {num_selected} points in time range {start_time:.0f}-{end_time:.0f}ms.")

                if self.dragging_action_idx != -1 and allow_editing_timeline:
                    dragged_action_ref = actions_list[self.dragging_action_idx]
                    actions_list.sort(key=lambda a: a["at"])
                    try:
                        new_idx = actions_list.index(dragged_action_ref)
                        self.selected_action_idx = new_idx
                        self.multi_selected_action_indices = {new_idx}
                    except ValueError:
                        self.selected_action_idx = -1
                        self.multi_selected_action_indices.clear()

                    if self.drag_undo_recorded:
                        fs_proc._finalize_action_and_update_ui(self.timeline_num, "Point Dragged")
                        self.drag_undo_recorded = False
                    self.dragging_action_idx = -1

            # --- Context Menu & Tooltip ---
            if imgui.begin_popup(context_popup_id):
                imgui.text(f"Timeline {self.timeline_num} @ Time: {self.new_point_candidate_at}ms, Pos: {self.new_point_candidate_pos}")
                imgui.separator()

                if allow_editing_timeline:
                    if imgui.menu_item(f"Add Point Here##CTXMenuAdd{window_id_suffix}")[0]:
                        op_desc = "Added Point (Menu)"
                        fs_proc._record_timeline_action(self.timeline_num, op_desc)

                        primary_pos, secondary_pos = None, None
                        if axis_name_for_render == 'primary':
                            primary_pos = self.new_point_candidate_pos
                        elif axis_name_for_render == 'secondary':
                            secondary_pos = self.new_point_candidate_pos

                        target_funscript_instance_for_render.add_action(
                            timestamp_ms=self.new_point_candidate_at,
                            primary_pos=primary_pos,
                            secondary_pos=secondary_pos,
                            is_from_live_tracker=False
                        )

                        fs_proc._finalize_action_and_update_ui(self.timeline_num, op_desc)
                        imgui.close_current_popup()
                else:
                    imgui.menu_item("Add Point Here", enabled=False)

                imgui.separator()

                other_timeline_num = 2 if self.timeline_num == 1 else 1

                # Option 1: Copy this entire timeline over to the other one
                if imgui.menu_item(f"Copy All to Timeline {other_timeline_num}##CTXCopyFull",
                                   enabled=allow_editing_timeline)[0]:
                    if allow_editing_timeline:
                        self._handle_copy_full_timeline_to_other()
                    imgui.close_current_popup()

                # Option 2: Swap the contents of the two timelines
                if imgui.menu_item(f"Swap with Timeline {other_timeline_num}##CTXSwap", enabled=allow_editing_timeline)[
                    0]:
                    if allow_editing_timeline:
                        self._handle_swap_with_other_timeline()
                    imgui.close_current_popup()
                imgui.separator()

                can_copy_to_other = allow_editing_timeline and bool(self.multi_selected_action_indices)
                copy_dest_t_num = 2 if self.timeline_num == 1 else 1

                if imgui.menu_item(f"Copy Selected to Timeline {copy_dest_t_num}", enabled=can_copy_to_other)[0]:
                    if can_copy_to_other:
                        self._handle_copy_to_other_timeline()
                    imgui.close_current_popup()
                imgui.separator()

                # --- Selection Filtering ---
                can_filter_selection = len(self.multi_selected_action_indices) >= 3
                if imgui.menu_item("Select Top Points", enabled=can_filter_selection)[0]:
                    if can_filter_selection:
                        self._handle_selection_filtering(self._select_top_points)
                    imgui.close_current_popup()

                if imgui.menu_item("Select Bottom Points", enabled=can_filter_selection)[0]:
                    if can_filter_selection:
                        self._handle_selection_filtering(self._select_bottom_points)
                    imgui.close_current_popup()

                if imgui.menu_item("Select Mid Points", enabled=can_filter_selection)[0]:
                    if can_filter_selection:
                        self._handle_selection_filtering(self._select_mid_points)
                    imgui.close_current_popup()
                # --- End Selection Filtering ---

                imgui.separator()

                # --- New "Start Selection" / "End Selection" Context Menu Items ---
                # Use self.context_menu_point_idx here
                if self.context_menu_point_idx != -1:
                    # Only show "Start selection" if no anchor is set or if clicking the same anchor point again
                    if self.selection_anchor_idx == -1 or self.selection_anchor_idx == self.context_menu_point_idx:
                        if imgui.menu_item(f"Start Selection Here##CTXMenuStartSel{window_id_suffix}")[0]:
                            self.selection_anchor_idx = self.context_menu_point_idx
                            self.selected_action_idx = self.selection_anchor_idx
                            self.multi_selected_action_indices.clear()
                            self.multi_selected_action_indices.add(self.selection_anchor_idx)
                            self.app.logger.info(
                                f"T{self.timeline_num}: Selection start point set at index {self.selection_anchor_idx}.")
                            imgui.close_current_popup()
                    # Only show "End selection" if an anchor is set AND it's a different point
                    elif self.selection_anchor_idx != -1 and self.selection_anchor_idx != self.context_menu_point_idx:
                        if imgui.menu_item(f"End Selection Here##CTXMenuEndSel{window_id_suffix}")[0]:
                            start_idx = min(self.selection_anchor_idx, self.context_menu_point_idx)
                            end_idx = max(self.selection_anchor_idx, self.context_menu_point_idx)

                            new_selection = set(range(start_idx, end_idx + 1))
                            self.multi_selected_action_indices = new_selection
                            self.selected_action_idx = self.context_menu_point_idx  # Set the last clicked as primary
                            self.selection_anchor_idx = -1  # Reset anchor
                            self.app.logger.info(f"T{self.timeline_num}: Selected points from {start_idx} to {end_idx}.")
                            imgui.close_current_popup()
                else:  # If not hovering over a point when the context menu was opened
                    imgui.menu_item(f"Start Selection Here", enabled=False)
                    imgui.menu_item(f"End Selection Here", enabled=False)
                # --- End new items ---
                imgui.separator()

                can_copy = allow_editing_timeline and (bool(self.multi_selected_action_indices) or self.selected_action_idx != -1)
                if imgui.menu_item(f"Copy Selected##CTXMenuCopy", shortcut=shortcuts.get("copy_selection", "Ctrl+C"), enabled=can_copy)[0]:
                    if can_copy: self._handle_copy_selection()
                    imgui.close_current_popup()
                can_paste = allow_editing_timeline and self.app.funscript_processor.clipboard_has_actions()
                if imgui.menu_item(f"Paste at Cursor##CTXMenuPaste", shortcut=shortcuts.get("paste_selection", "Ctrl+V"), enabled=can_paste)[0]:
                    if can_paste: self._handle_paste_actions(self.new_point_candidate_at)
                    imgui.close_current_popup()
                imgui.separator()
                if imgui.menu_item(f"Cancel##CTXMenuCancel")[0]: imgui.close_current_popup()
                imgui.end_popup()

            if hovered_action_idx_current_timeline != -1 and self.dragging_action_idx == -1 and not imgui.is_popup_open(
                    context_popup_id):
                if 0 <= hovered_action_idx_current_timeline < len(actions_list):
                    action_hovered = actions_list[hovered_action_idx_current_timeline]

                    imgui.begin_tooltip()
                    imgui.text(f"Time: {action_hovered['at']} ms ({action_hovered['at'] / 1000.0:.2f}s) | Pos: {action_hovered['pos']}")

                    if video_loaded and video_fps_for_calc > 0:
                        imgui.text(f"Frame: {int(round((action_hovered['at'] / 1000.0) * video_fps_for_calc))}")

                    if hovered_action_idx_current_timeline > 0:
                        dt = action_hovered['at'] - actions_list[hovered_action_idx_current_timeline - 1]['at']

                        speed = (
                            abs(action_hovered['pos'] - actions_list[hovered_action_idx_current_timeline - 1]['pos']) / (dt / 1000.0)
                            if dt > 0 else 0
                        )
                        imgui.text(f"In-Speed: {speed:.1f} pos/s")

                    if hovered_action_idx_current_timeline < len(actions_list) - 1:
                        dt = actions_list[hovered_action_idx_current_timeline + 1]['at'] - action_hovered['at']
                        speed = (
                            abs(actions_list[hovered_action_idx_current_timeline + 1]['pos'] - action_hovered['pos']) / (dt / 1000.0)
                            if dt > 0 else 0
                        )
                        imgui.text(f"Out-Speed: {speed:.1f} pos/s")
                    imgui.end_tooltip()

        # --- Window End ---
        imgui.end()
