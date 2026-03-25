import numpy as np
from typing import Tuple, Optional
from config.constants import ProcessingSpeedMode, DEFAULT_TRACKER_NAME
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE


class AppStateUI:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.app_settings = self.app.app_settings
        defaults = self.app_settings.get_default_settings()

        self.ui_layout_mode = self.app_settings.get("ui_layout_mode", defaults.get("ui_layout_mode", "fixed"))
        self.show_control_panel_window = self.app_settings.get("show_control_panel_window", True)
        # Note: show_video_display_window only applies to floating mode
        # In fixed mode, video display is always shown regardless of this setting
        self.show_video_display_window = self.app_settings.get("show_video_display_window", True)
        self.show_video_navigation_window = self.app_settings.get("show_video_navigation_window", True)
        self.show_info_graphs_window = self.app_settings.get("show_info_graphs_window", True)

        # Window dimensions
        self.window_width = self.app_settings.get("window_width", defaults.get("window_width", 1800))
        self.window_height = self.app_settings.get("window_height", defaults.get("window_height", 1000))

        # Video Zoom and Pan State
        self.video_zoom_factor = 1.0
        self.video_pan_normalized = [0.0, 0.0]  # [pan_x_norm, pan_y_norm] top-left of visible UV
        self.video_pan_step = 0.05  # Percentage of visible video width/height

        # Cached content UV rect (invalidated on video load / settings change)
        self._cached_content_uv = None

        # Status Message
        self.status_message: str = ""
        self.status_message_time: float = 0.0

        # Load last used tracker using dynamic discovery
        _TRACKER_MIGRATION = {
            # Legacy offline → new offline
            "OFFLINE_3_STAGE": "OFFLINE_GUIDED_FLOW",
            "OFFLINE_3_STAGE_MIXED": "OFFLINE_GUIDED_FLOW",
            "OFFLINE_2_STAGE": "OFFLINE_CONTACT_ANALYSIS",
            # Old snake_case live → new SCREAMING_SNAKE
            "oscillation": "LIVE_OSCILLATION",
            "yolo_roi": "LIVE_YOLO_ROI",
            "user_roi": "LIVE_USER_ROI",
            "vr_chapter_flow": "LIVE_VR_CHAPTER_FLOW",
            "vr_focused": "LIVE_VR_FOCUSED",
            # Old legacy names → new LEGACY_ prefix
            "oscillation_legacy": "LEGACY_OSCILLATION",
            "axis_projection_enhanced": "LEGACY_AXIS_PROJECTION_ENHANCED",
            "axis_projection_working": "LEGACY_AXIS_PROJECTION_WORKING",
            "hybrid_intelligence": "LEGACY_HYBRID_INTELLIGENCE",
            "relative_distance": "LEGACY_RELATIVE_DISTANCE",
            # Old community names
            "multi_axis_stress_test": "COMMUNITY_MULTI_AXIS_STRESS_TEST",
        }
        saved_tracker = self.app_settings.get(
            "selected_tracker_name",
            defaults.get("selected_tracker_name", DEFAULT_TRACKER_NAME)
        )
        if saved_tracker in _TRACKER_MIGRATION:
            saved_tracker = _TRACKER_MIGRATION[saved_tracker]
            self.app_settings.set("selected_tracker_name", saved_tracker)
        # Validate saved tracker still exists — fall back to default if not
        try:
            from config.tracker_discovery import TrackerDiscovery
            discovery = TrackerDiscovery()
            if saved_tracker not in discovery.get_all_trackers():
                self.logger.warning(f"Saved tracker '{saved_tracker}' not found, falling back to default")
                saved_tracker = DEFAULT_TRACKER_NAME
                self.app_settings.set("selected_tracker_name", saved_tracker)
        except Exception:
            pass  # Discovery not ready yet, trust the saved value
        self.selected_tracker_name: str = saved_tracker
        # Load saved processing speed mode, default to REALTIME
        saved_speed_mode = self.app_settings.get("selected_processing_speed_mode", "REALTIME")
        try:
            self.selected_processing_speed_mode: ProcessingSpeedMode = ProcessingSpeedMode[saved_speed_mode]
        except KeyError:
            self.selected_processing_speed_mode: ProcessingSpeedMode = ProcessingSpeedMode.REALTIME
        # Slow-motion target FPS (1–30, default 10)
        self.slow_motion_fps: float = self.app_settings.get("slow_motion_fps", 10.0)

        # UI visibility states
        self.show_toolbar = True
        self.show_simulator_3d = self.app_settings.get("show_simulator_3d", defaults.get("show_simulator_3d", True))
        self.show_script_gauge = self.app_settings.get("show_script_gauge", False)
        self.show_plugin_pipeline = False
        self.show_funscript_timeline = self.app_settings.get("show_funscript_timeline", defaults.get("show_funscript_timeline", True))
        self.show_heatmap = self.app_settings.get("show_heatmap", defaults.get("show_heatmap", True))
        self.show_audio_waveform = self.app_settings.get("show_audio_waveform", False)
        self.show_funscript_interactive_timeline = self.app_settings.get("show_funscript_interactive_timeline", defaults.get("show_funscript_interactive_timeline", True))
        self.show_funscript_interactive_timeline2 = self.app_settings.get("show_funscript_interactive_timeline2", defaults.get("show_funscript_interactive_timeline2", False))
        # Extra timeline visibility (supporter feature, default False)
        for _t_num in EXTRA_TIMELINE_RANGE:
            _attr = f"show_funscript_interactive_timeline{_t_num}"
            setattr(self, _attr, self.app_settings.get(_attr, defaults.get(_attr, False)))
        self.show_stage2_overlay = self.app_settings.get("show_stage2_overlay", defaults.get("show_stage2_overlay", True))
        self.show_timeline_editor_buttons = self.app_settings.get("show_timeline_editor_buttons", defaults.get("show_timeline_editor_buttons", False))
        self.show_advanced_options = self.app_settings.get("show_advanced_options", defaults.get("show_advanced_options", False))
        self.show_video_feed = self.app_settings.get("show_video_feed", defaults.get("show_video_feed", True))

        self.show_video_controls_overlay = self.app_settings.get("show_video_controls_overlay", defaults.get("show_video_controls_overlay", True))

        self.show_generated_file_manager = False

        # Visualization Feature Settings
        self.speed_limit_threshold: float = self.app_settings.get("speed_limit_threshold", 400.0)
        self.show_bpm_overlay: bool = self.app_settings.get("show_bpm_overlay", False)

        # FPS Override (manual scripting — snap grid at custom FPS)
        self.fps_override_enabled: bool = False
        self.fps_override_value: float = 60.0

        # Tracker UI visibility flags (app state that tracker reads via app_logic)
        # These will be updated by app_logic based on tracker's actual state if needed,
        # or directly if UI controls them.
        self.ui_show_masks = False
        self.ui_show_flow = False
        self.ui_show_stats_on_video = False
        self.ui_show_funscript_preview_on_video = False

        # InfoGraphsUI sections visibility
        self.show_video_info_section = True
        self.show_video_settings_section = True
        self.show_funscript_info_t1_section = True
        self.show_funscript_info_t2_section = False
        self.show_undo_redo_history_section = True

        # ControlPanelUI sections visibility
        self.show_general_tracking_settings_section = False
        self.show_tracking_control_section = True
        self.show_range_selection_section = True

        self.show_funscript_window_controls_section = True

        # FPS Slider Range (for UI elements)
        self.fps_slider_min_val = 1.0
        self.fps_slider_max_val = 200.0

        # Timeline Interaction Properties
        self.timeline_base_height = 180  # Constant for layout
        self.timeline_zoom_factor_ms_per_px = self.app_settings.get("timeline_zoom_factor_ms_per_px",
                                                                    defaults.get("timeline_zoom_factor_ms_per_px",
                                                                                 20.0))
        self.timeline_pan_offset_ms = self.app_settings.get("timeline_pan_offset_ms",
                                                            defaults.get("timeline_pan_offset_ms", 0.0))
        self.timeline_point_radius = 4.0  # Constant for timeline drawing
        self.snap_to_grid_time_ms = 20  # ms for time snapping
        self.snap_to_grid_pos = 5  # pos units for position snapping
        self.is_manual_panning = False  # Shared by timelines via app

        self.show_auto_post_processing_section = True

        # Preview/Heatmap constants and state for re-render checks by GUI
        self.funscript_preview_draw_height = 50  # For legacy funscript preview bar
        self.timeline_heatmap_height = 15  # For heatmap below interactive timeline
        self.heatmap_texture_fixed_height = self.timeline_heatmap_height
        self.funscript_preview_texture_fixed_height = self.funscript_preview_draw_height

        self.heatmap_dirty = True
        self.last_heatmap_bar_width = 0
        self.last_heatmap_video_duration_s = 0.0
        self.last_heatmap_action_count = -1  # For primary timeline's heatmap

        self.funscript_preview_dirty = True  # For legacy preview bar
        self.last_funscript_preview_bar_width = 0
        self.last_funscript_preview_duration_s = 0.0
        self.last_funscript_preview_action_count = -1

        self.fixed_layout_geometry = {}
        self.just_switched_to_floating = False

        # Timeline sync state with video playback
        self.timeline_interaction_active = False  # True if user is dragging points/selection box in a timeline
        self.last_synced_frame_index_timeline = -1  # Last video frame index timeline was auto-panned to
        self.force_timeline_pan_to_current_frame = False  # Flag to command timeline to pan to current video frame
        self.interactive_refinement_mode_enabled: bool = False

        # Active timeline tracking for shortcuts (which timeline receives number key input)
        # Default to 1 (primary). Updated when user clicks/focuses a timeline.
        self.active_timeline_num: int = 1
        
        # Update Settings Dialog
        self.show_update_settings_dialog = False

    def sync_tracker_ui_flags(self):
        """Ensure AppStateUI flags match the actual tracker state."""
        if self.app.tracker:
            self.ui_show_masks = self.app.tracker.show_all_boxes
            self.ui_show_flow = self.app.tracker.show_flow
            self.ui_show_stats_on_video = self.app.tracker.show_stats
            self.ui_show_funscript_preview_on_video = self.app.tracker.show_funscript_preview
        else:
            self.ui_show_masks = False
            self.ui_show_flow = False
            self.ui_show_stats_on_video = False
            self.ui_show_funscript_preview_on_video = False

    def set_tracker_ui_flag(self, flag_name: str, value: bool):
        """Sets a UI flag and attempts to update the tracker's corresponding property."""
        if flag_name == "show_masks":
            self.ui_show_masks = value
            if self.app.tracker: self.app.tracker.show_all_boxes = value
        elif flag_name == "show_flow":
            self.ui_show_flow = value
            if self.app.tracker: self.app.tracker.show_flow = value
        elif flag_name == "show_stats_on_video":
            self.ui_show_stats_on_video = value
            if self.app.tracker: self.app.tracker.show_stats = value
        elif flag_name == "show_funscript_preview_on_video":
            self.ui_show_funscript_preview_on_video = value
            if self.app.tracker: self.app.tracker.show_funscript_preview = value
        else:
            self.logger.warning(f"Attempted to set unknown tracker UI flag: {flag_name}")
        self.app.energy_saver.reset_activity_timer()

    def adjust_video_zoom(self, zoom_multiplier: float, mouse_pos_normalized: Optional[Tuple[float, float]] = None):
        old_zoom = self.video_zoom_factor
        self.video_zoom_factor *= zoom_multiplier
        self.video_zoom_factor = max(1.0, min(self.video_zoom_factor, 10.0))  # Clamp zoom factor

        if abs(old_zoom - self.video_zoom_factor) < 1e-6: return  # No change

        self.app.energy_saver.reset_activity_timer()

        # Target UV dimensions based on new zoom
        uv_target_width = 1.0 / self.video_zoom_factor
        uv_target_height = 1.0 / self.video_zoom_factor

        if self.video_zoom_factor == 1.0:
            self.video_pan_normalized = [0.0, 0.0]
        else:
            if mouse_pos_normalized:
                # Calculate where the mouse pointer was in the texture's UV space before zoom
                # mouse_pos_normalized is (u,v) relative to the displayed video panel
                # old_uv_x, old_uv_y are the top-left corner of the *previous* zoomed view in texture space
                old_uv_x, old_uv_y = self.video_pan_normalized
                old_uv_w = 1.0 / old_zoom
                old_uv_h = 1.0 / old_zoom

                # Mouse position in texture's full UV space (0-1)
                tex_mouse_u = old_uv_x + mouse_pos_normalized[0] * old_uv_w
                tex_mouse_v = old_uv_y + mouse_pos_normalized[1] * old_uv_h

                # New top-left corner to keep tex_mouse_u, tex_mouse_v at the same relative spot
                # mouse_pos_normalized[0] = (tex_mouse_u - new_uv_x) / new_uv_w
                # new_uv_x = tex_mouse_u - mouse_pos_normalized[0] * new_uv_w
                self.video_pan_normalized[0] = tex_mouse_u - mouse_pos_normalized[0] * uv_target_width
                self.video_pan_normalized[1] = tex_mouse_v - mouse_pos_normalized[1] * uv_target_height
            else:  # If no mouse pos, zoom relative to center of current view
                current_center_x = self.video_pan_normalized[0] + (1.0 / old_zoom) / 2.0
                current_center_y = self.video_pan_normalized[1] + (1.0 / old_zoom) / 2.0
                self.video_pan_normalized[0] = current_center_x - uv_target_width / 2.0
                self.video_pan_normalized[1] = current_center_y - uv_target_height / 2.0

            # Clamp panning to ensure the view stays within the 0-1 UV bounds
            self.video_pan_normalized[0] = np.clip(self.video_pan_normalized[0], 0.0, 1.0 - uv_target_width)
            self.video_pan_normalized[1] = np.clip(self.video_pan_normalized[1], 0.0, 1.0 - uv_target_height)

        self.app.project_manager.project_dirty = True

    def reset_video_zoom_pan(self):
        self.video_zoom_factor = 1.0
        self.video_pan_normalized = [0.0, 0.0]
        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def pan_video_normalized_delta(self, dx_norm_view: float, dy_norm_view: float):
        """Pans the video by a delta normalized to the current view dimensions."""
        if self.video_zoom_factor <= 1.0:
            return

        # Convert view-normalized delta to texture-normalized delta
        # A full pan of the view (dx_norm_view = 1.0) corresponds to moving
        # by (1.0 / self.video_zoom_factor) in the texture's UV space.
        dx_tex = dx_norm_view * (1.0 / self.video_zoom_factor)
        dy_tex = dy_norm_view * (1.0 / self.video_zoom_factor)

        uv_width = 1.0 / self.video_zoom_factor
        uv_height = 1.0 / self.video_zoom_factor

        self.video_pan_normalized[0] = np.clip(self.video_pan_normalized[0] + dx_tex, 0.0, 1.0 - uv_width)
        self.video_pan_normalized[1] = np.clip(self.video_pan_normalized[1] + dy_tex, 0.0, 1.0 - uv_height)
        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def invalidate_content_uv_cache(self):
        """Call when video changes (load, settings reapply) to recompute content UV rect."""
        self._cached_content_uv = None

    def get_content_uv_rect(self) -> Tuple[float, float, float, float]:
        """Returns (uv_left, uv_top, uv_right, uv_bottom) of actual video content within the texture.

        Cached per-video — only recomputes when invalidated (video load/settings change).
        """
        if self._cached_content_uv is not None:
            return self._cached_content_uv

        result = self._compute_content_uv_rect()
        self._cached_content_uv = result
        return result

    def _compute_content_uv_rect(self) -> Tuple[float, float, float, float]:
        """Compute content UV rect (uncached)."""
        proc = self.app.processor
        if not proc or not proc.video_info:
            return (0.0, 0.0, 1.0, 1.0)

        # HD mode: display frame is already at native AR, no padding
        if proc.is_hd_active:
            return (0.0, 0.0, 1.0, 1.0)

        # VR videos fill the entire square (v360 / GPU unwarp / crop+scale)
        if proc.determined_video_type == 'VR':
            return (0.0, 0.0, 1.0, 1.0)

        # Preprocessed videos have no padding
        if hasattr(proc, '_is_using_preprocessed_video') and proc._is_using_preprocessed_video():
            return (0.0, 0.0, 1.0, 1.0)

        width = proc.video_info.get('width', 0)
        height = proc.video_info.get('height', 0)
        if width <= 0 or height <= 0:
            return (0.0, 0.0, 1.0, 1.0)

        aspect = width / height
        size = proc.yolo_input_size

        if 0.95 < aspect < 1.05:
            return (0.0, 0.0, 1.0, 1.0)  # Square — no padding
        elif aspect > 1.05:
            # Landscape — padded top/bottom
            scaled_h = size / aspect
            scaled_h = int(scaled_h) & ~1
            scaled_h = min(scaled_h, size)
            pad = (size - scaled_h) / 2.0
            return (0.0, pad / size, 1.0, (size - pad) / size)
        else:
            # Portrait — padded left/right
            scaled_w = size * aspect
            scaled_w = int(scaled_w) & ~1
            scaled_w = min(scaled_w, size)
            pad = (size - scaled_w) / 2.0
            return (pad / size, 0.0, (size - pad) / size, 1.0)

    def calculate_video_display_dimensions(self, available_w: float, available_h: float) -> Tuple[
        float, float, float, float]:
        if not (self.app.processor and self.app.processor.current_frame is not None and \
                self.app.processor.current_frame.shape[0] > 0 and self.app.processor.current_frame.shape[1] > 0):
            return 0, 0, 0, 0  # width, height, offset_x, offset_y

        frame_h_orig, frame_w_orig = self.app.processor.current_frame.shape[:2]
        if frame_h_orig == 0 or frame_w_orig == 0: return 0, 0, 0, 0

        # Use content aspect ratio (without black padding) for display sizing
        c_left, c_top, c_right, c_bottom = self.get_content_uv_rect()
        content_w_frac = c_right - c_left
        content_h_frac = c_bottom - c_top
        if content_w_frac <= 0 or content_h_frac <= 0:
            return 0, 0, 0, 0
        aspect_ratio = (frame_w_orig * content_w_frac) / (frame_h_orig * content_h_frac)

        # Calculate display dimensions maintaining content aspect ratio
        display_w = available_w
        display_h = display_w / aspect_ratio

        if display_h > available_h:
            display_h = available_h
            display_w = display_h * aspect_ratio

        # Calculate offsets to center the video
        offset_x = (available_w - display_w) / 2
        offset_y = (available_h - display_h) / 2

        return display_w, display_h, offset_x, offset_y

    def get_video_uv_coords(self) -> Tuple[float, float, float, float]:
        """Returns (uv0_x, uv0_y, uv1_x, uv1_y) for the zoomed/panned video texture.

        Maps pan/zoom through the content UV rect so only actual video content
        is displayed (black padding from 640x640 scaling is cropped out).
        """
        c_left, c_top, c_right, c_bottom = self.get_content_uv_rect()
        c_w = c_right - c_left
        c_h = c_bottom - c_top

        # Pan/zoom operate in content-relative [0,1] space
        uv_span_x = c_w / self.video_zoom_factor
        uv_span_y = c_h / self.video_zoom_factor

        uv0_x = c_left + self.video_pan_normalized[0] * c_w
        uv0_y = c_top + self.video_pan_normalized[1] * c_h
        uv1_x = uv0_x + uv_span_x
        uv1_y = uv0_y + uv_span_y
        return uv0_x, uv0_y, uv1_x, uv1_y

    def update_current_script_display_values(self):
        """Updates gauge and L/R dial values based on current video time and funscript data."""
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript and \
                self.app.processor.video_info and self.app.processor.current_frame_index >= 0:

            fps = self.app.processor.video_info.get('fps', 30.0)
            if fps <= 0: fps = 30.0  # Avoid division by zero or negative FPS

            current_time_ms = int(round((self.app.processor.current_frame_index / fps) * 1000.0))

            script_val_primary = self.app.processor.tracker.funscript.get_value(current_time_ms, axis='primary')
            self.script_position_t1 = float(script_val_primary)

            script_val_secondary = self.app.processor.tracker.funscript.get_value(current_time_ms, axis='secondary')
            self.script_position_t2 = float(script_val_secondary)
        else:
            self.script_position_t1 = 0.0
            self.script_position_t2 = 0.0

    def update_settings_from_app(self):
        """Called by AppLogic when settings are loaded or project is loaded."""
        defaults = self.app_settings.get_default_settings()
        self.window_width = self.app_settings.get("window_width", defaults.get("window_width", self.window_width))
        self.window_height = self.app_settings.get("window_height", defaults.get("window_height", self.window_height))

        self.timeline_zoom_factor_ms_per_px = self.app_settings.get("timeline_zoom_factor_ms_per_px", defaults.get("timeline_zoom_factor_ms_per_px", self.timeline_zoom_factor_ms_per_px))
        self.timeline_pan_offset_ms = self.app_settings.get("timeline_pan_offset_ms", defaults.get("timeline_pan_offset_ms", self.timeline_pan_offset_ms))

        self.show_funscript_interactive_timeline = self.app_settings.get("show_funscript_interactive_timeline", defaults.get("show_funscript_interactive_timeline", self.show_funscript_interactive_timeline))
        self.show_funscript_interactive_timeline2 = self.app_settings.get("show_funscript_interactive_timeline2", defaults.get("show_funscript_interactive_timeline2", self.show_funscript_interactive_timeline2))
        for _t_num in EXTRA_TIMELINE_RANGE:
            _attr = f"show_funscript_interactive_timeline{_t_num}"
            setattr(self, _attr, self.app_settings.get(_attr, defaults.get(_attr, getattr(self, _attr, False))))
        self.show_funscript_timeline = self.app_settings.get("show_funscript_timeline", defaults.get("show_funscript_timeline", self.show_funscript_timeline))
        self.show_heatmap = self.app_settings.get("show_heatmap", defaults.get("show_heatmap", self.show_heatmap))
        self.show_stage2_overlay = self.app_settings.get("show_stage2_overlay", defaults.get("show_stage2_overlay", self.show_stage2_overlay))
        self.show_timeline_editor_buttons = self.app_settings.get("show_timeline_editor_buttons", defaults.get("show_timeline_editor_buttons", self.show_timeline_editor_buttons))
        self.show_advanced_options = self.app_settings.get("show_advanced_options", defaults.get("show_advanced_options", self.show_advanced_options))

        self.show_audio_waveform = False

        self.full_width_nav = self.app_settings.get("full_width_nav", defaults.get("full_width_nav", True))

        self.ui_layout_mode = self.app_settings.get("ui_layout_mode", defaults.get("ui_layout_mode", self.ui_layout_mode))
        self.show_control_panel_window = self.app_settings.get("show_control_panel_window", self.show_control_panel_window)
        self.show_video_display_window = self.app_settings.get("show_video_display_window", self.show_video_display_window)
        self.show_video_navigation_window = self.app_settings.get("show_video_navigation_window", self.show_video_navigation_window)
        self.show_info_graphs_window = self.app_settings.get("show_info_graphs_window", self.show_info_graphs_window)

        # If project data has specific values, they override app_settings
        if hasattr(self.app, 'project_data_on_load') and self.app.project_data_on_load:
            project_data = self.app.project_data_on_load
            self.ui_layout_mode = project_data.get("ui_layout_mode", self.ui_layout_mode)
            self.timeline_pan_offset_ms = project_data.get("timeline_pan_offset_ms", self.timeline_pan_offset_ms)

        # If project data has specific values, they override app_settings
        if hasattr(self.app, 'project_data_on_load') and self.app.project_data_on_load:
            project_data = self.app.project_data_on_load
            self.timeline_pan_offset_ms = project_data.get("timeline_pan_offset_ms", self.timeline_pan_offset_ms)
            self.timeline_zoom_factor_ms_per_px = project_data.get("timeline_zoom_factor_ms_per_px", self.timeline_zoom_factor_ms_per_px)
            self.show_funscript_interactive_timeline = project_data.get("show_funscript_interactive_timeline", self.show_funscript_interactive_timeline)
            self.show_funscript_interactive_timeline2 = project_data.get("show_funscript_interactive_timeline2", self.show_funscript_interactive_timeline2)
            for _t_num in EXTRA_TIMELINE_RANGE:
                _attr = f"show_funscript_interactive_timeline{_t_num}"
                setattr(self, _attr, project_data.get(_attr, getattr(self, _attr, False)))
            self.show_heatmap = project_data.get("show_heatmap", self.show_heatmap)
            self.show_stage2_overlay = project_data.get("show_stage2_overlay", self.show_stage2_overlay)
            self.interactive_refinement_mode_enabled = project_data.get("interactive_refinement_mode_enabled", False)

        self.sync_tracker_ui_flags()

    def save_settings_to_app(self):
        """Called by AppLogic when settings are to be saved."""
        self.app_settings.set("window_width", int(self.window_width))
        self.app_settings.set("window_height", int(self.window_height))
        self.app_settings.set("timeline_zoom_factor_ms_per_px", self.timeline_zoom_factor_ms_per_px)
        self.app_settings.set("timeline_pan_offset_ms", self.timeline_pan_offset_ms)

        self.app_settings.set("ui_layout_mode", self.ui_layout_mode)
        self.app_settings.set("show_control_panel_window", self.show_control_panel_window)
        self.app_settings.set("show_video_display_window", self.show_video_display_window)
        self.app_settings.set("show_video_navigation_window", self.show_video_navigation_window)
        self.app_settings.set("show_info_graphs_window", self.show_info_graphs_window)


        self.app_settings.set("show_funscript_interactive_timeline", self.show_funscript_interactive_timeline)
        self.app_settings.set("show_funscript_interactive_timeline2", self.show_funscript_interactive_timeline2)
        for _t_num in EXTRA_TIMELINE_RANGE:
            _attr = f"show_funscript_interactive_timeline{_t_num}"
            self.app_settings.set(_attr, getattr(self, _attr, False))
        self.app_settings.set("show_funscript_timeline", self.show_funscript_timeline)  # Legacy
        self.app_settings.set("show_heatmap", self.show_heatmap)
        self.app_settings.set("show_stage2_overlay", self.show_stage2_overlay)
        self.app_settings.set("show_timeline_editor_buttons", self.show_timeline_editor_buttons)
        self.app_settings.set("show_advanced_options", self.show_advanced_options)
        self.app_settings.set("show_video_feed", self.show_video_feed)
        self.app_settings.set("show_video_controls_overlay", self.show_video_controls_overlay)
        self.app_settings.set("show_script_gauge", self.show_script_gauge)

        self.app_settings.set("interactive_refinement_mode_enabled", self.interactive_refinement_mode_enabled)
        self.app_settings.set("selected_processing_speed_mode", self.selected_processing_speed_mode.name)
        self.app_settings.set("slow_motion_fps", self.slow_motion_fps)
