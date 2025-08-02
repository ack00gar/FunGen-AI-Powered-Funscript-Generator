import numpy as np
from typing import Tuple, Optional
from config.constants import TrackerMode

class AppStateUI:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.app_settings = self.app.app_settings
        defaults = self.app_settings.get_default_settings()

        self.ui_layout_mode = self.app_settings.get("ui_layout_mode", defaults.get("ui_layout_mode", "fixed"))
        self.show_control_panel_window = self.app_settings.get("show_control_panel_window", True)
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

        # Status Message
        self.status_message: str = ""
        self.status_message_time: float = 0.0

        self.selected_tracker_mode: TrackerMode = TrackerMode.LIVE_YOLO_ROI

        # UI visibility states
        self.show_lr_dial_graph = self.app_settings.get("show_lr_dial_graph", defaults.get("show_lr_dial_graph", True))
        self.show_funscript_timeline = self.app_settings.get("show_funscript_timeline", defaults.get("show_funscript_timeline", True))  # Legacy preview
        self.show_gauge = self.app_settings.get("show_gauge_window", defaults.get("show_gauge_window", True))
        self.show_heatmap = self.app_settings.get("show_heatmap", defaults.get("show_heatmap", False))
        self.show_funscript_interactive_timeline = self.app_settings.get("show_funscript_interactive_timeline", defaults.get("show_funscript_interactive_timeline", True))
        self.show_funscript_interactive_timeline2 = self.app_settings.get("show_funscript_interactive_timeline2", defaults.get("show_funscript_interactive_timeline2", False))
        self.show_stage2_overlay = self.app_settings.get("show_stage2_overlay", defaults.get("show_stage2_overlay", True))

        self.show_generated_file_manager = False

        # Tracker UI visibility flags (app state that tracker reads via app_logic)
        # These will be updated by app_logic based on tracker's actual state if needed,
        # or directly if UI controls them.
        self.ui_show_masks = False
        self.ui_show_flow = False

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
        self.show_funscript_processing_tools_section = False
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
        self.funscript_preview_draw_height = 30  # For legacy funscript preview bar
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

        # Gauge Window Attributes
        self.show_gauge_window_timeline1 = self.app_settings.get("show_gauge_window_timeline1",
                                                                 defaults.get("show_gauge_window_timeline1", True))
        self.show_gauge_window_timeline2 = self.app_settings.get("show_gauge_window_timeline2",
                                                                 defaults.get("show_gauge_window_timeline2", False))

        default_gauge_w = self.app_settings.get("gauge_window_size_w", defaults.get("gauge_window_size_w", 100))
        default_gauge_h = self.app_settings.get("gauge_window_size_h", defaults.get("gauge_window_size_h", 220))
        # Default Y needs main menu bar height, which is usually set after GUI init.
        # AppLogic will call initialize_gauge_default_y once GUI provides that height.
        menu_bar_h_for_default = self.app_settings.get("main_menu_bar_height_for_gauge_default_y", 25)
        default_gauge_x = self.window_width - default_gauge_w - 20
        default_gauge_y = menu_bar_h_for_default + 10

        # Timeline 1 Gauge
        self.gauge_window_pos_t1 = (
            self.app_settings.get("gauge_window_pos_t1_x", default_gauge_x),
            self.app_settings.get("gauge_window_pos_t1_y", default_gauge_y)
        )
        self.gauge_window_size_t1 = (
            self.app_settings.get("gauge_window_size_t1_w", default_gauge_w),
            self.app_settings.get("gauge_window_size_t1_h", default_gauge_h)
        )

        # Timeline 2 Gauge (with staggered default)
        default_gauge_t2_x = default_gauge_x - default_gauge_w - 10 # Place it to the left of T1
        self.gauge_window_pos_t2 = (
            self.app_settings.get("gauge_window_pos_t2_x", default_gauge_t2_x),
            self.app_settings.get("gauge_window_pos_t2_y", default_gauge_y)
        )
        self.gauge_window_size_t2 = (
            self.app_settings.get("gauge_window_size_t2_w", default_gauge_w),
            self.app_settings.get("gauge_window_size_t2_h", default_gauge_h)
        )

        self.gauge_value_t1 = 0.0  # Live value from script for T1
        self.gauge_value_t2 = 0.0  # Live value from script for T2
        self.gauge_pos_initialized = False

        # L/R Dial Window Attributes
        self.show_lr_dial_graph = self.app_settings.get("show_lr_dial_graph", defaults.get("show_lr_dial_graph", True))
        default_lr_dial_size_w = self.app_settings.get("lr_dial_window_size_w",
                                                       defaults.get("lr_dial_window_size_w", 150))
        default_lr_dial_size_h = self.app_settings.get("lr_dial_window_size_h",
                                                       defaults.get("lr_dial_window_size_h", 180))
        gauge_w_for_lr_default = self.app_settings.get("gauge_window_size_w", defaults.get("gauge_window_size_w", 100))
        default_lr_dial_x = self.window_width - gauge_w_for_lr_default - default_lr_dial_size_w - 30
        default_lr_dial_y = menu_bar_h_for_default + 10

        self.lr_dial_window_pos = (
            self.app_settings.get("lr_dial_window_pos_x", default_lr_dial_x),
            self.app_settings.get("lr_dial_window_pos_y", default_lr_dial_y)
        )
        self.lr_dial_window_size = (default_lr_dial_size_w, default_lr_dial_size_h)
        self.lr_dial_value = 50.0  # Live value from script (0-100, default 50)

        self.fixed_layout_geometry = {}
        self.just_switched_to_floating = False

        # Timeline sync state with video playback
        self.timeline_interaction_active = False  # True if user is dragging points/selection box in a timeline
        self.last_synced_frame_index_timeline = -1  # Last video frame index timeline was auto-panned to
        self.force_timeline_pan_to_current_frame = False  # Flag to command timeline to pan to current video frame
        self.interactive_refinement_mode_enabled: bool = False
        
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

    def initialize_gauge_default_y(self, menu_bar_height: float):
        # This method is called by AppLogic once the GUI main menu bar height is known.
        if not self.gauge_pos_initialized:
            defaults = self.app_settings.get_default_settings()

            # Update T1 Gauge Y
            current_default_t1_y = self.app_settings.get("gauge_window_pos_t1_y", defaults.get("gauge_window_pos_y", 35))
            if self.gauge_window_pos_t1[1] == current_default_t1_y or self.gauge_window_pos_t1[1] <= menu_bar_height:
                self.gauge_window_pos_t1 = (self.gauge_window_pos_t1[0], menu_bar_height + 10)

            # Update T2 Gauge Y
            current_default_t2_y = self.app_settings.get("gauge_window_pos_t2_y", defaults.get("gauge_window_pos_y", 35))
            if self.gauge_window_pos_t2[1] == current_default_t2_y or self.gauge_window_pos_t2[1] <= menu_bar_height:
                self.gauge_window_pos_t2 = (self.gauge_window_pos_t2[0], menu_bar_height + 10)

            current_default_lr_dial_y_setting = self.app_settings.get("lr_dial_window_pos_y", defaults.get("lr_dial_window_pos_y", 35))
            if self.lr_dial_window_pos[1] == current_default_lr_dial_y_setting or self.lr_dial_window_pos[
                1] <= menu_bar_height:
                self.lr_dial_window_pos = (self.lr_dial_window_pos[0], menu_bar_height + 10)

            self.gauge_pos_initialized = True

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

    def calculate_video_display_dimensions(self, available_w: float, available_h: float) -> Tuple[
        float, float, float, float]:
        if not (self.app.processor and self.app.processor.current_frame is not None and \
                self.app.processor.current_frame.shape[0] > 0 and self.app.processor.current_frame.shape[1] > 0):
            return 0, 0, 0, 0  # width, height, offset_x, offset_y

        frame_h_orig, frame_w_orig = self.app.processor.current_frame.shape[:2]
        if frame_h_orig == 0 or frame_w_orig == 0: return 0, 0, 0, 0
        aspect_ratio = frame_w_orig / frame_h_orig

        # Calculate display dimensions maintaining aspect ratio
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
        """Returns (uv0_x, uv0_y, uv1_x, uv1_y) for the zoomed/panned video texture."""
        uv0_x = self.video_pan_normalized[0]
        uv0_y = self.video_pan_normalized[1]
        uv_width = 1.0 / self.video_zoom_factor
        uv_height = 1.0 / self.video_zoom_factor
        uv1_x = uv0_x + uv_width
        uv1_y = uv0_y + uv_height
        return uv0_x, uv0_y, uv1_x, uv1_y

    def update_current_script_display_values(self):
        """Updates gauge and L/R dial values based on current video time and funscript data."""
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript and \
                self.app.processor.video_info and self.app.processor.current_frame_index >= 0:

            fps = self.app.processor.video_info.get('fps', 30.0)
            if fps <= 0: fps = 30.0  # Avoid division by zero or negative FPS

            current_time_ms = int(round((self.app.processor.current_frame_index / fps) * 1000.0))

            script_val_primary = self.app.processor.tracker.funscript.get_value(current_time_ms, axis='primary')
            self.gauge_value_t1 = float(script_val_primary)

            script_val_secondary = self.app.processor.tracker.funscript.get_value(current_time_ms, axis='secondary')
            self.gauge_value_t2 = float(script_val_secondary)
            self.lr_dial_value = float(script_val_secondary)
        else:
            self.gauge_value_t1 = 0.0
            self.gauge_value_t2 = 0.0
            self.lr_dial_value = 50.0

    def update_settings_from_app(self):
        """Called by AppLogic when settings are loaded or project is loaded."""
        defaults = self.app_settings.get_default_settings()
        self.window_width = self.app_settings.get("window_width", defaults.get("window_width", self.window_width))
        self.window_height = self.app_settings.get("window_height", defaults.get("window_height", self.window_height))

        self.timeline_zoom_factor_ms_per_px = self.app_settings.get("timeline_zoom_factor_ms_per_px", defaults.get("timeline_zoom_factor_ms_per_px", self.timeline_zoom_factor_ms_per_px))
        self.timeline_pan_offset_ms = self.app_settings.get("timeline_pan_offset_ms", defaults.get("timeline_pan_offset_ms", self.timeline_pan_offset_ms))

        self.show_funscript_interactive_timeline = self.app_settings.get("show_funscript_interactive_timeline", defaults.get("show_funscript_interactive_timeline", self.show_funscript_interactive_timeline))
        self.show_funscript_interactive_timeline2 = self.app_settings.get("show_funscript_interactive_timeline2", defaults.get("show_funscript_interactive_timeline2", self.show_funscript_interactive_timeline2))
        self.show_funscript_timeline = self.app_settings.get("show_funscript_timeline", defaults.get("show_funscript_timeline", self.show_funscript_timeline))
        self.show_heatmap = self.app_settings.get("show_heatmap", defaults.get("show_heatmap", self.show_heatmap))
        self.show_stage2_overlay = self.app_settings.get("show_stage2_overlay", defaults.get("show_stage2_overlay", self.show_stage2_overlay))

        self.show_audio_waveform = False

        self.ui_view_mode = self.app_settings.get("ui_view_mode", defaults.get("ui_view_mode", "simple"))
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

        self.show_gauge_window_timeline1 = self.app_settings.get("show_gauge_window_timeline1",
                                                                 defaults.get("show_gauge_window_timeline1",
                                                                              self.show_gauge_window_timeline1))
        self.show_gauge_window_timeline2 = self.app_settings.get("show_gauge_window_timeline2",
                                                                 defaults.get("show_gauge_window_timeline2",
                                                                              self.show_gauge_window_timeline2))

        default_gauge_w = self.app_settings.get("gauge_window_size_w", defaults.get("gauge_window_size_w", 100))
        default_gauge_h = self.app_settings.get("gauge_window_size_h", defaults.get("gauge_window_size_h", 220))
        menu_bar_h_for_default = self.app_settings.get("main_menu_bar_height_for_gauge_default_y", 25)
        default_gauge_x = self.window_width - default_gauge_w - 20
        default_gauge_y = menu_bar_h_for_default + 10

        self.gauge_window_pos_t1 = (
            self.app_settings.get("gauge_window_pos_t1_x", default_gauge_x),
            self.app_settings.get("gauge_window_pos_t1_y", default_gauge_y)
        )
        self.gauge_window_size_t1 = (
            self.app_settings.get("gauge_window_size_t1_w", default_gauge_w),
            self.app_settings.get("gauge_window_size_t1_h", default_gauge_h)
        )

        default_gauge_t2_x = default_gauge_x - default_gauge_w - 10
        self.gauge_window_pos_t2 = (
            self.app_settings.get("gauge_window_pos_t2_x", default_gauge_t2_x),
            self.app_settings.get("gauge_window_pos_t2_y", default_gauge_y)
        )
        self.gauge_window_size_t2 = (
            self.app_settings.get("gauge_window_size_t2_w", default_gauge_w),
            self.app_settings.get("gauge_window_size_t2_h", default_gauge_h)
        )

        self.show_lr_dial_graph = self.app_settings.get("show_lr_dial_graph", defaults.get("show_lr_dial_graph", self.show_lr_dial_graph))
        default_lr_dial_w = self.app_settings.get("lr_dial_window_size_w", defaults.get("lr_dial_window_size_w", 150))
        default_lr_dial_h = self.app_settings.get("lr_dial_window_size_h", defaults.get("lr_dial_window_size_h", 180))
        gauge_w_curr = self.gauge_window_size_t1[0]  # Use current gauge width for positioning

        calc_def_lr_dial_x = self.window_width - gauge_w_curr - default_lr_dial_w - 30
        default_lr_dial_y = menu_bar_h_for_default + 10

        self.lr_dial_window_pos = (
            self.app_settings.get("lr_dial_window_pos_x", calc_def_lr_dial_x),
            self.app_settings.get("lr_dial_window_pos_y", default_lr_dial_y)
        )
        self.lr_dial_window_size = (default_lr_dial_w, default_lr_dial_h)

        # If project data has specific values, they override app_settings
        if hasattr(self.app, 'project_data_on_load') and self.app.project_data_on_load:
            project_data = self.app.project_data_on_load
            self.timeline_pan_offset_ms = project_data.get("timeline_pan_offset_ms", self.timeline_pan_offset_ms)
            self.timeline_zoom_factor_ms_per_px = project_data.get("timeline_zoom_factor_ms_per_px", self.timeline_zoom_factor_ms_per_px)
            self.show_funscript_interactive_timeline = project_data.get("show_funscript_interactive_timeline", self.show_funscript_interactive_timeline)
            self.show_funscript_interactive_timeline2 = project_data.get("show_funscript_interactive_timeline2", self.show_funscript_interactive_timeline2)
            self.show_lr_dial_graph = project_data.get("show_lr_dial_graph", self.show_lr_dial_graph)
            self.show_heatmap = project_data.get("show_heatmap", self.show_heatmap)
            self.show_gauge_window_timeline1 = project_data.get("show_gauge_window_timeline1",
                                                                self.show_gauge_window_timeline1)
            self.show_gauge_window_timeline2 = project_data.get("show_gauge_window_timeline2",
                                                                self.show_gauge_window_timeline2)
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
        self.app_settings.set("show_funscript_timeline", self.show_funscript_timeline)  # Legacy
        self.app_settings.set("show_heatmap", self.show_heatmap)
        self.app_settings.set("show_stage2_overlay", self.show_stage2_overlay)

        self.app_settings.set("show_gauge_window_timeline1", self.show_gauge_window_timeline1)
        self.app_settings.set("show_gauge_window_timeline2", self.show_gauge_window_timeline2)

        self.app_settings.set("gauge_window_pos_t1_x", int(self.gauge_window_pos_t1[0]))
        self.app_settings.set("gauge_window_pos_t1_y", int(self.gauge_window_pos_t1[1]))
        self.app_settings.set("gauge_window_size_t1_w", int(self.gauge_window_size_t1[0]))
        self.app_settings.set("gauge_window_size_t1_h", int(self.gauge_window_size_t1[1]))

        self.app_settings.set("gauge_window_pos_t2_x", int(self.gauge_window_pos_t2[0]))
        self.app_settings.set("gauge_window_pos_t2_y", int(self.gauge_window_pos_t2[1]))
        self.app_settings.set("gauge_window_size_t2_w", int(self.gauge_window_size_t2[0]))
        self.app_settings.set("gauge_window_size_t2_h", int(self.gauge_window_size_t2[1]))

        self.app_settings.set("show_lr_dial_graph", self.show_lr_dial_graph)
        self.app_settings.set("lr_dial_window_pos_x", int(self.lr_dial_window_pos[0]))
        self.app_settings.set("lr_dial_window_pos_y", int(self.lr_dial_window_pos[1]))
        self.app_settings.set("lr_dial_window_size_w", int(self.lr_dial_window_size[0]))
        self.app_settings.set("lr_dial_window_size_h", int(self.lr_dial_window_size[1]))

        self.app_settings.set("interactive_refinement_mode_enabled", self.interactive_refinement_mode_enabled)
