import imgui
import os
import config
from application.utils import get_icon_texture_manager, primary_button_style, destructive_button_style

# Import dynamic tracker discovery
try:
    from .dynamic_tracker_ui import DynamicTrackerUI
    from config.tracker_discovery import get_tracker_discovery, TrackerCategory
except ImportError:
    DynamicTrackerUI = None
    TrackerCategory = None

def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)

class _DisabledScope:
    __slots__ = ("active",)

    def __init__(self, active):
        self.active = active
        if active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()


def _readonly_input(label_id, value, width=-1):
    if width is not None and width >= 0:
        imgui.push_item_width(width)
    imgui.input_text(label_id, value or "Not set", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
    if width is not None and width >= 0:
        imgui.pop_item_width()


class ControlPanelUI:
    __slots__ = (
        "app",
        "timeline_editor1",
        "timeline_editor2",
        "ControlPanelColors",
        "GeneralColors",
        "constants",
        "AI_modelExtensionsFilter",
        "AI_modelTooltipExtensions",
        "tracker_ui",
        # Performance optimization attributes
        "_last_tab_hash",
        "_cached_tab_content", 
        "_widget_visibility_cache",
        "_update_throttle_counter",
        "_heavy_operation_frame_skip",
        # Device Control attributes (supporter feature)
        "device_manager",
        "param_manager", 
        "_device_control_initialized",
        "_first_frame_rendered",
        "device_list",
        "_available_osr_ports",
        "_osr_scan_performed",
        # Buttplug UI state attributes
        "_discovered_buttplug_devices",
        "_buttplug_discovery_performed",
        # Bridge attributes for live control
        "video_playback_bridge",
        "live_tracker_bridge",
        # Streamer attributes (supporter feature)
        "_native_sync_manager",
        "_prev_client_count",
        "_native_sync_status_cache",
        "_native_sync_status_time",
        # Local video browser attributes
        "_local_video_folder",
        "_local_video_files",
        "_selected_local_video",
    )

    def __init__(self, app):
        self.app = app
        self.timeline_editor1 = None
        self.timeline_editor2 = None
        self.ControlPanelColors = config.ControlPanelColors
        self.GeneralColors = config.GeneralColors
        
        # PERFORMANCE OPTIMIZATIONS: Smart rendering and caching
        self._last_tab_hash = None  # Track tab changes
        self._cached_tab_content = {}  # Cache expensive tab rendering
        self._widget_visibility_cache = {}  # Cache widget visibility states
        self._update_throttle_counter = 0  # Throttle expensive updates
        self._heavy_operation_frame_skip = 0  # Skip frames during heavy ops
        self.constants = config.constants
        self.AI_modelExtensionsFilter = self.constants.AI_MODEL_EXTENSIONS_FILTER
        self.AI_modelTooltipExtensions = self.constants.AI_MODEL_TOOLTIP_EXTENSIONS
        
        # Initialize dynamic tracker UI helper
        self.tracker_ui = None
        self._try_reinitialize_tracker_ui()
        
        # Initialize device control attributes (supporter feature)
        self.device_manager = None
        self.param_manager = None
        self._device_control_initialized = False
        self._first_frame_rendered = False
        self.video_playback_bridge = None  # Video playback bridge for live control
        self.live_tracker_bridge = None    # Live tracker bridge for real-time control
        self.device_list = []  # List of discovered devices
        self._available_osr_ports = []
        self._osr_scan_performed = False
        
        # Buttplug device discovery UI state
        self._discovered_buttplug_devices = []
        self._buttplug_discovery_performed = False

        # Streamer attributes (supporter feature)
        self._native_sync_manager = None
        self._prev_client_count = 0
        self._native_sync_status_cache = None
        self._native_sync_status_time = 0
        
        # Local video browser attributes
        self._local_video_folder = ""
        self._local_video_files = []
        self._selected_local_video = ""

    # ------- Helpers -------
    
    def _try_reinitialize_tracker_ui(self):
        """Try to initialize or reinitialize the dynamic tracker UI."""
        if self.tracker_ui is not None:
            return  # Already initialized
        
        try:
            if DynamicTrackerUI:
                self.tracker_ui = DynamicTrackerUI()
                if hasattr(self.app, 'logger'):
                    self.app.logger.debug("Dynamic tracker UI initialized successfully")
            else:
                if hasattr(self.app, 'logger'):
                    self.app.logger.warning("DynamicTrackerUI class not available (import failed)")
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Failed to initialize dynamic tracker UI: {e}")
            self.tracker_ui = None

    def _is_tracker_category(self, tracker_name: str, category) -> bool:
        """Check if tracker belongs to specific category using dynamic discovery."""
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tracker_name)
        return tracker_info and tracker_info.category == category

    def _is_live_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a live tracker (LIVE or LIVE_INTERVENTION)."""
        from config.tracker_discovery import get_tracker_discovery, TrackerCategory
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tracker_name)
        return tracker_info and tracker_info.category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION]

    def _is_offline_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is an offline tracker."""
        from config.tracker_discovery import TrackerCategory
        return self._is_tracker_category(tracker_name, TrackerCategory.OFFLINE)

    def _is_specific_tracker(self, tracker_name: str, target_name: str) -> bool:
        """Check if tracker matches a specific name."""
        return tracker_name == target_name

    def _tracker_in_list(self, tracker_name: str, target_list: list) -> bool:
        """Check if tracker is in a list of specific tracker names."""
        return tracker_name in target_list

    def _is_stage2_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a 2-stage offline tracker."""
        if not self.tracker_ui:
            # Try to reinitialize if it failed during __init__
            self._try_reinitialize_tracker_ui()
        
        if self.tracker_ui:
            return self.tracker_ui.is_stage2_tracker(tracker_name)
        
        # If still failing, log error but don't crash
        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available, cannot check if '{tracker_name}' is stage2 tracker")
        return False

    def _is_stage3_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a 3-stage offline tracker."""
        if not self.tracker_ui:
            self._try_reinitialize_tracker_ui()
        
        if self.tracker_ui:
            return self.tracker_ui.is_stage3_tracker(tracker_name)
        
        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available, cannot check if '{tracker_name}' is stage3 tracker")
        return False

    def _is_mixed_stage3_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a mixed 3-stage offline tracker."""
        if not self.tracker_ui:
            self._try_reinitialize_tracker_ui()
        
        if self.tracker_ui:
            return self.tracker_ui.is_mixed_stage3_tracker(tracker_name)
        
        if hasattr(self.app, 'logger'):
            self.app.logger.warning(f"Dynamic tracker UI not available, cannot check if '{tracker_name}' is mixed stage3 tracker")
        return False

    def _get_tracker_lists_for_ui(self, simple_mode=False):
        """Get tracker lists for UI combo boxes using dynamic discovery."""
        try:
            if simple_mode:
                # Simple mode: only live trackers
                display_names, internal_names = self.tracker_ui.get_simple_mode_trackers()
            else:
                # Full mode: all trackers
                display_names, internal_names = self.tracker_ui.get_gui_display_list()
            
            # Return display names, internal names, and internal names for tooltip generation
            return display_names, internal_names, internal_names
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Dynamic tracker discovery failed: {e}")
            
            # Return empty lists on failure
            return [], [], []
    
    
    def _generate_combined_tooltip(self, tracker_names):
        """Generate combined tooltip for discovered trackers."""
        if not tracker_names:
            return "No trackers available. Please check your tracker_modules installation."
        
        return self.tracker_ui.get_combined_tooltip(tracker_names)

    def _help_tooltip(self, text):
        if imgui.is_item_hovered():
            imgui.set_tooltip(text)

    def _section_header(self, text, help_text=None):
        imgui.spacing()
        imgui.push_style_color(imgui.COLOR_TEXT, *self.ControlPanelColors.SECTION_HEADER)
        imgui.text(text)
        imgui.pop_style_color()
        if help_text:
            _tooltip_if_hovered(help_text)
        imgui.separator()

    def _status_indicator(self, text, status, help_text=None):
        c = self.ControlPanelColors
        if status == "ready":
            color, icon = c.STATUS_READY, "[OK]"
        elif status == "warning":
            color, icon = c.STATUS_WARNING, "[!]"
        elif status == "error":
            color, icon = c.STATUS_ERROR, "[X]"
        else:
            color, icon = c.STATUS_INFO, "[i]"
        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        imgui.text("%s %s" % (icon, text))
        imgui.pop_style_color()
        if help_text:
            _tooltip_if_hovered(help_text)

    # ------- Model path updates -------

    def _update_detection_model_path(self, path):
        app = self.app
        tracker = app.tracker
        if not path or (tracker and path == tracker.det_model_path):
            return
        app.cached_class_names = None
        app.yolo_detection_model_path_setting = path
        app.app_settings.set("yolo_det_model_path", path)
        app.yolo_det_model_path = path
        app.project_manager.project_dirty = True
        app.logger.info("Detection model path updated to: %s. Reloading models." % path)
        if tracker:
            tracker.det_model_path = path
            tracker._load_models()

    def _update_pose_model_path(self, path):
        app = self.app
        tracker = app.tracker
        if not path or (tracker and path == tracker.pose_model_path):
            return
        app.cached_class_names = None
        app.yolo_pose_model_path_setting = path
        app.app_settings.set("yolo_pose_model_path", path)
        app.yolo_pose_model_path = path
        app.project_manager.project_dirty = True
        app.logger.info("Pose model path updated to: %s. Reloading models." % path)
        if tracker:
            tracker.pose_model_path = path
            tracker._load_models()

    def _update_artifacts_dir_path(self, path):
        app = self.app
        if not path or path == app.pose_model_artifacts_dir_setting:
            return
        app.pose_model_artifacts_dir_setting = path
        app.app_settings.set("pose_model_artifacts_dir", path)
        app.project_manager.project_dirty = True
        app.logger.info("Pose Model Artifacts directory updated to: %s." % path)

    # ------- Main render -------

    def render(self, control_panel_w=None, available_height=None):
        app = self.app
        app_state = app.app_state_ui
        calibration_mgr = app.calibration

        if calibration_mgr.is_calibration_mode_active:
            self._render_calibration_window(calibration_mgr, app_state)
            return

        is_simple_mode = (getattr(app_state, "ui_view_mode", "expert") == "simple")
        if is_simple_mode:
            self._render_simple_mode_ui()
            return

        floating = (app_state.ui_layout_mode == "floating")
        if floating:
            if not getattr(app_state, "show_control_panel_window", True):
                return
            is_open, new_vis = imgui.begin("Control Panel##ControlPanelFloating", closable=True)
            if new_vis != app_state.show_control_panel_window:
                app_state.show_control_panel_window = new_vis
            if not is_open:
                imgui.end()
                return
        else:
            flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE
            imgui.begin("Control Panel##MainControlPanel", flags=flags)

        tab_selected = None
        if imgui.begin_tab_bar("ControlPanelTabs"):
            if imgui.begin_tab_item("Run Control")[0]:
                tab_selected = "run_control"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Configuration")[0]:
                tab_selected = "configuration"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Post-Processing")[0]:
                tab_selected = "post_processing"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Settings")[0]:
                tab_selected = "settings"
                imgui.end_tab_item()
            
            # Device Control tab (supporter feature)
            try:
                from application.utils.feature_detection import is_feature_available
                if is_feature_available("device_control"):
                    if imgui.begin_tab_item("Device Control")[0]:
                        tab_selected = "device_control"
                        imgui.end_tab_item()
            except ImportError:
                pass

            # Streamer tab (supporter feature)
            try:
                from application.utils.feature_detection import is_feature_available
                if is_feature_available("streamer"):
                    if imgui.begin_tab_item("Streamer")[0]:
                        tab_selected = "native_sync"
                        imgui.end_tab_item()
            except ImportError:
                pass

            imgui.end_tab_bar()

        avail = imgui.get_content_region_available()
        imgui.begin_child("TabContentRegion", width=0, height=avail[1], border=False)
        if tab_selected == "run_control":
            self._render_run_control_tab()
        elif tab_selected == "configuration":
            self._render_configuration_tab()
        elif tab_selected == "post_processing":
            self._render_post_processing_tab()
        elif tab_selected == "settings":
            self._render_settings_tab()
        elif tab_selected == "device_control":
            self._render_device_control_tab()
        elif tab_selected == "native_sync":
            self._render_native_sync_tab()
        imgui.end_child()
        imgui.end()

    # ------- Tabs -------

    def _render_simple_mode_ui(self):
        app = self.app
        app_state = app.app_state_ui
        # TrackerMode removed - using dynamic discovery system

        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE
        imgui.begin("FunGen Simple##SimpleControlPanel", flags=flags)

        imgui.text("FunGen Simple Workflow")

        if app_state.show_advanced_options:
            c = self.ControlPanelColors
            imgui.push_style_color(imgui.COLOR_TEXT, *c.STATUS_WARNING)
            imgui.text("[ADV] Advanced Options Enabled")
            imgui.pop_style_color()
            imgui.same_line()
            imgui.text_disabled("(View menu > Show Advanced Options)")

        processor = app.processor
        if processor and processor.video_info:
            self._status_indicator(
                "Video loaded",
                "ready",
                "Loaded: %s" % os.path.basename(processor.video_path or "Unknown"),
            )
        else:
            self._status_indicator(
                "Drag & drop a video onto the window",
                "info",
                "Supported formats: MP4, AVI, MOV, MKV",
            )

        imgui.text_wrapped("2. Choose an analysis method below.")
        imgui.text_wrapped("3. Click Start.")
        self._section_header(
            ">> Step 2: Choose Analysis Method",
            "Select the best analysis method for your video content type",
        )

        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor
        if stage_proc.full_analysis_active:
            self._status_indicator("Analysis in progress...", "info", "Processing your video. Please wait.")
        else:
            acts = fs_proc.get_actions("primary")
            if acts:
                self._status_indicator(
                    "Analysis complete - %d points generated" % len(acts),
                    "ready",
                    "Ultimate Autotune preview is visible in timeline. Ready for next step.",
                )
            else:
                self._status_indicator(
                    "Ready to analyze",
                    "info",
                    "Click Start when you're ready to begin analysis",
                )

        # Use dynamic tracker discovery
        modes_display, modes_enum, discovered_trackers = self._get_tracker_lists_for_ui(simple_mode=True)
        try:
            cur_idx = modes_enum.index(app_state.selected_tracker_name)
        except ValueError:
            cur_idx = 0
            # Set default to first available tracker or fallback to default
            from config.constants import DEFAULT_TRACKER_NAME
            default_tracker = modes_enum[cur_idx] if modes_enum else DEFAULT_TRACKER_NAME
            app_state.selected_tracker_name = default_tracker

        imgui.push_item_width(-1)
        clicked, new_idx = imgui.combo(
            "Analysis Method##SimpleTrackerMode", cur_idx, modes_display
        )
        imgui.pop_item_width()
        self._help_tooltip(self._generate_combined_tooltip(discovered_trackers))
        if clicked and new_idx != cur_idx:
            new_mode = modes_enum[new_idx]
            # Clear overlays only when switching modes
            if app_state.selected_tracker_name != new_mode:
                if hasattr(app, 'logger') and app.logger:
                    app.logger.info(f"UI(Simple): Mode change requested {app_state.selected_tracker_name} -> {new_mode}. Clearing overlays.")
                if hasattr(app, 'clear_all_overlays_and_ui_drawings'):
                    app.clear_all_overlays_and_ui_drawings()
            app_state.selected_tracker_name = new_mode
            # Persist user choice (store tracker name directly)
            if hasattr(app, 'app_settings') and hasattr(app.app_settings, 'set'):
                app.app_settings.set("selected_tracker_name", new_mode)

        self._render_execution_progress_display()
        self._render_start_stop_buttons(stage_proc, fs_proc, app.event_handlers)
        imgui.end()

    def _render_processing_speed_controls(self, app_state):
        app = self.app
        processor = app.processor
        selected_mode = app_state.selected_tracker_name
        
        # Always show processing speed controls as they affect basic video playback
        # Check if current tracker is a live mode for tooltip information
        from config.tracker_discovery import get_tracker_discovery, TrackerCategory
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(selected_mode)
        is_live_mode = tracker_info and tracker_info.category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION]
        
        # Update tooltip based on context
        if is_live_mode:
            tooltip = "Control the processing speed for live analysis and video playback"
        else:
            tooltip = "Control the video playback speed"
        
        self._section_header(">> Processing Speed", tooltip)

        current_speed_mode = app_state.selected_processing_speed_mode
 
        if imgui.radio_button("Real Time", current_speed_mode == config.ProcessingSpeedMode.REALTIME):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.REALTIME
        imgui.same_line()
        if imgui.radio_button("Slow-mo", current_speed_mode == config.ProcessingSpeedMode.SLOW_MOTION):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.SLOW_MOTION
        imgui.same_line()
        if imgui.radio_button("Max Speed", current_speed_mode == config.ProcessingSpeedMode.MAX_SPEED):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.MAX_SPEED

    def _render_run_control_tab(self):
        app = self.app
        app_state = app.app_state_ui
        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor
        events = app.event_handlers
        # TrackerMode removed - using dynamic discovery system

        # Ensure this is always defined before any conditional UI blocks use it
        processor = app.processor
        disable_combo = (
            stage_proc.full_analysis_active
            or app.is_setting_user_roi_mode
            or (processor and processor.is_processing and not processor.pause_event.is_set())
        )

        # Use dynamic tracker discovery for full mode
        modes_display_full, modes_enum, discovered_trackers_full = self._get_tracker_lists_for_ui(simple_mode=False)

        open_, _ = imgui.collapsing_header(
            "Choose Analysis Method##SimpleAnalysisMethod",
            flags=imgui.TREE_NODE_DEFAULT_OPEN,
        )
        if open_:
            modes_display = modes_display_full

            processor = app.processor
            disable_combo = (
                stage_proc.full_analysis_active
                or app.is_setting_user_roi_mode
                or (processor and processor.is_processing and not processor.pause_event.is_set())
            )
            with _DisabledScope(disable_combo):
                try:
                    cur_idx = modes_enum.index(app_state.selected_tracker_name)
                except ValueError:
                    cur_idx = 0
                    app_state.selected_tracker_name = modes_enum[cur_idx]

                clicked, new_idx = imgui.combo("##TrackerModeCombo", cur_idx, modes_display)
                self._help_tooltip(self._generate_combined_tooltip(discovered_trackers_full))

            if clicked and new_idx != cur_idx:
                new_mode = modes_enum[new_idx]
                # Clear all overlays when switching to a different mode
                if app_state.selected_tracker_name != new_mode:
                    if hasattr(app, 'logger') and app.logger:
                        app.logger.info(f"UI(RunTab): Mode change requested {app_state.selected_tracker_name} -> {new_mode}. Clearing overlays.")
                    if hasattr(app, 'clear_all_overlays_and_ui_drawings'):
                        app.clear_all_overlays_and_ui_drawings()
                app_state.selected_tracker_name = new_mode
                # Persist user choice (store tracker name directly)
                if hasattr(app, 'app_settings') and hasattr(app.app_settings, 'set'):
                    app.app_settings.set("selected_tracker_name", new_mode)
                
                # Set tracker mode using dynamic discovery
                tr = app.tracker
                if tr:
                    tr.set_tracking_mode(new_mode)

            proc = app.processor
            video_loaded = proc and proc.is_video_open()
            processing_active = stage_proc.full_analysis_active
            disable_after = (not video_loaded) or processing_active

            with _DisabledScope(disable_after):

                self._render_execution_progress_display()
            
        # Always show processing speed controls as they affect basic video playback
        self._render_processing_speed_controls(app_state)

        open_, _ = imgui.collapsing_header(
            "Tracking##SimpleTracking",
            flags=imgui.TREE_NODE_DEFAULT_OPEN,
        )
        if open_:
            self._render_tracking_axes_mode(stage_proc)

        mode = app_state.selected_tracker_name
        if mode and (self._is_offline_tracker(mode) or self._is_live_tracker(mode)):
            if app_state.show_advanced_options:
                open_, _ = imgui.collapsing_header(
                    "Analysis Options##RunControlAnalysisOptions",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    imgui.text("Analysis Range")
                    self._render_range_selection(stage_proc, fs_proc, events)

                    if self._is_offline_tracker(mode):
                        imgui.text("Stage Reruns:")
                        with _DisabledScope(disable_combo):
                            _, stage_proc.force_rerun_stage1 = imgui.checkbox(
                                "Force Re-run Stage 1##ForceRerunS1",
                                stage_proc.force_rerun_stage1,
                            )
                            imgui.same_line()
                            _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox(
                                "Force Re-run Stage 2##ForceRerunS2",
                                stage_proc.force_rerun_stage2_segmentation,
                            )
                            if not hasattr(stage_proc, "save_preprocessed_video"):
                                stage_proc.save_preprocessed_video = app.app_settings.get("save_preprocessed_video", False)
                            changed, new_val = imgui.checkbox("Save/Reuse Preprocessed Video##SavePreprocessedVideo", stage_proc.save_preprocessed_video)
                            if changed:
                                stage_proc.save_preprocessed_video = new_val
                                app.app_settings.set("save_preprocessed_video", new_val)
                                if new_val:
                                    stage_proc.num_producers_stage1 = 1
                                    app.app_settings.set("num_producers_stage1", 1)
                            _tooltip_if_hovered(
                                "Saves a preprocessed (resized/unwarped) video for faster re-runs.\n"
                                "This enables Optical Flow recovery in Stage 2 and is RECOMMENDED for Stage 3 speed.\n"
                                "Forces the number of Producer threads to 1."
                            )
                        
                        # Database Retention Option
                        with _DisabledScope(disable_combo):
                            retain_database = self.app.app_settings.get("retain_stage2_database", True)
                            changed_db, new_db_val = imgui.checkbox("Keep Stage 2 Database##RetainStage2Database", retain_database)
                            if changed_db:
                                self.app.app_settings.set("retain_stage2_database", new_db_val)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip(
                                "Keep the Stage 2 database file after processing completes.\n"
                                "Disable to save disk space (database is automatically deleted).\n" 
                                "Note: Database is always kept during 3-stage pipelines until Stage 3 completes."
                            )

        proc = app.processor
        video_loaded = proc and proc.is_video_open()
        processing_active = stage_proc.full_analysis_active
        disable_after = (not video_loaded) or processing_active

        self._render_start_stop_buttons(stage_proc, fs_proc, events)

        self._render_interactive_refinement_controls()

        chapters = getattr(app.funscript_processor, "video_chapters", [])
        if chapters:
            # Clear All Chapters button (DESTRUCTIVE - deletes all chapters)
            with destructive_button_style():
                if imgui.button("Clear All Chapters", width=-1):
                    imgui.open_popup("ConfirmClearChapters")
            opened, _ = imgui.begin_popup_modal("ConfirmClearChapters")
            if opened:
                w = imgui.get_window_width()
                text = "Are you sure you want to clear all chapters? This cannot be undone."
                tw = imgui.calc_text_size(text)[0]
                imgui.set_cursor_pos_x((w - tw) * 0.5)
                imgui.text(text)
                imgui.spacing()
                bw, cw = 150, 100
                total = bw + cw + imgui.get_style().item_spacing[0]
                imgui.set_cursor_pos_x((w - total) * 0.5)
                # Confirm button (DESTRUCTIVE - irreversible action)
                with destructive_button_style():
                    if imgui.button("Yes, clear all", width=bw):
                        app.funscript_processor.video_chapters.clear()
                        app.project_manager.project_dirty = True
                        imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel", width=cw):
                    imgui.close_current_popup()
                imgui.end_popup()

        if disable_after and imgui.is_item_hovered():
            imgui.set_tooltip("Requires a video to be loaded and no other process to be active.")

    def _render_configuration_tab(self):
        app = self.app
        app_state = app.app_state_ui
        tmode = app_state.selected_tracker_name

        imgui.text("Configure settings for the selected mode.")
        imgui.spacing()

        # Show AI model settings for live and offline trackers
        if self._is_live_tracker(tmode) or self._is_offline_tracker(tmode):
            if imgui.collapsing_header("AI Models & Inference##ConfigAIModels")[0]:
                self._render_ai_model_settings()

        adv = app.app_state_ui.show_advanced_options
        if self._is_live_tracker(tmode) and adv:
            self._render_live_tracker_settings()

        # TEMPORARILY DISABLE SECTIONS WITH HARDCODED TRACKERMODE REFERENCES
        # TODO: Replace with dynamic discovery logic
        
        # Class filtering for advanced users
        if (self._is_live_tracker(tmode) or self._is_offline_tracker(tmode)) and adv:
            if imgui.collapsing_header("Class Filtering##ConfigClassFilterHeader")[0]:
                self._render_class_filtering_content()

        # Oscillation detector settings for oscillation trackers
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(tmode)
        if tracker_info and 'oscillation' in tracker_info.display_name.lower():
            if imgui.collapsing_header("Oscillation Detector Settings##ConfigOscillationDetector", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_oscillation_detector_settings()

        # Stage 3 specific settings (temporarily disabled - needs proper stage detection)
        # if tmode == "stage3_optical_flow":
        #     if imgui.collapsing_header("Stage 3 Oscillation Detector Mode##ConfigStage3OD", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        #         self._render_stage3_oscillation_detector_mode_settings()

        # Check if configuration is available for this tracker
        has_config = self._is_live_tracker(tmode) or self._is_offline_tracker(tmode)
        if not has_config:
            imgui.text_disabled("No configuration available for this mode.")

    def _render_settings_tab(self):
        app = self.app
        app_state = app.app_state_ui

        imgui.text("Global application settings. Saved in settings.json.")
        imgui.spacing()

        if imgui.collapsing_header(
            "Interface & Performance##SettingsMenuPerfInterface",
            flags=imgui.TREE_NODE_DEFAULT_OPEN,
        )[0]:
            self._render_settings_interface_perf()

        if imgui.collapsing_header(
            "File & Output##SettingsMenuOutput", flags=imgui.TREE_NODE_DEFAULT_OPEN
        )[0]:
            self._render_settings_file_output()

        if app_state.show_advanced_options:
            if imgui.collapsing_header("Logging & Autosave##SettingsMenuLogging")[0]:
                self._render_settings_logging_autosave()

            if imgui.collapsing_header("View/Edit Hotkeys##FSHotkeysMenuSettingsDetail")[0]:
                self._render_settings_hotkeys()
        imgui.spacing()

        # Reset All Settings button (DESTRUCTIVE - resets all settings)
        with destructive_button_style():
            if imgui.button("Reset All Settings to Default##ResetAllSettingsButton", width=-1):
                imgui.open_popup("Confirm Reset##ResetSettingsPopup")

        if imgui.begin_popup_modal(
            "Confirm Reset##ResetSettingsPopup", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )[0]:
            imgui.text(
                "This will reset all application settings to their defaults.\n"
                "Your projects will not be affected.\n"
                "This action cannot be undone."
            )

            avail_w = imgui.get_content_region_available_width()
            pw = (avail_w - imgui.get_style().item_spacing[0]) / 2.0

            if imgui.button("Confirm Reset", width=pw):
                app.app_settings.reset_to_defaults()
                app.logger.info("All settings have been reset to default.", extra={"status_message": True})
                imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", width=pw):
                imgui.close_current_popup()
            imgui.end_popup()

    def _render_post_processing_tab(self):
        app = self.app
        if imgui.collapsing_header(
            "Manual Adjustments##PostProcManual", flags=imgui.TREE_NODE_DEFAULT_OPEN
        )[0]:
            self._render_funscript_processing_tools(app.funscript_processor, app.event_handlers)
        if imgui.collapsing_header("Automated Post-Processing##PostProcAuto")[0]:
            self._render_automatic_post_processing_new(app.funscript_processor)

    # ------- AI model settings -------

    def _render_ai_model_settings(self):
        app = self.app
        stage_proc = app.stage_processor
        settings = app.app_settings
        style = imgui.get_style()

        def show_model_file_dialog(title, current_path, callback):
            gi = getattr(app, "gui_instance", None)
            if not gi:
                return
            init_dir = os.path.dirname(current_path) if current_path else None
            gi.file_dialog.show(
                title=title,
                is_save=False,
                callback=callback,
                extension_filter=self.AI_modelExtensionsFilter,
                initial_path=init_dir,
            )

        # Precompute widths
        tp = style.frame_padding.x * 2
        browse_w = imgui.calc_text_size("Browse").x + tp
        unload_w = imgui.calc_text_size("Unload").x + tp
        total_btn_w = browse_w + unload_w + style.item_spacing.x
        avail_w = imgui.get_content_region_available_width()
        input_w = avail_w - total_btn_w - style.item_spacing.x

        # Detection model
        imgui.text("Detection Model")
        _readonly_input("##S1YOLOPath", app.yolo_detection_model_path_setting, input_w)
        imgui.same_line()
        # Browse button with folder-open icon
        icon_mgr = get_icon_texture_manager()
        folder_open_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
        btn_size = imgui.get_frame_height()
        if folder_open_tex and imgui.image_button(folder_open_tex, btn_size, btn_size):
            show_model_file_dialog(
                "Select YOLO Detection Model",
                app.yolo_detection_model_path_setting,
                self._update_detection_model_path,
            )
        elif not folder_open_tex and imgui.button("Browse##S1YOLOBrowse"):
            show_model_file_dialog(
                "Select YOLO Detection Model",
                app.yolo_detection_model_path_setting,
                self._update_detection_model_path,
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Browse for detection model file")
        imgui.same_line()
        # Unload button (DESTRUCTIVE - unloads model from memory)
        with destructive_button_style():
            if imgui.button("Unload##S1YOLOUnload"):
                app.unload_model("detection")
        _tooltip_if_hovered("Path to the YOLO object detection model file (%s)." % self.AI_modelTooltipExtensions)

        # Pose model
        imgui.text("Pose Model")
        _readonly_input("##PoseYOLOPath", app.yolo_pose_model_path_setting, input_w)
        imgui.same_line()
        # Browse button with folder-open icon
        folder_open_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
        if folder_open_tex and imgui.image_button(folder_open_tex, btn_size, btn_size):
            show_model_file_dialog(
                "Select YOLO Pose Model",
                app.yolo_pose_model_path_setting,
                self._update_pose_model_path,
            )
        elif not folder_open_tex and imgui.button("Browse##PoseYOLOBrowse"):
            show_model_file_dialog(
                "Select YOLO Pose Model",
                app.yolo_pose_model_path_setting,
                self._update_pose_model_path,
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Browse for pose model file")
        imgui.same_line()
        # Unload button (DESTRUCTIVE - unloads model from memory)
        with destructive_button_style():
            if imgui.button("Unload##PoseYOLOUnload"):
                app.unload_model("pose")
        _tooltip_if_hovered("Path to the YOLO pose estimation model file (%s). This model is optional." % self.AI_modelTooltipExtensions)

        imgui.text("Pose Model Artifacts Dir")
        dir_input_w = avail_w - browse_w - style.item_spacing.x if avail_w > browse_w else -1
        _readonly_input("##PoseArtifactsDirPath", app.pose_model_artifacts_dir, dir_input_w)
        imgui.same_line()
        # Browse button with folder-open icon
        folder_open_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
        if folder_open_tex and imgui.image_button(folder_open_tex, btn_size, btn_size):
            gi = getattr(app, "gui_instance", None)
            if gi:
                gi.file_dialog.show(
                    title="Select Pose Model Artifacts Directory",
                    callback=self._update_artifacts_dir_path,
                    is_folder_dialog=True,
                    initial_path=app.pose_model_artifacts_dir,
                )
        elif not folder_open_tex and imgui.button("Browse##PoseArtifactsDirBrowse"):
            gi = getattr(app, "gui_instance", None)
            if gi:
                gi.file_dialog.show(
                    title="Select Pose Model Artifacts Directory",
                    callback=self._update_artifacts_dir_path,
                    is_folder_dialog=True,
                    initial_path=app.pose_model_artifacts_dir,
                )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Browse for pose model artifacts directory")
        _tooltip_if_hovered(
            "Path to the folder containing your trained classifier,\n"
            "imputer, and other .joblib model artifacts."
        )

        mode = app.app_state_ui.selected_tracker_name
        if self._is_offline_tracker(mode):
            imgui.text("Stage 1 Inference Workers:")
            imgui.push_item_width(100)
            is_save_pre = getattr(stage_proc, "save_preprocessed_video", False)
            with _DisabledScope(is_save_pre):
                ch_p, n_p = imgui.input_int(
                    "Producers##S1Producers", stage_proc.num_producers_stage1
                )
                if ch_p and not is_save_pre:
                    v = max(1, n_p)
                    if v != stage_proc.num_producers_stage1:
                        stage_proc.num_producers_stage1 = v
                        settings.set("num_producers_stage1", v)
            if is_save_pre:
                _tooltip_if_hovered("Producers are forced to 1 when 'Save/Reuse Preprocessed Video' is enabled.")
            else:
                _tooltip_if_hovered("Number of threads for video decoding & preprocessing.")

            imgui.same_line()
            ch_c, n_c = imgui.input_int("Consumers##S1Consumers", stage_proc.num_consumers_stage1)
            if ch_c:
                v = max(1, n_c)
                if v != stage_proc.num_consumers_stage1:
                    stage_proc.num_consumers_stage1 = v
                    settings.set("num_consumers_stage1", v)
            _tooltip_if_hovered("Number of threads for AI model inference. Match to available cores for best performance.")
            imgui.pop_item_width()

            imgui.text("Stage 2 OF Workers")
            imgui.same_line()
            imgui.push_item_width(120)
            cur_s2 = settings.get("num_workers_stage2_of", self.constants.DEFAULT_S2_OF_WORKERS)
            ch, new_s2 = imgui.input_int("##S2OFWorkers", cur_s2)
            if ch:
                v = max(1, new_s2)
                if v != cur_s2:
                    settings.set("num_workers_stage2_of", v)
            imgui.pop_item_width()
            _tooltip_if_hovered(
                "Number of processes for Stage 2 Optical Flow gap recovery.\n"
                "More may be faster on high-core CPUs."
            )

    # ------- Settings: interface/perf -------

    def _render_settings_interface_perf(self):
        app = self.app
        energy = app.energy_saver
        settings = app.app_settings

        imgui.text("Font Scale")
        imgui.same_line()
        imgui.push_item_width(120)
        labels = config.constants.FONT_SCALE_LABELS
        values = config.constants.FONT_SCALE_VALUES
        cur_val = settings.get("global_font_scale", config.constants.DEFAULT_FONT_SCALE)
        try:
            cur_idx = min(range(len(values)), key=lambda i: abs(values[i] - cur_val))
        except (ValueError, IndexError):
            cur_idx = 3
        ch, new_idx = imgui.combo("##GlobalFontScale", cur_idx, labels)
        if ch:
            nv = values[new_idx]
            if nv != cur_val:
                settings.set("global_font_scale", nv)
                energy.reset_activity_timer()
        imgui.pop_item_width()
        _tooltip_if_hovered("Adjust the global UI font size. Applied instantly.")
        
        # Automatic system scaling option
        imgui.same_line()
        auto_scaling_enabled = settings.get("auto_system_scaling_enabled", True)
        ch, auto_scaling_enabled = imgui.checkbox("Auto System Scaling", auto_scaling_enabled)
        if ch:
            settings.set("auto_system_scaling_enabled", auto_scaling_enabled)
            if auto_scaling_enabled:
                # Apply system scaling immediately when enabled
                try:
                    from application.utils.system_scaling import apply_system_scaling_to_settings
                    scaling_applied = apply_system_scaling_to_settings(settings)
                    if scaling_applied:
                        app.logger.info("System scaling applied to application settings")
                        energy.reset_activity_timer()
                except Exception as e:
                    app.logger.warning(f"Failed to apply system scaling: {e}")
            else:
                app.logger.info("Automatic system scaling disabled")
        _tooltip_if_hovered("Automatically detect and apply system DPI/scaling settings at startup. "
                           "When enabled, the application will adjust the UI font size based on your "
                           "system's display scaling settings (e.g., 125%, 150%, etc.).")
        
        # Manual system scaling detection button
        if imgui.button("Detect System Scaling Now"):
            try:
                from application.utils.system_scaling import get_system_scaling_info, get_recommended_font_scale
                scaling_factor, dpi, platform_name = get_system_scaling_info()
                recommended_scale = get_recommended_font_scale(scaling_factor)
                current_scale = settings.get("global_font_scale", config.constants.DEFAULT_FONT_SCALE)
                
                app.logger.info(f"System scaling detected: {scaling_factor:.2f}x ({dpi:.0f} DPI on {platform_name})")
                app.logger.info(f"Recommended font scale: {recommended_scale} (current: {current_scale})")
                
                if abs(recommended_scale - current_scale) > 0.05:  # Only update if significantly different
                    settings.set("global_font_scale", recommended_scale)
                    energy.reset_activity_timer()
                    app.logger.info(f"Font scale updated to {recommended_scale} based on system scaling")
                else:
                    app.logger.info("System scaling matches current font scale setting")
            except Exception as e:
                app.logger.warning(f"Failed to detect system scaling: {e}")
        _tooltip_if_hovered("Manually detect and apply current system DPI/scaling settings.")

        imgui.text("Timeline Pan Speed")
        imgui.same_line()
        imgui.push_item_width(120)
        cur_speed = settings.get("timeline_pan_speed_multiplier", config.constants.DEFAULT_TIMELINE_PAN_SPEED)
        ch, new_speed = imgui.slider_int("##TimelinePanSpeed", cur_speed, config.constants.TIMELINE_PAN_SPEED_MIN, config.constants.TIMELINE_PAN_SPEED_MAX)
        if ch and new_speed != cur_speed:
            settings.set("timeline_pan_speed_multiplier", new_speed)
        imgui.pop_item_width()
        _tooltip_if_hovered("Multiplier for keyboard-based timeline panning speed.")

        # --- Timeline Performance & GPU Settings ---
        imgui.text("Timeline Performance")
        
        # GPU Enable/Disable
        gpu_enabled = settings.get("timeline_gpu_enabled", False)
        changed, gpu_enabled = imgui.checkbox("Enable GPU Rendering##GPUTimeline", gpu_enabled)
        if changed:
            settings.set("timeline_gpu_enabled", gpu_enabled)
            app.energy_saver.reset_activity_timer()
            # Reinitialize GPU if being enabled
            if gpu_enabled and hasattr(app, '_initialize_gpu_timeline'):
                app._initialize_gpu_timeline()
            app.logger.info(f"GPU timeline rendering {'enabled' if gpu_enabled else 'disabled'}", extra={"status_message": True})
        _tooltip_if_hovered(
            "Enable GPU-accelerated timeline rendering for massive performance improvements.\n"
            "Best for datasets with 10,000+ points. Automatic fallback to CPU if GPU fails."
        )
        
        if gpu_enabled:
            imgui.text("GPU Threshold")
            imgui.same_line()
            imgui.push_item_width(120)
            gpu_threshold = settings.get("timeline_gpu_threshold_points", 5000)
            changed, gpu_threshold = imgui.input_int("##GPUThreshold", gpu_threshold)
            if changed:
                gpu_threshold = max(1000, min(100000, gpu_threshold))  # Clamp between 1k-100k
                settings.set("timeline_gpu_threshold_points", gpu_threshold)
            imgui.pop_item_width()
            _tooltip_if_hovered("Use GPU rendering when timeline has more than this many points")
        
        # Performance indicators
        show_perf = settings.get("show_timeline_optimization_indicator", False)
        changed, show_perf = imgui.checkbox("Show Performance Info##PerfIndicator", show_perf)
        if changed:
            settings.set("show_timeline_optimization_indicator", show_perf)
        _tooltip_if_hovered("Display performance indicators on timeline (render time, optimization modes)")
        
        # Performance stats (if GPU enabled and available)
        if gpu_enabled and hasattr(app, 'gpu_integration') and app.gpu_integration:
            try:
                stats = app.gpu_integration.get_performance_summary()
                imgui.text(f"GPU Backend: {stats.get('current_backend', 'Unknown')}")
                
                if 'gpu_details' in stats:
                    gpu_stats = stats['gpu_details']
                    render_time = gpu_stats.get('render_time_ms', 0)
                    points_rendered = gpu_stats.get('points_rendered', 0)
                    imgui.text(f"Last Render: {render_time:.2f}ms, {points_rendered:,} pts")
                    
                    # Show GPU performance color coding
                    if render_time < 5.0:
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.0, 1.0)  # Green
                        imgui.text("Excellent Performance")
                    elif render_time < 16.67:  # 60fps threshold
                        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0, 1.0)  # Yellow
                        imgui.text("Good Performance")
                    else:
                        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.5, 0.0, 1.0)  # Orange
                        imgui.text("High Load")
                    imgui.pop_style_color()
            except Exception as e:
                imgui.text_disabled(f"GPU stats unavailable: {str(e)[:30]}...")
        elif gpu_enabled:
            imgui.text_disabled("GPU not available - using CPU fallback")

        imgui.text("Video Decoding")
        imgui.same_line()
        imgui.push_item_width(180)
        opts = app.available_ffmpeg_hwaccels
        disp = [o.replace("videotoolbox", "VideoToolbox (macOS)") for o in opts]
        try:
            hw_idx = opts.index(app.hardware_acceleration_method)
        except ValueError:
            hw_idx = 0
        ch, nidx = imgui.combo("HW Acceleration##HWAccelMethod", hw_idx, disp)
        if ch:
            method = opts[nidx]
            if method != app.hardware_acceleration_method:
                app.hardware_acceleration_method = method
                settings.set("hardware_acceleration_method", method)
                app.logger.info("Hardware acceleration set to: %s. Reload video to apply." % method, extra={"status_message": True})
        imgui.pop_item_width()
        _tooltip_if_hovered("Select FFmpeg hardware acceleration. Requires video reload to apply.")

        imgui.text("Energy Saver Mode:")
        ch_es, v_es = imgui.checkbox("Enable##EnableES", energy.energy_saver_enabled)
        if ch_es and v_es != energy.energy_saver_enabled:
            energy.energy_saver_enabled = v_es
            settings.set("energy_saver_enabled", v_es)

        if energy.energy_saver_enabled:
            imgui.push_item_width(100)
            imgui.text("Normal FPS")
            imgui.same_line()
            nf = int(energy.main_loop_normal_fps_target)
            ch, val = imgui.input_int("##NormalFPS", nf)
            if ch:
                v = max(config.constants.ENERGY_SAVER_NORMAL_FPS_MIN, val)
                if v != nf:
                    energy.main_loop_normal_fps_target = v
                    settings.set("main_loop_normal_fps_target", v)

            imgui.text("Idle After (s)")
            imgui.same_line()
            th = int(energy.energy_saver_threshold_seconds)
            ch, val = imgui.input_int("##ESThreshold", th)
            if ch:
                v = float(max(config.constants.ENERGY_SAVER_THRESHOLD_MIN, val))
                if v != energy.energy_saver_threshold_seconds:
                    energy.energy_saver_threshold_seconds = v
                    settings.set("energy_saver_threshold_seconds", v)

            imgui.text("Idle FPS")
            imgui.same_line()
            ef = int(energy.energy_saver_fps)
            ch, val = imgui.input_int("##ESFPS", ef)
            if ch:
                v = max(config.constants.ENERGY_SAVER_IDLE_FPS_MIN, val)
                if v != ef:
                    energy.energy_saver_fps = v
                    settings.set("energy_saver_fps", v)
            imgui.pop_item_width()

    # ------- Settings: file/output -------

    def _render_settings_file_output(self):
        settings = self.app.app_settings

        imgui.text("Output Folder:")
        imgui.push_item_width(-1)
        cur = settings.get("output_folder_path", "output")
        ch, new_val = imgui.input_text("##OutputFolder", cur, 256)
        if ch and new_val != cur:
            settings.set("output_folder_path", new_val)
        imgui.pop_item_width()
        _tooltip_if_hovered("Root folder for all generated files (projects, analysis data, etc.).")

        imgui.text("Funscript Output:")
        ch, v = imgui.checkbox(
            "Autosave final script next to video",
            settings.get("autosave_final_funscript_to_video_location", True),
        )
        if ch:
            settings.set("autosave_final_funscript_to_video_location", v)

        ch, v = imgui.checkbox("Generate .roll file (from Timeline 2)", settings.get("generate_roll_file", True))
        if ch:
            settings.set("generate_roll_file", v)

        imgui.text("Batch Processing Default:")
        cur = settings.get("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Process All (skips own matching version)", cur == 0):
            if cur != 0:
                settings.set("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Skip if Funscript Exists", cur == 1):
            if cur != 1:
                settings.set("batch_mode_overwrite_strategy", 1)

    # ------- Settings: logging/autosave -------

    def _render_settings_logging_autosave(self):
        app = self.app
        settings = app.app_settings

        imgui.text("Logging Level")
        imgui.same_line()
        imgui.push_item_width(150)
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            idx = levels.index(app.logging_level_setting.upper())
        except ValueError:
            idx = 1
        ch, nidx = imgui.combo("##LogLevel", idx, levels)
        if ch:
            new_level = levels[nidx]
            if new_level != app.logging_level_setting.upper():
                app.set_application_logging_level(new_level)
        imgui.pop_item_width()

        imgui.text("Project Autosave:")
        ch, v = imgui.checkbox(
            "Enable##EnableAutosave", settings.get("autosave_enabled", True)
        )
        if ch:
            settings.set("autosave_enabled", v)

        if settings.get("autosave_enabled"):
            imgui.push_item_width(100)
            imgui.text("Interval (s)")
            imgui.same_line()
            interval = settings.get("autosave_interval_seconds", 300)
            ch_int, new_interval = imgui.input_int("##AutosaveInterval", interval)
            if ch_int:
                nv = max(30, new_interval)
                if nv != interval:
                    settings.set("autosave_interval_seconds", nv)
            imgui.pop_item_width()

# ------- Settings: hotkeys -------

    def _render_settings_hotkeys(self):
        app = self.app
        shortcuts_settings = app.app_settings.get("funscript_editor_shortcuts", {})
        sm = app.shortcut_manager
        c = self.ControlPanelColors.ACTIVE_PROGRESS

        for action_name, key_str in list(shortcuts_settings.items()):
            disp = action_name.replace("_", " ").title()
            imgui.text("%s: " % disp)
            imgui.same_line()

            recording = (sm.is_recording_shortcut_for == action_name)
            display_key = "PRESS KEY..." if recording else key_str
            btn_text = "Cancel" if recording else "Record"

            imgui.text_colored(display_key, *c)
            imgui.same_line()
            if imgui.button("%s##record_btn_%s" % (btn_text, action_name)):
                if recording:
                    sm.cancel_shortcut_recording()
                else:
                    sm.start_shortcut_recording(action_name)

# ------- Execution/progress -------

    def _render_execution_progress_display(self):
        app = self.app
        stage_proc = app.stage_processor
        app_state = app.app_state_ui
        mode = app_state.selected_tracker_name

        if self._is_offline_tracker(mode):
            self._render_stage_progress_ui(stage_proc)
            return

        if self._is_live_tracker(mode):
            tr = app.tracker
            imgui.text(">> Tracker Status")
            imgui.separator()
            # Show video processor FPS (more accurate for MAX_SPEED performance)
            video_fps = (app.processor.actual_fps if app.processor and hasattr(app.processor, 'actual_fps') else 0.0)
            tracker_fps = (tr.current_fps if tr else 0.0)
            
            # Prefer video processor FPS for accuracy, fallback to tracker FPS
            display_fps = video_fps if video_fps > 0 else tracker_fps
            imgui.text(" - Video FPS: %.1f" % (video_fps if isinstance(video_fps, (int, float)) else 0.0))
            imgui.text(" - Tracker FPS: %.1f" % (tracker_fps if isinstance(tracker_fps, (int, float)) else 0.0))
            roi_status = "Not Set"
            if tr:
                if mode == "yolo_roi":
                    roi_status = (
                        "Tracking '%s'" % tr.main_interaction_class
                        if getattr(tr, "main_interaction_class", None)
                        else "Searching..."
                    )
                elif mode == "user_roi":
                    roi_status = "Set" if getattr(tr, "user_roi_fixed", False) else "Not Set"
                elif mode in ["oscillation_experimental", "oscillation_legacy", "oscillation_experimental_2"]:
                    roi_status = "Set" if getattr(tr, "oscillation_area_fixed", None) else "Not Set"
            imgui.text(" - ROI Status: %s" % roi_status)

            if mode == "user_roi":
                self._render_user_roi_controls_for_run_tab()
            return

# ------- Live tracker settings -------

    def _render_live_tracker_settings(self):
        app = self.app
        tr = app.tracker
        if not tr:
            imgui.text_disabled("Tracker not initialized.")
            return

        settings = app.app_settings

        if imgui.collapsing_header("Detection & ROI Definition##ROIDetectionTrackerMenu")[0]:
            cur_conf = settings.get("live_tracker_confidence_threshold")
            ch, new_conf = imgui.slider_float("Obj. Confidence##ROIConfTrackerMenu", cur_conf, 0.1, 0.95, "%.2f")
            if ch and new_conf != cur_conf:
                settings.set("live_tracker_confidence_threshold", new_conf)
                tr.confidence_threshold = new_conf

            cur_pad = settings.get("live_tracker_roi_padding")
            ch, new_pad = imgui.input_int("ROI Padding##ROIPadTrackerMenu", cur_pad)
            if ch:
                v = max(0, new_pad)
                if v != cur_pad:
                    settings.set("live_tracker_roi_padding", v)
                    tr.roi_padding = v

            cur_int = settings.get("live_tracker_roi_update_interval")
            ch, new_int = imgui.input_int("ROI Update Interval (frames)##ROIIntervalTrackerMenu", cur_int)
            if ch:
                v = max(1, new_int)
                if v != cur_int:
                    settings.set("live_tracker_roi_update_interval", v)
                    tr.roi_update_interval = v

            cur_sm = settings.get("live_tracker_roi_smoothing_factor")
            ch, new_sm = imgui.slider_float("ROI Smoothing Factor##ROISmoothTrackerMenu", cur_sm, 0.0, 1.0, "%.2f")
            if ch and new_sm != cur_sm:
                settings.set("live_tracker_roi_smoothing_factor", new_sm)
                tr.roi_smoothing_factor = new_sm

            cur_persist = settings.get("live_tracker_roi_persistence_frames")
            ch, new_pf = imgui.input_int("ROI Persistence (frames)##ROIPersistTrackerMenu", cur_persist)
            if ch:
                v = max(0, new_pf)
                if v != cur_persist:
                    settings.set("live_tracker_roi_persistence_frames", v)
                    tr.max_frames_for_roi_persistence = v

        if imgui.collapsing_header("Optical Flow##ROIFlowTrackerMenu")[0]:
            cur_sparse = settings.get("live_tracker_use_sparse_flow")
            ch, new_sparse = imgui.checkbox("Use Sparse Optical Flow##ROISparseFlowTrackerMenu", cur_sparse)
            if ch:
                settings.set("live_tracker_use_sparse_flow", new_sparse)
                tr.use_sparse_flow = new_sparse

            imgui.text("DIS Dense Flow Settings:")
            if cur_sparse:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            presets = ["ULTRAFAST", "FAST", "MEDIUM"]
            cur_p = settings.get("live_tracker_dis_flow_preset").upper()
            try:
                p_idx = presets.index(cur_p)
            except ValueError:
                p_idx = 0
            ch, nidx = imgui.combo("DIS Preset##ROIDISPresetTrackerMenu", p_idx, presets)
            if ch:
                nv = presets[nidx]
                if nv != cur_p:
                    settings.set("live_tracker_dis_flow_preset", nv)
                    tr.update_dis_flow_config(preset=nv)

            cur_scale = settings.get("live_tracker_dis_finest_scale")
            ch, new_scale = imgui.input_int("DIS Finest Scale (0-10, 0=auto)##ROIDISFineScaleTrackerMenu", cur_scale)
            if ch and new_scale != cur_scale:
                settings.set("live_tracker_dis_finest_scale", new_scale)
                tr.update_dis_flow_config(finest_scale=new_scale)

            if cur_sparse:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()

            if imgui.collapsing_header("Output Signal Generation##ROISignalTrackerMenu"):
                cur_sens = settings.get("live_tracker_sensitivity")
                ch, ns = imgui.slider_float("Output Sensitivity##ROISensTrackerMenu", cur_sens, 0.0, 100.0, "%.1f")
                if ch and ns != cur_sens:
                    settings.set("live_tracker_sensitivity", ns)
                    tr.sensitivity = ns

                cur_amp = settings.get("live_tracker_base_amplification")
                ch, na = imgui.slider_float("Base Amplification##ROIBaseAmpTrackerMenu", cur_amp, 0.1, 5.0, "%.2f")
                if ch:
                    v = max(0.1, na)
                    if v != cur_amp:
                        settings.set("live_tracker_base_amplification", v)
                        tr.base_amplification_factor = v

                imgui.text("Class-Specific Amplification Multipliers:")
                cur = settings.get("live_tracker_class_amp_multipliers", {})
                changed = False

                face = cur.get("face", 1.0)
                ch, nv = imgui.slider_float("Face Amp. Mult.##ROIFaceAmpTrackerMenu", face, 0.1, 5.0, "%.2f")
                if ch:
                    cur["face"] = max(0.1, nv)
                    changed = True

                hand = cur.get("hand", 1.0)
                ch, nv = imgui.slider_float("Hand Amp. Mult.##ROIHandAmpTrackerMenu", hand, 0.1, 5.0, "%.2f")
                if ch:
                    cur["hand"] = max(0.1, nv)
                    changed = True

                if changed:
                    settings.set("live_tracker_class_amp_multipliers", cur)
                    tr.class_specific_amplification_multipliers = cur

            cur_smooth = settings.get("live_tracker_flow_smoothing_window")
            ch, nv = imgui.input_int("Flow Smoothing Window##ROIFlowSmoothWinTrackerMenu", cur_smooth)
            if ch:
                v = max(1, nv)
                if v != cur_smooth:
                    settings.set("live_tracker_flow_smoothing_window", v)
                    tr.flow_history_window_smooth = v

            imgui.text("Output Delay (frames):")
            cur_delay = settings.get("funscript_output_delay_frames")
            ch, nd = imgui.slider_int("##OutputDelayFrames", cur_delay, 0, 20)
            if ch and nd != cur_delay:
                settings.set("funscript_output_delay_frames", nd)
                app.calibration.funscript_output_delay_frames = nd
                app.calibration.update_tracker_delay_params()

# ------- Oscillation detector -------

    def _render_calibration_window(self, calibration_mgr, app_state):
        """Renders the dedicated latency calibration window."""
        window_title = "Latency Calibration"
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        # In fixed mode, embed it in the main panel area without a title bar
        if app_state.ui_layout_mode == 'fixed':
            imgui.begin("Modular Control Panel##LeftControlsModular", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
            self._render_latency_calibration(calibration_mgr)
            imgui.end()
        else: # Floating mode
            if imgui.begin(window_title, closable=False, flags=flags):
                self._render_latency_calibration(calibration_mgr)
                imgui.end()

    def _render_start_stop_buttons(self, stage_proc, fs_proc, event_handlers):
        is_batch_mode = self.app.is_batch_processing_active
        is_analysis_running = stage_proc.full_analysis_active

        # A "Live Tracking" session is only running if the processor is active
        # AND tracker processing has been explicitly enabled.
        is_live_tracking_running = (self.app.processor and
                                    self.app.processor.is_processing and
                                    self.app.processor.enable_tracker_processing)

        is_setting_roi = self.app.is_setting_user_roi_mode
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi

        if is_batch_mode:
            imgui.text_ansi_colored("--- BATCH PROCESSING ACTIVE ---", 1.0, 0.7, 0.3) # TODO: move to theme, orange
            total_videos = len(self.app.batch_video_paths)
            current_idx = self.app.current_batch_video_index
            if 0 <= current_idx < total_videos:
                current_video_name = os.path.basename(self.app.batch_video_paths[current_idx]["path"])
                imgui.text_wrapped(f"Processing {current_idx + 1}/{total_videos}:")
                imgui.text_wrapped(f"{current_video_name}")
            # Abort button (DESTRUCTIVE - stops batch process)
            with destructive_button_style():
                if imgui.button("Abort Batch Process", width=-1):
                    self.app.abort_batch_processing()
            return

        selected_mode = self.app.app_state_ui.selected_tracker_name
        button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) / 2

        if is_any_process_active:
            status_text = "Processing..."
            if is_analysis_running:
                status_text = "Aborting..." if stage_proc.current_analysis_stage == -1 else f"Stage {stage_proc.current_analysis_stage} Running..."
            elif is_live_tracking_running:
                # This logic is now correctly guarded by the new is_live_tracking_running flag
                if self.app.processor.pause_event.is_set():
                    if imgui.button("Resume Tracking", width=button_width):
                        self.app.processor.start_processing()
                        if not self.app.tracker.tracking_active:
                            self.app.tracker.start_tracking()
                else:
                    if imgui.button("Pause Tracking", width=button_width):
                        self.app.processor.pause_processing()

                status_text = None
            elif is_setting_roi:
                status_text = "Setting ROI..."
            if status_text: imgui.button(status_text, width=button_width)
        else:
            start_text = "Start"
            handler = None
            
            # Check for resumable tasks
            resumable_checkpoint = None
            if self._is_offline_tracker(selected_mode) and self.app.file_manager.video_path:
                resumable_checkpoint = stage_proc.can_resume_video(self.app.file_manager.video_path)
            
            if self._is_offline_tracker(selected_mode):
                start_text = "Start AI Analysis (Range)" if fs_proc.scripting_range_active else "Start Full AI Analysis"
                handler = event_handlers.handle_start_ai_cv_analysis
            elif self._is_live_tracker(selected_mode):
                imgui.new_line()
                start_text = "Start Live Tracking (Range)" if fs_proc.scripting_range_active else "Start Live Tracking"
                handler = event_handlers.handle_start_live_tracker_click
            
            # Show resume button if checkpoint exists
            if resumable_checkpoint:
                button_width_third = (imgui.get_content_region_available()[0] - 2 * imgui.get_style().item_spacing[0]) / 3

                # Resume button (PRIMARY - positive action)
                with primary_button_style():
                    if imgui.button(f"Resume ({resumable_checkpoint.progress_percentage:.0f}%)", width=button_width_third):
                        if stage_proc.start_resume_from_checkpoint(resumable_checkpoint):
                            self.app.logger.info("Resumed processing from checkpoint", extra={'status_message': True})

                imgui.same_line()

                # Start fresh button (PRIMARY - positive action)
                with primary_button_style():
                    if imgui.button("Start Fresh", width=button_width_third):
                        # Delete checkpoint and start fresh
                        stage_proc.delete_checkpoint_for_video(self.app.file_manager.video_path)
                        if handler: handler()

                imgui.same_line()

                # Delete checkpoint button (DESTRUCTIVE - deletes data)
                with destructive_button_style():
                    if imgui.button("Clear Resume", width=button_width_third):
                        stage_proc.delete_checkpoint_for_video(self.app.file_manager.video_path)

            else:
                # Normal start button (PRIMARY - positive action)
                with primary_button_style():
                    if imgui.button(start_text, width=button_width):
                        if self._is_live_tracker(selected_mode):
                            self._start_live_tracking()
                        elif handler: handler()

        imgui.same_line()
        if not is_any_process_active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        # Abort/Stop button (DESTRUCTIVE - stops process)
        with destructive_button_style():
            if imgui.button("Abort/Stop Process##AbortGeneral", width=button_width):
                event_handlers.handle_abort_process_click()

        if not is_any_process_active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_stage_progress_ui(self, stage_proc):
        is_analysis_running = stage_proc.full_analysis_active
        selected_mode = self.app.app_state_ui.selected_tracker_name

        active_progress_color = self.ControlPanelColors.ACTIVE_PROGRESS # Vibrant blue for active
        completed_progress_color = self.ControlPanelColors.COMPLETED_PROGRESS # Vibrant green for completed

        # Stage 1
        imgui.text("Stage 1: YOLO Object Detection")
        if is_analysis_running and stage_proc.current_analysis_stage == 1:
            imgui.text(f"Time: {stage_proc.stage1_time_elapsed_str} | ETA: {stage_proc.stage1_eta_str} | Avg Speed:  {stage_proc.stage1_processing_fps_str}")
            imgui.text_wrapped(f"Progress: {stage_proc.stage1_progress_label}")

            # Apply active color
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *active_progress_color)
            imgui.progress_bar(stage_proc.stage1_progress_value, size=(-1, 0), overlay=f"{stage_proc.stage1_progress_value * 100:.0f}% | {stage_proc.stage1_instant_fps_str}" if stage_proc.stage1_progress_value >= 0 else "")
            imgui.pop_style_color()

            frame_q_size = stage_proc.stage1_frame_queue_size
            frame_q_max = self.constants.STAGE1_FRAME_QUEUE_MAXSIZE
            frame_q_fraction = frame_q_size / frame_q_max if frame_q_max > 0 else 0.0
            suggestion_message, bar_color = "", (0.2, 0.8, 0.2) # TODO: move to theme, green
            if frame_q_fraction > 0.9:
                bar_color, suggestion_message = (0.9, 0.3, 0.3), "Suggestion: Add consumer if resources allow" # TODO: move to theme, red
            elif frame_q_fraction > 0.2:
                bar_color, suggestion_message = (1.0, 0.5, 0.0), "Balanced" # TODO: move to theme, yellow
            else:
                bar_color, suggestion_message = (0.2, 0.8, 0.2), "Suggestion: Lessen consumers or add producer" # TODO: move to theme, green
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *bar_color)
            imgui.progress_bar(frame_q_fraction, size=(-1, 0), overlay=f"Frame Queue: {frame_q_size}/{frame_q_max}")
            imgui.pop_style_color()
            if suggestion_message: imgui.text(suggestion_message)

            if getattr(stage_proc, 'save_preprocessed_video', False):
                # The encoding queue (OS pipe buffer) isn't directly measurable.
                # However, its fill rate is entirely dependent on the producer, which is
                # throttled by the main frame queue. Therefore, the main frame queue's
                # size is an excellent proxy for the encoding backpressure.
                encoding_q_fraction = frame_q_fraction # Use the same fraction
                encoding_bar_color = bar_color # Use the same color logic

                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *encoding_bar_color)
                imgui.progress_bar(encoding_q_fraction, size=(-1, 0), overlay=f"Encoding Queue: ~{frame_q_size}/{frame_q_max}")
                imgui.pop_style_color()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "This is an estimate of the video encoding buffer.\n"
                        "It is based on the main analysis frame queue, which acts as a throttle for the encoder."
                    )

            imgui.text(f"Result Queue Size: ~{stage_proc.stage1_result_queue_size}")
        elif stage_proc.stage1_final_elapsed_time_str:
            imgui.text_wrapped(f"Last Run: {stage_proc.stage1_final_elapsed_time_str} | Avg Speed: {stage_proc.stage1_final_fps_str or 'N/A'}")
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *completed_progress_color)
            imgui.progress_bar(1.0, size=(-1, 0), overlay="Completed")
            imgui.pop_style_color()
        else:
            imgui.text_wrapped(f"Status: {stage_proc.stage1_status_text}")

        # Stage 2
        s2_title = "Stage 2: Contact Analysis & Funscript" if self._is_stage2_tracker(selected_mode) else "Stage 2: Segmentation"
        imgui.text(s2_title)
        if is_analysis_running and stage_proc.current_analysis_stage == 2:
            imgui.text_wrapped(f"Main: {stage_proc.stage2_main_progress_label}")

            # Apply active color
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *active_progress_color)
            imgui.progress_bar(stage_proc.stage2_main_progress_value, size=(-1, 0), overlay=f"{stage_proc.stage2_main_progress_value * 100:.0f}%" if stage_proc.stage2_main_progress_value >= 0 else "")
            imgui.pop_style_color()

            # Show this bar only when a sub-task is actively reporting progress.
            is_sub_task_active = stage_proc.stage2_sub_progress_value > 0.0 and stage_proc.stage2_sub_progress_value < 1.0
            if is_sub_task_active:
                # Add timing gauges if the data is available
                if stage_proc.stage2_sub_time_elapsed_str:
                    imgui.text(f"Time: {stage_proc.stage2_sub_time_elapsed_str} | ETA: {stage_proc.stage2_sub_eta_str} | Speed: {stage_proc.stage2_sub_processing_fps_str}")

                sub_progress_color = self.ControlPanelColors.SUB_PROGRESS
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *sub_progress_color)

                # Construct the overlay text with a percentage.
                overlay_text = f"{stage_proc.stage2_sub_progress_value * 100:.0f}%"
                imgui.progress_bar(stage_proc.stage2_sub_progress_value, size=(-1, 0), overlay=overlay_text)
                imgui.pop_style_color()

        elif stage_proc.stage2_final_elapsed_time_str:
            imgui.text_wrapped(f"Status: Completed in {stage_proc.stage2_final_elapsed_time_str}")
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *completed_progress_color)
            imgui.progress_bar(1.0, size=(-1, 0), overlay="Completed")
            imgui.pop_style_color()
        else:
            imgui.text_wrapped(f"Status: {stage_proc.stage2_status_text}")

        # Stage 3
        if self._is_stage3_tracker(selected_mode) or self._is_mixed_stage3_tracker(selected_mode):
            if self._is_mixed_stage3_tracker(selected_mode):
                imgui.text("Stage 3: Mixed Processing")
            else:
                imgui.text("Stage 3: Per-Segment Optical Flow")
            if is_analysis_running and stage_proc.current_analysis_stage == 3:
                imgui.text(f"Time: {stage_proc.stage3_time_elapsed_str} | ETA: {stage_proc.stage3_eta_str} | Speed: {stage_proc.stage3_processing_fps_str}")

                # Display chapter and chunk progress on separate lines for clarity
                imgui.text_wrapped(stage_proc.stage3_current_segment_label) # e.g., "Chapter: 1/5 (Cowgirl)"
                imgui.text_wrapped(stage_proc.stage3_overall_progress_label) # e.g., "Overall Task: Chunk 12/240"

                # Apply active color to both S3 progress bars
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *active_progress_color)

                # Overall Progress bar remains tied to total frames processed
                overlay_text = f"{stage_proc.stage3_overall_progress_value * 100:.0f}%"
                imgui.progress_bar(stage_proc.stage3_overall_progress_value, size=(-1, 0), overlay=overlay_text)

                imgui.pop_style_color()

            elif stage_proc.stage3_final_elapsed_time_str:
                imgui.text_wrapped(f"Last Run: {stage_proc.stage3_final_elapsed_time_str} | Avg Speed: {stage_proc.stage3_final_fps_str or 'N/A'}")
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *completed_progress_color)
                imgui.progress_bar(1.0, size=(-1, 0), overlay="Completed")
                imgui.pop_style_color()
            else:
                imgui.text_wrapped(f"Status: {stage_proc.stage3_status_text}")
        imgui.spacing()

    # ------- Common actions -------
    def _start_live_tracking(self):
        """Unified start flow for all live tracking modes."""
        try:
            self.app.event_handlers.handle_start_live_tracker_click()
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Failed to start live tracking: {e}")

    def _render_tracking_axes_mode(self, stage_proc):
        """Renders UI elements for tracking axis mode."""
        axis_modes = ["Both Axes (Up/Down + Left/Right)", "Up/Down Only (Vertical)", "Left/Right Only (Horizontal)"]
        current_axis_mode_idx = 0
        if self.app.tracking_axis_mode == "vertical":
            current_axis_mode_idx = 1
        elif self.app.tracking_axis_mode == "horizontal":
            current_axis_mode_idx = 2

        processor = self.app.processor
        disable_axis_controls = (
            stage_proc.full_analysis_active
            or self.app.is_setting_user_roi_mode
            or (processor and processor.is_processing and not processor.pause_event.is_set())
        )
        if disable_axis_controls:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        axis_mode_changed, new_axis_mode_idx = imgui.combo("##TrackingAxisModeComboGlobal", current_axis_mode_idx, axis_modes)
        if axis_mode_changed:
            old_mode = self.app.tracking_axis_mode
            if new_axis_mode_idx == 0:
                self.app.tracking_axis_mode = "both"
            elif new_axis_mode_idx == 1:
                self.app.tracking_axis_mode = "vertical"
            else:
                self.app.tracking_axis_mode = "horizontal"
            if old_mode != self.app.tracking_axis_mode:
                self.app.project_manager.project_dirty = True
                self.app.logger.info(f"Tracking axis mode set to: {self.app.tracking_axis_mode}", extra={'status_message': True})
                self.app.app_settings.set("tracking_axis_mode", self.app.tracking_axis_mode) # Auto-save
                self.app.energy_saver.reset_activity_timer()

        if self.app.tracking_axis_mode != "both":
            imgui.text("Output Single Axis To:")
            output_targets = ["Timeline 1 (Primary)", "Timeline 2 (Secondary)"]
            current_output_target_idx = 1 if self.app.single_axis_output_target == "secondary" else 0

            output_target_changed, new_output_target_idx = imgui.combo("##SingleAxisOutputComboGlobal", current_output_target_idx, output_targets)
            if output_target_changed:
                old_target = self.app.single_axis_output_target
                self.app.single_axis_output_target = "secondary" if new_output_target_idx == 1 else "primary"
                if old_target != self.app.single_axis_output_target:
                    self.app.project_manager.project_dirty = True
                    self.app.logger.info(f"Single axis output target set to: {self.app.single_axis_output_target}", extra={'status_message': True})
                    self.app.app_settings.set("single_axis_output_target", self.app.single_axis_output_target) # Auto-save
                    self.app.energy_saver.reset_activity_timer()
        if disable_axis_controls:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_oscillation_detector_settings(self):
        app = self.app
        settings = app.app_settings

        imgui.text("Analysis Grid Size")
        _tooltip_if_hovered(
            "Finer grids (higher numbers) are more precise but use more CPU.\n"
            "8=Very Coarse\n"
            "20=Balanced\n"
            "40=Fine\n"
            "80=Very Fine"
        )

        cur_grid = settings.get("oscillation_detector_grid_size", 20)
        imgui.push_item_width(200)
        ch, nv = imgui.slider_int("##GridSize", cur_grid, 8, 80)
        if ch:
            valid = [8, 10, 16, 20, 32, 40, 64, 80]
            closest = min(valid, key=lambda x: abs(x - nv))
            if closest != cur_grid:
                settings.set("oscillation_detector_grid_size", closest)
                tr = app.tracker
                if tr:
                    tr.update_oscillation_grid_size()
        imgui.same_line()
        if imgui.button("Reset##ResetGridSize"):
            default_grid = 20
            if cur_grid != default_grid:
                settings.set("oscillation_detector_grid_size", default_grid)
                tr = app.tracker
                if tr:
                    tr.update_oscillation_grid_size()
        imgui.pop_item_width()

        imgui.text("Detection Sensitivity")
        _tooltip_if_hovered(
            "Adjusts how sensitive the oscillation detector is to motion.\n"
            "Lower values = less sensitive, Higher values = more sensitive"
        )

        cur_sens = settings.get("oscillation_detector_sensitivity", 1.0)
        imgui.push_item_width(200)
        ch, nv = imgui.slider_float("##Sensitivity", cur_sens, 0.1, 3.0, "%.2f")
        if ch and nv != cur_sens:
            settings.set("oscillation_detector_sensitivity", nv)
            tr = app.tracker
            if tr:
                tr.update_oscillation_sensitivity()
        imgui.same_line()
        if imgui.button("Reset##ResetSensitivity"):
            default_sens = 1.0
            if cur_sens != default_sens:
                settings.set("oscillation_detector_sensitivity", default_sens)
                tr = app.tracker
                if tr:
                    tr.update_oscillation_sensitivity()
        imgui.pop_item_width()

        imgui.text("Oscillation Area Selection")
        _tooltip_if_hovered("Select a specific area for oscillation detection instead of the full frame.")

        tr = app.tracker
        has_area = tr and tr.oscillation_area_fixed
        btn_count = 2 if has_area else 1
        avail_w = imgui.get_content_region_available_width()
        btn_w = (
            (avail_w - imgui.get_style().item_spacing.x * (btn_count - 1)) / btn_count
            if btn_count > 1
            else -1
        )

        set_text = "Cancel Set Oscillation Area" if app.is_setting_oscillation_area_mode else "Set Oscillation Area"
        # Set Oscillation Area button - PRIMARY when starting, DESTRUCTIVE when canceling
        if app.is_setting_oscillation_area_mode:
            with destructive_button_style():
                if imgui.button("%s##SetOscillationArea" % set_text, width=btn_w):
                    app.exit_set_oscillation_area_mode()
        else:
            with primary_button_style():
                if imgui.button("%s##SetOscillationArea" % set_text, width=btn_w):
                    app.enter_set_oscillation_area_mode()

        if has_area:
            imgui.same_line()
            # Clear Oscillation Area button (DESTRUCTIVE - clears user data)
            with destructive_button_style():
                if imgui.button("Clear Oscillation Area##ClearOscillationArea", width=btn_w):
                    tr.clear_oscillation_area_and_point()
                if hasattr(app, "is_setting_oscillation_area_mode"):
                    app.is_setting_oscillation_area_mode = False
                gi = getattr(app, "gui_instance", None)
                if gi and hasattr(gi, "video_display_ui"):
                    v = gi.video_display_ui
                    v.is_drawing_oscillation_area = False
                    v.drawn_oscillation_area_video_coords = None
                    v.waiting_for_oscillation_point_click = False
                    v.oscillation_area_draw_start_screen_pos = (0, 0)
                    v.oscillation_area_draw_current_screen_pos = (0, 0)
                app.logger.info("Oscillation area cleared.", extra={"status_message": True})
        # Overlays
        imgui.text("Overlays")
        _tooltip_if_hovered("Visualization layers for the Oscillation Detector.")
        cur_overlay = settings.get("oscillation_show_overlay", getattr(tr, "show_masks", False))
        ch, nv_overlay = imgui.checkbox("Show Oscillation Overlay##OscShowOverlay", cur_overlay)
        if ch and nv_overlay != cur_overlay:
            settings.set("oscillation_show_overlay", nv_overlay)
            if hasattr(tr, "show_masks"):
                tr.show_masks = nv_overlay
        # Default ROI rectangle to enabled on first launch (True)
        cur_roi_overlay = settings.get("oscillation_show_roi_overlay", True)
        has_osc_area = bool(tr and getattr(tr, "oscillation_area_fixed", None))
        with _DisabledScope(not has_osc_area):
            ch, nv_roi_overlay = imgui.checkbox("Show ROI Rectangle##OscShowROIOverlay", cur_roi_overlay)
        if has_osc_area and ch and nv_roi_overlay != cur_roi_overlay:
            settings.set("oscillation_show_roi_overlay", nv_roi_overlay)
            if hasattr(tr, "show_roi"):
                tr.show_roi = nv_roi_overlay
        # Static grid blocks toggle (processed-frame grid visualization)
        cur_grid_blocks = settings.get("oscillation_show_grid_blocks", False)
        ch, nv_grid_blocks = imgui.checkbox("Show Static Grid Blocks##OscShowGridBlocks", cur_grid_blocks)
        if ch and nv_grid_blocks != cur_grid_blocks:
            settings.set("oscillation_show_grid_blocks", nv_grid_blocks)
            if hasattr(tr, "show_grid_blocks"):
                tr.show_grid_blocks = nv_grid_blocks

        imgui.text("Live Signal Amplification")
        _tooltip_if_hovered("Stretches the live signal to use the full 0-100 range based on recent motion.")

        en = settings.get("live_oscillation_dynamic_amp_enabled", True)
        ch, nv = imgui.checkbox("Enable Dynamic Amplification##EnableLiveAmp", en)
        if ch and nv != en:
            settings.set("live_oscillation_dynamic_amp_enabled", nv)

        # Legacy improvements settings
        imgui.separator()
        imgui.text("Signal Processing Improvements")
        
        # Simple amplification mode
        cur_simple_amp = settings.get("oscillation_use_simple_amplification", False)
        ch, nv_simple = imgui.checkbox("Use Simple Amplification##UseSimpleAmp", cur_simple_amp)
        if ch and nv_simple != cur_simple_amp:
            settings.set("oscillation_use_simple_amplification", nv_simple)
        _tooltip_if_hovered("Use legacy-style fixed multipliers (dy*-10, dx*10) instead of dynamic scaling")
        
        # Decay mechanism
        cur_decay = settings.get("oscillation_enable_decay", True)
        ch, nv_decay = imgui.checkbox("Enable Decay Mechanism##EnableDecay", cur_decay)
        if ch and nv_decay != cur_decay:
            settings.set("oscillation_enable_decay", nv_decay)
        _tooltip_if_hovered("Gradually return to center when no motion is detected")
        
        if cur_decay:
            # Hold duration
            imgui.text("Hold Duration (ms)")
            cur_hold = settings.get("oscillation_hold_duration_ms", 250)
            imgui.push_item_width(150)
            ch, nv_hold = imgui.slider_int("##HoldDuration", cur_hold, 50, 1000)
            if ch and nv_hold != cur_hold:
                settings.set("oscillation_hold_duration_ms", nv_hold)
            imgui.pop_item_width()
            _tooltip_if_hovered("How long to hold position before starting decay")
            
            # Decay factor
            imgui.text("Decay Factor")
            cur_decay_factor = settings.get("oscillation_decay_factor", 0.95)
            imgui.push_item_width(150)
            ch, nv_decay_factor = imgui.slider_float("##DecayFactor", cur_decay_factor, 0.85, 0.99, "%.3f")
            if ch and nv_decay_factor != cur_decay_factor:
                settings.set("oscillation_decay_factor", nv_decay_factor)
            imgui.pop_item_width()
            _tooltip_if_hovered("How quickly to decay towards center (0.95 = slow, 0.85 = fast)")

        imgui.new_line()
        imgui.text_ansi_colored("Note: Detection Sensitivity and Dynamic\nAmplification are currently not yet working.", 0.25, 0.88, 0.82)

        # TODO: Move values to constants
        if settings.get("live_oscillation_dynamic_amp_enabled", True):
            imgui.text("Analysis Window (ms)")
            cur_ms = settings.get("live_oscillation_amp_window_ms", 4000)
            imgui.push_item_width(200)
            ch, nv = imgui.slider_int("##LiveAmpWindow", cur_ms, 1000, 10000)
            if ch and nv != cur_ms:
                settings.set("live_oscillation_amp_window_ms", nv)
            imgui.same_line()
            if imgui.button("Reset##ResetAmpWindow"):
                default_ms = 4000
                if cur_ms != default_ms:
                    settings.set("live_oscillation_amp_window_ms", default_ms)
            imgui.pop_item_width()

    def _render_stage3_oscillation_detector_mode_settings(self):
        """Render UI for selecting oscillation detector mode in Stage 3"""
        app = self.app
        settings = app.app_settings
        
        imgui.text("Stage 3 Oscillation Detector Mode")
        _tooltip_if_hovered(
            "Choose which oscillation detector algorithm to use in Stage 3:\n\n"
            "Current: Uses the experimental oscillation detector with\n"
            "  adaptive motion detection and dynamic scaling\n\n"
            "Legacy: Uses the legacy oscillation detector from commit f5ae40f\n"
            "  with fixed amplification and explicit decay mechanisms\n\n"
            "Hybrid: Combines benefits from both approaches (future feature)"
        )
        
        current_mode = settings.get("stage3_oscillation_detector_mode", "current")
        mode_options = ["current", "legacy", "hybrid"]
        mode_display = ["Current (Experimental)", "Legacy (f5ae40f)", "Hybrid (Coming Soon)"]
        
        try:
            current_idx = mode_options.index(current_mode)
        except ValueError:
            current_idx = 0
            
        imgui.push_item_width(200)
        
        # Disable hybrid for now
        with _DisabledScope(current_idx == 2):  # hybrid not implemented yet
            clicked, new_idx = imgui.combo("##Stage3ODMode", current_idx, mode_display)
            
        if clicked and new_idx != current_idx and new_idx != 2:  # Don't allow selecting hybrid
            new_mode = mode_options[new_idx]
            settings.set("stage3_oscillation_detector_mode", new_mode)
            app.logger.info(f"Stage 3 Oscillation Detector mode set to: {new_mode}", extra={"status_message": True})
            
        imgui.pop_item_width()
        
        # Show current selection info
        if current_mode == "current":
            imgui.text_ansi_colored("Using experimental oscillation detector", 0.0, 0.8, 0.0)
        elif current_mode == "legacy":
            imgui.text_ansi_colored("Using legacy oscillation detector (f5ae40f)", 0.0, 0.6, 0.8)
        else:
            imgui.text_ansi_colored("Hybrid mode (not yet implemented)", 0.8, 0.6, 0.0)

# ------- Class filtering -------

    def _render_class_filtering_content(self):
        app = self.app
        classes = app.get_available_tracking_classes()
        if not classes:
            imgui.text_disabled("No classes available (model not loaded or no classes defined).")
            return

        imgui.text_wrapped("Select classes to DISCARD from tracking and analysis.")
        discarded = set(app.discarded_tracking_classes)
        changed_any = False
        num_cols = 3
        if imgui.begin_table("ClassFilterTable", num_cols, flags=imgui.TABLE_SIZING_STRETCH_SAME):
            col = 0
            for cls in classes:
                if col == 0:
                    imgui.table_next_row()
                imgui.table_set_column_index(col)
                is_discarded = (cls in discarded)
                imgui.push_id("discard_cls_%s" % cls)
                clicked, new_val = imgui.checkbox(" %s" % cls, is_discarded)
                imgui.pop_id()
                if clicked:
                    changed_any = True
                    if new_val:
                        discarded.add(cls)
                    else:
                        discarded.discard(cls)
                col = (col + 1) % num_cols
            imgui.end_table()

        if changed_any:
            new_list = sorted(list(discarded))
            if new_list != app.discarded_tracking_classes:
                app.discarded_tracking_classes = new_list
                app.app_settings.set("discarded_tracking_classes", new_list)
                app.project_manager.project_dirty = True
                app.logger.info("Discarded classes updated: %s" % new_list, extra={"status_message": True})
                app.energy_saver.reset_activity_timer()

        imgui.spacing()
        if imgui.button(
            "Clear All Discards##ClearDiscardFilters",
            width=imgui.get_content_region_available_width(),
        ):
            if app.discarded_tracking_classes:
                app.discarded_tracking_classes.clear()
                app.app_settings.set("discarded_tracking_classes", [])
                app.project_manager.project_dirty = True
                app.logger.info("All class discard filters cleared.", extra={"status_message": True})
                app.energy_saver.reset_activity_timer()
        _tooltip_if_hovered("Unchecks all classes, enabling all classes for tracking/analysis.")

# ------- ROI controls -------

    def _render_user_roi_controls_for_run_tab(self):
        app = self.app
        sp = app.stage_processor
        proc = app.processor

        imgui.spacing()

        set_disabled = sp.full_analysis_active or not (proc and proc.is_video_open())
        with _DisabledScope(set_disabled):
            tr = app.tracker
            has_roi = tr and tr.user_roi_fixed
            btn_count = 2 if has_roi else 1
            avail_w = imgui.get_content_region_available_width()
            btn_w = (
                (avail_w - imgui.get_style().item_spacing.x * (btn_count - 1)) / btn_count
                if btn_count > 1
                else -1
            )

            set_text = "Cancel Set ROI" if app.is_setting_user_roi_mode else "Set ROI & Point"
            # Set ROI button - PRIMARY when starting, DESTRUCTIVE when canceling
            if app.is_setting_user_roi_mode:
                with destructive_button_style():
                    if imgui.button("%s##UserSetROI_RunTab" % set_text, width=btn_w):
                        app.exit_set_user_roi_mode()
            else:
                with primary_button_style():
                    if imgui.button("%s##UserSetROI_RunTab" % set_text, width=btn_w):
                        app.enter_set_user_roi_mode()

            if has_roi:
                imgui.same_line()
                # Clear ROI button (DESTRUCTIVE - clears user data)
                with destructive_button_style():
                    if imgui.button("Clear ROI##UserClearROI_RunTab", width=btn_w):
                        if tr and hasattr(tr, "clear_user_defined_roi_and_point"):
                            tr.stop_tracking()
                            tr.clear_user_defined_roi_and_point()
                        app.logger.info("User ROI cleared.", extra={"status_message": True})

        if app.is_setting_user_roi_mode:
            col = self.ControlPanelColors.STATUS_WARNING
            imgui.text_ansi_colored("Selection Active: Draw ROI then click point on video.", *col)

# ------- Interactive refinement -------

    def _render_interactive_refinement_controls(self):
        app = self.app
        sp = app.stage_processor
        if not sp.stage2_overlay_data_map:
            return

        imgui.text("Interactive Refinement")
        disabled = sp.full_analysis_active or sp.refinement_analysis_active
        is_enabled = app.app_state_ui.interactive_refinement_mode_enabled

        with _DisabledScope(disabled):
            if is_enabled:
                g = self.GeneralColors
                imgui.push_style_color(imgui.COLOR_BUTTON, *g.RED_DARK)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *g.RED_LIGHT)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *g.RED)
                btn_text = "Disable Refinement Mode"
            else:
                btn_text = "Enable Refinement Mode"

            if imgui.button("%s##ToggleInteractiveRefinement" % btn_text, width=-1):
                app.app_state_ui.interactive_refinement_mode_enabled = not is_enabled

            if is_enabled:
                imgui.pop_style_color(3)

            _tooltip_if_hovered("Enables clicking on object boxes in the video to refine the script for that chapter.")

            if is_enabled:
                col = (
                    self.ControlPanelColors.STATUS_WARNING
                    if sp.refinement_analysis_active
                    else self.ControlPanelColors.STATUS_INFO
                )
                msg = "Refining chapter..." if sp.refinement_analysis_active else "Click a box in the video to start."
                imgui.text_ansi_colored(msg, *col)

        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Refinement is disabled while another process is active.")

# ------- Post-processing helpers -------

    def _render_post_processing_profile_row(self, long_name, profile_params, config_copy):
        changed_cfg = False
        imgui.push_id("profile_%s" % long_name)
        is_open = imgui.tree_node(long_name)

        if is_open:
            imgui.columns(2, "profile_settings", border=False)

            imgui.text("Amplification")

            imgui.text("Scale")
            imgui.next_column()
            imgui.push_item_width(-1)
            val = profile_params.get("scale_factor", 1.0)
            ch, nv = imgui.slider_float("##scale", val, 0.1, 5.0, "%.2f")
            if ch:
                profile_params["scale_factor"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Center")
            imgui.next_column()
            imgui.push_item_width(-1)
            val = profile_params.get("center_value", 50)
            ch, nv = imgui.slider_int("##amp_center", val, 0, 100)
            if ch:
                profile_params["center_value"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            clamp_low = profile_params.get("clamp_lower", 10)
            clamp_high = profile_params.get("clamp_upper", 90)

            imgui.text("Clamp Low")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch_l, nv_l = imgui.slider_int("##clamp_low", clamp_low, 0, 100)
            if ch_l:
                clamp_low = min(nv_l, clamp_high)
                profile_params["clamp_lower"] = clamp_low
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Clamp High")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch_h, nv_h = imgui.slider_int("##clamp_high", clamp_high, 0, 100)
            if ch_h:
                clamp_high = max(nv_h, clamp_low)
                profile_params["clamp_upper"] = clamp_high
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.columns(1)
            imgui.spacing()
            imgui.columns(2, "profile_settings_2", border=False)

            imgui.text("Smoothing (SG Filter)")

            imgui.text("Window")
            imgui.next_column()
            imgui.push_item_width(-1)
            sg_win = profile_params.get("sg_window", 7)
            ch, nv = imgui.slider_int("##sg_win", sg_win, 3, 99)
            if ch:
                nv = max(3, nv + 1 if nv % 2 == 0 else nv)
                if nv != sg_win:
                    profile_params["sg_window"] = nv
                    changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Polyorder")
            imgui.next_column()
            imgui.push_item_width(-1)
            sg_poly = profile_params.get("sg_polyorder", 3)
            max_poly = max(1, profile_params.get("sg_window", 7) - 1)
            cur_poly = min(sg_poly, max_poly)
            ch, nv = imgui.slider_int("##sg_poly", cur_poly, 1, max_poly)
            if ch and nv != sg_poly:
                profile_params["sg_polyorder"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Simplification (RDP)")

            imgui.text("Epsilon")
            imgui.next_column()
            imgui.push_item_width(-1)
            rdp_eps = profile_params.get("rdp_epsilon", 1.0)
            ch, nv = imgui.slider_float("##rdp_eps", rdp_eps, 0.1, 20.0, "%.2f")
            if ch and nv != rdp_eps:
                profile_params["rdp_epsilon"] = nv
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            out_min = profile_params.get("output_min", 0)
            out_max = profile_params.get("output_max", 100)

            imgui.text("Output Min")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch, nv = imgui.slider_int("##out_min", out_min, 0, 100)
            if ch:
                out_min = min(nv, out_max)
                profile_params["output_min"] = out_min
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Output Max")
            imgui.next_column()
            imgui.push_item_width(-1)
            ch, nv = imgui.slider_int("##out_max", out_max, 0, 100)
            if ch:
                out_max = max(nv, out_min)
                profile_params["output_max"] = out_max
                changed_cfg = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.columns(1)
            imgui.tree_pop()

        if changed_cfg:
            config_copy[long_name] = profile_params
        imgui.pop_id()
        return changed_cfg

    def _render_automatic_post_processing_new(self, fs_proc):
        app = self.app
        sp = app.stage_processor
        proc = app.processor

        proc_tools_disabled = sp.full_analysis_active or (proc and proc.is_processing) or app.is_setting_user_roi_mode
        with _DisabledScope(proc_tools_disabled):
            enabled = app.app_settings.get("enable_auto_post_processing", False)
            ch, nv = imgui.checkbox("Enable Automatic Post-Processing on Completion", enabled)
            if ch and nv != enabled:
                app.app_settings.set("enable_auto_post_processing", nv)
                app.project_manager.project_dirty = True
                app.logger.info("Automatic post-processing on completion %s." % ("enabled" if nv else "disabled"), extra={"status_message": True})
            _tooltip_if_hovered("If checked, the profiles below will be applied automatically\nafter an offline analysis or live tracking session finishes.")

            # Run post-processing button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Run Post-Processing Now##RunAutoPostProcessButton", width=-1):
                    if hasattr(fs_proc, "apply_automatic_post_processing"):
                        fs_proc.apply_automatic_post_processing()

            use_chapter = app.app_settings.get("auto_processing_use_chapter_profiles", True)
            ch, nv = imgui.checkbox("Apply Per-Chapter Settings (if available)", use_chapter)
            if ch and nv != use_chapter:
                app.app_settings.set("auto_processing_use_chapter_profiles", nv)
            _tooltip_if_hovered("If checked, applies specific profiles below to each chapter.\nIf unchecked, applies only the 'Default' profile to the entire script.")

            config = app.app_settings.get("auto_post_processing_amplification_config", {})
            config_copy = config.copy()
            master_changed = False

            if app.app_settings.get("auto_processing_use_chapter_profiles", True):
                imgui.text("Per-Position Processing Profiles")
                all_pos = ["Default"] + sorted(
                    list({info["long_name"] for info in self.constants.POSITION_INFO_MAPPING.values()})
                )
                default_profile = self.constants.DEFAULT_AUTO_POST_AMP_CONFIG.get("Default", {})
                for name in all_pos:
                    if not name:
                        continue
                    params = config_copy.get(name, default_profile).copy()
                    if self._render_post_processing_profile_row(name, params, config_copy):
                        master_changed = True
            else:
                imgui.text("Default Processing Profile (applies to all)")
                name = "Default"
                default_profile = self.constants.DEFAULT_AUTO_POST_AMP_CONFIG.get(name, {})
                params = config_copy.get(name, default_profile).copy()
                if self._render_post_processing_profile_row(name, params, config_copy):
                    master_changed = True

            if master_changed:
                app.app_settings.set("auto_post_processing_amplification_config", config_copy)
                app.project_manager.project_dirty = True

            if imgui.button("Reset All Profiles to Defaults##ResetAutoPostProcessing", width=-1):
                app.app_settings.set(
                    "auto_post_processing_amplification_config",
                    self.constants.DEFAULT_AUTO_POST_AMP_CONFIG,
                )
                app.project_manager.project_dirty = True
                app.logger.info("All post-processing profiles reset to defaults.", extra={"status_message": True})

            imgui.text("Final Smoothing Pass")
            en = app.app_settings.get("auto_post_proc_final_rdp_enabled", False)
            ch, nv = imgui.checkbox("Run Final RDP Pass to Seam Chapters", en)
            if ch and nv != en:
                app.app_settings.set("auto_post_proc_final_rdp_enabled", nv)
                app.project_manager.project_dirty = True
            _tooltip_if_hovered(
                "After all other processing, run one final simplification pass\n"
                "on the entire script. This can help smooth out the joints\n"
                "between chapters that used different processing settings."
            )

            if app.app_settings.get("auto_post_proc_final_rdp_enabled", False):
                imgui.same_line()
                imgui.push_item_width(120)
                cur_eps = app.app_settings.get("auto_post_proc_final_rdp_epsilon", 10.0)
                ch, nv = imgui.slider_float("Epsilon##FinalRDPEpsilon", cur_eps, 0.1, 20.0, "%.2f")
                if ch and nv != cur_eps:
                    app.app_settings.set("auto_post_proc_final_rdp_epsilon", nv)
                    app.project_manager.project_dirty = True
                imgui.pop_item_width()

        # Disabled tooltip
        if proc_tools_disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")

    # ------- Calibration -------

    def _render_latency_calibration(self, calibration_mgr):
        col = self.ControlPanelColors.STATUS_WARNING
        imgui.text_ansi_colored("--- LATENCY CALIBRATION MODE ---", *col)
        if not calibration_mgr.calibration_reference_point_selected:
            imgui.text_wrapped("1. Start the live tracker for 10s of action then pause it.")
            imgui.text_wrapped("   Select a clear action point on Timeline 1.")
        else:
            imgui.text_wrapped("1. Point at %.0fms selected." % calibration_mgr.calibration_timeline_point_ms)
            imgui.text_wrapped("2. Now, use video controls (seek, frame step) to find the")
            imgui.text_wrapped("   EXACT visual moment corresponding to the selected point.")
            imgui.text_wrapped("3. Press 'Confirm Visual Match' below.")
        if imgui.button("Confirm Visual Match##ConfirmCalibration", width=-1):
            if calibration_mgr.calibration_reference_point_selected:
                calibration_mgr.confirm_latency_calibration()
            else:
                self.app.logger.info("Please select a reference point on Timeline 1 first.", extra={"status_message": True})
        if imgui.button("Cancel Calibration##CancelCalibration", width=-1):
            calibration_mgr.is_calibration_mode_active = False
            calibration_mgr.calibration_reference_point_selected = False
            self.app.logger.info("Latency calibration cancelled.", extra={"status_message": True})
            self.app.energy_saver.reset_activity_timer()

    # ------- Range selection -------

    def _render_range_selection(self, stage_proc, fs_proc, event_handlers):
        app = self.app
        disabled = stage_proc.full_analysis_active or (app.processor and app.processor.is_processing) or app.is_setting_user_roi_mode

        with _DisabledScope(disabled):
            ch, new_active = imgui.checkbox("Enable Range Processing", fs_proc.scripting_range_active)
            if ch:
                event_handlers.handle_scripting_range_active_toggle(new_active)

            if fs_proc.scripting_range_active:
                imgui.text("Set Frames Range Manually (-1 = End):")
                imgui.push_item_width(imgui.get_content_region_available()[0] * 0.4)
                ch, nv = imgui.input_int(
                    "Start##SR_InputStart",
                    fs_proc.scripting_start_frame,
                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                )
                if ch:
                    event_handlers.handle_scripting_start_frame_input(nv)
                imgui.same_line()
                imgui.text(" ")
                imgui.same_line()
                ch, nv = imgui.input_int(
                    "End (-1)##SR_InputEnd",
                    fs_proc.scripting_end_frame,
                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                )
                if ch:
                    event_handlers.handle_scripting_end_frame_input(nv)
                imgui.pop_item_width()

                start_disp, end_disp = fs_proc.get_scripting_range_display_text()
                imgui.text("Active Range: Frames: %s to %s" % (start_disp, end_disp))
                sel_ch = fs_proc.selected_chapter_for_scripting
                if sel_ch:
                    imgui.text("Chapter: %s (%s)" % (sel_ch.class_name, sel_ch.segment_type))
                if imgui.button("Clear Range Selection##ClearRangeButton"):
                    event_handlers.clear_scripting_range_selection()
            else:
                imgui.text_disabled("Range processing not active. Enable checkbox or select a chapter.")

        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")

    

    # ------- Post-processing manual tools -------

    def _render_funscript_processing_tools(self, fs_proc, event_handlers):
        app = self.app
        sp = app.stage_processor
        proc = app.processor
        disabled = sp.full_analysis_active or (proc and proc.is_processing) or app.is_setting_user_roi_mode

        with _DisabledScope(disabled):
            axis_opts = ["Primary Axis", "Secondary Axis"]
            cur_idx = 0 if fs_proc.selected_axis_for_processing == "primary" else 1
            ch, nidx = imgui.combo("Target Axis##ProcAxis", cur_idx, axis_opts)
            if ch and nidx != cur_idx:
                event_handlers.set_selected_axis_for_processing("primary" if nidx == 0 else "secondary")
            # imgui.separator()

            imgui.text("Apply To:")
            range_label = fs_proc.get_operation_target_range_label()
            if imgui.radio_button(
                "%s##OpTargetRange" % range_label,
                fs_proc.operation_target_mode == "apply_to_scripting_range",
            ):
                fs_proc.operation_target_mode = "apply_to_scripting_range"
            imgui.same_line()
            if imgui.radio_button(
                "Selected Points##OpTargetSelect",
                fs_proc.operation_target_mode == "apply_to_selected_points",
            ):
                fs_proc.operation_target_mode = "apply_to_selected_points"

            def prep_op():
                if fs_proc.operation_target_mode == "apply_to_selected_points":
                    editor = (
                        self.timeline_editor1
                        if fs_proc.selected_axis_for_processing == "primary"
                        else self.timeline_editor2
                    )
                    fs_proc.current_selection_indices = list(
                        editor.multi_selected_action_indices
                    ) if editor else []
                    if not fs_proc.current_selection_indices:
                        app.logger.info("No points selected for operation.", extra={"status_message": True})

            imgui.text("Points operations")
            if imgui.button("Clamp to 0##Clamp0"):
                prep_op()
                fs_proc.handle_funscript_operation("clamp_0")
            imgui.same_line()
            if imgui.button("Clamp to 100##Clamp100"):
                prep_op()
                fs_proc.handle_funscript_operation("clamp_100")
            imgui.same_line()
            if imgui.button("Invert##InvertPoints"):
                prep_op()
                fs_proc.handle_funscript_operation("invert")
            imgui.same_line()
            # Clear button (DESTRUCTIVE - deletes all points)
            with destructive_button_style():
                if imgui.button("Clear##ClearPoints"):
                    prep_op()
                    fs_proc.handle_funscript_operation("clear")

            imgui.text("Amplify Values")
            ch, nv = imgui.slider_float("Factor##AmplifyFactor", fs_proc.amplify_factor_input, 0.1, 3.0, "%.2f")
            if ch:
                fs_proc.amplify_factor_input = nv
            ch, nv = imgui.slider_int("Center##AmplifyCenter", fs_proc.amplify_center_input, 0, 100)
            if ch:
                fs_proc.amplify_center_input = nv
            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply Amplify##ApplyAmplify"):
                    prep_op()
                    fs_proc.handle_funscript_operation("amplify")

            imgui.text("Savitzky-Golay Filter")
            ch, nv = imgui.slider_int("Window Length##SGWin", fs_proc.sg_window_length_input, 3, 99)
            if ch:
                event_handlers.update_sg_window_length(nv)
            max_po = max(1, fs_proc.sg_window_length_input - 1)
            po_val = min(fs_proc.sg_polyorder_input, max_po)
            ch, nv = imgui.slider_int("Polyorder##SGPoly", po_val, 1, max_po)
            if ch:
                fs_proc.sg_polyorder_input = nv
            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply Savitzky-Golay##ApplySG"):
                    prep_op()
                    fs_proc.handle_funscript_operation("apply_sg")

            imgui.text("RDP Simplification")
            ch, nv = imgui.slider_float("Epsilon##RDPEps", fs_proc.rdp_epsilon_input, 0.01, 20.0, "%.2f")
            if ch:
                fs_proc.rdp_epsilon_input = nv
            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply RDP##ApplyRDP"):
                    prep_op()
                    fs_proc.handle_funscript_operation("apply_rdp")

            imgui.text("Dynamic Amplification")
            if not hasattr(fs_proc, "dynamic_amp_window_ms_input"):
                fs_proc.dynamic_amp_window_ms_input = 4000
            ch, nv = imgui.slider_int("Window (ms)##DynAmpWin", fs_proc.dynamic_amp_window_ms_input, 500, 10000)
            if ch:
                fs_proc.dynamic_amp_window_ms_input = nv
            _tooltip_if_hovered("The size of the 'before/after' window in milliseconds to consider for amplification.")

            # Apply button (PRIMARY - positive action)
            with primary_button_style():
                if imgui.button("Apply Dynamic Amplify##ApplyDynAmp"):
                    prep_op()
                    fs_proc.handle_funscript_operation("apply_dynamic_amp")

        # If disabled, show a tooltip on hover (outside the disabled scope)
        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")

    def _render_device_control_tab(self):
        """Render device control tab content."""
        try:
            # Safety check: Don't initialize during first frame to avoid segfault
            # The app needs to be fully initialized before creating device manager
            if not hasattr(self, '_first_frame_rendered'):
                self._first_frame_rendered = False
            
            if not self._first_frame_rendered:
                imgui.text("Device Control initializing...")
                imgui.text("Please wait for application to fully load.")
                self._first_frame_rendered = True
                return
            
            # Initialize device control system lazily
            if not self._device_control_initialized:
                self._initialize_device_control()
                
            # If device control is available, render the UI
            if self.device_manager and self.param_manager:
                self._render_device_control_content()
            else:
                imgui.text("Device Control system failed to initialize.")
                imgui.text_colored("Check logs for details.", 1.0, 0.5, 0.0)
                if imgui.button("Retry Initialization"):
                    # Reset initialization flag to try again
                    self._device_control_initialized = False
                
        except Exception as e:
            imgui.text(f"Error in Device Control: {e}")
            imgui.text_colored("See logs for full details.", 1.0, 0.0, 0.0)
    
    def _initialize_device_control(self):
        """Initialize device control system for the control panel."""
        try:
            from device_control.device_manager import DeviceManager, DeviceControlConfig
            from device_control.device_parameterization import DeviceParameterManager
            
            self.app.logger.info("Device Control: Starting initialization...")
            
            # Create device manager with default config
            config = DeviceControlConfig(
                enable_live_tracking=True,
                enable_funscript_playback=True,
                preferred_backend="auto",
                log_device_commands=False  # Disable excessive logging in production
            )
            
            self.app.logger.info("Device Control: Creating DeviceManager...")
            self.device_manager = DeviceManager(config)
            
            # Share device manager with app for TrackerManager integration
            self.app.device_manager = self.device_manager
            self.app.logger.info("Device Control: DeviceManager created and shared with app")
            
            # Update existing tracker managers to use the shared device manager
            self._update_existing_tracker_managers()
            
            self.app.logger.info("Device Control: Creating DeviceParameterManager...")
            self.param_manager = DeviceParameterManager()
            self.app.logger.info("Device Control: DeviceParameterManager created successfully")
            
            # UI state already initialized in __init__
            
            self._device_control_initialized = True
            self.app.logger.info("Device Control initialized in Control Panel successfully")
            
        except Exception as e:
            self.app.logger.error(f"Failed to initialize Device Control: {e}")
            import traceback
            self.app.logger.error(f"Full traceback: {traceback.format_exc()}")
            self._device_control_initialized = True  # Mark as attempted
    
    def _update_existing_tracker_managers(self):
        """Update existing TrackerManagers to use the shared device manager."""
        try:
            # Check if app has tracker managers
            found_any = False
            for timeline_id in range(1, 3):  # Timeline 1 and 2
                tracker_manager = getattr(self.app, f'tracker_manager_{timeline_id}', None)
                if tracker_manager:
                    found_any = True
                    self.app.logger.info(f"Updating TrackerManager {timeline_id} with shared device manager")
                    # Re-initialize the device bridge with shared device manager
                    tracker_manager._init_device_bridge()
                    
                    # Also update live device control setting from current settings
                    live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
                    if live_tracking_enabled:
                        tracker_manager.set_live_device_control_enabled(True)
                        self.app.logger.info(f"TrackerManager {timeline_id} live control enabled from settings")
            
            if not found_any:
                self.app.logger.info("No existing TrackerManagers found to update")
                
        except Exception as e:
            self.app.logger.warning(f"Failed to update existing tracker managers: {e}")
            import traceback
            self.app.logger.warning(f"Traceback: {traceback.format_exc()}")
    
    def _render_device_control_content(self):
        """Render the main device control interface with improved UX."""
        # Connection Status Section
        if imgui.collapsing_header("Connection Status##DeviceConnectionStatus", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_connection_status_section()
        
        # Device Types (without header)
        imgui.indent(15)
        
        # OSR2/OSR6 Devices
        if imgui.collapsing_header("OSR2/OSR6 Devices (USB/Serial)##OSRDevices")[0]:
            self._render_osr_controls()
            
        # Buttplug.io Universal Devices  
        if imgui.collapsing_header("Buttplug.io Devices (Universal)##ButtplugDevices")[0]:
            self._render_buttplug_controls()
        
        # Handy Direct Control
        if imgui.collapsing_header("Handy (Direct API)##HandyDirect")[0]:
            self._render_handy_controls()
            
        imgui.unindent(15)
        
        # Settings Section (only show if connected)
        if self.device_manager.is_connected():
            if imgui.collapsing_header("Device Settings##DeviceSettings")[0]:
                self._render_device_settings_section()
    
    def _render_connection_status_section(self):
        """Render connection status section with consistent UX."""
        imgui.indent(15)
        
        if self.device_manager.is_connected():
            device_name = self.device_manager.get_connected_device_name()
            self._status_indicator(f"Connected to {device_name}", "ready", "Device is connected and ready")
            
            # Connection info
            device_info = self.device_manager.get_connected_device_info()
            if device_info:
                imgui.text(f"Device ID: {device_info.device_id}")
                imgui.text(f"Type: {device_info.device_type.value.title()}")
                
                # Quick position test
                imgui.separator()
                imgui.text("Quick Test:")
                current_pos = self.device_manager.current_position
                changed, new_pos = imgui.slider_float("Position##QuickTest", current_pos, 0.0, 100.0, "%.1f")
                if changed:
                    self.device_manager.update_position(new_pos, 50.0)
                
                _tooltip_if_hovered("Drag to test device movement")
            
            imgui.separator()
            if imgui.button("Disconnect Device"):
                self._disconnect_current_device()
        else:
            self._status_indicator("No device connected", "warning", "Connect a device below")
            imgui.text("Select and connect a device from the types below.")
        
        imgui.unindent(15)
    
    def _render_device_types_section(self):
        """Render device types section with consistent UX."""
        imgui.indent(15)
        
        # OSR2/OSR6 Devices
        if imgui.collapsing_header("OSR2/OSR6 Devices (USB/Serial)##OSRDevices")[0]:
            self._render_osr_controls()
            
        # Buttplug.io Universal Devices  
        if imgui.collapsing_header("Buttplug.io Devices (Universal)##ButtplugDevices")[0]:
            self._render_buttplug_controls()
        
        # Handy Direct Control
        if imgui.collapsing_header("Handy (Direct API)##HandyDirect")[0]:
            self._render_handy_controls()
            
        imgui.unindent(15)
    
    def _render_osr_controls(self):
        """Render OSR device controls."""
        imgui.indent(10)
            
        # Check OSR connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_osr_connected = connected_device and "osr" in connected_device.device_id.lower()
        
        if is_osr_connected:
            self._status_indicator(f"Connected to {connected_device.device_id}", "ready", "OSR device connected and ready")
            
            # Advanced OSR Settings
            if imgui.collapsing_header("OSR Performance Settings##OSRPerformance")[0]:
                self._render_osr_performance_settings()
                
            if imgui.collapsing_header("OSR Axis Configuration##OSRAxis")[0]:
                self._render_osr_axis_configuration()
                
            if imgui.collapsing_header("OSR Test Functions##OSRTest")[0]:
                imgui.indent(10)
                if imgui.button("Run Movement Test##OSR"):
                    self._test_osr_movement()
                _tooltip_if_hovered("Test OSR device with predefined movement sequence")
                imgui.unindent(10)
                
        else:
            imgui.text("Connect your OSR2/OSR6 device via USB cable.")
            
            imgui.separator()
            if imgui.button("Scan for OSR Devices##OSRScan"):
                self._scan_osr_devices()
            _tooltip_if_hovered("Search for connected OSR devices on serial ports")
            
            # Show available ports
            if self._available_osr_ports:
                imgui.spacing()
                imgui.text("Available devices:")
                for port_info in self._available_osr_ports:
                    port_name = port_info.get('device', 'Unknown')
                    description = port_info.get('description', 'No description')
                    
                    if imgui.button(f"Connect##OSR_{port_name}"):
                        self._connect_osr_device(port_name)
                    imgui.same_line()
                    imgui.text(f"{port_name} ({description})")
                    
            elif self._osr_scan_performed:
                imgui.spacing()
                self._status_indicator("No OSR devices found", "warning", "Try troubleshooting steps below")
                imgui.text("Troubleshooting:")
                imgui.bullet_text("Ensure OSR2/OSR6 is connected via USB")
                imgui.bullet_text("Check device is powered on")
                imgui.bullet_text("Try different USB cable or port")
        
        imgui.unindent(10)
    
    def _render_handy_controls(self):
        """Render Handy direct API controls."""
        imgui.indent(10)
        
        # Check Handy connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_handy_connected = connected_device and "handy" in connected_device.device_id.lower()
        
        if is_handy_connected:
            # Connected state
            self._status_indicator(f"Connected to {connected_device.name}", "ready", "Handy connected and ready")
            
            # Disconnect button
            if imgui.button("Disconnect Handy##HandyDisconnect"):
                self._disconnect_handy()
            _tooltip_if_hovered("Disconnect from Handy device")
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # Handy settings while connected
            imgui.text("Performance Settings:")
            imgui.indent(10)
            
            # Minimum interval between commands
            changed, value = imgui.slider_int(
                "Min Command Interval (ms)##HandyMinInterval",
                self.app.app_settings.get("handy_min_interval", 60),
                20, 200
            )
            if changed:
                self.app.app_settings.set("handy_min_interval", value)
            _tooltip_if_hovered("Minimum time between position commands (60ms recommended)")
            
            # Lookahead for smooth movement
            changed, value = imgui.slider_int(
                "Lookahead Time (ms)##HandyLookahead",
                self.app.app_settings.get("handy_lookahead_ms", 500),
                100, 2000
            )
            if changed:
                self.app.app_settings.set("handy_lookahead_ms", value)
            _tooltip_if_hovered("How far ahead to look for next position")
            
            imgui.unindent(10)
            
            # Test functions
            if imgui.collapsing_header("Test Functions##HandyTest")[0]:
                imgui.indent(10)
                if imgui.button("Test Movement##HandyTestMove"):
                    self._test_handy_movement()
                _tooltip_if_hovered("Test Handy with movement sequence")
                imgui.unindent(10)
                
        else:
            # Disconnected state - show connection controls
            imgui.text("Enter your Handy connection key:")
            
            # Connection key input
            connection_key = self.app.app_settings.get("handy_connection_key", "")
            changed, new_key = imgui.input_text(
                "##HandyConnectionKey",
                connection_key,
                256
            )
            if changed:
                self.app.app_settings.set("handy_connection_key", new_key)
            _tooltip_if_hovered("Your Handy connection key (e.g., 'DH7Hc')")
            
            imgui.spacing()
            
            # Connect button
            if connection_key and len(connection_key) > 0:
                if imgui.button("Connect to Handy##HandyConnect"):
                    self._connect_handy(connection_key)
                _tooltip_if_hovered("Connect to your Handy device")
            else:
                imgui.text_disabled("Enter connection key to enable connect button")
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # Help text
            imgui.text("How to get your connection key:")
            imgui.indent(10)
            imgui.bullet_text("Open the Handy app")
            imgui.bullet_text("Go to Settings > Connection")
            imgui.bullet_text("Copy the connection key")
            imgui.unindent(10)
            
            imgui.spacing()
            
            # Advanced settings even when disconnected
            if imgui.collapsing_header("Advanced Settings##HandyAdvanced")[0]:
                imgui.indent(10)
                
                # Minimum interval setting
                changed, value = imgui.slider_int(
                    "Min Command Interval (ms)##HandyMinIntervalAdv",
                    self.app.app_settings.get("handy_min_interval", 60),
                    20, 200
                )
                if changed:
                    self.app.app_settings.set("handy_min_interval", value)
                _tooltip_if_hovered("Minimum time between position commands (60ms recommended)")
                
                imgui.unindent(10)
        
        imgui.unindent(10)
    
    def _render_buttplug_controls(self):
        """Render Buttplug.io device controls."""
        imgui.indent(10)
        
        # Check Buttplug connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_buttplug_connected = connected_device and "buttplug" in connected_device.device_id.lower()
        
        if is_buttplug_connected:
            self._status_indicator(f"Connected to {connected_device.name}", "ready", "Buttplug device connected and ready")
            
            # Device capabilities
            if hasattr(connected_device, 'capabilities') and connected_device.capabilities:
                caps = connected_device.capabilities
                imgui.text("Device capabilities:")
                imgui.indent(10)
                if caps.supports_linear:
                    imgui.bullet_text(f"Linear motion: {caps.linear_channels} axis")
                if caps.supports_vibration:
                    imgui.bullet_text(f"Vibration: {caps.vibration_channels} motors")
                if caps.supports_rotation:
                    imgui.bullet_text(f"Rotation: {caps.rotation_channels} axis")
                imgui.bullet_text(f"Update rate: {caps.max_position_rate_hz} Hz")
                imgui.unindent(10)
            
            # Advanced Buttplug Settings
            if imgui.collapsing_header("Buttplug Test Functions##ButtplugTest")[0]:
                imgui.indent(10)
                if imgui.button("Run Movement Test##Buttplug"):
                    self._test_buttplug_movement()
                _tooltip_if_hovered("Test device with predefined movement sequence")
                imgui.unindent(10)
                
        else:
            imgui.text("Connect devices via Intiface Central")
            imgui.text("Supports 100+ devices: Handy, Lovense, Kiiroo, OSR2, and more")
            
            # Server configuration
            if imgui.collapsing_header("Buttplug Server Configuration##ButtplugServer")[0]:
                imgui.indent(10)
                
                # Server address
                current_address = self.app.app_settings.get("buttplug_server_address", "localhost")
                changed, new_address = imgui.input_text("Server Address##ButtplugAddr", current_address, 256)
                if changed:
                    self.app.app_settings.set("buttplug_server_address", new_address)
                _tooltip_if_hovered("IP address or hostname of Intiface Central server")
                
                # Server port
                current_port = self.app.app_settings.get("buttplug_server_port", 12345)
                changed, new_port = imgui.input_int("Port##ButtplugPort", current_port)
                if changed and 1024 <= new_port <= 65535:
                    self.app.app_settings.set("buttplug_server_port", new_port)
                _tooltip_if_hovered("WebSocket port (default: 12345)")
                
                imgui.unindent(10)
            
            imgui.separator()
            if imgui.button("Discover Devices##ButtplugDiscover"):
                self._discover_buttplug_devices()
            _tooltip_if_hovered("Search for devices through Intiface Central")
            
            imgui.same_line()
            if imgui.button("Check Server##ButtplugStatus"):
                self._check_buttplug_server_status()
            _tooltip_if_hovered("Test connection to Intiface Central server")
            
            # Show discovered devices
            if hasattr(self, '_discovered_buttplug_devices') and self._discovered_buttplug_devices:
                imgui.spacing()
                imgui.text(f"Found {len(self._discovered_buttplug_devices)} device(s):")
                
                for i, device_info in enumerate(self._discovered_buttplug_devices):
                    if imgui.button(f"Connect##buttplug_{i}"):
                        self._connect_specific_buttplug_device(device_info.device_id)
                    imgui.same_line()
                    imgui.text(f"{device_info.name} ({device_info.device_type.name})")
                
            elif hasattr(self, '_buttplug_discovery_performed') and self._buttplug_discovery_performed:
                imgui.spacing()
                self._status_indicator("No devices found", "warning", "Check troubleshooting steps below")
                imgui.text("Troubleshooting:")
                imgui.bullet_text("Start Intiface Central application")
                imgui.bullet_text("Enable Server Mode in Intiface")
                imgui.bullet_text("Connect and pair your devices")
                
        
        imgui.unindent(10)
    
    def _render_device_settings_section(self):
        """Render device settings section with consistent UX."""
        imgui.indent(15)
            
        # Performance Settings
        if imgui.collapsing_header("Device Performance##DevicePerformance")[0]:
            imgui.indent(10)
            config = self.device_manager.config
            
            # Update rate
            changed, new_rate = imgui.slider_float("Update Rate##DeviceRate", config.max_position_rate_hz, 1.0, 50.0, "%.1f Hz")
            if changed:
                config.max_position_rate_hz = new_rate
            _tooltip_if_hovered("How often device position is updated per second")
            
            # Position smoothing
            changed, new_smoothing = imgui.slider_float("Position Smoothing##DeviceSmooth", config.position_smoothing, 0.0, 1.0, "%.2f")
            if changed:
                config.position_smoothing = new_smoothing
            _tooltip_if_hovered("Smooths position changes to reduce jerkiness (0=no smoothing, 1=maximum smoothing)")
            
            # Latency compensation
            changed, new_latency = imgui.slider_int("Latency Compensation##DeviceLatency", config.latency_compensation_ms, 0, 200)
            if changed:
                config.latency_compensation_ms = new_latency
            _tooltip_if_hovered("Compensates for device response delay in milliseconds")
            
            imgui.unindent(10)
            
        
        # Live Control Integration
        if imgui.collapsing_header("Live Control Integration##DeviceLiveControl")[0]:
            imgui.indent(10)
            
            # Live tracking device control
            live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
            changed, new_live_tracking = imgui.checkbox("Live Tracking Control##DeviceLiveTracking", live_tracking_enabled)
            if changed:
                self.app.app_settings.set("device_control_live_tracking", new_live_tracking)
                self.app.app_settings.save_settings()
                self._update_live_tracking_control(new_live_tracking)
            _tooltip_if_hovered("Stream live tracker data directly to device in real-time")
            
            # Video playback device control  
            video_playback_enabled = self.app.app_settings.get("device_control_video_playback", False)
            changed, new_video_playback = imgui.checkbox("Video Playback Control##DeviceVideoPlayback", video_playback_enabled)
            if changed:
                self.app.app_settings.set("device_control_video_playback", new_video_playback)
                self.app.app_settings.save_settings()
                self._update_video_playback_control(new_video_playback)
            _tooltip_if_hovered("Sync device with video timeline and funscript playback")
            
            imgui.unindent(10)
            
        
        # Advanced Settings (only show if live control enabled)
        live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
        video_playback_enabled = self.app.app_settings.get("device_control_video_playback", False)
        
        if live_tracking_enabled or video_playback_enabled:
            if imgui.collapsing_header("Advanced Control Settings##DeviceAdvanced")[0]:
                imgui.indent(10)
                
                # Control intensity
                live_intensity = self.app.app_settings.get("device_control_live_intensity", 1.0)
                changed, new_intensity = imgui.slider_float("Control Intensity##DeviceIntensity", live_intensity, 0.1, 2.0, "%.2fx")
                if changed:
                    self.app.app_settings.set("device_control_live_intensity", new_intensity)
                    self.app.app_settings.save_settings()
                _tooltip_if_hovered("Multiplier for device movement intensity")
                
                imgui.unindent(10)
        
        imgui.unindent(15)
    
    def _disconnect_current_device(self):
        """Disconnect the currently connected device."""
        try:
            import threading
            import asyncio
            
            def run_disconnect():
                try:
                    # Try to use existing event loop first
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule disconnect in the existing loop
                            future = asyncio.run_coroutine_threadsafe(self.device_manager.stop(), loop)
                            future.result(timeout=10)  # Wait up to 10 seconds
                        else:
                            # Use the existing loop if not running
                            loop.run_until_complete(self.device_manager.stop())
                    except RuntimeError:
                        # No event loop exists, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self.device_manager.stop())
                        finally:
                            loop.close()
                    
                    self.app.logger.info("Device disconnected successfully")
                except Exception as e:
                    self.app.logger.error(f"Error during disconnect: {e}")
            
            thread = threading.Thread(target=run_disconnect, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to disconnect device: {e}")
    
    def _scan_osr_devices(self):
        """Scan for OSR devices specifically."""
        try:
            import threading
            def run_osr_scan():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Get OSR backend and scan
                    osr_backend = self.device_manager.available_backends.get('osr')
                    if osr_backend:
                        devices = loop.run_until_complete(osr_backend.discover_devices())
                        # Convert to simple format for UI
                        self._available_osr_ports = []
                        for device in devices:
                            self._available_osr_ports.append({
                                'device': device.device_id,
                                'description': device.name,
                                'manufacturer': getattr(device, 'manufacturer', 'Unknown')
                            })
                        self.app.logger.info(f"Found {len(devices)} potential OSR devices")
                        self._osr_scan_performed = True
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_osr_scan, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to scan OSR devices: {e}")
    
    def _connect_osr_device(self, port_name):
        """Connect to specific OSR device."""
        try:
            import threading
            import asyncio
            
            def run_osr_connect_and_loop():
                """Connect to OSR device and keep the async loop running."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def connect_and_run():
                    try:
                        success = await self.device_manager.connect(port_name)
                        if success:
                            self.app.logger.info(f"Connected to OSR device on {port_name}")
                            self.app.logger.info("Async loop running for device control - keeping alive for live tracking")
                            
                            # Keep the loop running forever to maintain the position update task
                            # This will only end when the application shuts down
                            try:
                                while True:
                                    await asyncio.sleep(1)  # Keep loop alive
                            except asyncio.CancelledError:
                                self.app.logger.info("Device manager loop cancelled")
                        else:
                            self.app.logger.error(f"Failed to connect to OSR device on {port_name}")
                    except Exception as e:
                        self.app.logger.error(f"Error in device connection loop: {e}")
                
                try:
                    # Store loop reference for potential cleanup
                    self.device_manager.loop = loop
                    loop.run_until_complete(connect_and_run())
                finally:
                    loop.close()
            
            # Start the persistent connection thread
            thread = threading.Thread(target=run_osr_connect_and_loop, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to connect OSR device: {e}")
    
    def _render_osr_performance_settings(self):
        """Render OSR performance tuning controls."""
        try:
            imgui.separator()
            imgui.text("Performance Settings:")
            
            # Get current settings or defaults
            sensitivity = self.app.app_settings.get("osr_sensitivity", 2.0)
            speed = self.app.app_settings.get("osr_speed", 2.0)
            
            # Sensitivity slider
            imgui.text("Sensitivity (how small movements trigger device):")
            changed_sens, new_sensitivity = imgui.slider_float("##osr_sensitivity", sensitivity, 0.5, 5.0, "%.1fx")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Higher = more responsive to small position changes\nLower = only responds to large movements")
            
            if changed_sens:
                self.app.app_settings.set("osr_sensitivity", new_sensitivity)
                self._update_osr_performance(new_sensitivity, speed)
            
            # Speed slider  
            imgui.text("Speed (how fast the device moves):")
            changed_speed, new_speed = imgui.slider_float("##osr_speed", speed, 0.5, 5.0, "%.1fx")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Higher = faster movements\nLower = slower, smoother movements")
            
            if changed_speed:
                self.app.app_settings.set("osr_speed", new_speed)
                self._update_osr_performance(sensitivity, new_speed)
            
            # Video playback amplification
            imgui.separator()
            imgui.text("Video Playback Amplification:")
            video_amp = self.app.app_settings.get("video_playback_amplification", 1.5)
            changed_amp, new_amp = imgui.slider_float("##video_amp", video_amp, 1.0, 3.0, "%.1fx")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Amplifies funscript movement during video playback\nHigher = more dramatic movement\n1.0x = original funscript range")
            
            if changed_amp:
                self.app.app_settings.set("video_playback_amplification", new_amp)
                self.app.logger.info(f"Video playback amplification set to {new_amp:.1f}x")
            
            # Reset button
            if imgui.button("Reset to Defaults##OSR_Performance"):
                self.app.app_settings.set("osr_sensitivity", 2.0)
                self.app.app_settings.set("osr_speed", 2.0)
                self.app.app_settings.set("video_playback_amplification", 1.5)
                self._update_osr_performance(2.0, 2.0)
                
        except Exception as e:
            self.app.logger.error(f"Error rendering OSR performance settings: {e}")
    
    def _update_osr_performance(self, sensitivity: float, speed: float):
        """Update OSR device performance settings."""
        try:
            # Get the OSR backend
            osr_backend = self.device_manager.available_backends.get('osr')
            if osr_backend and hasattr(osr_backend, 'set_performance_settings'):
                osr_backend.set_performance_settings(sensitivity, speed)
                self.app.logger.info(f"Updated OSR performance: sensitivity={sensitivity:.1f}x, speed={speed:.1f}x")
            else:
                self.app.logger.debug("OSR backend not available for performance update")
                
        except Exception as e:
            self.app.logger.error(f"Failed to update OSR performance: {e}")
    
    def _test_osr_movement(self):
        """Test OSR movement with a simple pattern."""
        try:
            import threading
            def run_test():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Check device manager state
                    if not self.device_manager:
                        self.app.logger.error("Device manager not initialized")
                        return
                    
                    # Check if any device is connected
                    if not self.device_manager.is_connected():
                        self.app.logger.error("No device connected. Please connect an OSR device first.")
                        return
                    
                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        self.app.logger.error("No connected backend available")
                        return
                    
                    # Check if the backend is actually connected
                    if not backend.is_connected():
                        self.app.logger.error("Backend reports not connected")
                        return
                    
                    self.app.logger.info("Starting OSR test movement pattern...")
                    self.app.logger.info(f"Using backend: {type(backend).__name__}")
                    
                    # Test pattern: center -> up -> center -> down -> center
                    test_positions = [
                        (50, "Center"),
                        (10, "Up"),  
                        (50, "Center"),
                        (90, "Down"),
                        (50, "Center")
                    ]
                    
                    for position, label in test_positions:
                        # Use the correct backend method
                        self.app.logger.info(f"Sending {label} position ({position}%) to device...")
                        success = loop.run_until_complete(backend.set_position(position, 50))
                        if success:
                            self.app.logger.debug(f"OSR test: {label} position ({position}%) - Success")
                        else:
                            self.app.logger.error(f"❌ OSR test: {label} position ({position}%) - Failed")
                        time.sleep(1.0)  # Hold position for 1 second
                    
                    self.app.logger.info("OSR test movement completed")
                        
                except Exception as e:
                    self.app.logger.error(f"Error during OSR test: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_test, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start OSR test movement: {e}")
    
    def _preview_axis_position(self, axis_key, tcode_position, message):
        """Preview a specific axis position in real-time."""
        try:
            if not self.device_manager.is_connected():
                self.app.logger.warning("No device connected for preview")
                return
            
            connected_device = self.device_manager.get_connected_device_info()
            if not connected_device or "osr" not in connected_device.device_id.lower():
                self.app.logger.warning("Preview only available for OSR devices")
                return
            
            # Get the OSR backend
            backend = self.device_manager.get_connected_backend()
            if not backend:
                self.app.logger.warning("No connected backend available for preview")
                return
            
            # Check backend connection status
            if not backend.is_connected():
                self.app.logger.warning("Backend not connected for preview")
                return
            
            self.app.logger.debug(f"Using backend: {type(backend).__name__} for axis preview")
            
            # Map axis key to TCode axis identifier (standard T-code protocol)
            axis_mapping = {
                'up_down': 'L0',         # Linear axis 0 (up/down stroke) 
                'left_right': 'L1',      # Linear axis 1 (left/right)
                'front_back': 'L2',      # Linear axis 2 (front/back)
                'twist': 'R0',           # Rotation axis 0 (twist/yaw)
                'roll': 'R1',            # Rotation axis 1 (roll)
                'pitch': 'R2',           # Rotation axis 2 (pitch)
                'vibration': 'V0',       # Vibration axis 0 (primary)
                'aux_vibration': 'V1'    # Vibration axis 1 (auxiliary)
            }
            
            tcode_axis = axis_mapping.get(axis_key)
            if not tcode_axis:
                self.app.logger.warning(f"Unknown axis key: {axis_key}")
                return
            
            # Convert TCode position (0-9999) to percentage (0-100) for backend
            position_percent = (tcode_position / 9999.0) * 100.0
            
            # Send command through backend's standardized axis method
            import threading
            def run_preview():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Log what we're about to send
                    self.app.logger.debug(f"Sending command: {tcode_axis} to {position_percent:.1f}% via {type(backend).__name__}")
                    
                    # Check if backend is still connected before sending
                    if not backend.is_connected():
                        self.app.logger.error(f"Backend disconnected before sending {tcode_axis} command")
                        return
                    
                    # Use backend's set_axis_position method instead of direct TCode
                    success = loop.run_until_complete(backend.set_axis_position(tcode_axis, position_percent))
                    
                    if success:
                        self.app.logger.info(f"{message}: {tcode_axis} axis to {position_percent:.1f}%")
                    else:
                        self.app.logger.error(f"Failed to set {tcode_axis} axis position - backend returned False")
                        
                except Exception as e:
                    self.app.logger.error(f"Failed to preview axis position: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_preview, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to preview axis position: {e}")
    
    def _demo_axis_range(self, axis_key, min_pos, max_pos, inverted, axis_label):
        """Demonstrate the full range of an axis with current settings."""
        try:
            if not self.device_manager.is_connected():
                self.app.logger.warning("No device connected for demo")
                return
            
            import threading
            def run_demo():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Get the OSR backend
                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        return
                    
                    # Map axis key to TCode axis identifier (standard T-code protocol)
                    axis_mapping = {
                        'up_down': 'L0',         # Linear axis 0 (up/down stroke) 
                        'left_right': 'L1',      # Linear axis 1 (left/right)
                        'front_back': 'L2',      # Linear axis 2 (front/back)
                        'twist': 'R0',           # Rotation axis 0 (twist/yaw)
                        'roll': 'R1',            # Rotation axis 1 (roll)
                        'pitch': 'R2',           # Rotation axis 2 (pitch)
                        'vibration': 'V0',       # Vibration axis 0 (primary)
                        'aux_vibration': 'V1'    # Vibration axis 1 (auxiliary)
                    }
                    
                    tcode_axis = axis_mapping.get(axis_key)
                    if not tcode_axis:
                        self.app.logger.warning(f"Unknown axis: {axis_key}")
                        return
                    
                    self.app.logger.info(f"Demonstrating {axis_label} range...")
                    
                    # Demo sequence: min → max → center (respecting inversion)
                    # Convert TCode positions (0-9999) to percentages (0-100) for backend
                    if inverted:
                        sequence = [
                            ((max_pos / 9999.0) * 100.0, "0% (inverted)"),
                            ((min_pos / 9999.0) * 100.0, "100% (inverted)"), 
                            (((min_pos + max_pos) / 2 / 9999.0) * 100.0, "50% (center)")
                        ]
                    else:
                        sequence = [
                            ((min_pos / 9999.0) * 100.0, "0% (normal)"),
                            ((max_pos / 9999.0) * 100.0, "100% (normal)"),
                            (((min_pos + max_pos) / 2 / 9999.0) * 100.0, "50% (center)")
                        ]
                    
                    for position_percent, label in sequence:
                        # Use backend's standardized axis method instead of direct TCode
                        success = loop.run_until_complete(backend.set_axis_position(tcode_axis, position_percent))
                        if success:
                            self.app.logger.info(f"{axis_label} demo: {label} → {tcode_axis} axis to {position_percent:.1f}%")
                        else:
                            self.app.logger.error(f"Failed to set {tcode_axis} axis to {position_percent:.1f}%")
                        time.sleep(2.0)  # Wait between movements
                    
                    self.app.logger.info(f"{axis_label} range demonstration complete")
                    
                except Exception as e:
                    self.app.logger.error(f"Failed to demo axis range: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_demo, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to start axis demo: {e}")
    
    def _simulate_axis_pattern(self, axis_key: str, pattern_type: str, min_pos: int, max_pos: int, inverted: bool, axis_label: str):
        """Simulate various motion patterns for axis testing."""
        try:
            import threading
            import math
            import random
            
            def run_pattern():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        return
                    
                    # Map axis key to TCode axis identifier
                    axis_mapping = {
                        'up_down': 'L0', 'left_right': 'L1', 'front_back': 'L2',
                        'twist': 'R0', 'roll': 'R1', 'pitch': 'R2',
                        'vibration': 'V0', 'aux_vibration': 'V1'
                    }
                    
                    tcode_axis = axis_mapping.get(axis_key)
                    if not tcode_axis:
                        self.app.logger.warning(f"Unknown axis: {axis_key}")
                        return
                    
                    self.app.logger.info(f"Starting {pattern_type} pattern for {axis_label}...")
                    
                    # Generate pattern positions
                    center_pos = (min_pos + max_pos) // 2
                    amplitude = (max_pos - min_pos) // 2
                    duration = 10.0  # 10 seconds
                    steps = 50  # Number of steps
                    dt = duration / steps
                    
                    positions = []
                    
                    if pattern_type == "sine_wave":
                        for i in range(steps):
                            t = (i / steps) * 4 * math.pi  # 2 full cycles
                            offset = amplitude * math.sin(t)
                            pos = center_pos + offset
                            positions.append((int(pos), f"Sine {i}/{steps}"))
                    
                    elif pattern_type == "square_wave":
                        for i in range(steps):
                            t = (i / steps) * 4  # 2 full cycles
                            pos = max_pos if (t % 2) < 1 else min_pos
                            positions.append((int(pos), f"Square {i}/{steps}"))
                    
                    elif pattern_type == "triangle_wave":
                        for i in range(steps):
                            t = (i / steps) * 4  # 2 full cycles
                            cycle_pos = t % 2
                            if cycle_pos < 1:
                                # Rising
                                pos = min_pos + (max_pos - min_pos) * cycle_pos
                            else:
                                # Falling
                                pos = max_pos - (max_pos - min_pos) * (cycle_pos - 1)
                            positions.append((int(pos), f"Triangle {i}/{steps}"))
                    
                    elif pattern_type == "random":
                        for i in range(20):  # Shorter for random
                            pos = random.randint(min_pos, max_pos)
                            positions.append((int(pos), f"Random {i}/20"))
                    
                    elif pattern_type == "pulse":
                        for i in range(10):  # 10 pulses
                            # Pulse out and back
                            positions.append((max_pos, f"Pulse {i} - Out"))
                            positions.append((center_pos, f"Pulse {i} - Back"))
                    
                    # Execute pattern
                    for pos, label in positions:
                        if inverted:
                            # Invert the position mapping
                            display_pos = max_pos + min_pos - pos
                        else:
                            display_pos = pos
                        
                        # Convert to percentage for backend
                        position_percent = (display_pos / 9999.0) * 100.0
                        
                        success = loop.run_until_complete(backend.set_axis_position(tcode_axis, position_percent))
                        if success:
                            self.app.logger.info(f"{axis_label} {pattern_type}: {label} → {tcode_axis} at {position_percent:.1f}%")
                        else:
                            self.app.logger.error(f"Failed to set {tcode_axis} to {position_percent:.1f}%")
                        
                        time.sleep(dt)
                    
                    # Return to center
                    center_percent = ((center_pos if not inverted else center_pos) / 9999.0) * 100.0
                    loop.run_until_complete(backend.set_axis_position(tcode_axis, center_percent))
                    self.app.logger.info(f"{axis_label} {pattern_type} pattern complete - returned to center")
                    
                except Exception as e:
                    self.app.logger.error(f"Error in {pattern_type} pattern: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_pattern, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to start {pattern_type} pattern: {e}")
    
    def _update_live_tracking_control(self, enabled: bool):
        """Update live tracking control setting in tracker manager."""
        try:
            # Get tracker manager from app
            tracker_manager = getattr(self.app, 'tracker_manager', None)
            self.app.logger.info(f"Updating live tracking control: enabled={enabled}, tracker_manager={tracker_manager is not None}")
            
            if tracker_manager and hasattr(tracker_manager, 'set_live_device_control_enabled'):
                tracker_manager.set_live_device_control_enabled(enabled)
                self.app.logger.info(f"Live tracking device control {'enabled' if enabled else 'disabled'}")
            else:
                self.app.logger.warning(f"Tracker manager not available for live device control: {tracker_manager}")
                
                # Try to find tracker managers by timeline ID
                for timeline_id in range(1, 3):
                    tm = getattr(self.app, f'tracker_manager_{timeline_id}', None)
                    if tm:
                        self.app.logger.info(f"Found tracker_manager_{timeline_id}, updating...")
                        tm.set_live_device_control_enabled(enabled)
                        
        except Exception as e:
            self.app.logger.error(f"Failed to update live tracking control: {e}")
            import traceback
            self.app.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_video_playback_control(self, enabled: bool):
        """Update video playback control setting."""
        try:
            # Setting is automatically picked up by timeline during video playback
            self.app.logger.info(f"Video playback device control {'enabled' if enabled else 'disabled'}")
            
            if enabled:
                # Verify device manager is available
                device_manager = getattr(self.app, 'device_manager', None)
                if device_manager and device_manager.is_connected():
                    self.app.logger.info("Device control ready for video playback")
                else:
                    self.app.logger.warning("No connected devices - video playback control will be inactive")
                    
        except Exception as e:
            self.app.logger.error(f"Failed to update video playback control: {e}")
    
    def _initialize_video_playback_bridge(self):
        """Initialize video playback bridge."""
        try:
            if self.device_manager:
                from device_control.bridges.video_playback_bridge import create_video_playback_bridge
                self.video_playback_bridge = create_video_playback_bridge(self.device_manager)
                self.app.logger.info("Video playback bridge initialized")
            else:
                self.app.logger.warning("Device manager not available for video playback bridge")
        except Exception as e:
            self.app.logger.error(f"Failed to initialize video playback bridge: {e}")
            self.video_playback_bridge = None
    
    
    def _render_osr_axis_configuration(self):
        """Render OSR axis configuration UI."""
        try:
            imgui.separator()
            imgui.text("OSR Axis Configuration")
            
            # Load current OSR settings
            current_profile_name = self.app.app_settings.get("device_control_selected_profile", "Balanced")
            osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})
            
            if current_profile_name not in osr_profiles:
                imgui.text_colored("No OSR profile found in settings", 1.0, 0.5, 0.0)
                return
            
            profile_data = osr_profiles[current_profile_name]
            
            # Profile selection
            imgui.text("Profile:")
            imgui.same_line()
            profile_names = list(osr_profiles.keys())
            current_index = profile_names.index(current_profile_name) if current_profile_name in profile_names else 0
            
            changed, new_index = imgui.combo("##profile_selector", current_index, profile_names)
            if changed and 0 <= new_index < len(profile_names):
                new_profile_name = profile_names[new_index]
                self.app.app_settings.set("device_control_selected_profile", new_profile_name)
                profile_data = osr_profiles[new_profile_name]
                self._load_osr_profile_to_device(new_profile_name, profile_data)
            
            imgui.text(f"Description: {profile_data.get('description', 'No description')}")
            
            # Axis configurations
            imgui.separator()
            imgui.text("Axis Settings:")
            
            axes_to_show = [
                # Linear axes
                ("up_down", "Up/Down Stroke", "L0"),
                ("left_right", "Left/Right", "L1"),
                ("front_back", "Front/Back", "L2"),
                # Rotation axes
                ("twist", "Twist", "R0"),
                ("roll", "Roll", "R1"), 
                ("pitch", "Pitch", "R2"),
                # Vibration axes
                ("vibration", "Vibration", "V0"),
                ("aux_vibration", "Aux Vibration", "V1")
            ]
            
            settings_changed = False
            
            for axis_key, axis_label, tcode in axes_to_show:
                if axis_key not in profile_data:
                    continue
                    
                axis_data = profile_data[axis_key]
                
                # Axis header with enable checkbox
                enabled = axis_data.get("enabled", False)
                changed, new_enabled = imgui.checkbox(f"{axis_label} ({tcode})", enabled)
                if changed:
                    axis_data["enabled"] = new_enabled
                    settings_changed = True
                
                if enabled:
                    imgui.indent(20)
                    
                    # Min/Max position sliders with real-time preview
                    min_pos = axis_data.get("min_position", 0)
                    max_pos = axis_data.get("max_position", 9999)
                    
                    imgui.text(f"{axis_label} Range:")
                    imgui.text_colored("Drag sliders to feel the limits in real-time", 0.7, 0.7, 0.7)
                    
                    changed, new_min = imgui.slider_int(f"Min Position##{axis_key}", min_pos, 0, 9999, f"%d (0%% limit)")
                    if changed:
                        axis_data["min_position"] = new_min
                        settings_changed = True
                        # Real-time preview: move to min position
                        self._preview_axis_position(axis_key, new_min, f"Previewing {axis_label} minimum")
                    
                    changed, new_max = imgui.slider_int(f"Max Position##{axis_key}", max_pos, 0, 9999, f"%d (100%% limit)")
                    if changed:
                        axis_data["max_position"] = new_max
                        settings_changed = True
                        # Real-time preview: move to max position
                        self._preview_axis_position(axis_key, new_max, f"Previewing {axis_label} maximum")
                    
                    # Range validation
                    if new_min >= new_max:
                        imgui.text_colored("Warning: Min must be less than Max", 1.0, 0.5, 0.0)
                    
                    # Preview buttons for testing limits
                    imgui.text("Test Range:")
                    if imgui.button(f"Test Min##{axis_key}"):
                        self._preview_axis_position(axis_key, new_min, f"Testing {axis_label} minimum (0%)")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Move to minimum position (funscript 0%)")
                    
                    imgui.same_line()
                    if imgui.button(f"Test Max##{axis_key}"):
                        self._preview_axis_position(axis_key, new_max, f"Testing {axis_label} maximum (100%)")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Move to maximum position (funscript 100%)")
                    
                    imgui.same_line()
                    if imgui.button(f"Center##{axis_key}"):
                        center_pos = (new_min + new_max) // 2
                        self._preview_axis_position(axis_key, center_pos, f"Centering {axis_label} (50%)")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Move to center position (funscript 50%)")
                    
                    # Speed multiplier
                    speed_mult = axis_data.get("speed_multiplier", 1.0)
                    changed, new_speed = imgui.slider_float(f"Speed Multiplier##{axis_key}", speed_mult, 0.1, 3.0, "%.2f")
                    if changed:
                        axis_data["speed_multiplier"] = new_speed
                        settings_changed = True
                    
                    # Invert checkbox with preview
                    invert = axis_data.get("invert", False)
                    changed, new_invert = imgui.checkbox(f"Invert Direction##{axis_key}", invert)
                    if changed:
                        axis_data["invert"] = new_invert
                        settings_changed = True
                        # Preview inversion by showing the effect
                        if new_invert:
                            # Show inverted max (funscript 0% → device max)
                            self._preview_axis_position(axis_key, new_max, f"Previewing {axis_label} INVERTED: funscript 0% → device max")
                        else:
                            # Show normal min (funscript 0% → device min)
                            self._preview_axis_position(axis_key, new_min, f"Previewing {axis_label} NORMAL: funscript 0% → device min")
                    
                    # Pattern simulation buttons
                    imgui.separator()
                    imgui.text(f"{axis_label} Simulation Patterns:")
                    
                    # Row 1: Basic patterns
                    if imgui.button(f"Demo Range##{axis_key}"):
                        self._demo_axis_range(axis_key, new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Test min → max → center positions")
                    
                    imgui.same_line()
                    if imgui.button(f"Sine Wave##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "sine_wave", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Smooth sine wave motion")
                    
                    imgui.same_line()
                    if imgui.button(f"Square Wave##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "square_wave", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Sharp min/max transitions")
                    
                    # Row 2: Complex patterns
                    if imgui.button(f"Triangle Wave##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "triangle_wave", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Linear ramp up/down motion")
                    
                    imgui.same_line()
                    if imgui.button(f"Random Pattern##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "random", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Random positions for testing")
                    
                    imgui.same_line()
                    if imgui.button(f"Pulse Pattern##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "pulse", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Quick pulses from center")
                    
                    # Smoothing
                    smoothing = axis_data.get("smoothing_factor", 0.8)
                    changed, new_smoothing = imgui.slider_float(f"Smoothing##{axis_key}", smoothing, 0.0, 1.0, "%.2f")
                    if changed:
                        axis_data["smoothing_factor"] = new_smoothing
                        settings_changed = True
                    
                    # Pattern generation settings for this axis
                    imgui.separator()
                    imgui.text(f"{axis_label} Pattern Generation:")
                    
                    # Pattern type dropdown (generalized for all axes)
                    pattern_types = ["disabled", "wave", "follow", "auto"]
                    pattern_labels = ["Disabled", "Wave (Smooth)", "Follow Primary", "Auto-Select"]
                    current_pattern = axis_data.get("pattern_type", "disabled")
                    current_pattern_index = pattern_types.index(current_pattern) if current_pattern in pattern_types else 0
                    
                    changed, new_pattern_index = imgui.combo(f"Pattern Type##{axis_key}", current_pattern_index, pattern_labels)
                    if changed and 0 <= new_pattern_index < len(pattern_types):
                        axis_data["pattern_type"] = pattern_types[new_pattern_index]
                        settings_changed = True
                    
                    # Pattern intensity (only if not disabled)
                    if axis_data.get("pattern_type", "disabled") != "disabled":
                        intensity = axis_data.get("pattern_intensity", 1.0)
                        changed, new_intensity = imgui.slider_float(f"Pattern Intensity##{axis_key}", intensity, 0.0, 2.0, "%.2f")
                        if changed:
                            axis_data["pattern_intensity"] = new_intensity
                            settings_changed = True
                        
                        # Pattern frequency (only if not disabled)
                        frequency = axis_data.get("pattern_frequency", 1.0)
                        changed, new_frequency = imgui.slider_float(f"Pattern Frequency##{axis_key}", frequency, 0.1, 5.0, "%.2f")
                        if changed:
                            axis_data["pattern_frequency"] = new_frequency
                            settings_changed = True
                    
                    imgui.unindent(20)
                
                imgui.separator()
            
            # Global settings
            imgui.text("Global Settings:")
            
            # Update rate
            update_rate = profile_data.get("update_rate_hz", 20.0)
            changed, new_rate = imgui.slider_float("Update Rate (Hz)", update_rate, 5.0, 50.0, "%.1f")
            if changed:
                profile_data["update_rate_hz"] = new_rate
                settings_changed = True
            
            # Safety limits
            safety_enabled = profile_data.get("safety_limits_enabled", True)
            changed, new_safety = imgui.checkbox("Safety Limits Enabled", safety_enabled)
            if changed:
                profile_data["safety_limits_enabled"] = new_safety
                settings_changed = True
            
            # Apply button
            imgui.separator()
            if imgui.button("Apply Configuration"):
                self._load_osr_profile_to_device(current_profile_name, profile_data)
                settings_changed = True
            
            imgui.same_line()
            if imgui.button("Test Axis Movement"):
                self._test_osr_axes()
            
            # Save settings if changed
            if settings_changed:
                osr_profiles[current_profile_name] = profile_data
                self.app.app_settings.set("device_control_osr_profiles", osr_profiles)
                self.app.app_settings.save_settings()
                
        except Exception as e:
            imgui.text_colored(f"Error in OSR configuration: {e}", 1.0, 0.0, 0.0)
    
    def _load_osr_profile_to_device(self, profile_name: str, profile_data: dict):
        """Load OSR profile to the connected device."""
        try:
            # Import axis control here to avoid circular imports
            from device_control.axis_control import load_profile_from_settings
            
            # Convert settings to OSRControlProfile
            profile = load_profile_from_settings(profile_data)
            
            # Get the OSR backend and load the profile
            backend = self.device_manager.get_connected_backend()
            if backend and hasattr(backend, 'load_axis_profile'):
                success = backend.load_axis_profile(profile)
                if success:
                    self.app.logger.info(f"Loaded OSR profile '{profile_name}' to device")
                else:
                    self.app.logger.error(f"Failed to load OSR profile '{profile_name}' to device")
                        
        except Exception as e:
            self.app.logger.error(f"Error loading OSR profile to device: {e}")
    
    def _test_osr_axes(self):
        """Test OSR axes with a simple movement pattern."""
        try:
            import threading
            def run_test():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._test_osr_movement_async())
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_test, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start OSR test: {e}")
    
    async def _test_osr_movement_async(self):
        """Test OSR movement pattern."""
        try:
            backend = self.device_manager.get_connected_backend()
            if backend and hasattr(backend, 'set_position_with_profile'):
                # Test pattern: 0 -> 50 -> 100 -> 50 -> 0
                test_positions = [0.0, 50.0, 100.0, 50.0, 0.0]
                
                import asyncio
                for pos in test_positions:
                    await backend.set_position_with_profile(pos)
                    await asyncio.sleep(1.0)  # Hold position for 1 second
                
                self.app.logger.info("OSR axis test completed")
                    
        except Exception as e:
            self.app.logger.error(f"OSR test movement failed: {e}")
    
    
    def _open_intiface_download(self):
        """Open Intiface Central download page."""
        try:
            import webbrowser
            webbrowser.open("https://intiface.com/central/")
            self.app.logger.info("Opened Intiface Central download page")
        except Exception as e:
            self.app.logger.error(f"Failed to open Intiface download page: {e}")
    
    def _discover_buttplug_devices(self):
        """Discover available Buttplug devices using current server settings."""
        try:
            import threading
            def run_buttplug_discovery():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Use the device manager's existing Buttplug backend to ensure consistency
                    if self.device_manager and 'buttplug' in self.device_manager.available_backends:
                        backend = self.device_manager.available_backends['buttplug']
                        server_url = backend.server_address
                        self.app.logger.info(f"Discovering Buttplug devices at {server_url}...")
                        devices = loop.run_until_complete(backend.discover_devices())
                    else:
                        # Fallback: Create temporary backend for discovery
                        from device_control.backends.buttplug_backend_direct import DirectButtplugBackend
                        
                        server_address = self.app.app_settings.get("buttplug_server_address", "localhost")
                        server_port = self.app.app_settings.get("buttplug_server_port", 12345)
                        server_url = f"ws://{server_address}:{server_port}"
                        
                        self.app.logger.info(f"Discovering Buttplug devices at {server_url}...")
                        backend = DirectButtplugBackend(server_url)
                        devices = loop.run_until_complete(backend.discover_devices())
                    
                    # Store discovered devices for UI display
                    self._discovered_buttplug_devices = devices
                    self._buttplug_discovery_performed = True
                    
                    if devices:
                        self.app.logger.debug(f"Found {len(devices)} Buttplug device(s):")
                        for device in devices:
                            caps = []
                            if device.capabilities.supports_linear:
                                caps.append(f"Linear({device.capabilities.linear_channels}ch)")
                            if device.capabilities.supports_vibration:
                                caps.append(f"Vibration({device.capabilities.vibration_channels}ch)")
                            if device.capabilities.supports_rotation:
                                caps.append(f"Rotation({device.capabilities.rotation_channels}ch)")
                            
                            self.app.logger.info(f"  • {device.name} - {', '.join(caps) if caps else 'No capabilities'}")
                    else:
                        self.app.logger.info("❌ No Buttplug devices found")
                        self.app.logger.info("Make sure Intiface Central is running and devices are connected")
                        
                except Exception as e:
                    self._buttplug_discovery_performed = True
                    if "Connection refused" in str(e) or "Connect call failed" in str(e):
                        self.app.logger.info(f"❌ Cannot connect to Intiface Central at {server_url}")
                        self.app.logger.info("Please start Intiface Central and enable server mode")
                    else:
                        self.app.logger.error(f"Buttplug discovery error: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_buttplug_discovery, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start Buttplug discovery: {e}")
    
    def _connect_specific_buttplug_device(self, device_id):
        """Connect to a specific Buttplug device by ID."""
        try:
            import threading
            def run_buttplug_connection():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(self.device_manager.connect(device_id))
                    if success:
                        # Find the device name for logging
                        device_name = "Unknown Device"
                        if hasattr(self, '_discovered_buttplug_devices'):
                            for device in self._discovered_buttplug_devices:
                                if device.device_id == device_id:
                                    device_name = device.name
                                    break
                        
                        self.app.logger.info(f"Connected to {device_name}")
                    else:
                        self.app.logger.error(f"❌ Failed to connect to device {device_id}")
                        
                except Exception as e:
                    self.app.logger.error(f"Buttplug connection failed: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_buttplug_connection, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to connect to Buttplug device: {e}")
    
    def _check_buttplug_server_status(self):
        """Check if Buttplug server is running at configured address/port."""
        try:
            import threading
            def run_status_check():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def check_server():
                    try:
                        server_address = self.app.app_settings.get("buttplug_server_address", "localhost")
                        server_port = self.app.app_settings.get("buttplug_server_port", 12345)
                        server_url = f"ws://{server_address}:{server_port}"
                        
                        # Try to connect briefly to check status
                        try:
                            import websockets
                            import json
                            
                            import asyncio
                            websocket = await asyncio.wait_for(
                                websockets.connect(server_url), timeout=5
                            )
                            
                            # Send handshake
                            handshake = {
                                "RequestServerInfo": {
                                    "Id": 1,
                                    "ClientName": "VR-Funscript-AI-Generator-StatusCheck",
                                    "MessageVersion": 3
                                }
                            }
                            
                            await websocket.send(json.dumps([handshake]))
                            response = await websocket.recv()
                            response_data = json.loads(response)
                            
                            await websocket.close()
                            
                            if response_data and len(response_data) > 0 and 'ServerInfo' in response_data[0]:
                                server_info = response_data[0]['ServerInfo']
                                server_name = server_info.get('ServerName', 'Unknown')
                                server_version = server_info.get('MessageVersion', 'Unknown')
                                
                                self.app.logger.debug(f"Buttplug server running at {server_url}")
                                self.app.logger.info(f"   Server: {server_name} (Protocol v{server_version})")
                            else:
                                self.app.logger.debug(f"Connected to {server_url} but unexpected response")
                                
                        except Exception as connection_error:
                            if "Connection refused" in str(connection_error):
                                self.app.logger.info(f"❌ Buttplug server not running at {server_url}")
                                self.app.logger.info("Please start Intiface Central and enable server mode")
                            else:
                                self.app.logger.error(f"Server status check failed: {connection_error}")
                            
                    except Exception as e:
                        self.app.logger.error(f"Failed to check server status: {e}")
                
                try:
                    loop.run_until_complete(check_server())
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_status_check, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start server status check: {e}")
    
    def _test_buttplug_movement(self):
        """Test movement for connected Buttplug device."""
        try:
            import threading
            def run_movement_test():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if not self.device_manager.is_connected():
                        self.app.logger.info("No device connected for movement test")
                        return
                    
                    self.app.logger.debug("Testing Buttplug device movement...")
                    
                    # Test sequence with good timing
                    positions = [0, 100, 25, 75, 50]
                    for i, pos in enumerate(positions):
                        loop.run_until_complete(asyncio.sleep(0.8))  # Wait between positions
                        self.device_manager.update_position(pos, 50.0)
                        self.app.logger.info(f"   Step {i+1}/{len(positions)}: Position {pos}%")
                    
                    # Return to center
                    loop.run_until_complete(asyncio.sleep(0.8))
                    self.device_manager.update_position(50.0, 50.0)
                    self.app.logger.debug("Movement test complete")
                    
                except Exception as e:
                    self.app.logger.error(f"Movement test failed: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_movement_test, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start movement test: {e}")
    
    def _connect_handy(self, connection_key: str):
        """Connect to Handy device with given connection key."""
        try:
            import threading
            def connect_handy_async():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    from device_control.backends.handy_hdsp_backend import HandyHDSPBackend, HandyHDSPConfig
                    
                    # Create configuration
                    config = HandyHDSPConfig(
                        connection_key=connection_key,
                        use_time_based_control=True,
                        immediate_response=False,
                        stop_on_target=False
                    )
                    
                    # Initialize backend
                    backend = HandyHDSPBackend(config)
                    
                    # Discover and connect
                    devices = loop.run_until_complete(backend.discover_devices())
                    if devices:
                        device = devices[0]
                        success = loop.run_until_complete(backend.connect(device.device_id))
                        if success:
                            # Properly register with device manager
                            self.device_manager.connected_devices[device.device_id] = backend
                            
                            # Import the proper state enum
                            from device_control.device_manager import DeviceManagerState
                            self.device_manager.state = DeviceManagerState.CONNECTED
                            
                            # Store backend reference for other components
                            self.device_manager._handy_backend = backend
                            
                            self.app.logger.info(f"Connected to Handy: {device.name}")
                        else:
                            self.app.logger.error("Failed to connect to Handy")
                    else:
                        self.app.logger.error("No Handy device found with this connection key")
                        
                except Exception as e:
                    self.app.logger.error(f"Handy connection failed: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=connect_handy_async, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to start Handy connection: {e}")
    
    def _disconnect_handy(self):
        """Disconnect from Handy device."""
        try:
            import threading
            def disconnect_handy_async():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if hasattr(self.device_manager, '_handy_backend') and self.device_manager._handy_backend:
                        backend = self.device_manager._handy_backend
                        loop.run_until_complete(backend.disconnect())
                        
                        # Properly clean up device manager state
                        device_info = backend.get_device_info()
                        if device_info and device_info.device_id in self.device_manager.connected_devices:
                            del self.device_manager.connected_devices[device_info.device_id]
                        
                        # Import the proper state enum
                        from device_control.device_manager import DeviceManagerState
                        self.device_manager.state = DeviceManagerState.DISCONNECTED
                        
                        self.device_manager._handy_backend = None
                        
                        self.app.logger.info("Disconnected from Handy")
                except Exception as e:
                    self.app.logger.error(f"Handy disconnection failed: {e}")
                finally:
                    loop.close()
            
            thread = threading.Thread(target=disconnect_handy_async, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to disconnect Handy: {e}")
    
    def _test_handy_movement(self):
        """Test Handy device movement."""
        try:
            import threading
            def test_handy_async():
                import asyncio
                
                async def run_test():
                    try:
                        self.app.logger.info(f"Device manager has _handy_backend: {hasattr(self.device_manager, '_handy_backend')}")
                        if hasattr(self.device_manager, '_handy_backend'):
                            self.app.logger.info(f"_handy_backend value: {self.device_manager._handy_backend}")
                        
                        if not hasattr(self.device_manager, '_handy_backend') or not self.device_manager._handy_backend:
                            self.app.logger.error("No Handy connected")
                            return
                        
                        backend = self.device_manager._handy_backend
                        self.app.logger.info(f"Backend type: {type(backend)}")
                        self.app.logger.info(f"Backend connected: {backend.is_connected()}")
                        self.app.logger.info("Testing Handy movement...")
                        
                        # Test sequence: position, duration_ms (short durations for immediate testing)
                        positions = [(20, 50), (80, 50), (50, 50), (30, 50), (70, 50), (50, 50)]
                        
                        for i, (pos, duration) in enumerate(positions):
                            try:
                                self.app.logger.info(f"   Calling set_position_enhanced({pos}, duration_ms={duration})")
                                success = await backend.set_position_enhanced(
                                    primary=pos,
                                    duration_ms=duration,
                                    movement_type="test"
                                )
                                self.app.logger.info(f"   set_position_enhanced returned: {success}")
                                
                                if success:
                                    self.app.logger.info(f"   Step {i+1}/{len(positions)}: Position {pos}% in {duration}ms")
                                else:
                                    self.app.logger.error(f"   Step {i+1} failed")
                                
                                # Wait for movement to complete
                                await asyncio.sleep(duration / 1000.0 + 0.2)
                                
                            except Exception as e:
                                self.app.logger.error(f"   Step {i+1} error: {e}")
                                import traceback
                                self.app.logger.error(f"   Traceback: {traceback.format_exc()}")
                        
                        # Return to center
                        try:
                            await backend.stop()
                            self.app.logger.info("Handy movement test complete")
                        except Exception as e:
                            self.app.logger.error(f"Failed to stop Handy: {e}")
                        
                    except Exception as e:
                        self.app.logger.error(f"Handy test failed: {e}")
                
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(run_test())
                finally:
                    loop.close()
            
            thread = threading.Thread(target=test_handy_async, daemon=True)
            thread.start()
            
        except Exception as e:
            self.app.logger.error(f"Failed to start Handy test: {e}")

    def _render_native_sync_tab(self):
        """Render streamer tab content."""
        try:
            # Initialize streamer manager lazily
            if self._native_sync_manager is None:
                from streamer.integration_manager import NativeSyncManager
                try:
                    # Try with app_logic parameter (newer streamer versions)
                    self._native_sync_manager = NativeSyncManager(
                        self.app.processor,
                        logger=self.app.logger,
                        app_logic=self.app  # For HereSphere auto-load functionality
                    )
                except TypeError:
                    # Fall back to old signature (older streamer versions)
                    self.app.logger.warning("Streamer version doesn't support app_logic parameter - using backward-compatible initialization")
                    self._native_sync_manager = NativeSyncManager(
                        self.app.processor,
                        logger=self.app.logger
                    )
            
            # Initialize local video browser attributes if they don't exist
            if not hasattr(self, '_local_video_folder'):
                self._local_video_folder = ""
            if not hasattr(self, '_local_video_files'):
                self._local_video_files = []
            if not hasattr(self, '_selected_local_video'):
                self._selected_local_video = ""

            # Cache status to avoid expensive lookups every frame (throttle to 500ms)
            import time
            current_time = time.time()

            # Update cache if stale (> 500ms)
            if self._native_sync_status_cache is None or (current_time - self._native_sync_status_time) > 0.5:
                self._native_sync_status_cache = self._native_sync_manager.get_status()
                self._native_sync_status_time = current_time

            # Use cached status
            status = self._native_sync_status_cache
            is_running = status.get('is_running', False)
            client_count = status.get('connected_clients', 0)

            # Auto-hide/show video feed based on client connections
            if is_running:
                # Initialize setting if not exists
                if not hasattr(self.app.app_settings, '_streamer_auto_hide_video'):
                    self.app.app_settings._streamer_auto_hide_video = True

                auto_hide_enabled = getattr(self.app.app_settings, '_streamer_auto_hide_video', True)

                # Track previous client count
                if not hasattr(self, '_prev_client_count'):
                    self._prev_client_count = 0

                # Client connected (0 -> >0)
                if auto_hide_enabled and client_count > 0 and self._prev_client_count == 0:
                    self.app.app_state_ui.show_video_feed = False
                    self.app.app_settings.set("show_video_feed", False)
                    self.app.logger.info("🎥 Auto-hiding video feed (streamer active)")

                # All clients disconnected (>0 -> 0)
                elif client_count == 0 and self._prev_client_count > 0:
                    self.app.app_state_ui.show_video_feed = True
                    self.app.app_settings.set("show_video_feed", True)
                    self.app.logger.info("🎥 Restoring video feed (no clients)")

                self._prev_client_count = client_count

            # Control Section
            open_, _ = imgui.collapsing_header(
                "Server Control##NativeSyncControl",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )
            if open_:
                # Description
                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                imgui.text_colored(
                    "Stream video to browsers/VR headsets with frame-perfect synchronization. "
                    "Supports zoom/pan controls, speed modes, and interactive device control.",
                    0.7, 0.7, 0.7
                )
                imgui.pop_text_wrap_pos()
                imgui.spacing()

                # Start/Stop button
                if is_running:
                    # Running - show stop button (DESTRUCTIVE - stops server)
                    with destructive_button_style():
                        if imgui.button("Stop Streaming Server", width=-1):
                            self._stop_native_sync()
                else:
                    # Not running - show start button (PRIMARY - positive action)
                    with primary_button_style():
                        if imgui.button("Start Streaming Server", width=-1):
                            self._start_native_sync()

            # Connection Info Section (only when running)
            if is_running:
                open_, _ = imgui.collapsing_header(
                    "Connection Info##NativeSyncConnection",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    # FunGen Viewer URL
                    viewer_url = f"http://{self._get_local_ip()}:{status.get('http_port', 8080)}/fungen"
                    imgui.text("FunGen Viewer URL:")
                    imgui.same_line()
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.5)
                    imgui.text(viewer_url)
                    imgui.pop_style_color()

                    # Buttons row
                    button_width = imgui.get_content_region_available_width() / 2 - 5
                    if imgui.button("Open in Browser", width=button_width):
                        self._open_in_browser(viewer_url)
                    imgui.same_line()
                    if imgui.button("Copy URL", width=button_width):
                        self._copy_to_clipboard(viewer_url)

                    imgui.spacing()

                    # WebSocket URL
                    ws_url = f"ws://{self._get_local_ip()}:{status.get('sync_port', 8765)}"
                    imgui.text("WebSocket Sync:")
                    imgui.same_line()
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.5)
                    imgui.text(ws_url)
                    imgui.pop_style_color()

                    # HereSphere URLs (if enabled)
                    if status.get('heresphere_enabled', False):
                        imgui.spacing()
                        imgui.separator()
                        imgui.text_colored("HereSphere Integration:", 0.5, 0.8, 1.0)
                        imgui.spacing()

                        # HereSphere API URL
                        local_ip = self._get_local_ip()
                        heresphere_api_port = status.get('heresphere_api_port', 8091)
                        heresphere_event_port = status.get('heresphere_event_port', 8090)

                        api_url = f"http://{local_ip}:{heresphere_api_port}/heresphere"
                        imgui.text("API Server (POST):")
                        imgui.same_line()
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 1.0, 0.8)
                        imgui.text(api_url)
                        imgui.pop_style_color()

                        # Copy button for API URL
                        if imgui.button("Copy API URL", width=-1):
                            self._copy_to_clipboard(api_url)

                        imgui.spacing()

                        # HereSphere Event URL
                        event_url = f"http://{local_ip}:{heresphere_event_port}/heresphere/event"
                        imgui.text("Event Server (POST):")
                        imgui.same_line()
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 1.0, 0.8)
                        imgui.text(event_url)
                        imgui.pop_style_color()

                        # Copy button for Event URL
                        if imgui.button("Copy Event URL", width=-1):
                            self._copy_to_clipboard(event_url)

                        imgui.spacing()
                        imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                        imgui.text_colored(
                            "Configure HereSphere to use the API URL above as a library source. "
                            "The Event URL is automatically provided to HereSphere via video metadata.",
                            0.6, 0.6, 0.6
                        )
                        imgui.pop_text_wrap_pos()

                # Status Section
                open_, _ = imgui.collapsing_header(
                    "Status##NativeSyncStatus",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    # Server status
                    sync_active = status.get('sync_server_active', False)
                    video_active = status.get('video_server_active', False)

                    imgui.text("Streamer:")
                    imgui.same_line()
                    if sync_active:
                        imgui.text_colored("Active", 0.0, 1.0, 0.0)
                    else:
                        imgui.text_colored("Inactive", 1.0, 0.0, 0.0)

                    imgui.text("Video Server:")
                    imgui.same_line()
                    if video_active:
                        imgui.text_colored("Active", 0.0, 1.0, 0.0)
                    else:
                        imgui.text_colored("Inactive", 1.0, 0.0, 0.0)

                    # Connected clients
                    client_count = status.get('connected_clients', 0)
                    imgui.text(f"Connected Clients:")
                    imgui.same_line()
                    if client_count > 0:
                        imgui.text_colored(str(client_count), 0.0, 1.0, 0.5)
                    else:
                        imgui.text_colored("0", 0.7, 0.7, 0.7)

                    imgui.spacing()

                    # Browser Playback Progress (when clients connected)
                    if client_count > 0:
                        imgui.separator()
                        imgui.text("Browser Playback Position:")

                        # Get sync server for browser position
                        if self._native_sync_manager and self._native_sync_manager.sync_server:
                            sync_server = self._native_sync_manager.sync_server
                            browser_frame = sync_server.target_frame_index
                            processor_frame = self.app.processor.current_frame_index
                            total_frames = self.app.processor.total_frames

                            if browser_frame is not None and total_frames > 0:
                                # Calculate progress percentages
                                browser_progress = (browser_frame / total_frames) * 100.0
                                processor_progress = (processor_frame / total_frames) * 100.0

                                # Browser progress bar
                                imgui.text("  Browser:")
                                imgui.same_line(120)
                                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.0, 0.7, 1.0)
                                imgui.progress_bar(browser_progress / 100.0, (200, 0), f"{browser_frame} / {total_frames}")
                                imgui.pop_style_color()

                                # Processor progress bar
                                imgui.text("  Processing:")
                                imgui.same_line(120)
                                if processor_frame > browser_frame:
                                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.0, 1.0, 0.5)
                                else:
                                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 1.0, 0.5, 0.0)
                                imgui.progress_bar(processor_progress / 100.0, (200, 0), f"{processor_frame} / {total_frames}")
                                imgui.pop_style_color()

                                # Show drift if exists
                                drift = processor_frame - browser_frame
                                if abs(drift) > 10:
                                    imgui.spacing()
                                    if drift > 0:
                                        imgui.text_colored(f"  Processing is {drift} frames ahead", 0.0, 1.0, 0.5)
                                    else:
                                        imgui.text_colored(f"  Processing is {abs(drift)} frames behind!", 1.0, 0.5, 0.0)
                            else:
                                imgui.text_colored("  Waiting for browser position updates...", 0.7, 0.7, 0.7)
                        else:
                            imgui.text_colored("  No sync data available", 0.7, 0.7, 0.7)

                # Video Display Options (when streaming)
                open_, _ = imgui.collapsing_header(
                    "Display Options##NativeSyncDisplay",
                    flags=0,  # Collapsed by default
                )
                if open_:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "When streaming to browsers, you can hide the local video feed to save GPU resources.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    # Auto-hide video feed checkbox
                    if not hasattr(self.app.app_settings, '_streamer_auto_hide_video'):
                        self.app.app_settings._streamer_auto_hide_video = True

                    auto_hide = getattr(self.app.app_settings, '_streamer_auto_hide_video', True)
                    clicked, new_val = imgui.checkbox("Auto-hide Video Feed while streaming", auto_hide)
                    if clicked:
                        self.app.app_settings._streamer_auto_hide_video = new_val
                        # Apply immediately
                        if new_val and client_count > 0:
                            # Hide video
                            self.app.app_state_ui.show_video_feed = False
                            self.app.app_settings.set("show_video_feed", False)
                        elif not new_val:
                            # Show video
                            self.app.app_state_ui.show_video_feed = True
                            self.app.app_settings.set("show_video_feed", True)

                    imgui.same_line()
                    imgui.text_colored("(?)", 0.7, 0.7, 0.7)
                    if imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.text("When enabled, the video feed will be hidden\nwhen clients are connected, and restored when\nall clients disconnect.")
                        imgui.end_tooltip()

            # XBVR Configuration Section
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            open_, _ = imgui.collapsing_header(
                "XBVR Integration##XBVRSettings",
                flags=imgui.TREE_NODE_DEFAULT_OPEN if not is_running else 0,
            )
            if open_:
                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                imgui.text_colored(
                    "Browse and load videos from your XBVR library directly in the VR viewer. "
                    "Displays scene thumbnails, funscript availability, and enables remote playback control.",
                    0.7, 0.7, 0.7
                )
                imgui.pop_text_wrap_pos()
                imgui.spacing()

                # Get current settings (XBVR always enabled by default)
                xbvr_host = self.app.app_settings.get('xbvr_host', 'localhost')
                xbvr_port = self.app.app_settings.get('xbvr_port', 9999)

                if True:
                    imgui.spacing()

                    # XBVR Host
                    imgui.text("XBVR Host/IP:")
                    imgui.push_item_width(200)
                    changed, new_host = imgui.input_text(
                        "##xbvr_host",
                        str(xbvr_host),
                        256
                    )
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        self.app.app_settings.set('xbvr_host', new_host)
                        self.app.app_settings.save_settings()

                    # XBVR Port
                    imgui.text("XBVR Port:")
                    imgui.push_item_width(100)
                    changed, new_port_str = imgui.input_text(
                        "##xbvr_port",
                        str(xbvr_port),
                        256
                    )
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        try:
                            new_port = int(new_port_str)
                            self.app.app_settings.set('xbvr_port', new_port)
                            self.app.app_settings.save_settings()
                        except ValueError:
                            pass  # Ignore invalid port input

                    imgui.spacing()
                    imgui.text_colored(
                        f"XBVR URL: http://{xbvr_host}:{xbvr_port}",
                        0.5, 0.8, 1.0
                    )

                    imgui.spacing()
                    # Discover XBVR button
                    if imgui.button("Discover XBVR Address", width=-1):
                        self._discover_xbvr_address()

            # Local Folder Browser Section
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            open_, _ = imgui.collapsing_header(
                "Local Video Browser##LocalVideoBrowser",
                flags=imgui.TREE_NODE_DEFAULT_OPEN if not is_running else 0,
            )
            if open_:
                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                imgui.text_colored(
                    "Browse and load videos from your local folders directly in the VR viewer. "
                    "Displays video thumbnails and enables remote playback control.",
                    0.7, 0.7, 0.7
                )
                imgui.pop_text_wrap_pos()
                imgui.spacing()

                # Browse Local Folders button
                if imgui.button("Browse Local Folders", width=-1):
                    # Open file dialog to select a video folder
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and self.app.gui_instance.file_dialog:
                        self.app.gui_instance.file_dialog.show(
                            title="Select Video Folder",
                            is_folder_dialog=True,
                            callback=lambda folder_path: self._browse_local_videos(folder_path)
                        )
                
                # Show currently selected folder and available videos if any
                if self._local_video_folder:
                    imgui.spacing()
                    imgui.text_colored(f"Selected Folder: {self._local_video_folder}", 0.5, 0.8, 1.0)
                    
                    if self._local_video_files:
                        imgui.text(f"Found {len(self._local_video_files)} videos:")
                        
                        # Show video selection list
                        imgui.begin_child("LocalVideoList", height=150, border=True)
                        for i, video_path in enumerate(self._local_video_files):
                            video_name = os.path.basename(video_path)
                            is_selected = self._selected_local_video == video_path
                            
                            if imgui.selectable(f"{i+1:2d}. {video_name}", is_selected, flags=imgui.SELECTABLE_ALLOW_DOUBLE_CLICK):
                                self._selected_local_video = video_path
                                
                                # If double-clicked, load the video
                                if imgui.is_mouse_double_clicked(0):
                                    self._load_selected_local_video()
                        
                        imgui.end_child()
                        
                        # Load selected video button
                        can_load = bool(self._selected_local_video)
                        if not can_load:
                            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                        
                        if imgui.button("Load Selected Video", width=-1) and self._selected_local_video:
                            self._load_selected_local_video()
                        
                        if not can_load:
                            imgui.pop_style_var()
                            imgui.internal.pop_item_flag()
                        
                        # Or load first video button if none selected
                        if not self._selected_local_video and self._local_video_files:
                            if imgui.button("Load First Video", width=-1):
                                self._selected_local_video = self._local_video_files[0]
                                self._load_selected_local_video()
                    else:
                        imgui.text_colored("No video files found in this folder.", 1.0, 0.5, 0.5)

            # Info Section (only when not running)
            if not is_running:
                open_, _ = imgui.collapsing_header(
                    "Requirements##NativeSyncRequirements",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "This will start HTTP and WebSocket servers for native video playback "
                        "in browsers and VR headsets. Your video will be served at full quality "
                        "with frame-perfect synchronization.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    imgui.bullet_text("Ports 8080 (HTTP) and 8765 (WebSocket) available")
                    imgui.bullet_text("Browser with HTML5 video support")
                    imgui.bullet_text("Video can be loaded before or after starting the server")

                # Features Section
                open_, _ = imgui.collapsing_header(
                    "Features##NativeSyncFeatures",
                    flags=0,  # Collapsed by default
                )
                if open_:
                    imgui.bullet_text("Native hardware H.265/AV1 decode")
                    imgui.bullet_text("Zoom/Pan controls (+/- and WASD keys)")
                    imgui.bullet_text("Speed modes (Real Time / Slo Mo)")
                    imgui.bullet_text("Real-time FPS and resolution stats")
                    imgui.bullet_text("Interactive device control")
                    imgui.bullet_text("Funscript visualization graph")

        except Exception as e:
            imgui.text(f"Error in Streamer: {e}")
            imgui.text_colored("See logs for details.", 1.0, 0.0, 0.0)
            import traceback
            self.app.logger.error(f"Streamer tab error: {e}")
            self.app.logger.error(traceback.format_exc())

    def _start_native_sync(self):
        """Start streamer servers."""
        try:
            # Streamer can start without a video loaded (video can be loaded later)
            self.app.logger.info("Starting streamer...")

            # Enable HereSphere and XBVR browser by default
            if self._native_sync_manager:
                self._native_sync_manager.enable_heresphere = True
                self._native_sync_manager.enable_xbvr_browser = True

            self._native_sync_manager.start()

        except Exception as e:
            self.app.logger.error(f"Failed to start streamer: {e}")
            import traceback
            self.app.logger.error(traceback.format_exc())

    def _stop_native_sync(self):
        """Stop streamer servers."""
        try:
            self.app.logger.info("Stopping streamer...")
            self._native_sync_manager.stop()

        except Exception as e:
            self.app.logger.error(f"Failed to stop streamer: {e}")

    def _open_in_browser(self, url: str):
        """Open URL in system default browser."""
        try:
            import webbrowser
            webbrowser.open(url)
            self.app.logger.info(f"Opening in browser: {url}")
        except Exception as e:
            self.app.logger.error(f"Failed to open browser: {e}")

    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            self.app.logger.info(f"Copied to clipboard: {text}")
        except:
            # Fallback - just log
            self.app.logger.info(f"URL to copy: {text}")

    def _browse_local_videos(self, folder_path):
        """Browse local video folder and handle video selection."""
        try:
            self.app.logger.info(f"Scanning folder for videos: {folder_path}")
            
            # Use the file manager's existing scan method
            video_files = self.app.file_manager._scan_folder_for_videos(folder_path)
            
            if not video_files:
                self.app.logger.warning(f"No video files found in: {folder_path}")
                self._local_video_folder = ""
                self._local_video_files = []
                self._selected_local_video = ""
                return
            
            self.app.logger.info(f"Found {len(video_files)} video files in: {folder_path}")
            
            self._local_video_folder = folder_path
            self._local_video_files = video_files
            self._selected_local_video = ""
                
        except Exception as e:
            self.app.logger.error(f"Error browsing local videos: {e}")
            import traceback
            self.app.logger.error(traceback.format_exc())

    def _load_selected_local_video(self):
        """Load the currently selected local video."""
        try:
            if not self._selected_local_video or not os.path.exists(self._selected_local_video):
                self.app.logger.warning("No valid video selected to load")
                return
            
            video_path = self._selected_local_video
            self.app.logger.info(f"Loading selected video: {os.path.basename(video_path)}")
            
            # Use the file manager to load the video
            success = self.app.file_manager.open_video_from_path(video_path)
            
            if success:
                self.app.logger.info(f"Successfully loaded video: {os.path.basename(video_path)}", 
                                   extra={'status_message': True})
                
                # If streamer is running, the video will be available for streaming
                if self._native_sync_manager and self._native_sync_manager.get_status().get('is_running', False):
                    self.app.logger.info("Video is now available for streaming", extra={'status_message': True})
            else:
                self.app.logger.error(f"Failed to load video: {os.path.basename(video_path)}", 
                                    extra={'status_message': True})
                
        except Exception as e:
            self.app.logger.error(f"Error loading selected local video: {e}")
            import traceback
            self.app.logger.error(traceback.format_exc())

    def _discover_xbvr_address(self):
        """Attempt to discover XBVR on the local network."""
        import socket
        import requests
        from threading import Thread

        def scan_network():
            # Get local IP to determine subnet
            local_ip = self._get_local_ip()
            if not local_ip or not local_ip.startswith('192.168.'):
                self.app.logger.info("Could not determine local network subnet", extra={'status_message': True})
                return

            # Extract subnet (e.g., 192.168.1.x)
            subnet = '.'.join(local_ip.split('.')[:-1])
            self.app.logger.info(f"Scanning {subnet}.0/24 for XBVR on port 9999...", extra={'status_message': True})

            found = False
            for i in range(1, 255):
                if found:
                    break
                test_ip = f"{subnet}.{i}"
                try:
                    # Quick connection test on port 9999
                    response = requests.get(f"http://{test_ip}:9999", timeout=0.5)
                    if response.status_code == 200 or 'xbvr' in response.text.lower():
                        self.app.logger.info(f"✅ Found XBVR at {test_ip}:9999", extra={'status_message': True})
                        self.app.app_settings.set('xbvr_host', test_ip)
                        self.app.app_settings.save_settings()
                        found = True
                except:
                    pass  # Connection failed, continue

            if not found:
                self.app.logger.info("No XBVR server found on local network", extra={'status_message': True})

        # Run scan in background thread
        Thread(target=scan_network, daemon=True).start()

    def _get_local_ip(self) -> str:
        """Get local network IP address (prefer 192.168.x.x over VPN)."""
        import socket
        try:
            # Get all network interfaces
            hostname = socket.gethostname()
            addresses = socket.getaddrinfo(hostname, None, socket.AF_INET)

            # Prefer 192.168.x.x addresses
            for addr in addresses:
                ip = addr[4][0]
                if ip.startswith('192.168.'):
                    return ip

            # Fallback to any non-loopback address
            for addr in addresses:
                ip = addr[4][0]
                if not ip.startswith('127.'):
                    return ip

            return "localhost"
        except:
            return "localhost"
