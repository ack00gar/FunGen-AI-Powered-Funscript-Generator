import imgui
import os
import config

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
        "TrackerMode",
        "AI_modelExtensionsFilter",
        "AI_modelTooltipExtensions",
    )

    def __init__(self, app):
        self.app = app
        self.timeline_editor1 = None
        self.timeline_editor2 = None
        self.ControlPanelColors = config.ControlPanelColors
        self.GeneralColors = config.GeneralColors
        self.constants = config.constants
        self.TrackerMode = self.constants.TrackerMode
        self.AI_modelExtensionsFilter = self.constants.AI_MODEL_EXTENSIONS_FILTER
        self.AI_modelTooltipExtensions = self.constants.AI_MODEL_TOOLTIP_EXTENSIONS

    # ---------------- Helpers ----------------

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

    # ---------------- Model path updates ----------------

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

    # ---------------- Main render ----------------

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
        imgui.end_child()
        imgui.end()

    # ---------------- Tabs ----------------

    def _render_simple_mode_ui(self):
        app = self.app
        app_state = app.app_state_ui
        tracker_mode = self.TrackerMode

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
            self._status_indicator(
                "Analysis in progress...", "info", "Processing your video. Please wait."
            )
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

        modes_display = [
            "Live Oscillation Detector",
            "Offline AI Analysis (3-Stage)",
            "Live Tracking (YOLO ROI)",
        ]
        modes_enum = [
            tracker_mode.OSCILLATION_DETECTOR,
            tracker_mode.OFFLINE_3_STAGE,
            tracker_mode.LIVE_YOLO_ROI,
        ]
        try:
            cur_idx = modes_enum.index(app_state.selected_tracker_mode)
        except ValueError:
            cur_idx = 0
            app_state.selected_tracker_mode = tracker_mode.OSCILLATION_DETECTOR

        imgui.push_item_width(-1)
        clicked, new_idx = imgui.combo(
            "Analysis Method##SimpleTrackerMode", cur_idx, modes_display
        )
        imgui.pop_item_width()
        self._help_tooltip(
            "Live Oscillation Detector: Fast & simple, best for rhythmic motion\n"
            "Offline AI Analysis: High quality, uses AI for object detection\n"
            "Live Tracking: Real-time AI tracking with immediate preview"
        )
        if clicked and new_idx != cur_idx:
            app_state.selected_tracker_mode = modes_enum[new_idx]

        imgui.separator()
        self._render_start_stop_buttons(stage_proc, fs_proc, app.event_handlers)
        imgui.separator()
        self._render_execution_progress_display()
        imgui.end()

    def _render_processing_speed_controls(self, app_state):
        app = self.app
        processor = app.processor
        selected_mode = app_state.selected_tracker_mode
        is_live_mode = selected_mode in (self.TrackerMode.LIVE_YOLO_ROI, self.TrackerMode.LIVE_USER_ROI, self.TrackerMode.OSCILLATION_DETECTOR)
        is_playback_active = processor and processor.is_processing and not processor.enable_tracker_processing

        if not (is_live_mode or is_playback_active):
            return

        if processor:
            imgui.text(f"Actual FPS: {processor.actual_fps:.1f}")

        if not app_state.show_video_feed:  # video feed not visible
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.MAX_SPEED
            return


        self._section_header(">> Processing Speed", "Control the processing speed for live analysis")

        current_speed_mode = app_state.selected_processing_speed_mode
        
        if imgui.radio_button("Real Time", current_speed_mode == config.ProcessingSpeedMode.REALTIME):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.REALTIME
        imgui.same_line()
        if imgui.radio_button("Slow-mo", current_speed_mode == config.ProcessingSpeedMode.SLOW_MOTION):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.SLOW_MOTION
        imgui.same_line()
        if imgui.radio_button("Max Speed", current_speed_mode == config.ProcessingSpeedMode.MAX_SPEED):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.MAX_SPEED

        #if processor:
        #    imgui.text(f"Actual FPS: {processor.actual_fps:.1f}")

    def _render_run_control_tab(self):
        app = self.app
        app_state = app.app_state_ui
        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor
        events = app.event_handlers
        tracker_mode = self.TrackerMode

        self._section_header(
            ">> Analysis Method", "Choose how the application will analyze your video"
        )

        modes_enum = [
            tracker_mode.OSCILLATION_DETECTOR,
            tracker_mode.LIVE_YOLO_ROI,
            tracker_mode.LIVE_USER_ROI,
            tracker_mode.OFFLINE_2_STAGE,
            tracker_mode.OFFLINE_3_STAGE,
        ]
        modes_display = [m.value for m in modes_enum]

        processor = app.processor
        disable_combo = (
            stage_proc.full_analysis_active
            or app.is_setting_user_roi_mode
            or (processor and processor.is_processing and not processor.pause_event.is_set())
        )
        with _DisabledScope(disable_combo):
            try:
                cur_idx = modes_enum.index(app_state.selected_tracker_mode)
            except ValueError:
                cur_idx = 0
                app_state.selected_tracker_mode = modes_enum[cur_idx]

            clicked, new_idx = imgui.combo(
                "Tracker Type##TrackerModeCombo", cur_idx, modes_display
            )
            self._help_tooltip(
                "Choose analysis method:\n"
                "• Live Oscillation Detector: Fast, real-time analysis for rhythmic motion\n"
                "• Live YOLO ROI: AI-powered object detection with real-time tracking\n"
                "• Live User ROI: Manual region selection for custom tracking\n"
                "• Offline 2-Stage: GPU-accelerated batch processing\n"
                "• Offline 3-Stage: Full pipeline with advanced post-processing"
            )

        if clicked and new_idx != cur_idx:
            new_mode = modes_enum[new_idx]
            app_state.selected_tracker_mode = new_mode
            tr = app.tracker
            if tr:
                if new_mode == tracker_mode.LIVE_USER_ROI:
                    tr.set_tracking_mode("USER_FIXED_ROI")
                elif new_mode == tracker_mode.OSCILLATION_DETECTOR:
                    tr.set_tracking_mode("OSCILLATION_DETECTOR")
                else:
                    tr.set_tracking_mode("YOLO_ROI")

        self._render_processing_speed_controls(app_state)

        self._section_header(
            ">> Output Configuration", "Configure which movement axes to track and output"
        )
        self._render_tracking_axes_mode(stage_proc)

        mode = app_state.selected_tracker_mode
        if mode in (
            tracker_mode.OFFLINE_2_STAGE,
            tracker_mode.OFFLINE_3_STAGE,
            tracker_mode.LIVE_YOLO_ROI,
            tracker_mode.LIVE_USER_ROI,
            tracker_mode.OSCILLATION_DETECTOR,
        ):
            if app_state.show_advanced_options:
                open_, _ = imgui.collapsing_header(
                    "Analysis Options##RunControlAnalysisOptions",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    imgui.text("Analysis Range")
                    self._render_range_selection(stage_proc, fs_proc, events)

                    if mode in (tracker_mode.OFFLINE_2_STAGE, tracker_mode.OFFLINE_3_STAGE):
                        imgui.separator()
                        imgui.text("Stage Reruns:")
                        with _DisabledScope(disable_combo):
                            _, stage_proc.force_rerun_stage1 = imgui.checkbox(
                                "Force Re-run Stage 1##ForceRerunS1",
                                stage_proc.force_rerun_stage1,
                            )
                            imgui.same_line()
                            _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox(
                                "Force Re-run S2 Chapter Creation##ForceRerunS2",
                                stage_proc.force_rerun_stage2_segmentation,
                            )
                            imgui.separator()
                            if not hasattr(stage_proc, "save_preprocessed_video"):
                                stage_proc.save_preprocessed_video = app.app_settings.get(
                                    "save_preprocessed_video", False
                                )
                            changed, new_val = imgui.checkbox(
                                "Save/Reuse Preprocessed Video##SavePreprocessedVideo",
                                stage_proc.save_preprocessed_video,
                            )
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
            imgui.separator()

        self._render_start_stop_buttons(stage_proc, fs_proc, events)
        imgui.separator()

        proc = app.processor
        video_loaded = proc and proc.is_video_open()
        processing_active = stage_proc.full_analysis_active
        disable_after = (not video_loaded) or processing_active

        with _DisabledScope(disable_after):
            self._render_execution_progress_display()
        imgui.separator()

        self._render_interactive_refinement_controls()

        chapters = getattr(app.funscript_processor, "video_chapters", [])
        if chapters:
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
                if imgui.button("Yes, clear all", width=bw):
                    app.funscript_processor.video_chapters.clear()
                    app.project_manager.project_dirty = True
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel", width=cw):
                    imgui.close_current_popup()
                imgui.end_popup()

        if disable_after and imgui.is_item_hovered():
            imgui.set_tooltip(
                "Requires a video to be loaded and no other process to be active."
            )
        imgui.separator()

    def _render_configuration_tab(self):
        app = self.app
        app_state = app.app_state_ui
        tmode = app_state.selected_tracker_mode

        imgui.text("Configure settings for the selected mode.")
        imgui.spacing()

        if tmode in (
            self.TrackerMode.LIVE_YOLO_ROI,
            self.TrackerMode.OFFLINE_2_STAGE,
            self.TrackerMode.OFFLINE_3_STAGE,
        ):
            if imgui.collapsing_header("AI Models & Inference##ConfigAIModels")[0]:
                self._render_ai_model_settings()
            imgui.separator()

        adv = app.app_state_ui.show_advanced_options
        if tmode in (self.TrackerMode.LIVE_YOLO_ROI, self.TrackerMode.LIVE_USER_ROI) and adv:
            self._render_live_tracker_settings()
            imgui.separator()

        if tmode in (
            self.TrackerMode.LIVE_YOLO_ROI,
            self.TrackerMode.OFFLINE_2_STAGE,
            self.TrackerMode.OFFLINE_3_STAGE,
        ) and adv:
            if imgui.collapsing_header("Class Filtering##ConfigClassFilterHeader")[0]:
                self._render_class_filtering_content()

        if tmode == self.TrackerMode.OSCILLATION_DETECTOR:
            if imgui.collapsing_header("Oscillation Detector Settings##ConfigOscillationDetector")[0]:
                self._render_oscillation_detector_settings()

        with_config = {
            self.TrackerMode.LIVE_YOLO_ROI,
            self.TrackerMode.LIVE_USER_ROI,
            self.TrackerMode.OFFLINE_2_STAGE,
            self.TrackerMode.OFFLINE_3_STAGE,
        }
        if tmode not in with_config:
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
            imgui.separator()
            # self._render_performance_info()  # Removed from settings tab
        imgui.separator()

        if imgui.collapsing_header(
            "File & Output##SettingsMenuOutput", flags=imgui.TREE_NODE_DEFAULT_OPEN
        )[0]:
            self._render_settings_file_output()
        imgui.separator()

        if app_state.show_advanced_options:
            if imgui.collapsing_header("Logging & Autosave##SettingsMenuLogging")[0]:
                self._render_settings_logging_autosave()
            imgui.separator()

            if imgui.collapsing_header("View/Edit Hotkeys##FSHotkeysMenuSettingsDetail")[0]:
                self._render_settings_hotkeys()
            imgui.separator()
        imgui.spacing()

        if imgui.button("Reset All Settings to Default##ResetAllSettingsButton", width=-1):
            imgui.open_popup("Confirm Reset##ResetSettingsPopup")

        if imgui.begin_popup_modal(
            "Confirm Reset##ResetSettingsPopup", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )[0]:
            imgui.text("This will reset all application settings to their defaults.")
            imgui.text("Your projects will not be affected.")
            imgui.text("This action cannot be undone.")
            imgui.separator()

            avail_w = imgui.get_content_region_available_width()
            pw = (avail_w - imgui.get_style().item_spacing[0]) / 2.0

            if imgui.button("Confirm Reset", width=pw):
                app.app_settings.reset_to_defaults()
                app.logger.info(
                    "All settings have been reset to default.", extra={"status_message": True}
                )
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
        imgui.separator()
        if imgui.collapsing_header("Automated Post-Processing##PostProcAuto")[0]:
            self._render_automatic_post_processing_new(app.funscript_processor)

    # ---------------- AI model settings ----------------

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
        if imgui.button("Browse##S1YOLOBrowse"):
            show_model_file_dialog(
                "Select YOLO Detection Model",
                app.yolo_detection_model_path_setting,
                self._update_detection_model_path,
            )
        imgui.same_line()
        if imgui.button("Unload##S1YOLOUnload"):
            app.unload_model("detection")
        _tooltip_if_hovered(
            "Path to the YOLO object detection model file (%s)." % self.AI_modelTooltipExtensions
        )

        # Pose model
        imgui.text("Pose Model")
        _readonly_input("##PoseYOLOPath", app.yolo_pose_model_path_setting, input_w)
        imgui.same_line()
        if imgui.button("Browse##PoseYOLOBrowse"):
            show_model_file_dialog(
                "Select YOLO Pose Model",
                app.yolo_pose_model_path_setting,
                self._update_pose_model_path,
            )
        imgui.same_line()
        if imgui.button("Unload##PoseYOLOUnload"):
            app.unload_model("pose")
        _tooltip_if_hovered(
            "Path to the YOLO pose estimation model file (%s). This model is optional."
            % self.AI_modelTooltipExtensions
        )

        imgui.separator()
        imgui.text("Pose Model Artifacts Dir")
        dir_input_w = avail_w - browse_w - style.item_spacing.x if avail_w > browse_w else -1
        _readonly_input("##PoseArtifactsDirPath", app.pose_model_artifacts_dir, dir_input_w)
        imgui.same_line()
        if imgui.button("Browse##PoseArtifactsDirBrowse"):
            gi = getattr(app, "gui_instance", None)
            if gi:
                gi.file_dialog.show(
                    title="Select Pose Model Artifacts Directory",
                    callback=self._update_artifacts_dir_path,
                    is_folder_dialog=True,
                    initial_path=app.pose_model_artifacts_dir,
                )
        _tooltip_if_hovered(
            "Path to the folder containing your trained classifier,\n"
            "imputer, and other .joblib model artifacts."
        )
        imgui.separator()

        mode = app.app_state_ui.selected_tracker_mode
        if mode in (self.TrackerMode.OFFLINE_2_STAGE, self.TrackerMode.OFFLINE_3_STAGE):
            imgui.separator()
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
                _tooltip_if_hovered(
                    "Producers are forced to 1 when 'Save/Reuse Preprocessed Video' is enabled."
                )
            else:
                _tooltip_if_hovered("Number of threads for video decoding & preprocessing.")

            imgui.same_line()
            ch_c, n_c = imgui.input_int(
                "Consumers##S1Consumers", stage_proc.num_consumers_stage1
            )
            if ch_c:
                v = max(1, n_c)
                if v != stage_proc.num_consumers_stage1:
                    stage_proc.num_consumers_stage1 = v
                    settings.set("num_consumers_stage1", v)
            _tooltip_if_hovered(
                "Number of threads for AI model inference. Match to available cores for best performance."
            )
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
                "Number of processes for Stage 2 Optical Flow gap recovery. "
                "More may be faster on high-core CPUs."
            )

    # ---------------- Settings: interface/perf ----------------

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
        imgui.separator()
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
            app.logger.info(
                f"GPU timeline rendering {'enabled' if gpu_enabled else 'disabled'}",
                extra={"status_message": True}
            )
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
        
        imgui.separator()

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
                app.logger.info(
                    "Hardware acceleration set to: %s. Reload video to apply." % method,
                    extra={"status_message": True},
                )
        imgui.pop_item_width()
        _tooltip_if_hovered(
            "Select FFmpeg hardware acceleration. Requires video reload to apply."
        )
        imgui.separator()

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

    # ---------------- Settings: file/output ----------------

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
        imgui.separator()

        imgui.text("Funscript Output:")
        ch, v = imgui.checkbox(
            "Autosave final script next to video",
            settings.get("autosave_final_funscript_to_video_location", True),
        )
        if ch:
            settings.set("autosave_final_funscript_to_video_location", v)

        ch, v = imgui.checkbox(
            "Generate .roll file (from Timeline 2)", settings.get("generate_roll_file", True)
        )
        if ch:
            settings.set("generate_roll_file", v)
        imgui.separator()

        imgui.text("Batch Processing Default:")
        cur = settings.get("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Process All (skips own matching version)", cur == 0):
            if cur != 0:
                settings.set("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Skip if Funscript Exists", cur == 1):
            if cur != 1:
                settings.set("batch_mode_overwrite_strategy", 1)

    # ---------------- Settings: logging/autosave ----------------

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
        imgui.separator()

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

# ---------------- Settings: hotkeys ----------------

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

# ---------------- Execution/progress ----------------

    def _render_execution_progress_display(self):
        app = self.app
        stage_proc = app.stage_processor
        app_state = app.app_state_ui
        mode = app_state.selected_tracker_mode

        if mode in (self.TrackerMode.OFFLINE_2_STAGE, self.TrackerMode.OFFLINE_3_STAGE):
            self._render_stage_progress_ui(stage_proc)
            return

        if mode in (self.TrackerMode.LIVE_YOLO_ROI, self.TrackerMode.LIVE_USER_ROI):
            tr = app.tracker
            imgui.text("Live Tracker Status:")
            fps = (tr.current_fps if tr else 0.0)
            imgui.text("  - Actual FPS: %.1f" % (fps if isinstance(fps, (int, float)) else 0.0))
            roi_status = "Not Set"
            if tr:
                if mode == self.TrackerMode.LIVE_YOLO_ROI:
                    roi_status = (
                        "Tracking '%s'" % tr.main_interaction_class
                        if getattr(tr, "main_interaction_class", None)
                        else "Searching..."
                    )
                else:
                    roi_status = "Set" if getattr(tr, "user_roi_fixed", False) else "Not Set"
            imgui.text("  - ROI Status: %s" % roi_status)

            if mode == self.TrackerMode.LIVE_USER_ROI:
                self._render_user_roi_controls_for_run_tab()
            return

        imgui.text_disabled("No execution monitoring for this mode.")

# ---------------- Live tracker settings ----------------

    def _render_live_tracker_settings(self):
        app = self.app
        tr = app.tracker
        if not tr:
            imgui.text_disabled("Tracker not initialized.")
            return

        settings = app.app_settings

        if imgui.collapsing_header("Detection & ROI Definition##ROIDetectionTrackerMenu"):
            cur_conf = settings.get("live_tracker_confidence_threshold")
            ch, new_conf = imgui.slider_float(
                "Obj. Confidence##ROIConfTrackerMenu", cur_conf, 0.1, 0.95, "%.2f"
            )
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
            ch, new_int = imgui.input_int(
                "ROI Update Interval (frames)##ROIIntervalTrackerMenu", cur_int
            )
            if ch:
                v = max(1, new_int)
                if v != cur_int:
                    settings.set("live_tracker_roi_update_interval", v)
                    tr.roi_update_interval = v

            cur_sm = settings.get("live_tracker_roi_smoothing_factor")
            ch, new_sm = imgui.slider_float(
                "ROI Smoothing Factor##ROISmoothTrackerMenu", cur_sm, 0.0, 1.0, "%.2f"
            )
            if ch and new_sm != cur_sm:
                settings.set("live_tracker_roi_smoothing_factor", new_sm)
                tr.roi_smoothing_factor = new_sm

            cur_persist = settings.get("live_tracker_roi_persistence_frames")
            ch, new_pf = imgui.input_int(
                "ROI Persistence (frames)##ROIPersistTrackerMenu", cur_persist
            )
            if ch:
                v = max(0, new_pf)
                if v != cur_persist:
                    settings.set("live_tracker_roi_persistence_frames", v)
                    tr.max_frames_for_roi_persistence = v

        if imgui.collapsing_header("Optical Flow##ROIFlowTrackerMenu"):
            cur_sparse = settings.get("live_tracker_use_sparse_flow")
            ch, new_sparse = imgui.checkbox(
                "Use Sparse Optical Flow##ROISparseFlowTrackerMenu", cur_sparse
            )
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
            ch, new_scale = imgui.input_int(
                "DIS Finest Scale (0-10, 0=auto)##ROIDISFineScaleTrackerMenu", cur_scale
            )
            if ch and new_scale != cur_scale:
                settings.set("live_tracker_dis_finest_scale", new_scale)
                tr.update_dis_flow_config(finest_scale=new_scale)

            if cur_sparse:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()

            if imgui.collapsing_header("Output Signal Generation##ROISignalTrackerMenu"):
                cur_sens = settings.get("live_tracker_sensitivity")
                ch, ns = imgui.slider_float(
                    "Output Sensitivity##ROISensTrackerMenu", cur_sens, 0.0, 100.0, "%.1f"
                )
                if ch and ns != cur_sens:
                    settings.set("live_tracker_sensitivity", ns)
                    tr.sensitivity = ns

                cur_amp = settings.get("live_tracker_base_amplification")
                ch, na = imgui.slider_float(
                    "Base Amplification##ROIBaseAmpTrackerMenu", cur_amp, 0.1, 5.0, "%.2f"
                )
                if ch:
                    v = max(0.1, na)
                    if v != cur_amp:
                        settings.set("live_tracker_base_amplification", v)
                        tr.base_amplification_factor = v

                imgui.text("Class-Specific Amplification Multipliers:")
                cur = settings.get("live_tracker_class_amp_multipliers", {})
                changed = False

                face = cur.get("face", 1.0)
                ch, nv = imgui.slider_float(
                    "Face Amp. Mult.##ROIFaceAmpTrackerMenu", face, 0.1, 5.0, "%.2f"
                )
                if ch:
                    cur["face"] = max(0.1, nv)
                    changed = True

                hand = cur.get("hand", 1.0)
                ch, nv = imgui.slider_float(
                    "Hand Amp. Mult.##ROIHandAmpTrackerMenu", hand, 0.1, 5.0, "%.2f"
                )
                if ch:
                    cur["hand"] = max(0.1, nv)
                    changed = True

                if changed:
                    settings.set("live_tracker_class_amp_multipliers", cur)
                    tr.class_specific_amplification_multipliers = cur

                imgui.separator()

            cur_smooth = settings.get("live_tracker_flow_smoothing_window")
            ch, nv = imgui.input_int(
                "Flow Smoothing Window##ROIFlowSmoothWinTrackerMenu", cur_smooth
            )
            if ch:
                v = max(1, nv)
                if v != cur_smooth:
                    settings.set("live_tracker_flow_smoothing_window", v)
                    tr.flow_history_window_smooth = v

            imgui.separator()
            imgui.text("Output Delay (frames):")
            cur_delay = settings.get("funscript_output_delay_frames")
            ch, nd = imgui.slider_int("##OutputDelayFrames", cur_delay, 0, 20)
            if ch and nd != cur_delay:
                settings.set("funscript_output_delay_frames", nd)
                app.calibration.funscript_output_delay_frames = nd
                app.calibration.update_tracker_delay_params()

# ---------------- Oscillation detector ----------------

    def _render_calibration_window(self, calibration_mgr, app_state):
        """Renders the dedicated latency calibration window."""
        window_title = "Latency Calibration"
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        if app_state.ui_layout_mode == 'fixed':
            # In fixed mode, embed it in the main panel area without a title bar
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
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi or stage_proc.scene_detection_active

        if is_batch_mode:
            imgui.text_ansi_colored("--- BATCH PROCESSING ACTIVE ---", 1.0, 0.7, 0.3) # TODO: move to theme, orange
            total_videos = len(self.app.batch_video_paths)
            current_idx = self.app.current_batch_video_index
            if 0 <= current_idx < total_videos:
                current_video_name = os.path.basename(self.app.batch_video_paths[current_idx]["path"])
                imgui.text_wrapped(f"Processing {current_idx + 1}/{total_videos}:")
                imgui.text_wrapped(f"{current_video_name}")
            if imgui.button("Abort Batch Process", width=-1):
                self.app.abort_batch_processing()
            return

        selected_mode = self.app.app_state_ui.selected_tracker_mode
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
            if selected_mode in [self.TrackerMode.OFFLINE_3_STAGE, self.TrackerMode.OFFLINE_2_STAGE] and self.app.file_manager.video_path:
                resumable_checkpoint = stage_proc.can_resume_video(self.app.file_manager.video_path)
            
            if selected_mode in [self.TrackerMode.OFFLINE_3_STAGE, self.TrackerMode.OFFLINE_2_STAGE]:
                start_text = "Start AI Analysis (Range)" if fs_proc.scripting_range_active else "Start Full AI Analysis"
                handler = event_handlers.handle_start_ai_cv_analysis
            elif selected_mode in [self.TrackerMode.LIVE_YOLO_ROI, self.TrackerMode.LIVE_USER_ROI, self.TrackerMode.OSCILLATION_DETECTOR]:
                start_text = "Start Live Tracking (Range)" if fs_proc.scripting_range_active else "Start Live Tracking"
                handler = event_handlers.handle_start_live_tracker_click
            
            # Show resume button if checkpoint exists
            if resumable_checkpoint:
                button_width_third = (imgui.get_content_region_available()[0] - 2 * imgui.get_style().item_spacing[0]) / 3
                
                # Resume button
                if imgui.button(f"Resume ({resumable_checkpoint.progress_percentage:.0f}%)", width=button_width_third):
                    if stage_proc.start_resume_from_checkpoint(resumable_checkpoint):
                        self.app.logger.info("Resumed processing from checkpoint", extra={'status_message': True})
                
                imgui.same_line()
                
                # Start fresh button  
                if imgui.button("Start Fresh", width=button_width_third):
                    # Delete checkpoint and start fresh
                    stage_proc.delete_checkpoint_for_video(self.app.file_manager.video_path)
                    if handler: handler()
                
                imgui.same_line()
                
                # Delete checkpoint button
                if imgui.button("Clear Resume", width=button_width_third):
                    stage_proc.delete_checkpoint_for_video(self.app.file_manager.video_path)
                    
            else:
                # Normal start button
                if imgui.button(start_text, width=button_width):
                    if handler: handler()

        imgui.same_line()
        if not is_any_process_active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button("Abort/Stop Process##AbortGeneral", width=button_width): event_handlers.handle_abort_process_click()
        if not is_any_process_active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_stage_progress_ui(self, stage_proc):
        is_analysis_running = stage_proc.full_analysis_active
        selected_mode = self.app.app_state_ui.selected_tracker_mode

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
        imgui.separator()

        # Stage 2
        s2_title = "Stage 2: Contact Analysis & Funscript" if selected_mode == self.TrackerMode.OFFLINE_2_STAGE else "Stage 2: Segmentation"
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
        imgui.separator()

        # Stage 3
        if selected_mode == self.TrackerMode.OFFLINE_3_STAGE:
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
            or (
                processor and processor.is_processing and not processor.pause_event.is_set() and not self._is_normal_playback_mode()
            )
        )
        if disable_axis_controls:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        axis_mode_changed, new_axis_mode_idx = imgui.combo("Tracking Axes##TrackingAxisModeComboGlobal", current_axis_mode_idx, axis_modes)
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
            "8=Very Coarse, 20=Balanced, 40=Fine, 80=Very Fine"
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
        imgui.separator()

        imgui.text("Oscillation Area Selection")
        _tooltip_if_hovered(
            "Select a specific area for oscillation detection instead of the full frame."
        )

        tr = app.tracker
        has_area = tr and tr.oscillation_area_fixed
        btn_count = 2 if has_area else 1
        avail_w = imgui.get_content_region_available_width()
        btn_w = (
            (avail_w - imgui.get_style().item_spacing.x * (btn_count - 1)) / btn_count
            if btn_count > 1
            else -1
        )

        set_text = (
            "Cancel Set Oscillation Area" if app.is_setting_oscillation_area_mode else "Set Oscillation Area"
        )
        if imgui.button("%s##SetOscillationArea" % set_text, width=btn_w):
            if app.is_setting_oscillation_area_mode:
                app.exit_set_oscillation_area_mode()
            else:
                app.enter_set_oscillation_area_mode()

        if has_area:
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
        imgui.separator()

        imgui.text("Live Signal Amplification")
        _tooltip_if_hovered(
            "Stretches the live signal to use the full 0-100 range based on recent motion."
        )

        en = settings.get("live_oscillation_dynamic_amp_enabled", True)
        ch, nv = imgui.checkbox("Enable Dynamic Amplification##EnableLiveAmp", en)
        if ch and nv != en:
            settings.set("live_oscillation_dynamic_amp_enabled", nv)

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

# ---------------- Class filtering ----------------

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
                app.logger.info(
                    "Discarded classes updated: %s" % new_list, extra={"status_message": True}
                )
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
                app.logger.info(
                    "All class discard filters cleared.", extra={"status_message": True}
                )
                app.energy_saver.reset_activity_timer()
        _tooltip_if_hovered(
            "Unchecks all classes, enabling all classes for tracking/analysis."
        )

# ---------------- ROI controls ----------------

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
            if imgui.button("%s##UserSetROI_RunTab" % set_text, width=btn_w):
                if app.is_setting_user_roi_mode:
                    app.exit_set_user_roi_mode()
                else:
                    app.enter_set_user_roi_mode()

            if has_roi:
                imgui.same_line()
                if imgui.button("Clear ROI##UserClearROI_RunTab", width=btn_w):
                    if tr and hasattr(tr, "clear_user_defined_roi_and_point"):
                        tr.stop_tracking()
                        tr.clear_user_defined_roi_and_point()
                        app.logger.info("User ROI cleared.", extra={"status_message": True})

        if app.is_setting_user_roi_mode:
            col = self.ControlPanelColors.STATUS_WARNING
            imgui.text_ansi_colored(
                "Selection Active: Draw ROI then click point on video.", *col
            )

# ---------------- Interactive refinement ----------------

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

            _tooltip_if_hovered(
                "Enables clicking on object boxes in the video to refine the script for that chapter."
            )

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
        imgui.separator()

# ---------------- Post-processing helpers ----------------

    def _render_post_processing_profile_row(self, long_name, profile_params, config_copy):
        changed_cfg = False
        imgui.push_id("profile_%s" % long_name)
        is_open = imgui.tree_node(long_name)

        if is_open:
            imgui.columns(2, "profile_settings", border=False)

            imgui.text("Amplification")
            imgui.separator()

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
            imgui.separator()

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
            imgui.separator()

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
            ch, nv = imgui.checkbox(
                "Enable Automatic Post-Processing on Completion", enabled
            )
            if ch and nv != enabled:
                app.app_settings.set("enable_auto_post_processing", nv)
                app.project_manager.project_dirty = True
                app.logger.info(
                    "Automatic post-processing on completion %s."
                    % ("enabled" if nv else "disabled"),
                    extra={"status_message": True},
                )
            _tooltip_if_hovered(
                "If checked, the profiles below will be applied automatically\n"
                "after an offline analysis or live tracking session finishes."
            )
            imgui.separator()

            if imgui.button("Run Post-Processing Now##RunAutoPostProcessButton", width=-1):
                if hasattr(fs_proc, "apply_automatic_post_processing"):
                    fs_proc.apply_automatic_post_processing()
            imgui.separator()

            use_chapter = app.app_settings.get("auto_processing_use_chapter_profiles", True)
            ch, nv = imgui.checkbox(
                "Apply Per-Chapter Settings (if available)", use_chapter
            )
            if ch and nv != use_chapter:
                app.app_settings.set("auto_processing_use_chapter_profiles", nv)
            _tooltip_if_hovered(
                "If checked, applies specific profiles below to each chapter.\n"
                "If unchecked, applies only the 'Default' profile to the entire script."
            )
            imgui.separator()

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

            imgui.separator()
            if imgui.button("Reset All Profiles to Defaults##ResetAutoPostProcessing", width=-1):
                app.app_settings.set(
                    "auto_post_processing_amplification_config",
                    self.constants.DEFAULT_AUTO_POST_AMP_CONFIG,
                )
                app.project_manager.project_dirty = True
                app.logger.info(
                    "All post-processing profiles reset to defaults.",
                    extra={"status_message": True},
                )
            imgui.separator()

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
                ch, nv = imgui.slider_float(
                    "Epsilon##FinalRDPEpsilon", cur_eps, 0.1, 20.0, "%.2f"
                )
                if ch and nv != cur_eps:
                    app.app_settings.set("auto_post_proc_final_rdp_epsilon", nv)
                    app.project_manager.project_dirty = True
                imgui.pop_item_width()

        # Disabled tooltip
        if proc_tools_disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")

    # ---------------- Calibration ----------------

    def _render_calibration_window(self, calibration_mgr, app_state):
        window_title = "Latency Calibration"
        if app_state.ui_layout_mode == "fixed":
            flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE
            imgui.begin("Modular Control Panel##LeftControlsModular", flags=flags)
            self._render_latency_calibration(calibration_mgr)
            imgui.end()
        else:
            if imgui.begin(window_title, closable=False, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE):
                self._render_latency_calibration(calibration_mgr)
                imgui.end()

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
                self.app.logger.info(
                    "Please select a reference point on Timeline 1 first.",
                    extra={"status_message": True},
                )
        if imgui.button("Cancel Calibration##CancelCalibration", width=-1):
            calibration_mgr.is_calibration_mode_active = False
            calibration_mgr.calibration_reference_point_selected = False
            self.app.logger.info(
                "Latency calibration cancelled.", extra={"status_message": True}
            )
            self.app.energy_saver.reset_activity_timer()

    # ---------------- Range selection ----------------

    def _render_range_selection(self, stage_proc, fs_proc, event_handlers):
        app = self.app
        disabled = stage_proc.full_analysis_active or (app.processor and app.processor.is_processing) or app.is_setting_user_roi_mode

        with _DisabledScope(disabled):
            ch, new_active = imgui.checkbox(
                "Enable Range Processing", fs_proc.scripting_range_active
            )
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
                    imgui.text(
                        "Chapter: %s (%s)"
                        % (sel_ch.class_name, sel_ch.segment_type)
                    )
                if imgui.button("Clear Range Selection##ClearRangeButton"):
                    event_handlers.clear_scripting_range_selection()
            else:
                imgui.text_disabled("Range processing not active. Enable checkbox or select a chapter.")

        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")

    def _render_tracking_axes_mode(self, stage_proc):
        """Render UI controls for selecting which movement axes to track and output."""
        app = self.app
        
        imgui.text("Tracking Axes")
        imgui.same_line()
        imgui.push_item_width(120)
        
        axis_mode_opts = ["Both Axes", "Vertical Only", "Horizontal Only"]
        current_mode_idx = 0
        if app.tracking_axis_mode == "vertical":
            current_mode_idx = 1
        elif app.tracking_axis_mode == "horizontal":
            current_mode_idx = 2
            
        changed, new_mode_idx = imgui.combo("##TrackingAxes", current_mode_idx, axis_mode_opts)
        if changed:
            if new_mode_idx == 0:
                app.tracking_axis_mode = "both"
            elif new_mode_idx == 1:
                app.tracking_axis_mode = "vertical"
            else:
                app.tracking_axis_mode = "horizontal"
            app.app_settings.set("tracking_axis_mode", app.tracking_axis_mode)
            
        imgui.pop_item_width()
        
        # Show single axis output target when not tracking both axes
        if app.tracking_axis_mode != "both":
            imgui.same_line()
            imgui.push_item_width(100)
            output_target_opts = ["Primary", "Secondary"]
            current_target_idx = 0 if app.single_axis_output_target == "primary" else 1
            
            changed, new_target_idx = imgui.combo("Output##SingleAxisOutput", current_target_idx, output_target_opts)
            if changed:
                app.single_axis_output_target = "primary" if new_target_idx == 0 else "secondary"
                app.app_settings.set("single_axis_output_target", app.single_axis_output_target)
                
            imgui.pop_item_width()
            
        _tooltip_if_hovered(
            "Select which movement axes to track during analysis.\n"
            "'Both Axes' tracks vertical and horizontal movement.\n"
            "'Vertical Only' or 'Horizontal Only' tracks only one axis.\n"
            "When tracking a single axis, choose which axis to output."
        )

    def _render_start_stop_buttons(self, stage_proc, fs_proc, event_handlers):
        """Render Start and Stop/Abort buttons for analysis."""
        app = self.app
        app_state = app.app_state_ui
        
        # Check if analysis is currently running
        analysis_active = stage_proc.full_analysis_active
        video_loaded = app.processor and app.processor.is_video_open()
        
        # Check if live tracking is running
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        
        # Determine button states and availability
        can_start = video_loaded and not analysis_active and not is_live_tracking_running and not app.is_setting_user_roi_mode
        can_stop = analysis_active or is_live_tracking_running
        
        # Start button
        with _DisabledScope(not can_start):
            if imgui.button("Start Analysis", width=120):
                if app_state.selected_tracker_mode in (self.TrackerMode.OFFLINE_2_STAGE, self.TrackerMode.OFFLINE_3_STAGE):
                    event_handlers.handle_start_ai_cv_analysis()
                else:
                    event_handlers.handle_start_live_tracker_click()
        
        if not can_start and imgui.is_item_hovered():
            if not video_loaded:
                imgui.set_tooltip("No video loaded. Please load a video first.")
            elif app.is_setting_user_roi_mode:
                imgui.set_tooltip("Please complete ROI selection first.")
            else:
                imgui.set_tooltip("Analysis already in progress.")
        
        imgui.same_line()
        
        # Stop/Abort button
        with _DisabledScope(not can_stop):
            if imgui.button("Stop Analysis", width=120):
                if analysis_active:
                    event_handlers.handle_abort_process_click()
                elif is_live_tracking_running:
                    event_handlers.handle_reset_live_tracker_click()
        
        if not can_stop and imgui.is_item_hovered():
            imgui.set_tooltip("No analysis currently running.")
        
        # Show current status
        if analysis_active:
            imgui.same_line()
            imgui.text_colored("Analysis Running...", *config.ControlPanelColors.STATUS_READY)
        elif is_live_tracking_running:
            imgui.same_line()
            imgui.text_colored("Live Tracking...", *config.ControlPanelColors.STATUS_INFO)

    # ---------------- Post-processing manual tools ----------------

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
            imgui.separator()

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
                        app.logger.info(
                            "No points selected for operation.",
                            extra={"status_message": True},
                        )

            imgui.separator()
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
            if imgui.button("Clear##ClearPoints"):
                prep_op()
                fs_proc.handle_funscript_operation("clear")

            imgui.separator()
            imgui.text("Amplify Values")
            ch, nv = imgui.slider_float(
                "Factor##AmplifyFactor", fs_proc.amplify_factor_input, 0.1, 3.0, "%.2f"
            )
            if ch:
                fs_proc.amplify_factor_input = nv
            ch, nv = imgui.slider_int(
                "Center##AmplifyCenter", fs_proc.amplify_center_input, 0, 100
            )
            if ch:
                fs_proc.amplify_center_input = nv
            if imgui.button("Apply Amplify##ApplyAmplify"):
                prep_op()
                fs_proc.handle_funscript_operation("amplify")

            imgui.separator()
            imgui.text("Savitzky-Golay Filter")
            ch, nv = imgui.slider_int("Window Length##SGWin", fs_proc.sg_window_length_input, 3, 99)
            if ch:
                event_handlers.update_sg_window_length(nv)
            max_po = max(1, fs_proc.sg_window_length_input - 1)
            po_val = min(fs_proc.sg_polyorder_input, max_po)
            ch, nv = imgui.slider_int("Polyorder##SGPoly", po_val, 1, max_po)
            if ch:
                fs_proc.sg_polyorder_input = nv
            if imgui.button("Apply Savitzky-Golay##ApplySG"):
                prep_op()
                fs_proc.handle_funscript_operation("apply_sg")

            imgui.separator()
            imgui.text("RDP Simplification")
            ch, nv = imgui.slider_float(
                "Epsilon##RDPEps", fs_proc.rdp_epsilon_input, 0.01, 20.0, "%.2f"
            )
            if ch:
                fs_proc.rdp_epsilon_input = nv
            if imgui.button("Apply RDP##ApplyRDP"):
                prep_op()
                fs_proc.handle_funscript_operation("apply_rdp")

            imgui.separator()
            imgui.text("Dynamic Amplification")
            if not hasattr(fs_proc, "dynamic_amp_window_ms_input"):
                fs_proc.dynamic_amp_window_ms_input = 4000
            ch, nv = imgui.slider_int(
                "Window (ms)##DynAmpWin", fs_proc.dynamic_amp_window_ms_input, 500, 10000
            )
            if ch:
                fs_proc.dynamic_amp_window_ms_input = nv
            _tooltip_if_hovered(
                "The size of the 'before/after' window in milliseconds to consider for amplification."
            )

            if imgui.button("Apply Dynamic Amplify##ApplyDynAmp"):
                prep_op()
                fs_proc.handle_funscript_operation("apply_dynamic_amp")

        # If disabled, show a tooltip on hover (outside the disabled scope)
        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")
