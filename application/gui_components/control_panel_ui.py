import imgui
import os
from config import constants
from config.constants import TrackerMode

class ControlPanelUI:
    def __init__(self, app):
        self.app = app
        self.timeline_editor1 = None
        self.timeline_editor2 = None

    # --- Main Render Method ---
    def render(self, control_panel_w=None, available_height=None):
        app_state = self.app.app_state_ui
        calibration_mgr = self.app.calibration

        # Handle the special calibration mode separately
        if calibration_mgr.is_calibration_mode_active:
            self._render_calibration_window(calibration_mgr, app_state)
            return

        window_title = "Control Panel##ControlPanelFloating"
        window_flags = 0

        if app_state.ui_layout_mode == 'floating':
            if not getattr(app_state, 'show_control_panel_window', True):
                return
            is_open, new_visibility = imgui.begin(window_title, closable=True)
            if new_visibility != app_state.show_control_panel_window:
                app_state.show_control_panel_window = new_visibility
            if not is_open:
                imgui.end()
                return
        else:
            window_flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE
            imgui.begin("Control Panel##MainControlPanel", flags=window_flags)

        # Create the main tab bar for the new workflow
        if imgui.begin_tab_bar("ControlPanelTabs"):
            if imgui.begin_tab_item("Run Control")[0]:
                self._render_run_control_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Configuration")[0]:
                self._render_configuration_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Post-Processing")[0]:
                self._render_post_processing_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Settings")[0]:
                self._render_settings_tab()
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

    # --- Tab Renderer Methods ---

    def _render_run_control_tab(self):
        """Renders Tab 1: Mode selection, axis config, execution, and progress."""
        stage_proc = self.app.stage_processor
        fs_proc = self.app.funscript_processor
        app_state = self.app.app_state_ui
        event_handlers = self.app.event_handlers

        imgui.text("Select mode, configure axes, and run.")
        imgui.spacing()

        # --- Tracker Type Selection ---
        tracking_modes_enums = [
            TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI,
            TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE
        ]
        tracking_modes_display = [mode.value for mode in tracking_modes_enums]

        disable_combo = stage_proc.full_analysis_active or (self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if disable_combo:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        current_mode_idx = tracking_modes_enums.index(app_state.selected_tracker_mode)
        clicked, new_idx = imgui.combo("Tracker Type##TrackerModeCombo", current_mode_idx, tracking_modes_display)

        if disable_combo:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

        if clicked and new_idx != current_mode_idx:
            new_mode = tracking_modes_enums[new_idx]
            app_state.selected_tracker_mode = new_mode

            if new_mode == TrackerMode.LIVE_USER_ROI:
                self.app.tracker.set_tracking_mode("USER_FIXED_ROI")
                self.app.enter_set_user_roi_mode()
            else:
                self.app.tracker.set_tracking_mode("YOLO_ROI")

        # --- Tracking Axes ---
        self._render_tracking_axes_mode(stage_proc)
        imgui.separator()

        # --- Analysis Range and Rerun Options ---
        if app_state.selected_tracker_mode in [TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE, TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI]:
            if imgui.collapsing_header("Analysis Options##RunControlAnalysisOptions", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                # --- Range Selection ---
                imgui.text("Analysis Range")
                self._render_range_selection(self.app.stage_processor, self.app.funscript_processor, self.app.event_handlers)
                imgui.separator()

                # --- Force Rerun ---
                if app_state.selected_tracker_mode in [TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE]:
                    imgui.text("Stage Reruns:")
                    _, stage_proc.force_rerun_stage1 = imgui.checkbox("Force Re-run Stage 1##ForceRerunS1", stage_proc.force_rerun_stage1)
                    imgui.same_line()
                    _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox("Force Re-run S2 Chapter Creation##ForceRerunS2", stage_proc.force_rerun_stage2_segmentation)
                imgui.separator()

        # --- Execution Buttons ---
        self._render_start_stop_buttons(stage_proc, fs_proc, event_handlers)
        imgui.separator()

        video_loaded = self.app.processor and self.app.processor.is_video_open()
        processing_active = stage_proc.full_analysis_active or stage_proc.scene_detection_active
        button_should_be_disabled = not video_loaded or processing_active

        if button_should_be_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        #if imgui.button("Detect Scenes & Create Chapters", width=-1):
        #    if not button_should_be_disabled:
        #        stage_proc.start_scene_detection_analysis()

        if button_should_be_disabled:
            if imgui.is_item_hovered():
                imgui.set_tooltip("Requires a video to be loaded and no other process to be active.")
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

        imgui.separator()

        # --- Execution & Progress Display ---
        self._render_execution_progress_display()

    def _render_configuration_tab(self):
        """Renders Tab 2: All mode-specific configurations."""
        app_state = self.app.app_state_ui
        selected_mode = app_state.selected_tracker_mode

        imgui.text("Configure settings for the selected mode.")
        imgui.spacing()

        # AI Models & Inference settings are shown for any mode that uses them.
        if selected_mode in [TrackerMode.LIVE_YOLO_ROI, TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE]:
            if imgui.collapsing_header("AI Models & Inference##ConfigAIModels", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_ai_model_settings()
            imgui.separator()

        # Live-specific settings are shown for live modes.
        if selected_mode in [TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI]:
            if selected_mode == TrackerMode.LIVE_USER_ROI:
                if imgui.collapsing_header("ROI Selection##ConfigUserROIHeader", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_user_roi_tracking_panel()
                imgui.separator()
            self._render_live_tracker_settings()
            imgui.separator()

        # Class filtering is available for all YOLO-based modes.
        if selected_mode in [TrackerMode.LIVE_YOLO_ROI, TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE]:
            if imgui.collapsing_header("Class Filtering##ConfigClassFilterHeader", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                self._render_class_filtering_content()

        # Fallback message for any mode that has no configuration options.
        modes_with_config = {
            TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI,
            TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE
        }
        if selected_mode not in modes_with_config:
            imgui.text_disabled("No configuration available for this mode.")



    def _render_settings_tab(self):
        """Renders the new global Application Settings tab."""
        imgui.text("Global application settings. Saved in settings.json.")
        imgui.spacing()

        if imgui.collapsing_header("Interface & Performance##SettingsMenuPerfInterface",
                                   flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_settings_interface_perf()
        imgui.separator()

        if imgui.collapsing_header("File & Output##SettingsMenuOutput", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_settings_file_output()
        imgui.separator()

        if imgui.collapsing_header("Logging & Autosave##SettingsMenuLogging", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_settings_logging_autosave()
        imgui.separator()

        if imgui.collapsing_header("View/Edit Hotkeys##FSHotkeysMenuSettingsDetail", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_settings_hotkeys()
        imgui.separator()

        imgui.spacing()

        # --- "Reset to Default" Button and Confirmation Popup ---
        if imgui.button("Reset All Settings to Default##ResetAllSettingsButton", width=-1):
            # This line opens the popup when the button is clicked
            imgui.open_popup("Confirm Reset##ResetSettingsPopup")

        # Define the modal popup
        if imgui.begin_popup_modal("Confirm Reset##ResetSettingsPopup", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("This will reset all application settings to their defaults.")
            imgui.text("Your projects will not be affected.")
            imgui.text("This action cannot be undone.")
            imgui.separator()

            popup_button_width = (imgui.get_content_region_available_width() - imgui.get_style().item_spacing[0]) / 2

            if imgui.button("Confirm Reset", width=popup_button_width):
                # Call the reset method if the user confirms
                self.app.app_settings.reset_to_defaults()
                self.app.logger.info("All settings have been reset to default.", extra={'status_message': True})
                imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", width=popup_button_width):
                imgui.close_current_popup()

            imgui.end_popup()

    def _render_post_processing_tab(self):
        if imgui.collapsing_header("Manual Adjustments##PostProcManual", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_funscript_processing_tools(self.app.funscript_processor, self.app.event_handlers)
        imgui.separator()

        if imgui.collapsing_header("Automated Post-Processing##PostProcAuto", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_automatic_post_processing_new(self.app.funscript_processor)

    def _render_ai_model_settings(self):
        stage_proc = self.app.stage_processor
        style = imgui.get_style()

        # --- Callback functions to update the model paths ---
        def update_detection_model_path(path: str):
            self.app.yolo_detection_model_path_setting = path
            self.app.app_settings.set("yolo_det_model_path", path)
            self.app.yolo_det_model_path = path
            self.app.project_manager.project_dirty = True
            self.app.logger.info(f"Detection model path selected: {path}. Setting has been saved.")
            if self.app.tracker:
                self.app.tracker.det_model_path = path
                self.app.tracker._load_models()

        def update_pose_model_path(path: str):
            self.app.yolo_pose_model_path_setting = path
            self.app.app_settings.set("yolo_pose_model_path", path)
            self.app.yolo_pose_model_path = path
            self.app.project_manager.project_dirty = True
            self.app.logger.info(f"Pose model path selected: {path}. Setting has been saved.")
            if self.app.tracker:
                self.app.tracker.pose_model_path = path
                self.app.tracker._load_models()

        def update_artifacts_dir_path(path: str):
            self.app.pose_model_artifacts_dir_setting = path
            self.app.app_settings.set("pose_model_artifacts_dir", path)
            self.app.project_manager.project_dirty = True
            self.app.logger.info(f"Pose Model Artifacts directory selected: {path}. Setting has been saved.")


        # --- Define fixed widths for buttons for consistent layout ---
        browse_button_width = imgui.calc_text_size("Browse").x + style.frame_padding.x * 2
        unload_button_width = imgui.calc_text_size("Unload").x + style.frame_padding.x * 2
        total_button_width = browse_button_width + unload_button_width + style.item_spacing.x

        # --- Calculate available width for the input text ---
        available_width = imgui.get_content_region_available_width()
        input_text_width = available_width - total_button_width - style.item_spacing.x

        # --- YOLO Detection Model ---
        imgui.text("Detection Model")
        imgui.push_item_width(input_text_width)
        current_yolo_path_display = self.app.yolo_detection_model_path_setting or "Not set"
        imgui.input_text("##S1YOLOPath", current_yolo_path_display, 256, flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Browse##S1YOLOBrowse"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                initial_dir = os.path.dirname(
                    self.app.yolo_detection_model_path_setting) if self.app.yolo_detection_model_path_setting else None
                self.app.gui_instance.file_dialog.show(
                    title="Select YOLO Detection Model", is_save=False, callback=update_detection_model_path,
                    extension_filter="AI Models (*.pt *.onnx *.mlpackage),*.pt;*.onnx;*.mlpackage|All Files,*.*",
                    initial_path=initial_dir
                )
        imgui.same_line()
        if imgui.button("Unload##S1YOLOUnload"):
            self.app.unload_model('detection')

        if imgui.is_item_hovered(): imgui.set_tooltip(
            "Path to the YOLO object detection model file (.pt, .onnx, .mlpackage).")

        # --- YOLO Pose Model ---
        imgui.text("Pose Model")
        imgui.push_item_width(input_text_width)
        current_pose_path_display = self.app.yolo_pose_model_path_setting or "Not set"
        imgui.input_text("##PoseYOLOPath", current_pose_path_display, 256, flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Browse##PoseYOLOBrowse"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                initial_dir = os.path.dirname(
                    self.app.yolo_pose_model_path_setting) if self.app.yolo_pose_model_path_setting else None
                self.app.gui_instance.file_dialog.show(
                    title="Select YOLO Pose Model", is_save=False, callback=update_pose_model_path,
                    extension_filter="AI Models (*.pt *.onnx *.mlpackage),*.pt;*.onnx;*.mlpackage|All Files,*.*",
                    initial_path=initial_dir
                )
        imgui.same_line()
        if imgui.button("Unload##PoseYOLOUnload"):
            self.app.unload_model('pose')

        if imgui.is_item_hovered(): imgui.set_tooltip(
            "Path to the YOLO pose estimation model file (.pt, .onnx, .mlpackage). This model is optional.")

        # UI for selecting the Pose Model Artifacts Directory
        imgui.separator()
        imgui.text("Pose Model Artifacts Dir")
        # For this one, the unload button doesn't make sense, so we give more space to the input text
        dir_input_width = available_width - browse_button_width - style.item_spacing.x if available_width > browse_button_width else -1
        imgui.push_item_width(dir_input_width)
        current_artifacts_path_display = self.app.pose_model_artifacts_dir or "Not set"
        imgui.input_text("##PoseArtifactsDirPath", current_artifacts_path_display, 256,
                         flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Browse##PoseArtifactsDirBrowse"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                self.app.gui_instance.file_dialog.show(
                    title="Select Pose Model Artifacts Directory",
                    callback=update_artifacts_dir_path,
                    is_folder_dialog=True,  # This tells the dialog to act as a folder picker
                    initial_path=self.app.pose_model_artifacts_dir
                )
        if imgui.is_item_hovered(): imgui.set_tooltip(
            "Path to the folder containing your trained classifier,\n"
            "imputer, and other .joblib model artifacts.")
        imgui.separator()

        # Conditionally render worker settings only for offline analysis modes
        if self.app.app_state_ui.selected_tracker_mode in [TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE]:
            imgui.separator()
            imgui.text("Stage 1 Inference Workers:")
            imgui.push_item_width(100)
            changed_prod, new_prod_s1_val = imgui.input_int("Producers##S1Producers", stage_proc.num_producers_stage1)
            if changed_prod:
                stage_proc.num_producers_stage1 = max(1, new_prod_s1_val)
                self.app.app_settings.set("num_producers_stage1", stage_proc.num_producers_stage1)
            if imgui.is_item_hovered(): imgui.set_tooltip("Number of threads for video decoding & preprocessing.")

            imgui.same_line()
            changed_cons, new_cons_s1_val = imgui.input_int("Consumers##S1Consumers", stage_proc.num_consumers_stage1)
            if changed_cons:
                stage_proc.num_consumers_stage1 = max(1, new_cons_s1_val)
                self.app.app_settings.set("num_consumers_stage1", stage_proc.num_consumers_stage1)
            if imgui.is_item_hovered(): imgui.set_tooltip(
                "Number of threads for AI model inference. Match to available cores for best performance.")
            imgui.pop_item_width()

    def _render_offline_analysis_settings(self, stage_proc, app_state):
        imgui.text("Stage Reruns:")
        _, stage_proc.force_rerun_stage1 = imgui.checkbox("Force Re-run Stage 1##ForceRerunS1",
                                                          stage_proc.force_rerun_stage1)

        imgui.same_line()
        _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox("Force Re-run S2 Chapter Creation##ForceRerunS2",
                                                                       stage_proc.force_rerun_stage2_segmentation)

    def _render_settings_interface_perf(self):
        energy_saver_mgr = self.app.energy_saver

        # Font Scale
        imgui.text("Font Scale")
        imgui.same_line()
        imgui.push_item_width(120)
        font_scale_options_display = ["70%", "80%", "90%", "100%", "110%", "125%", "150%", "175%", "200%"]
        font_scale_options_values = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
        current_scale_val = self.app.app_settings.get("global_font_scale", 1.0)
        try:
            current_scale_idx = min(range(len(font_scale_options_values)),
                                    key=lambda i: abs(font_scale_options_values[i] - current_scale_val))
        except (ValueError, IndexError):
            current_scale_idx = 3
        changed_font_scale, new_idx = imgui.combo("##GlobalFontScale", current_scale_idx, font_scale_options_display)
        if changed_font_scale:
            self.app.app_settings.set("global_font_scale", font_scale_options_values[new_idx])
            energy_saver_mgr.reset_activity_timer()
        imgui.pop_item_width()
        if imgui.is_item_hovered(): imgui.set_tooltip("Adjust the global UI font size. Applied instantly.")

        # Timeline Pan Speed Setting
        imgui.text("Timeline Pan Speed")
        imgui.same_line()
        imgui.push_item_width(120)
        current_pan_speed = self.app.app_settings.get("timeline_pan_speed_multiplier", 5)
        changed_pan_speed, new_pan_speed = imgui.slider_int("##TimelinePanSpeed", current_pan_speed, 1, 50)
        if changed_pan_speed:
            self.app.app_settings.set("timeline_pan_speed_multiplier", new_pan_speed)
        imgui.pop_item_width()
        if imgui.is_item_hovered(): imgui.set_tooltip("Multiplier for keyboard-based timeline panning speed.")

        # HW Accel
        imgui.text("Video Decoding")
        imgui.same_line();
        imgui.push_item_width(180)
        hw_accel_options = self.app.available_ffmpeg_hwaccels
        hw_accel_display = [opt.replace("videotoolbox", "VideoToolbox (macOS)") for opt in hw_accel_options]
        try:
            current_hw_idx = hw_accel_options.index(self.app.hardware_acceleration_method)
        except ValueError:
            current_hw_idx = 0
        changed_hw, new_hw_idx = imgui.combo("HW Acceleration##HWAccelMethod", current_hw_idx, hw_accel_display)
        if changed_hw:
            self.app.hardware_acceleration_method = hw_accel_options[new_hw_idx]
            self.app.app_settings.set("hardware_acceleration_method", hw_accel_options[new_hw_idx])
            self.app.logger.info(
                f"Hardware acceleration set to: {self.app.hardware_acceleration_method}. Reload video to apply.",
                extra={'status_message': True})
        imgui.pop_item_width()
        if imgui.is_item_hovered(): imgui.set_tooltip(
            "Select FFmpeg hardware acceleration. Requires video reload to apply.")

        imgui.separator()

        # Energy Saver
        imgui.text("Energy Saver Mode:")
        changed_es, val_es = imgui.checkbox("Enable##EnableES", energy_saver_mgr.energy_saver_enabled)
        if changed_es: energy_saver_mgr.energy_saver_enabled = val_es; self.app.app_settings.set("energy_saver_enabled",
                                                                                                 val_es)

        if energy_saver_mgr.energy_saver_enabled:
            imgui.push_item_width(100)
            imgui.text("Normal FPS")
            imgui.same_line()
            norm_fps = int(energy_saver_mgr.main_loop_normal_fps_target)
            ch_norm_fps, new_norm_fps = imgui.input_int("##NormalFPS", norm_fps)
            if ch_norm_fps: energy_saver_mgr.main_loop_normal_fps_target = max(10,
                                                                               new_norm_fps); self.app.app_settings.set(
                "main_loop_normal_fps_target", max(10, new_norm_fps))

            imgui.text("Idle After (s)")
            imgui.same_line()
            thresh = int(energy_saver_mgr.energy_saver_threshold_seconds)
            ch_thresh, new_thresh = imgui.input_int("##ESThreshold", thresh)
            if ch_thresh: energy_saver_mgr.energy_saver_threshold_seconds = float(
                max(10, new_thresh)); self.app.app_settings.set("energy_saver_threshold_seconds",
                                                                float(max(10, new_thresh)))

            imgui.text("Idle FPS")
            imgui.same_line()
            es_fps = int(energy_saver_mgr.energy_saver_fps)
            ch_es_fps, new_es_fps = imgui.input_int("##ESFPS", es_fps)
            if ch_es_fps: energy_saver_mgr.energy_saver_fps = max(1, new_es_fps); self.app.app_settings.set(
                "energy_saver_fps", max(1, new_es_fps))
            imgui.pop_item_width()

    def _render_settings_file_output(self):
        settings = self.app.app_settings

        # Output Folder
        imgui.text("Output Folder:")
        imgui.push_item_width(-1)
        current_output_folder = settings.get("output_folder_path", "output")
        changed, new_folder = imgui.input_text("##OutputFolder", current_output_folder, 256)
        if changed: settings.set("output_folder_path", new_folder)
        imgui.pop_item_width()
        if imgui.is_item_hovered(): imgui.set_tooltip(
            "Root folder for all generated files (projects, analysis data, etc.).")
        imgui.separator()

        imgui.text("Funscript Output:")
        c_auto_save_loc, val_auto_save_loc = imgui.checkbox("Autosave final script next to video",
                                                            settings.get("autosave_final_funscript_to_video_location",
                                                                         True))
        if c_auto_save_loc: settings.set("autosave_final_funscript_to_video_location", val_auto_save_loc)

        c_gen_roll, val_gen_roll = imgui.checkbox("Generate .roll file (from Timeline 2)",
                                                  settings.get("generate_roll_file", True))
        if c_gen_roll: settings.set("generate_roll_file", val_gen_roll)
        imgui.separator()

        imgui.text("Batch Processing Default:")
        current_overwrite = settings.get("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Process All (skips own matching version)", current_overwrite == 0): settings.set(
            "batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Skip if Funscript Exists", current_overwrite == 1): settings.set(
            "batch_mode_overwrite_strategy", 1)

    def _render_settings_logging_autosave(self):
        settings = self.app.app_settings

        # Logging
        imgui.text("Logging Level")
        imgui.same_line();
        imgui.push_item_width(150)
        logging_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            current_log_level_idx = logging_levels.index(self.app.logging_level_setting.upper())
        except ValueError:
            current_log_level_idx = 1  # INFO
        changed_log, new_log_idx = imgui.combo("##LogLevel", current_log_level_idx, logging_levels)
        if changed_log:
            new_level = logging_levels[new_log_idx]
            self.app.set_application_logging_level(new_level)  # This also sets the setting
        imgui.pop_item_width()
        imgui.separator()

        # Autosave
        imgui.text("Project Autosave:")
        c_auto_en, v_auto_en = imgui.checkbox("Enable##EnableAutosave", settings.get("autosave_enabled", True))
        if c_auto_en: settings.set("autosave_enabled", v_auto_en)

        if settings.get("autosave_enabled"):
            imgui.push_item_width(100)
            imgui.text("Interval (s)")
            imgui.same_line()
            interval = settings.get("autosave_interval_seconds", 300)
            c_interval, new_interval = imgui.input_int("##AutosaveInterval", interval)
            if c_interval: settings.set("autosave_interval_seconds", max(30, new_interval))
            imgui.pop_item_width()

    def _render_settings_hotkeys(self):
        shortcuts_settings = self.app.app_settings.get("funscript_editor_shortcuts", {})
        for action_name, key_str in list(shortcuts_settings.items()):
            action_display_name = action_name.replace('_', ' ').title()
            imgui.text(f"{action_display_name}: ")
            imgui.same_line()

            display_key = key_str
            button_text = "Record"
            if self.app.shortcut_manager.is_recording_shortcut_for == action_name:
                display_key = "PRESS KEY..."
                button_text = "Cancel"

            imgui.text_colored(display_key, 0.2, 0.8, 1.0, 1.0)
            imgui.same_line()

            if imgui.button(f"{button_text}##record_btn_{action_name}"):
                if self.app.shortcut_manager.is_recording_shortcut_for == action_name:
                    self.app.shortcut_manager.cancel_shortcut_recording()
                else:
                    self.app.shortcut_manager.start_shortcut_recording(action_name)

    def _render_execution_progress_display(self):
        """Renders the progress UI."""
        stage_proc = self.app.stage_processor
        app_state = self.app.app_state_ui
        selected_mode = app_state.selected_tracker_mode

        if selected_mode in [TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE]:
            if stage_proc.full_analysis_active or self.app.is_batch_processing_active or stage_proc.scene_detection_active:
                # Decide WHICH progress to show
                if stage_proc.scene_detection_active:
                    imgui.text("Scene Detection")
                    imgui.text(
                        f"Time: {stage_proc.scene_detection_time_elapsed_str} | ETA: {stage_proc.scene_detection_eta_str} | Speed: {stage_proc.scene_detection_processing_fps_str}")
                    imgui.text_wrapped(f"Status: {stage_proc.scene_detection_status}")
                    overlay_text = f"{stage_proc.scene_detection_progress * 100:.0f}%"
                    imgui.progress_bar(stage_proc.scene_detection_progress, size=(-1, 0), overlay=overlay_text)
                else: # Fallback to the main multi-stage analysis progress
                    self._render_stage_progress_ui(stage_proc)
            else:
                self._render_stage_progress_ui(stage_proc) # Also render persisted stats here

        elif selected_mode in [TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI]:
            imgui.text("Live Tracker Status:")
            imgui.text(f"  - Actual FPS: {self.app.tracker.current_fps if self.app.tracker else 'N/A':.1f}")
            roi_status = "Not Set"
            if self.app.tracker:
                if selected_mode == TrackerMode.LIVE_YOLO_ROI:
                    roi_status = f"Tracking '{self.app.tracker.main_interaction_class}'" if self.app.tracker.main_interaction_class else "Searching..."
                elif selected_mode == TrackerMode.LIVE_USER_ROI:
                    roi_status = "Set" if self.app.tracker.user_roi_fixed else "Not Set"
            imgui.text(f"  - ROI Status: {roi_status}")
        else:
            imgui.text_disabled("No execution monitoring for this mode.")

    # --- Helper & Content Renderer Methods ---

    def _render_live_tracker_settings(self):
        """Renders the live tracker configuration as global, persistent settings."""
        if not self.app.tracker:
            imgui.text_disabled("Tracker not initialized.")
            return

        tracker_instance = self.app.tracker
        settings = self.app.app_settings

        if imgui.collapsing_header("Detection & ROI Definition##ROIDetectionTrackerMenu",
                                   flags=imgui.TREE_NODE_DEFAULT_OPEN):
            # Confidence Threshold
            current_conf = settings.get("live_tracker_confidence_threshold")
            changed, new_conf = imgui.slider_float("Obj. Confidence##ROIConfTrackerMenu", current_conf, 0.1, 0.95,
                                                   "%.2f")
            if changed:
                settings.set("live_tracker_confidence_threshold", new_conf)
                tracker_instance.confidence_threshold = new_conf

            # ROI Padding
            current_padding = settings.get("live_tracker_roi_padding")
            changed, new_padding = imgui.input_int("ROI Padding##ROIPadTrackerMenu", current_padding)
            if changed:
                new_padding = max(0, new_padding)
                settings.set("live_tracker_roi_padding", new_padding)
                tracker_instance.roi_padding = new_padding

            # ROI Update Interval
            current_interval = settings.get("live_tracker_roi_update_interval")
            changed, new_interval = imgui.input_int("ROI Update Interval (frames)##ROIIntervalTrackerMenu",
                                                    current_interval)
            if changed:
                new_interval = max(1, new_interval)
                settings.set("live_tracker_roi_update_interval", new_interval)
                tracker_instance.roi_update_interval = new_interval

            # ROI Smoothing Factor
            current_smoothing = settings.get("live_tracker_roi_smoothing_factor")
            changed, new_smoothing = imgui.slider_float("ROI Smoothing Factor##ROISmoothTrackerMenu", current_smoothing,
                                                        0.0, 1.0, "%.2f")
            if changed:
                settings.set("live_tracker_roi_smoothing_factor", new_smoothing)
                tracker_instance.roi_smoothing_factor = new_smoothing

            # ROI Persistence
            current_persistence = settings.get("live_tracker_roi_persistence_frames")
            changed, new_persistence = imgui.input_int("ROI Persistence (frames)##ROIPersistTrackerMenu",
                                                       current_persistence)
            if changed:
                new_persistence = max(0, new_persistence)
                settings.set("live_tracker_roi_persistence_frames", new_persistence)
                tracker_instance.max_frames_for_roi_persistence = new_persistence

        if imgui.collapsing_header("Optical Flow##ROIFlowTrackerMenu", flags=imgui.TREE_NODE_DEFAULT_OPEN):
            # Use Sparse Flow
            current_sparse_flow = settings.get("live_tracker_use_sparse_flow")
            changed, new_sparse_flow = imgui.checkbox("Use Sparse Optical Flow##ROISparseFlowTrackerMenu",
                                                      current_sparse_flow)
            if changed:
                settings.set("live_tracker_use_sparse_flow", new_sparse_flow)
                tracker_instance.use_sparse_flow = new_sparse_flow

            # DIS Dense Flow Settings
            imgui.text("DIS Dense Flow Settings:")
            # Apply disable styling only if sparse flow is currently enabled
            if current_sparse_flow:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            dis_presets = ["ULTRAFAST", "FAST", "MEDIUM"]
            current_preset = settings.get("live_tracker_dis_flow_preset").upper()
            try:
                preset_idx = dis_presets.index(current_preset)
            except ValueError:
                preset_idx = 0
            changed, new_idx = imgui.combo("DIS Preset##ROIDISPresetTrackerMenu", preset_idx, dis_presets)
            if changed:
                new_preset = dis_presets[new_idx]
                settings.set("live_tracker_dis_flow_preset", new_preset)
                tracker_instance.update_dis_flow_config(preset=new_preset)

            current_scale = settings.get("live_tracker_dis_finest_scale")
            changed, new_scale = imgui.input_int("DIS Finest Scale (0-10, 0=auto)##ROIDISFineScaleTrackerMenu",
                                                 current_scale)
            if changed:
                settings.set("live_tracker_dis_finest_scale", new_scale)
                tracker_instance.update_dis_flow_config(finest_scale=new_scale)

            # Ensure pop is called when push was called
            if current_sparse_flow:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()

            if imgui.collapsing_header("Output Signal Generation##ROISignalTrackerMenu",
                                       flags=imgui.TREE_NODE_DEFAULT_OPEN):
                # Output Sensitivity
                current_sensitivity = settings.get("live_tracker_sensitivity")
                changed, new_sensitivity = imgui.slider_float("Output Sensitivity##ROISensTrackerMenu",
                                                              current_sensitivity,
                                                              0.0, 100.0, "%.1f")
                if changed:
                    settings.set("live_tracker_sensitivity", new_sensitivity)
                    tracker_instance.sensitivity = new_sensitivity

                # Base Amplification
                current_amp = settings.get("live_tracker_base_amplification")
                changed, new_amp = imgui.slider_float("Base Amplification##ROIBaseAmpTrackerMenu", current_amp, 0.1,
                                                      5.0,
                                                      "%.2f")
                if changed:
                    new_amp = max(0.1, new_amp)
                    settings.set("live_tracker_base_amplification", new_amp)
                    tracker_instance.base_amplification_factor = new_amp

                # Class-Specific Amplification Multipliers
                imgui.text("Class-Specific Amplification Multipliers:")
                current_class_amps = settings.get("live_tracker_class_amp_multipliers", {})
                class_amp_changed = False

                face_amp = current_class_amps.get("face", 1.0)
                ch_face, new_face_amp = imgui.slider_float("Face Amp. Mult.##ROIFaceAmpTrackerMenu", face_amp, 0.1, 5.0,
                                                           "%.2f")
                if ch_face:
                    current_class_amps["face"] = max(0.1, new_face_amp)
                    class_amp_changed = True

                hand_amp = current_class_amps.get("hand", 1.0)
                ch_hand, new_hand_amp = imgui.slider_float("Hand Amp. Mult.##ROIHandAmpTrackerMenu", hand_amp, 0.1, 5.0,
                                                           "%.2f")
                if ch_hand:
                    current_class_amps["hand"] = max(0.1, new_hand_amp)
                    class_amp_changed = True

                if class_amp_changed:
                    settings.set("live_tracker_class_amp_multipliers", current_class_amps)
                    tracker_instance.class_specific_amplification_multipliers = current_class_amps

                imgui.separator()

            # Flow Smoothing Window
            current_flow_smooth = settings.get("live_tracker_flow_smoothing_window")
            changed, new_flow_smooth = imgui.input_int("Flow Smoothing Window##ROIFlowSmoothWinTrackerMenu",
                                                       current_flow_smooth)
            if changed:
                new_flow_smooth = max(1, new_flow_smooth)
                settings.set("live_tracker_flow_smoothing_window", new_flow_smooth)
                tracker_instance.flow_history_window_smooth = new_flow_smooth

            imgui.separator()

            # Output Delay (This was already a global setting, just confirming the pattern)
            imgui.text("Output Delay (frames):")
            current_delay = settings.get("funscript_output_delay_frames")
            changed, new_delay = imgui.slider_int("##OutputDelayFrames", current_delay, 0, 20)
            if changed:
                settings.set("funscript_output_delay_frames", new_delay)
                self.app.calibration.funscript_output_delay_frames = new_delay
                self.app.calibration.update_tracker_delay_params()



    def _render_calibration_window(self, calibration_mgr, app_state):
        """Renders the dedicated latency calibration window."""
        window_title = "Latency Calibration"
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        if app_state.ui_layout_mode == 'fixed':
            # In fixed mode, embed it in the main panel area without a title bar
            imgui.begin("Modular Control Panel##LeftControlsModular",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
            self._render_latency_calibration(calibration_mgr)
            imgui.end()
        else: # Floating mode
            if imgui.begin(window_title, closable=False, flags=flags):
                self._render_latency_calibration(calibration_mgr)
                imgui.end()

    def _render_start_stop_buttons(self, stage_proc, fs_proc, event_handlers):
        is_batch_mode = self.app.is_batch_processing_active
        is_analysis_running = stage_proc.full_analysis_active
        is_live_tracking_running = self.app.processor and self.app.processor.is_processing
        is_setting_roi = self.app.is_setting_user_roi_mode
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi or stage_proc.scene_detection_active

        if is_batch_mode:
            imgui.text_ansi_colored("--- BATCH PROCESSING ACTIVE ---", 1.0, 0.7, 0.3)
            total_videos = len(self.app.batch_video_paths)
            current_idx = self.app.current_batch_video_index
            if 0 <= current_idx < total_videos:
                current_video_name = os.path.basename(self.app.batch_video_paths[current_idx])
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
                if self.app.processor.is_paused:
                    if imgui.button("Resume Tracking", width=button_width): self.app.processor.resume_processing()
                else:
                    if imgui.button("Pause Tracking", width=button_width): self.app.processor.pause_processing()
                status_text = None
            elif is_setting_roi:
                status_text = "Setting ROI..."
            if status_text: imgui.button(status_text, width=button_width)
        else:
            start_text = "Start"
            handler = None
            if selected_mode in [TrackerMode.OFFLINE_3_STAGE, TrackerMode.OFFLINE_2_STAGE]:
                start_text = "Start AI Analysis (Range)" if fs_proc.scripting_range_active else "Start Full AI Analysis"
                handler = event_handlers.handle_start_ai_cv_analysis
            elif selected_mode in [TrackerMode.LIVE_YOLO_ROI, TrackerMode.LIVE_USER_ROI]:
                start_text = "Start Live Tracking (Range)" if fs_proc.scripting_range_active else "Start Live Tracking"
                handler = event_handlers.handle_start_live_tracker_click
            if imgui.button(start_text, width=button_width):
                if handler: handler()

        imgui.same_line()
        if not is_any_process_active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button("Abort/Stop Process##AbortGeneral",
                        width=button_width): event_handlers.handle_abort_process_click()
        if not is_any_process_active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_stage_progress_ui(self, stage_proc):
        is_analysis_running = stage_proc.full_analysis_active
        selected_mode = self.app.app_state_ui.selected_tracker_mode

        # Stage 1
        imgui.text("Stage 1: YOLO Object Detection")
        if is_analysis_running and stage_proc.current_analysis_stage == 1:
            imgui.text(f"Time: {stage_proc.stage1_time_elapsed_str} | ETA: {stage_proc.stage1_eta_str} | Speed: {stage_proc.stage1_processing_fps_str}")
            imgui.text_wrapped(f"Progress: {stage_proc.stage1_progress_label}")
            imgui.progress_bar(stage_proc.stage1_progress_value, size=(-1, 0), overlay=f"{stage_proc.stage1_progress_value * 100:.0f}%" if stage_proc.stage1_progress_value > 0 else "")
            imgui.spacing()
            frame_q_size = stage_proc.stage1_frame_queue_size
            frame_q_max = constants.STAGE1_FRAME_QUEUE_MAXSIZE
            frame_q_fraction = frame_q_size / frame_q_max if frame_q_max > 0 else 0.0
            suggestion_message, bar_color = "", (0.2, 0.8, 0.2)
            if frame_q_fraction > 0.9:
                bar_color, suggestion_message = (0.9, 0.3, 0.3), "Suggestion: Add consumer if resources allow"
            elif frame_q_fraction > 0.2:
                bar_color, suggestion_message = (1.0, 0.5, 0.0), "Balanced"
            else:
                bar_color, suggestion_message = (0.2, 0.8, 0.2), "Suggestion: Lessen consumers or add producer"
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *bar_color)
            imgui.progress_bar(frame_q_fraction, size=(-1, 0), overlay=f"Frame Queue: {frame_q_size}/{frame_q_max}")
            imgui.pop_style_color()
            if suggestion_message: imgui.text(suggestion_message)
            imgui.text(f"Result Queue Size: ~{stage_proc.stage1_result_queue_size}")
        elif stage_proc.stage1_final_elapsed_time_str:
            imgui.text_wrapped(
                f"Last Run: {stage_proc.stage1_final_elapsed_time_str} | Avg Speed: {stage_proc.stage1_final_fps_str or 'N/A'}")
            imgui.progress_bar(1.0, size=(-1, 0), overlay="Completed")
        else:
            imgui.text_wrapped(f"Status: {stage_proc.stage1_status_text}")
        imgui.separator()

        # Stage 2
        s2_title = "Stage 2: Contact Analysis & Funscript" if selected_mode == TrackerMode.OFFLINE_2_STAGE else "Stage 2: Segmentation"
        imgui.text(s2_title)
        if is_analysis_running and stage_proc.current_analysis_stage == 2:
            imgui.text_wrapped(f"Main: {stage_proc.stage2_main_progress_label}")
            imgui.progress_bar(stage_proc.stage2_main_progress_value, size=(-1, 0), overlay=f"{stage_proc.stage2_main_progress_value * 100:.0f}%" if stage_proc.stage2_main_progress_value >= 0 else "")
        elif stage_proc.stage2_final_elapsed_time_str:
            imgui.text_wrapped(f"Status: Completed in {stage_proc.stage2_final_elapsed_time_str}")
            imgui.progress_bar(1.0, size=(-1, 0), overlay="Completed")
        else:
            imgui.text_wrapped(f"Status: {stage_proc.stage2_status_text}")
        imgui.separator()

        # Stage 3
        if selected_mode == TrackerMode.OFFLINE_3_STAGE:
            imgui.text("Stage 3: Per-Segment Optical Flow")
            if is_analysis_running and stage_proc.current_analysis_stage == 3:
                imgui.text(
                    f"Time: {stage_proc.stage3_time_elapsed_str} | ETA: {stage_proc.stage3_eta_str} | Speed: {stage_proc.stage3_processing_fps_str}")

                # --- Overall Progress ---
                imgui.text(stage_proc.stage3_overall_progress_label)
                overlay_text = f"{stage_proc.stage3_overall_progress_value * 100:.0f}%"
                imgui.progress_bar(stage_proc.stage3_overall_progress_value, size=(-1, 0), overlay=overlay_text)


                # --- Per-Segment Progress ---
                imgui.text_wrapped(f"Segment: {stage_proc.stage3_current_segment_label}")
                imgui.progress_bar(stage_proc.stage3_segment_progress_value, size=(-1, 0),
                                   overlay=f"{stage_proc.stage3_segment_progress_value * 100:.0f}%")

            elif stage_proc.stage3_final_elapsed_time_str:
                imgui.text_wrapped(
                    f"Last Run: {stage_proc.stage3_final_elapsed_time_str} | Avg Speed: {stage_proc.stage3_final_fps_str or 'N/A'}")
                imgui.progress_bar(1.0, size=(-1, 0), overlay="Completed")
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

        disable_axis_controls = stage_proc.full_analysis_active or \
                                (self.app.processor and self.app.processor.is_processing) or \
                                self.app.is_setting_user_roi_mode

        if disable_axis_controls:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        axis_mode_changed, new_axis_mode_idx = imgui.combo("Tracking Axes##TrackingAxisModeComboGlobal",
                                                           current_axis_mode_idx, axis_modes)
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

            output_target_changed, new_output_target_idx = imgui.combo("##SingleAxisOutputComboGlobal",
                                                                       current_output_target_idx, output_targets)
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

    def _render_class_filtering_content(self):
        available_classes = self.app.get_available_tracking_classes()
        if not available_classes:
            imgui.text_disabled("No classes available (model not loaded or no classes defined).")
            return
        imgui.text_wrapped("Select classes to DISCARD from tracking and analysis.")
        discarded_classes_set = set(self.app.discarded_tracking_classes)
        changed_any_class = False
        num_columns = 3
        table_flags = imgui.TABLE_SIZING_STRETCH_SAME
        if imgui.begin_table("ClassFilterTable", num_columns, flags=table_flags):
            col_idx = 0
            for class_name in available_classes:
                if col_idx == 0: imgui.table_next_row()
                imgui.table_set_column_index(col_idx)
                is_discarded = class_name in discarded_classes_set
                imgui.push_id(f"discard_cls_{class_name}")
                clicked, new_is_discarded = imgui.checkbox(f" {class_name}", is_discarded)
                imgui.pop_id()
                if clicked:
                    changed_any_class = True
                    if new_is_discarded:
                        discarded_classes_set.add(class_name)
                    else:
                        discarded_classes_set.remove(class_name)
                col_idx = (col_idx + 1) % num_columns
            imgui.end_table()
        if changed_any_class:
            self.app.discarded_tracking_classes = sorted(list(discarded_classes_set))
            self.app.app_settings.set("discarded_tracking_classes", self.app.discarded_tracking_classes) # Auto-save
            self.app.project_manager.project_dirty = True
            self.app.logger.info(f"Discarded classes updated: {self.app.discarded_tracking_classes}", extra={'status_message': True})
            self.app.energy_saver.reset_activity_timer()
        imgui.spacing()
        if imgui.button("Clear All Discards##ClearDiscardFilters", width=imgui.get_content_region_available_width()):
            if self.app.discarded_tracking_classes:
                self.app.discarded_tracking_classes.clear()
                self.app.app_settings.set("discarded_tracking_classes", self.app.discarded_tracking_classes) # Auto-save
                self.app.project_manager.project_dirty = True
                self.app.logger.info("All class discard filters cleared.", extra={'status_message': True})
                self.app.energy_saver.reset_activity_timer()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Unchecks all classes, enabling all classes for tracking/analysis.")

    def _render_user_roi_tracking_panel(self):
        set_roi_button_disabled = self.app.stage_processor.full_analysis_active or not (self.app.processor and self.app.processor.is_video_open())
        if set_roi_button_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True); imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        set_roi_text = "Set ROI & Point##UserSetROI"
        if self.app.is_setting_user_roi_mode: set_roi_text = "Cancel Set ROI##UserCancelSetROI"
        if imgui.button(set_roi_text, width=-1):
            if self.app.is_setting_user_roi_mode:
                self.app.exit_set_user_roi_mode()
            else:
                self.app.enter_set_user_roi_mode()
        if set_roi_button_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()
        if self.app.is_setting_user_roi_mode:
            imgui.text_ansi_colored("Selection Active: Draw ROI then click point on video.", 1.0, 0.7, 0.2)
        roi_status = "Set" if self.app.tracker and self.app.tracker.user_roi_fixed and self.app.tracker.user_roi_initial_point_relative else "Not Set"
        imgui.text(f"Selected ROI & Point: {roi_status}")

    def _render_post_processing_profile_row(self, long_name, profile_params, config_copy):
        """Helper to render a single, better-looking row for a post-processing profile."""
        config_changed = False
        imgui.push_id(f"profile_{long_name}")

        is_open = imgui.tree_node(long_name)

        if is_open:
            imgui.columns(2, "profile_settings", border=False)

            # --- Amplification Column ---
            imgui.text("Amplification")
            imgui.separator()

            imgui.text("Scale")
            imgui.next_column()
            imgui.push_item_width(-1)
            amp_scale = profile_params.get("scale_factor", 1.0)
            changed, new_val = imgui.slider_float("##scale", amp_scale, 0.1, 5.0, "%.2f")
            if changed:
                profile_params["scale_factor"] = new_val
                config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Center")
            imgui.next_column()
            imgui.push_item_width(-1)
            amp_center = profile_params.get("center_value", 50)
            changed, new_val = imgui.slider_int("##amp_center", amp_center, 0, 100)
            if changed:
                profile_params["center_value"] = new_val
                config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            # --- Clamping  ---
            clamp_low = profile_params.get("clamp_lower", 10)
            clamp_high = profile_params.get("clamp_upper", 90)

            imgui.text("Clamp Low")
            imgui.next_column()
            imgui.push_item_width(-1)
            changed_low, new_low_val = imgui.slider_int("##clamp_low", clamp_low, 0, 100)
            if changed_low:
                clamp_low = min(new_low_val, clamp_high)
                profile_params["clamp_lower"] = clamp_low
                config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Clamp High")
            imgui.next_column()
            imgui.push_item_width(-1)
            changed_high, new_high_val = imgui.slider_int("##clamp_high", clamp_high, 0, 100)
            if changed_high:
                clamp_high = max(new_high_val, clamp_low)
                profile_params["clamp_upper"] = clamp_high
                config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.columns(1)
            imgui.spacing()
            imgui.columns(2, "profile_settings_2", border=False)

            # --- Smoothing & Simplification Column ---
            imgui.text("Smoothing (SG Filter)")
            imgui.separator()

            imgui.text("Window")
            imgui.next_column()
            imgui.push_item_width(-1)
            sg_win = profile_params.get("sg_window", 7)
            changed, new_val = imgui.slider_int("##sg_win", sg_win, 3, 99)
            if changed: profile_params["sg_window"] = max(3,
                                                          new_val + 1 if new_val % 2 == 0 else new_val); config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Polyorder")
            imgui.next_column()
            imgui.push_item_width(-1)
            sg_poly = profile_params.get("sg_polyorder", 3)
            max_poly = max(1, profile_params.get("sg_window", 7) - 1)
            current_poly = min(sg_poly, max_poly)
            changed, new_val = imgui.slider_int("##sg_poly", current_poly, 1, max_poly)
            if changed:
                profile_params["sg_polyorder"] = new_val
                config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Simplification (RDP)")
            imgui.separator()

            imgui.text("Epsilon")
            imgui.next_column()
            imgui.push_item_width(-1)
            rdp_eps = profile_params.get("rdp_epsilon", 1.0)
            changed, new_val = imgui.slider_float("##rdp_eps", rdp_eps, 0.1, 20.0, "%.2f")
            if changed:
                profile_params["rdp_epsilon"] = new_val
                config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            # --- Output Range ---
            output_min = profile_params.get("output_min", 0)
            output_max = profile_params.get("output_max", 100)

            imgui.text("Output Min")
            imgui.next_column()
            imgui.push_item_width(-1)
            changed_out_min, new_out_min = imgui.slider_int("##out_min", output_min, 0, 100)
            if changed_out_min:
                output_min = min(new_out_min, output_max); profile_params["output_min"] = output_min; config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.text("Output Max")
            imgui.next_column()
            imgui.push_item_width(-1)
            changed_out_max, new_out_max = imgui.slider_int("##out_max", output_max, 0, 100)
            if changed_out_max:
                output_max = max(new_out_max, output_min); profile_params["output_max"] = output_max; config_changed = True
            imgui.pop_item_width()
            imgui.next_column()

            imgui.columns(1)
            imgui.tree_pop()

        if config_changed:
            config_copy[long_name] = profile_params

        imgui.pop_id()
        return config_changed

    def _render_automatic_post_processing_new(self, fs_proc):
        """Renders the improved automatic post-processing section."""
        proc_tools_disabled = self.app.stage_processor.full_analysis_active or (
                self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if proc_tools_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True);
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        # --- Master Switch for AUTOMATIC execution ---
        auto_post_proc_enabled = self.app.app_settings.get("enable_auto_post_processing", False)
        changed_master, new_master_enabled = imgui.checkbox("Enable Automatic Post-Processing on Completion",
                                                            auto_post_proc_enabled)
        if changed_master:
            self.app.app_settings.set("enable_auto_post_processing", new_master_enabled)
            self.app.project_manager.project_dirty = True
            self.app.logger.info(
                f"Automatic post-processing on completion {'enabled' if new_master_enabled else 'disabled'}.",
                extra={'status_message': True})
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "If checked, the profiles below will be applied automatically\n"
                "after an offline analysis or live tracking session finishes.")

        imgui.separator()

        # --- Manual Execution and Profile Configuration (Always Visible) ---
        if imgui.button("Run Post-Processing Now##RunAutoPostProcessButton", width=-1):
            if hasattr(fs_proc, 'apply_automatic_post_processing'): fs_proc.apply_automatic_post_processing()

        imgui.separator()

        # --- Secondary Switch for Per-Chapter Profiles ---
        use_chapter_profiles = self.app.app_settings.get("auto_processing_use_chapter_profiles", True)
        changed, new_use_chapters = imgui.checkbox("Apply Per-Chapter Settings (if available)", use_chapter_profiles)
        if changed:
            self.app.app_settings.set("auto_processing_use_chapter_profiles", new_use_chapters)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "If checked, applies specific profiles below to each chapter.\nIf unchecked, applies only the 'Default' profile to the entire script.")

        imgui.separator()

        config = self.app.app_settings.get("auto_post_processing_amplification_config", {})
        config_copy = config.copy()
        master_config_changed = False

        if use_chapter_profiles:
            imgui.text("Per-Position Processing Profiles")
            all_pos_long_names = ["Default"] + sorted(
                list(set(info["long_name"] for info in constants.POSITION_INFO_MAPPING.values())))
            default_profile_for_fallback = constants.DEFAULT_AUTO_POST_AMP_CONFIG.get("Default", {})
            for long_name in all_pos_long_names:
                if not long_name: continue
                profile_params = config_copy.get(long_name, default_profile_for_fallback).copy()
                if self._render_post_processing_profile_row(long_name, profile_params, config_copy):
                    master_config_changed = True
        else:
            imgui.text("Default Processing Profile (applies to all)")
            long_name = "Default"
            default_profile_for_fallback = constants.DEFAULT_AUTO_POST_AMP_CONFIG.get(long_name, {})
            profile_params = config_copy.get(long_name, default_profile_for_fallback).copy()
            if self._render_post_processing_profile_row(long_name, profile_params, config_copy):
                master_config_changed = True

        if master_config_changed:
            self.app.app_settings.set("auto_post_processing_amplification_config", config_copy)
            self.app.project_manager.project_dirty = True

        imgui.separator()
        if imgui.button("Reset All Profiles to Defaults##ResetAutoPostProcessing", width=-1):
            self.app.app_settings.set("auto_post_processing_amplification_config",
                                      constants.DEFAULT_AUTO_POST_AMP_CONFIG)
            self.app.project_manager.project_dirty = True
            self.app.logger.info("All post-processing profiles reset to defaults.", extra={'status_message': True})

        imgui.separator()

        # --- SECTION for Final RDP ---
        imgui.text("Final Smoothing Pass")
        final_rdp_enabled = self.app.app_settings.get("auto_post_proc_final_rdp_enabled", False)
        changed_final_rdp, new_final_rdp_enabled = imgui.checkbox("Run Final RDP Pass to Seam Chapters",
                                                                  final_rdp_enabled)
        if changed_final_rdp:
            self.app.app_settings.set("auto_post_proc_final_rdp_enabled", new_final_rdp_enabled)
            self.app.project_manager.project_dirty = True

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "After all other processing, run one final simplification pass\n"
                "on the entire script. This can help smooth out the joints\n"
                "between chapters that used different processing settings."
            )

        if final_rdp_enabled:
            imgui.same_line()
            imgui.push_item_width(120)
            final_rdp_epsilon = self.app.app_settings.get("auto_post_proc_final_rdp_epsilon", 10.0)
            changed_epsilon, new_epsilon = imgui.slider_float("Epsilon##FinalRDPEpsilon", final_rdp_epsilon, 0.1, 20.0,
                                                              "%.2f")
            if changed_epsilon:
                self.app.app_settings.set("auto_post_proc_final_rdp_epsilon", new_epsilon)
                self.app.project_manager.project_dirty = True
            imgui.pop_item_width()
        # --- END NEW SECTION ---

        if proc_tools_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_latency_calibration(self, calibration_mgr):
        imgui.text_ansi_colored("--- LATENCY CALIBRATION MODE ---", 1.0, 0.7, 0.3)
        if not calibration_mgr.calibration_reference_point_selected:
            imgui.text_wrapped("1. Start the live tracker for 10s of action then pause it.")
            imgui.text_wrapped("   Select a clear action point on Timeline 1.")
        else:
            imgui.text_wrapped(f"1. Point at {calibration_mgr.calibration_timeline_point_ms:.0f}ms selected.")
            imgui.text_wrapped("2. Now, use video controls (seek, frame step) to find the")
            imgui.text_wrapped("   EXACT visual moment corresponding to the selected point.")
            imgui.text_wrapped("3. Press 'Confirm Visual Match' below.")
        if imgui.button("Confirm Visual Match##ConfirmCalibration", width=-1):
            if calibration_mgr.calibration_reference_point_selected:
                calibration_mgr.confirm_latency_calibration()
            else:
                self.app.logger.info("Please select a reference point on Timeline 1 first.",
                                     extra={'status_message': True})
        if imgui.button("Cancel Calibration##CancelCalibration", width=-1):
            calibration_mgr.is_calibration_mode_active = False
            calibration_mgr.calibration_reference_point_selected = False
            self.app.logger.info("Latency calibration cancelled.", extra={'status_message': True})
            self.app.energy_saver.reset_activity_timer()

    def _render_range_selection(self, stage_proc, fs_proc, event_handlers):
        range_disabled = stage_proc.full_analysis_active or (self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if range_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        clicked_active, new_active = imgui.checkbox("Enable Range Processing", fs_proc.scripting_range_active)
        if clicked_active: event_handlers.handle_scripting_range_active_toggle(new_active)
        if fs_proc.scripting_range_active:
            imgui.text("Set Frames Range Manually (-1 = End):")
            imgui.push_item_width(imgui.get_content_region_available()[0] * 0.4)
            changed_start, new_start = imgui.input_int("Start##SR_InputStart", fs_proc.scripting_start_frame, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed_start: event_handlers.handle_scripting_start_frame_input(new_start)
            imgui.same_line()
            imgui.text(" ")
            imgui.same_line()
            changed_end, new_end = imgui.input_int("End (-1)##SR_InputEnd", fs_proc.scripting_end_frame,
                                                   flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed_end: event_handlers.handle_scripting_end_frame_input(new_end)
            imgui.pop_item_width()
            start_disp, end_disp = fs_proc.get_scripting_range_display_text()
            imgui.text(f"Active Range: Frames: {start_disp} to {end_disp}")
            if fs_proc.selected_chapter_for_scripting:
                imgui.text(f"Chapter: {fs_proc.selected_chapter_for_scripting.class_name} ({fs_proc.selected_chapter_for_scripting.segment_type})")
            if imgui.button("Clear Range Selection##ClearRangeButton"):
                event_handlers.clear_scripting_range_selection()
        else:
            imgui.text_disabled("Range processing not active. Enable checkbox or select a chapter.")
        if range_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_funscript_processing_tools(self, fs_proc, event_handlers):
        proc_tools_disabled = self.app.stage_processor.full_analysis_active or (self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if proc_tools_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        axis_options = ["Primary Axis", "Secondary Axis"]
        current_axis_idx = 0 if fs_proc.selected_axis_for_processing == 'primary' else 1
        changed_axis, new_axis_idx = imgui.combo("Target Axis##ProcAxis", current_axis_idx, axis_options)
        if changed_axis:
            event_handlers.set_selected_axis_for_processing('primary' if new_axis_idx == 0 else 'secondary')
        imgui.separator()
        imgui.text("Apply To:")
        range_label = fs_proc.get_operation_target_range_label()
        if imgui.radio_button(f"{range_label}##OpTargetRange", fs_proc.operation_target_mode == 'apply_to_scripting_range'):
            fs_proc.operation_target_mode = 'apply_to_scripting_range'
        imgui.same_line()
        if imgui.radio_button("Selected Points##OpTargetSelect", fs_proc.operation_target_mode == 'apply_to_selected_points'):
            fs_proc.operation_target_mode = 'apply_to_selected_points'
        def prep_op():
            if fs_proc.operation_target_mode == 'apply_to_selected_points':
                editor = self.timeline_editor1 if fs_proc.selected_axis_for_processing == 'primary' else self.timeline_editor2
                fs_proc.current_selection_indices = list(editor.multi_selected_action_indices) if editor else []
                if not fs_proc.current_selection_indices:
                    self.app.logger.info("No points selected for operation.", extra={'status_message': True})
        imgui.separator()
        imgui.text("Points operations")
        if imgui.button("Clamp to 0##Clamp0"):
            prep_op()
            fs_proc.handle_funscript_operation('clamp_0')
        imgui.same_line()
        if imgui.button("Clamp to 100##Clamp100"):
            prep_op()
            fs_proc.handle_funscript_operation('clamp_100')
        imgui.same_line()
        if imgui.button("Invert##InvertPoints"):
            prep_op()
            fs_proc.handle_funscript_operation('invert')
        imgui.same_line()
        if imgui.button("Clear##ClearPoints"):
            prep_op()
            fs_proc.handle_funscript_operation('clear')
        imgui.separator()
        imgui.text("Amplify Values")
        f_ch, f_new = imgui.slider_float("Factor##AmplifyFactor", fs_proc.amplify_factor_input, 0.1, 3.0, "%.2f")
        if f_ch:
            fs_proc.amplify_factor_input = f_new
        c_ch, c_new = imgui.slider_int("Center##AmplifyCenter", fs_proc.amplify_center_input, 0, 100)
        if c_ch:
            fs_proc.amplify_center_input = c_new
        if imgui.button("Apply Amplify##ApplyAmplify"):
            prep_op()
            fs_proc.handle_funscript_operation('amplify')
        imgui.separator()
        imgui.text("Savitzky-Golay Filter")
        wl_ch, wl_new = imgui.slider_int("Window Length##SGWin", fs_proc.sg_window_length_input, 3, 99)
        if wl_ch:
            event_handlers.update_sg_window_length(wl_new)
        max_po = max(1, fs_proc.sg_window_length_input - 1)
        po_val = min(fs_proc.sg_polyorder_input, max_po)
        po_ch, po_new = imgui.slider_int("Polyorder##SGPoly", po_val, 1, max_po)
        if po_ch:
            fs_proc.sg_polyorder_input = po_new
        if imgui.button("Apply Savitzky-Golay##ApplySG"):
            prep_op()
            fs_proc.handle_funscript_operation('apply_sg')
        imgui.separator()
        imgui.text("RDP Simplification")
        e_ch, e_new = imgui.slider_float("Epsilon##RDPEps", fs_proc.rdp_epsilon_input, 0.01, 20.0, "%.2f")
        if e_ch:
            fs_proc.rdp_epsilon_input = e_new
        if imgui.button("Apply RDP##ApplyRDP"):
            prep_op()
            fs_proc.handle_funscript_operation('apply_rdp')

        imgui.separator()
        imgui.text("Dynamic Amplification")

        # Add a new parameter to AppFunscriptProcessor for this
        if not hasattr(fs_proc, 'dynamic_amp_window_ms_input'):
            fs_proc.dynamic_amp_window_ms_input = 4000  # Default to 4 seconds

        win_ch, win_new = imgui.slider_int("Window (ms)##DynAmpWin", fs_proc.dynamic_amp_window_ms_input, 500, 10000)
        if win_ch:
            fs_proc.dynamic_amp_window_ms_input = win_new
        if imgui.is_item_hovered():
            imgui.set_tooltip("The size of the 'before/after' window in milliseconds to consider for amplification.")

        if imgui.button("Apply Dynamic Amplify##ApplyDynAmp"):
            prep_op()  # This is the existing helper function
            fs_proc.handle_funscript_operation('apply_dynamic_amp')

        if proc_tools_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()
