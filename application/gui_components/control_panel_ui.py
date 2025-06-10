import imgui
import os


import detection.cd.stage_1_cd as stage1_module
from config import constants

class ControlPanelUI:
    def __init__(self, app):
        self.app = app
        self.timeline_editor1 = None
        self.timeline_editor2 = None

    # --- Content Renderer Methods (Contain the actual UI widgets) ---

    def _render_content_general_tracking_settings(self):
        self._render_tracking_axes_mode(self.app.stage_processor)
        imgui.separator()
        imgui.spacing()
        if self.app.app_state_ui.selected_tracker_type_idx in [0, 1, 2]:
            imgui.text("Class Filtering")
            self._render_class_filtering_content()
        else:
            imgui.text_disabled("Class filtering not applicable to User Defined ROI mode.")

    def _render_content_tracking_control(self):
        app_state = self.app.app_state_ui
        stage_proc, fs_proc, event_handlers = self.app.stage_processor, self.app.funscript_processor, self.app.event_handlers
        tracking_modes = ["YOLO AI + Opt. Flow (3 Stages)", "YOLO AI (2 Stages)", "Live Optical Flow (YOLO ROI)", "Live Optical Flow (User ROI)"]
        disable_combo = stage_proc.full_analysis_active or (self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if disable_combo:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True); imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        clicked, new_idx = imgui.combo("Tracker Type##TrackerModeCombo", app_state.selected_tracker_type_idx, tracking_modes)
        if disable_combo:
            imgui.pop_style_var(); imgui.internal.pop_item_flag()
        if clicked and new_idx != app_state.selected_tracker_type_idx:
            app_state.selected_tracker_type_idx = new_idx
            mode_map = {0: "YOLO_ROI", 1: "YOLO_ROI", 2: "YOLO_ROI", 3: "USER_FIXED_ROI"}
            self.app.tracker.set_tracking_mode(mode_map.get(new_idx, "YOLO_ROI"))
        imgui.separator(); imgui.spacing()
        self._render_start_stop_buttons(stage_proc, fs_proc, event_handlers)
        imgui.spacing()
        if app_state.selected_tracker_type_idx in [0, 1]: self._render_offline_analysis_panel(stage_proc)
        elif app_state.selected_tracker_type_idx == 2: self._render_live_optical_flow_panel()
        elif app_state.selected_tracker_type_idx == 3: self._render_user_roi_tracking_panel()


    def _render_content_range_selection(self):
        self._render_range_selection(self.app.stage_processor, self.app.funscript_processor, self.app.event_handlers)

    def _render_content_funscript_processing_tools(self):
        self._render_funscript_processing_tools(self.app.funscript_processor, self.app.event_handlers)

    def _render_content_automatic_post_processing(self):
        self._render_automatic_post_processing(self.app.funscript_processor, self.app.event_handlers)

    def _render_content_funscript_window_controls(self):
        self._render_funscript_window_controls(self.app.app_state_ui)

    # --- Original Helper Methods (no changes needed for these) ---
    def _render_start_stop_buttons(self, stage_proc, fs_proc, event_handlers):
        """
        A centralized helper to render all Start/Pause/Abort/Stop buttons.
        This single function handles all tracking modes and application states.
        """
        # Determine overall state
        is_batch_mode = self.app.is_batch_processing_active
        is_analysis_running = stage_proc.full_analysis_active
        is_live_tracking_running = self.app.processor and self.app.processor.is_processing
        is_setting_roi = self.app.is_setting_user_roi_mode
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi

        # --- Batch Processing Mode UI ---
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
            return  # In batch mode, we don't show other buttons

        # --- Normal Mode UI (Non-Batch) ---
        selected_mode_idx = self.app.app_state_ui.selected_tracker_type_idx
        button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) / 2

        # --- Start/Pause/Status Button (Left) ---
        if is_any_process_active:
            # A process is running, show a status/pause button instead of "Start"
            status_text = "Processing..."
            if is_analysis_running:
                status_text = "Aborting..." if stage_proc.current_analysis_stage == -1 else f"Stage {stage_proc.current_analysis_stage} Running..."
            elif is_live_tracking_running:
                if imgui.button("Pause Tracking", width=button_width):
                    self.app.processor.pause_processing()
                status_text = None  # Button already drawn
            elif is_setting_roi:
                status_text = "Setting ROI..."

            if status_text:
                imgui.button(status_text, width=button_width)  # Display status as a disabled-like button
        else:
            # No process is running, show the appropriate "Start" button
            start_text = "Start Analysis"
            handler = None
            if selected_mode_idx == 0:
                start_text = "Start AI CV + OF (Range)" if fs_proc.scripting_range_active else "Start AI CV + Opt.Flow"
                handler = event_handlers.handle_start_ai_cv_analysis
            elif selected_mode_idx == 1:
                start_text = "Start AI CV (Range)" if fs_proc.scripting_range_active else "Start AI CV Analysis"
                handler = event_handlers.handle_start_ai_cv_analysis
            elif selected_mode_idx in [2, 3]:
                start_text = "Start Live Tracking (Range)" if fs_proc.scripting_range_active else "Start Live Tracking"
                handler = event_handlers.handle_start_live_tracker_click

            if imgui.button(start_text, width=button_width):
                if handler:
                    handler()

        imgui.same_line()

        # --- Abort/Stop Button (Right) ---
        if not is_any_process_active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        if imgui.button("Abort/Stop Process##AbortGeneral", width=button_width):
            event_handlers.handle_abort_process_click()

        if not is_any_process_active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_stage_progress_ui(self, stage_proc):
        """A centralized helper to render the progress bars for Stages 1, 2, and 3."""
        selected_mode_idx = self.app.app_state_ui.selected_tracker_type_idx
        is_analysis_running = stage_proc.full_analysis_active

        # --- Stage 1 ---
        imgui.text("Stage 1: YOLO Object Detection")
        imgui.text_wrapped(f"Status: {stage_proc.stage1_status_text}")

        if is_analysis_running and stage_proc.current_analysis_stage == 1:
            # --- Overall Stage 1 Progress ---
            imgui.text(
                f"Time: {stage_proc.stage1_time_elapsed_str} | ETA: {stage_proc.stage1_eta_str} | Speed: {stage_proc.stage1_processing_fps_str}")
            imgui.text_wrapped(f"Progress: {stage_proc.stage1_progress_label}")

            imgui.progress_bar(stage_proc.stage1_progress_value, size=(-1, 0),
                               overlay=f"{stage_proc.stage1_progress_value * 100:.0f}%" if stage_proc.stage1_progress_value > 0 else "")
            imgui.spacing()

            # --- Queue Visualization and Suggestions ---
            frame_q_size = stage_proc.stage1_frame_queue_size
            frame_q_max = constants.STAGE1_FRAME_QUEUE_MAXSIZE
            frame_q_fraction = frame_q_size / frame_q_max if frame_q_max > 0 else 0.0

            # Determine bar color and suggestion message based on user-defined logic
            suggestion_message = ""
            bar_color = (0.2, 0.8, 0.2)  # Default to Green

            if frame_q_fraction > 0.9:  # Red condition
                bar_color = (0.9, 0.3, 0.3)
                suggestion_message = "Suggestion: Add consumer if resources allow"
            elif frame_q_fraction > 0.2:  # Orange (Balanced) condition - adjusted threshold for a wider "balanced" range
                bar_color = (1.0, 0.5, 0.0)
                suggestion_message = "Balanced"
            else:  # Green condition
                bar_color = (0.2, 0.8, 0.2)
                suggestion_message = "Suggestion: Lessen consumers or add producer"

            # Render the queue bar
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *bar_color)
            imgui.progress_bar(frame_q_fraction, size=(-1, 0), overlay=f"Frame Queue: {frame_q_size}/{frame_q_max}")
            imgui.pop_style_color()

            # Render the suggestion message
            if suggestion_message:
                imgui.text(suggestion_message)

            # Result Queue (Consumers -> Logger)
            result_q_size = stage_proc.stage1_result_queue_size
            imgui.text(f"Result Queue Size: ~{result_q_size}")

        imgui.separator()

        # --- Stage 2 ---
        s2_title = "Stage 2: Contact Analysis & Funscript" if selected_mode_idx == 1 else "Stage 2: Segmentation"
        imgui.text(s2_title)
        imgui.text_wrapped(f"Status: {stage_proc.stage2_status_text}")
        if is_analysis_running and stage_proc.current_analysis_stage == 2:
            imgui.text_wrapped(f"Main: {stage_proc.stage2_main_progress_label}")
            imgui.progress_bar(stage_proc.stage2_main_progress_value, size=(-1, 0),
                               overlay=f"{stage_proc.stage2_main_progress_value * 100:.0f}%" if stage_proc.stage2_main_progress_value >= 0 else "")
            if selected_mode_idx == 1:
                imgui.text_wrapped(f"Sub: {stage_proc.stage2_sub_progress_label}")
                imgui.progress_bar(stage_proc.stage2_sub_progress_value, size=(-1, 0),
                                   overlay=f"{stage_proc.stage2_sub_progress_value * 100:.0f}%" if stage_proc.stage2_sub_progress_value >= 0 else "")
        imgui.separator()

        # --- Stage 3 (Only for AI CV + OF mode) ---
        if selected_mode_idx == 0:
            imgui.text("Stage 3: Per-Segment Optical Flow")
            imgui.text_wrapped(f"Status: {stage_proc.stage3_status_text}")
            if is_analysis_running and stage_proc.current_analysis_stage == 3:
                imgui.text(
                    f"Time: {stage_proc.stage3_time_elapsed_str} | ETA: {stage_proc.stage3_eta_str} | Speed: {stage_proc.stage3_processing_fps_str}")
                imgui.text_wrapped(f"Segment: {stage_proc.stage3_current_segment_label}")
                imgui.progress_bar(stage_proc.stage3_segment_progress_value, size=(-1, 0),
                                   overlay=f"{stage_proc.stage3_segment_progress_value * 100:.0f}%")
                imgui.text_wrapped(f"Overall: {stage_proc.stage3_overall_progress_label}")
                imgui.progress_bar(stage_proc.stage3_overall_progress_value, size=(-1, 0),
                                   overlay=f"{stage_proc.stage3_overall_progress_value * 100:.0f}%")
        imgui.spacing()

    # --- END: NEW HELPER METHODS ---

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
                self.app.logger.info(f"Tracking axis mode set to: {self.app.tracking_axis_mode}",
                                     extra={'status_message': True})
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
                    self.app.logger.info(f"Single axis output target set to: {self.app.single_axis_output_target}",
                                         extra={'status_message': True})
                    self.app.energy_saver.reset_activity_timer()

        if disable_axis_controls:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_class_filtering_content(self):
        """Renders the content for class filtering, meant to be embedded."""
        available_classes = self.app.get_available_tracking_classes()
        if not available_classes:
            imgui.text_disabled("No classes available (model not loaded or no classes defined).")
            return

        imgui.text_wrapped(
            "Select classes to DISCARD from tracking and analysis. This affects AI CV analysis (Stages 1 & 2), AI CV + Opt. Flow (Stages 1 & 2), and Optical Flow (YOLO ROI) mode's ROI selection.")
        discarded_classes_set = set(self.app.discarded_tracking_classes)
        changed_any_class = False

        num_columns = 3
        table_flags = imgui.TABLE_SIZING_STRETCH_SAME
        if imgui.begin_table("ClassFilterTable", num_columns, flags=table_flags):
            col_idx = 0
            for class_name in available_classes:
                if col_idx == 0:
                    imgui.table_next_row()
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
            self.app.project_manager.project_dirty = True
            self.app.logger.info(f"Discarded classes updated: {self.app.discarded_tracking_classes}",
                                 extra={'status_message': True})
            self.app.energy_saver.reset_activity_timer()

        imgui.spacing()
        if imgui.button("Clear All Discards##ClearDiscardFilters", width=imgui.get_content_region_available_width()):
            if self.app.discarded_tracking_classes:
                self.app.discarded_tracking_classes.clear()
                self.app.project_manager.project_dirty = True
                self.app.logger.info("All class discard filters cleared.", extra={'status_message': True})
                self.app.energy_saver.reset_activity_timer()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Unchecks all classes, enabling all classes for tracking/analysis.")

    def _render_offline_analysis_panel(self, stage_proc):
        """Renders controls for any multi-stage offline analysis."""
        self._render_stage_progress_ui(stage_proc)
        imgui.separator()
        _, stage_proc.force_rerun_stage1 = imgui.checkbox("Force Re-run Stage 1##ForceRerunS1_Unified",
                                                          stage_proc.force_rerun_stage1)
        if self.app.app_state_ui.selected_tracker_type_idx == 0:
            imgui.same_line()
            _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox(
                "Force Re-run S2 Segmentation##ForceRerunS2_Unified", stage_proc.force_rerun_stage2_segmentation)

    def _render_live_optical_flow_panel(self):
        """Renders controls for Live Optical Flow (YOLO ROI based)."""
        imgui.text_disabled("Settings for this mode are in Menu > Processing > Live Tracker.")
        imgui.text(f"Tracker Actual FPS: {self.app.tracker.current_fps if self.app.tracker else 'N/A':.1f}")
        imgui.spacing()

    def _render_user_roi_tracking_panel(self):
        """Renders controls specific to User Defined ROI Tracking."""
        is_live_user_roi_running = self.app.tracker and self.app.tracker.tracking_mode == "USER_FIXED_ROI" and self.app.processor and self.app.processor.is_processing
        set_roi_button_disabled = self.app.stage_processor.full_analysis_active or is_live_user_roi_running or not self.app.file_manager.video_path

        if set_roi_button_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        set_roi_text = "Set ROI & Point##UserSetROI"
        if self.app.is_setting_user_roi_mode:
            set_roi_text = "Cancel Set ROI##UserCancelSetROI"
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
        imgui.spacing()
        imgui.text(f"Tracker Actual FPS: {self.app.tracker.current_fps if self.app.tracker else 'N/A':.1f}")
        imgui.spacing()

    def _render_automatic_post_processing(self, fs_proc, event_handlers):
        """Renders controls for automatic post-processing."""
        proc_tools_disabled = self.app.stage_processor.full_analysis_active or (
                    self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if proc_tools_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        if imgui.button("Run Auto Post-Processing Now##RunAutoPostProcessButton", width=-1):
            if hasattr(fs_proc, 'apply_automatic_post_processing'):
                fs_proc.apply_automatic_post_processing()
            else:
                self.app.logger.error("Funscript processor is missing the apply_automatic_post_processing method.")
        imgui.separator()

        imgui.text("Settings for Automatic Post-Processing")
        changed_enable_app, new_enable_app_val = imgui.checkbox("Run automatically after analysis/tracking##EnableAPP",
                                                                self.app.app_settings.get("enable_auto_post_processing",
                                                                                          False))
        if changed_enable_app: self.app.app_settings.set("enable_auto_post_processing", new_enable_app_val)
        if imgui.is_item_hovered(): imgui.set_tooltip(
            "If checked, these processing steps will run automatically after a funscript is generated.")

        imgui.text("Savitzky-Golay Filter:")
        imgui.push_item_width(imgui.get_content_region_available()[0] * 0.4)
        sg_win = self.app.app_settings.get("auto_post_processing_sg_window", 7)
        changed_sg_win, new_sg_win = imgui.input_int("Window##APPSgWin", sg_win)
        if changed_sg_win:
            new_sg_win_clean = max(3, new_sg_win + 1 if new_sg_win % 2 == 0 else new_sg_win)
            self.app.app_settings.set("auto_post_processing_sg_window", new_sg_win_clean)

        imgui.same_line()
        sg_poly = self.app.app_settings.get("auto_post_processing_sg_polyorder", 3)
        max_poly = max(1, self.app.app_settings.get("auto_post_processing_sg_window", 7) - 1)
        current_poly = min(sg_poly, max_poly)
        changed_sg_poly, new_sg_poly = imgui.input_int("Polyorder##APPSgPoly", current_poly)
        if changed_sg_poly:
            new_sg_poly_clean = max(1, min(new_sg_poly, max_poly))
            self.app.app_settings.set("auto_post_processing_sg_polyorder", new_sg_poly_clean)
        imgui.pop_item_width()

        imgui.text("RDP Simplification:")
        imgui.push_item_width(-1)
        rdp_eps = self.app.app_settings.get("auto_post_processing_rdp_epsilon", 1.5)
        changed_rdp_eps, new_rdp_eps = imgui.slider_float("Epsilon##APPRdpEps", rdp_eps, 0.1, 20.0, "%.2f")
        if changed_rdp_eps: self.app.app_settings.set("auto_post_processing_rdp_epsilon", new_rdp_eps)
        imgui.pop_item_width()

        imgui.text("Primary Axis Clamping:")
        imgui.push_item_width(imgui.get_content_region_available()[0] * 0.4)
        cl_low = self.app.app_settings.get("auto_post_processing_clamp_lower_threshold_primary", 10)
        ch_cl_low, n_cl_low = imgui.slider_int("Lower Threshold (to 0)##APPClLow", cl_low, 0, 100)
        if ch_cl_low: self.app.app_settings.set("auto_post_processing_clamp_lower_threshold_primary", n_cl_low)


        cl_high = self.app.app_settings.get("auto_post_processing_clamp_upper_threshold_primary", 90)
        ch_cl_high, n_cl_high = imgui.slider_int("Upper Threshold (to 100)##APPClHigh", cl_high, 0, 100)
        if ch_cl_high: self.app.app_settings.set("auto_post_processing_clamp_upper_threshold_primary", n_cl_high)
        imgui.pop_item_width()

        imgui.separator()
        imgui.text("Segment Amplification Settings")
        amp_config = self.app.app_settings.get("auto_post_processing_amplification_config", {})
        config_copy = amp_config.copy()
        config_changed = False

        if imgui.begin_table("AmpConfigTable", 3, flags=imgui.TABLE_SIZING_STRETCH_SAME | imgui.TABLE_BORDERS_INNER):
            imgui.table_setup_column("Segment Type")
            imgui.table_setup_column("Scale Factor")
            imgui.table_setup_column("Center Value")
            imgui.table_headers_row()

            # FIXED LOGIC: Iterate over unique display names to prevent duplicate rows.
            unique_long_names = sorted(list(set(d.get("long_name") for d in constants.POSITION_INFO_MAPPING.values())))
            all_display_names = ["Default"] + unique_long_names

            for long_name in all_display_names:
                if long_name not in config_copy: config_copy[long_name] = amp_config.get("Default",
                                                                                         {"scale_factor": 1.0,
                                                                                          "center_value": 50})

                current_params = config_copy[long_name]
                scale = current_params.get("scale_factor", 1.0)
                center = current_params.get("center_value", 50)

                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(long_name)

                imgui.table_set_column_index(1)
                imgui.push_item_width(-1)
                changed_scale, new_scale = imgui.slider_float(f"##scale_{long_name}", scale, 0.1, 5.0, "%.2f")
                if changed_scale:
                    config_copy[long_name]["scale_factor"] = new_scale
                    config_changed = True
                imgui.pop_item_width()

                imgui.table_set_column_index(2)
                imgui.push_item_width(-1)
                changed_center, new_center = imgui.slider_int(f"##center_{long_name}", center, 0, 100)
                if changed_center:
                    config_copy[long_name]["center_value"] = new_center
                    config_changed = True
                imgui.pop_item_width()
            imgui.end_table()

        if config_changed:
            self.app.app_settings.set("auto_post_processing_amplification_config", config_copy)
            self.app.project_manager.project_dirty = True

        imgui.separator()
        if imgui.button("Reset All to Defaults##ResetAutoPostProcessing", width=-1):
            # Reset all APP settings to their default values
            self.app.app_settings.set("enable_auto_post_processing", False)
            self.app.app_settings.set("auto_post_processing_sg_window", 7)
            self.app.app_settings.set("auto_post_processing_sg_polyorder", 3)
            self.app.app_settings.set("auto_post_processing_rdp_epsilon", 15)
            self.app.app_settings.set("auto_post_processing_clamp_lower_threshold_primary", 10)
            self.app.app_settings.set("auto_post_processing_clamp_upper_threshold_primary", 90)
            self.app.app_settings.set("auto_post_processing_amplification_config", constants.DEFAULT_AUTO_POST_AMP_CONFIG)
            self.app.project_manager.project_dirty = True
            self.app.logger.info("Automatic post-processing settings have been reset to their defaults.",
                                 extra={'status_message': True})

        if proc_tools_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    # --- Main Render Method ---
    def render(self, control_panel_w=None, available_height=None):
        app_state = self.app.app_state_ui
        calibration_mgr = self.app.calibration

        if calibration_mgr.is_calibration_mode_active:
            window_title = "Latency Calibration"
            if app_state.ui_layout_mode == 'floating':
                if imgui.begin(window_title, closable=False, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE):
                    self._render_latency_calibration(calibration_mgr)
                    imgui.end()
            else:
                imgui.begin("Modular Control Panel##LeftControlsModular",
                            flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
                self._render_latency_calibration(calibration_mgr)
                imgui.end()
            return

        sections = {
            "show_general_tracking_settings_section": ("General Tracking Settings",
                                                       self._render_content_general_tracking_settings),
            "show_tracking_control_section": ("Tracking Control", self._render_content_tracking_control),
            "show_range_selection_section": ("Range Selection", self._render_content_range_selection),
            "show_funscript_processing_tools_section": ("Funscript Processing Tools",
                                                        self._render_content_funscript_processing_tools),
            "show_auto_post_processing_section": ("Automatic Post-Processing",
                                                  self._render_content_automatic_post_processing),
            "show_funscript_window_controls_section": ("Funscript Window Controls",
                                                       self._render_content_funscript_window_controls)
        }

        if app_state.ui_layout_mode == 'floating':
            for flag_name, (title, content_func) in sections.items():
                is_visible = getattr(app_state, flag_name, False)
                if is_visible:
                    window_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE

                    # Target this specific window to change its behavior
                    if title == "Automatic Post-Processing":
                        imgui.set_next_window_size(450, 500, condition=imgui.APPEARING)
                        # Remove the auto-resize flag for this window
                        window_flags = 0

                    is_open, new_visibility = imgui.begin(title, closable=True, flags=window_flags)

                    if new_visibility != is_visible:
                        setattr(app_state, flag_name, new_visibility)
                        self.app.project_manager.project_dirty = True
                    if is_open:
                        content_func()
                    imgui.end()

        else:  # 'fixed' mode
            imgui.begin("Modular Control Panel##LeftControlsModular",
                        flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
            for flag_name, (title, content_func) in sections.items():
                is_visible = getattr(app_state, flag_name, False)
                if is_visible:
                    header_title = f"{title}##{title.replace(' ', '')}FixedPanel"
                    expanded, new_visibility = imgui.collapsing_header(header_title, is_visible, flags=imgui.TREE_NODE_DEFAULT_OPEN)
                    if new_visibility != is_visible:
                        setattr(app_state, flag_name, new_visibility)
                        self.app.project_manager.project_dirty = True
                    if expanded:
                        content_func()
            imgui.end()

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
        range_disabled = stage_proc.full_analysis_active or (
                    self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if range_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        clicked_active, new_active = imgui.checkbox("Enable Range Processing", fs_proc.scripting_range_active)
        if clicked_active: event_handlers.handle_scripting_range_active_toggle(new_active)
        if fs_proc.scripting_range_active:
            imgui.text("Set Frames Range Manually (-1 = End):")
            imgui.push_item_width(imgui.get_content_region_available()[0] * 0.4)
            changed_start, new_start = imgui.input_int("Start##SR_InputStart", fs_proc.scripting_start_frame,
                                                       flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
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
            if fs_proc.selected_chapter_for_scripting: imgui.text(
                f"Chapter: {fs_proc.selected_chapter_for_scripting.class_name} ({fs_proc.selected_chapter_for_scripting.segment_type})")
            if imgui.button("Clear Range Selection##ClearRangeButton"): event_handlers.clear_scripting_range_selection()
        else:
            imgui.text_disabled("Range processing not active. Enable checkbox or select a chapter.")

        if range_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_funscript_processing_tools(self, fs_proc, event_handlers):
        proc_tools_disabled = self.app.stage_processor.full_analysis_active or (
                    self.app.processor and self.app.processor.is_processing) or self.app.is_setting_user_roi_mode
        if proc_tools_disabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(
            imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

        axis_options = ["Primary Axis", "Secondary Axis"]
        current_axis_idx = 0 if fs_proc.selected_axis_for_processing == 'primary' else 1
        changed_axis, new_axis_idx = imgui.combo("Target Axis##ProcAxis", current_axis_idx, axis_options)
        if changed_axis: event_handlers.set_selected_axis_for_processing(
            'primary' if new_axis_idx == 0 else 'secondary')

        imgui.separator()
        imgui.text("Apply To:")
        range_label = fs_proc.get_operation_target_range_label()
        if imgui.radio_button(f"{range_label}##OpTargetRange",
                              fs_proc.operation_target_mode == 'apply_to_scripting_range'): fs_proc.operation_target_mode = 'apply_to_scripting_range'
        imgui.same_line()
        if imgui.radio_button("Selected Points##OpTargetSelect",
                              fs_proc.operation_target_mode == 'apply_to_selected_points'): fs_proc.operation_target_mode = 'apply_to_selected_points'

        def prep_op():  # Helper to update selection before operation
            if fs_proc.operation_target_mode == 'apply_to_selected_points':
                editor = self.timeline_editor1 if fs_proc.selected_axis_for_processing == 'primary' else self.timeline_editor2
                fs_proc.current_selection_indices = list(editor.multi_selected_action_indices) if editor else []
                if not fs_proc.current_selection_indices: self.app.logger.info("No points selected for operation.",
                                                                               extra={'status_message': True})

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
        if f_ch: fs_proc.amplify_factor_input = f_new
        c_ch, c_new = imgui.slider_int("Center##AmplifyCenter", fs_proc.amplify_center_input, 0, 100)
        if c_ch: fs_proc.amplify_center_input = c_new
        if imgui.button("Apply Amplify##ApplyAmplify"):
            prep_op()
            fs_proc.handle_funscript_operation('amplify')

        imgui.separator()
        imgui.text("Savitzky-Golay Filter")
        wl_ch, wl_new = imgui.slider_int("Window Length##SGWin", fs_proc.sg_window_length_input, 3, 99)
        if wl_ch: event_handlers.update_sg_window_length(wl_new)
        max_po = max(1, fs_proc.sg_window_length_input - 1)
        po_val = min(fs_proc.sg_polyorder_input, max_po)
        po_ch, po_new = imgui.slider_int("Polyorder##SGPoly", po_val, 1, max_po)
        if po_ch: fs_proc.sg_polyorder_input = po_new
        if imgui.button("Apply Savitzky-Golay##ApplySG"):
            prep_op()
            fs_proc.handle_funscript_operation('apply_sg')

        imgui.separator()
        imgui.text("RDP Simplification")
        e_ch, e_new = imgui.slider_float("Epsilon##RDPEps", fs_proc.rdp_epsilon_input, 0.01, 20.0, "%.2f")
        if e_ch: fs_proc.rdp_epsilon_input = e_new
        if imgui.button("Apply RDP##ApplyRDP"):
            prep_op()
            fs_proc.handle_funscript_operation('apply_rdp')

        if proc_tools_disabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

    def _render_funscript_window_controls(self, app_state):
        imgui.text("Timeline Zoom (ms/px):")
        changed_zoom, new_zoom_val = imgui.slider_float("##TimelineZoomFactor",
                                                        app_state.timeline_zoom_factor_ms_per_px, 0.5, 500.0, "%.1f",
                                                        flags=imgui.SLIDER_FLAGS_LOGARITHMIC)
        if changed_zoom: app_state.timeline_zoom_factor_ms_per_px = new_zoom_val
        imgui.spacing()
