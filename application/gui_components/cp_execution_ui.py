"""Execution/Run tab UI mixin for ControlPanelUI."""
import imgui
import os
from config.constants_colors import CurrentTheme
from config.element_group_colors import ControlPanelColors as _CPColors
from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered


class ExecutionMixin:
    """Mixin providing execution progress and start/stop rendering methods."""

    def _render_execution_progress_display(self):
        app = self.app
        stage_proc = app.stage_processor
        app_state = app.app_state_ui
        mode = app_state.selected_tracker_name

        if self._is_offline_tracker(mode):
            self._render_stage_progress_ui(stage_proc)
            return

        if self._is_live_tracker(mode):
            # Tracker Status block removed

            tracker_info = app.tracker.get_tracker_info() if app.tracker else None
            if tracker_info and getattr(tracker_info, 'requires_intervention', False):
                self._render_user_roi_controls_for_run_tab()
            return

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
        # AND tracker processing has been explicitly enabled, OR if the tracker itself is active
        is_live_tracking_running = (self.app.processor and
                                    self.app.processor.is_processing and
                                    self.app.processor.enable_tracker_processing) or \
                                   (self.app.tracker and self.app.tracker.tracking_active)

        is_setting_roi = self.app.is_setting_user_roi_mode
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi

        if is_batch_mode:
            is_paused = self.app.is_batch_paused
            # Header
            if is_paused:
                imgui.text_ansi_colored("--- BATCH PAUSED ---", 1.0, 1.0, 0.3)  # yellow
            else:
                imgui.text_ansi_colored("--- BATCH PROCESSING ---", 1.0, 0.7, 0.3)  # orange

            total_videos = len(self.app.batch_video_paths)
            current_idx = self.app.current_batch_video_index

            # Overall batch progress bar
            if total_videos > 0:
                batch_frac = (current_idx + 1) / total_videos if current_idx >= 0 else 0.0
                imgui.text(f"Video {current_idx + 1} of {total_videos}")
                imgui.progress_bar(batch_frac, (-1, 0), f"{current_idx + 1}/{total_videos}")

            # Current video name
            if 0 <= current_idx < total_videos:
                current_video_name = os.path.basename(self.app.batch_video_paths[current_idx]["path"])
                imgui.text_wrapped(current_video_name)

            imgui.spacing()

            # Per-video progress (reuse simple progress display)
            if stage_proc.full_analysis_active:
                self._render_simple_progress_display()

            # Adaptive tuning status
            adaptive_state = self.app.adaptive_tuning_state
            if adaptive_state is not None:
                imgui.spacing()
                if adaptive_state.is_converged:
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.SUCCESS_TEXT)
                    imgui.text(f"Tuning converged: P={adaptive_state.best_producers}/C={adaptive_state.best_consumers} ({adaptive_state.best_fps:.0f} FPS)")
                    imgui.pop_style_color()
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.SUCCESS_TEXT)
                    imgui.text(f"Tuning: P={adaptive_state.current_producers}/C={adaptive_state.current_consumers}")
                    imgui.pop_style_color()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(adaptive_state.status_message)
                imgui.spacing()

            # Pause/Resume + Abort buttons side by side
            avail_w = imgui.get_content_region_available()[0]
            btn_w = (avail_w - imgui.get_style().item_spacing[0]) / 2
            if is_paused:
                with primary_button_style():
                    if imgui.button("Resume", width=btn_w):
                        self.app.resume_batch_processing()
            else:
                if imgui.button("Pause", width=btn_w):
                    self.app.pause_batch_processing()
            imgui.same_line()
            with destructive_button_style():
                if imgui.button("Abort Batch", width=btn_w):
                    self.app.abort_batch_processing()
            return

        selected_mode = self.app.app_state_ui.selected_tracker_name

        if is_any_process_active:
            # Status text for active processes
            if is_analysis_running:
                status_text = "Aborting..." if stage_proc.current_analysis_stage == -1 else f"Stage {stage_proc.current_analysis_stage} Running..."
                imgui.text(status_text)
            elif is_live_tracking_running:
                is_paused_live = self.app.processor.pause_event.is_set() if hasattr(self.app.processor, 'pause_event') else False
                imgui.text("Live Tracking Paused" if is_paused_live else "Live Tracking Active")
            elif is_setting_roi:
                imgui.text("Setting ROI...")
        else:
            # Resume-from-checkpoint controls (unique to Run tab)
            resumable_checkpoint = None
            if self._is_offline_tracker(selected_mode) and self.app.file_manager.video_path:
                resumable_checkpoint = stage_proc.can_resume_video(self.app.file_manager.video_path)

            if resumable_checkpoint:
                handler = event_handlers.handle_start_ai_cv_analysis
                button_width_third = (imgui.get_content_region_available()[0] - 2 * imgui.get_style().item_spacing[0]) / 3

                with primary_button_style():
                    if imgui.button(f"Resume ({resumable_checkpoint.progress_percentage:.0f}%)", width=button_width_third):
                        if stage_proc.start_resume_from_checkpoint(resumable_checkpoint):
                            self.app.logger.info("Resumed processing from checkpoint", extra={'status_message': True})
                _tooltip_if_hovered("Continue processing from the last checkpoint.")

                imgui.same_line()

                with primary_button_style():
                    if imgui.button("Start Fresh", width=button_width_third):
                        stage_proc.delete_checkpoint_for_video(self.app.file_manager.video_path)
                        handler()
                _tooltip_if_hovered("Delete the checkpoint and start a new analysis from scratch.")

                imgui.same_line()

                with destructive_button_style():
                    if imgui.button("Clear Resume", width=button_width_third):
                        stage_proc.delete_checkpoint_for_video(self.app.file_manager.video_path)
                _tooltip_if_hovered("Delete the saved checkpoint without starting a new analysis.")

            # Chapter overwrite setting (only for offline analysis)
            if self._is_offline_tracker(selected_mode):
                imgui.spacing()
                overwrite_setting = self.app.app_settings.get("overwrite_chapters_on_analysis", False)
                clicked, new_value = imgui.checkbox("Overwrite chapters during analysis", overwrite_setting)
                if clicked:
                    self.app.app_settings.set("overwrite_chapters_on_analysis", new_value)
                _tooltip_if_hovered("When checked, analysis will replace all existing chapters.\nWhen unchecked (default), existing chapters are preserved.")

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

            # Per-stage timing breakdown
            decode_ms = getattr(stage_proc, 'stage1_decode_ms', 0.0)
            unwarp_ms = getattr(stage_proc, 'stage1_unwarp_ms', 0.0)
            yolo_det_ms = getattr(stage_proc, 'stage1_yolo_det_ms', 0.0)
            yolo_pose_ms = getattr(stage_proc, 'stage1_yolo_pose_ms', 0.0)
            if decode_ms > 0 or yolo_det_ms > 0:
                timing_parts = [f"Decode: {decode_ms:.1f}ms"]
                if unwarp_ms > 0:
                    timing_parts.append(f"Unwarp: {unwarp_ms:.1f}ms")
                timing_parts.append(f"YOLO Det: {yolo_det_ms:.1f}ms")
                if yolo_pose_ms > 0:
                    timing_parts.append(f"Pose: {yolo_pose_ms:.1f}ms")
                imgui.text(" | ".join(timing_parts))

            frame_q_size = stage_proc.stage1_frame_queue_size
            frame_q_max = self.constants.STAGE1_FRAME_QUEUE_MAXSIZE
            frame_q_fraction = frame_q_size / frame_q_max if frame_q_max > 0 else 0.0
            suggestion_message, bar_color = "", CurrentTheme.GREEN[:3]
            if frame_q_fraction > 0.9:
                bar_color, suggestion_message = CurrentTheme.RED_LIGHT[:3], "Suggestion: Add consumer if resources allow"
            elif frame_q_fraction > 0.2:
                bar_color, suggestion_message = CurrentTheme.ORANGE[:3], "Balanced"
            else:
                bar_color, suggestion_message = CurrentTheme.GREEN[:3], "Suggestion: Lessen consumers or add producer"
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
                _tooltip_if_hovered("Cancel ROI selection mode.")
            else:
                with primary_button_style():
                    if imgui.button("%s##UserSetROI_RunTab" % set_text, width=btn_w):
                        app.enter_set_user_roi_mode()
                _tooltip_if_hovered("Draw a region of interest on the video, then click the tracking point.")

            if has_roi:
                imgui.same_line()
                # Clear ROI button (DESTRUCTIVE - clears user data)
                with destructive_button_style():
                    if imgui.button("Clear ROI##UserClearROI_RunTab", width=btn_w):
                        if tr and hasattr(tr, "clear_user_defined_roi_and_point"):
                            tr.stop_tracking()
                            tr.clear_user_defined_roi_and_point()
                        app.logger.info("User ROI cleared.", extra={"status_message": True})
                _tooltip_if_hovered("Remove the current ROI and tracking point.")

        if app.is_setting_user_roi_mode:
            col = self.ControlPanelColors.STATUS_WARNING
            imgui.text_ansi_colored("Selection Active: Draw ROI then click point on video.", *col)

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
                btn_text = "Disable Refinement Mode"
                with destructive_button_style():
                    if imgui.button("%s##ToggleInteractiveRefinement" % btn_text, width=-1):
                        app.app_state_ui.interactive_refinement_mode_enabled = not is_enabled
            else:
                btn_text = "Enable Refinement Mode"
                if imgui.button("%s##ToggleInteractiveRefinement" % btn_text, width=-1):
                    app.app_state_ui.interactive_refinement_mode_enabled = not is_enabled

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
