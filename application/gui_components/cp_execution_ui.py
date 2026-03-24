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

        # Batch progress bar (when batch processing is active)
        if app.is_batch_processing_active and app.batch_video_paths:
            total = len(app.batch_video_paths)
            current = app.current_batch_video_index + 1
            if current < 1:
                current = 1
            progress = current / max(1, total)

            video_name = ""
            if 0 <= app.current_batch_video_index < total:
                import os
                video_name = os.path.basename(app.batch_video_paths[app.current_batch_video_index].get("path", ""))

            imgui.text_colored(f"Batch: {current} / {total}", 0.4, 0.8, 1.0, 1.0)
            if video_name:
                imgui.text_wrapped(video_name)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.3, 0.65, 1.0, 1.0)
            imgui.progress_bar(progress, size=(-1, 0), overlay=f"{current}/{total}")
            imgui.pop_style_color()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        if self._is_offline_tracker(mode):
            self._render_stage_progress_ui(stage_proc)
            return

        if self._is_live_tracker(mode):
            # Tracker Status block removed

            tracker_info = app.tracker.get_tracker_info() if app.tracker else None
            if tracker_info and getattr(tracker_info, 'requires_intervention', False):
                self._render_user_roi_controls_for_run_tab()
            return

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

            # Developer details -- hidden unless View > Show Advanced Options
            if self.app.app_state_ui.show_advanced_options:
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
                if frame_q_fraction > 0.9:
                    bar_color = CurrentTheme.RED_LIGHT[:3]
                elif frame_q_fraction > 0.2:
                    bar_color = CurrentTheme.ORANGE[:3]
                else:
                    bar_color = CurrentTheme.GREEN[:3]
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *bar_color)
                imgui.progress_bar(frame_q_fraction, size=(-1, 0), overlay=f"Frame Queue: {frame_q_size}/{frame_q_max}")
                imgui.pop_style_color()

                if getattr(stage_proc, 'save_preprocessed_video', False):
                    encoding_q_fraction = frame_q_fraction
                    encoding_bar_color = bar_color
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

            # Per-component timing -- hidden unless View > Show Advanced Options
            if self.app.app_state_ui.show_advanced_options:
                timing_parts = []
                decode_ms = getattr(stage_proc, 'stage1_decode_ms', 0.0)
                yolo_ms = getattr(stage_proc, 'stage1_yolo_det_ms', 0.0)
                flow_ms = getattr(stage_proc, 'stage2_flow_ms', 0.0)
                if decode_ms > 0:
                    timing_parts.append(f"Decode: {decode_ms:.1f}ms")
                if yolo_ms > 0:
                    timing_parts.append(f"YOLO: {yolo_ms:.1f}ms")
                if flow_ms > 0:
                    timing_parts.append(f"Flow: {flow_ms:.1f}ms")
                if timing_parts:
                    imgui.text(" | ".join(timing_parts))

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
                        app.notify("User ROI cleared", "info", 2.0)
                _tooltip_if_hovered("Remove the current ROI and tracking point.")

        if app.is_setting_user_roi_mode:
            col = self.ControlPanelColors.STATUS_WARNING
            imgui.text_ansi_colored("Selection Active: Draw ROI then click point on video.", *col)

    def _render_simple_progress_display(self):
        """Render compact progress display (used during batch processing)."""
        stage_proc = self.app.stage_processor
        current_stage = stage_proc.current_analysis_stage

        _stage_metrics = {
            1: (stage_proc.stage1_progress_value, stage_proc.stage1_processing_fps_str, stage_proc.stage1_eta_str),
            2: (stage_proc.stage2_main_progress_value, stage_proc.stage2_sub_processing_fps_str or "", stage_proc.stage2_sub_eta_str or "N/A"),
            3: (stage_proc.stage3_overall_progress_value, stage_proc.stage3_processing_fps_str, stage_proc.stage3_eta_str),
        }
        stage_progress, fps_str, eta_str = _stage_metrics.get(current_stage, (0.0, "", "N/A"))
        overall = max(0.0, min(1.0, stage_progress))

        imgui.progress_bar(overall, (-1, 0), "%.0f%%" % (overall * 100))

        status_parts = []
        if fps_str and fps_str != "0 FPS":
            status_parts.append(fps_str)
        if overall > 0.01 and eta_str and eta_str != "N/A":
            status_parts.append("ETA: %s" % eta_str)
        if status_parts:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text(" | ".join(status_parts))
            imgui.pop_style_color()
