"""GUI event dispatch for the stage processing pipeline.

Extracted from AppStageProcessor to reduce file size.
Used as a mixin — all methods operate on the AppStageProcessor instance.
"""
import math
import os
from application.utils import VideoSegment


class StageGuiEventsMixin:
    """Mixin that handles dispatching queued GUI events from stage threads."""

    def process_gui_events(self):
        if self.full_analysis_active or self.refinement_analysis_active:
            if hasattr(self.app, 'energy_saver'):
                self.app.energy_saver.reset_activity_timer()

        fm = self.app.file_manager
        fs_proc = self.app.funscript_processor
        while not self.gui_event_queue.empty():
            try:
                queue_item = self.gui_event_queue.get_nowait()
                if not isinstance(queue_item, tuple) or len(queue_item) < 2:
                    continue

                event_type = queue_item[0]
                data1 = queue_item[1]
                data2 = queue_item[2] if len(queue_item) > 2 else None

                if event_type == "stage1_progress_update":
                    self._handle_stage1_progress(data1, data2)
                elif event_type == "stage1_status_update":
                    self.stage1_status_text = str(data1)
                    if data2 is not None:
                        self.stage1_progress_label = str(data2)
                elif event_type == "stage1_completed":
                    self.stage1_final_elapsed_time_str = str(data1)
                    self.stage1_final_fps_str = str(data2)
                    self.stage1_status_text = "Completed"
                    self.stage1_progress_value = 1.0
                elif event_type == "preprocessed_video_loaded":
                    if isinstance(data1, dict):
                        message = data1.get("message", "Preprocessed video loaded")
                        self.app.logger.info(message, extra={'status_message': True})
                        if hasattr(self.app, 'set_status_message'):
                            self.app.set_status_message(message)
                elif event_type == "stage1_queue_update":
                    if isinstance(data1, dict):
                        self.stage1_frame_queue_size = data1.get("frame_q_size", self.stage1_frame_queue_size)
                        self.stage1_result_queue_size = data1.get("result_q_size", self.stage1_result_queue_size)
                elif event_type == "stage2_dual_progress":
                    self._handle_stage2_dual_progress(data1, data2)
                elif event_type == "stage2_status_update":
                    self.stage2_status_text = str(data1)
                    if data2 is not None:
                        self.stage2_progress_label = str(data2)
                elif event_type == "stage2_completed":
                    self.stage2_final_elapsed_time_str = str(data1)
                    self.stage2_status_text = "Completed"
                    self.stage2_main_progress_value = 1.0
                    self.stage2_sub_progress_value = 1.0
                elif event_type == "stage2_results_success":
                    self._handle_stage2_results(data1, data2, fs_proc)
                elif event_type == "stage2_results_success_segments_only":
                    self._handle_stage2_segments_only(data1, fs_proc)
                elif event_type == "load_s2_overlay":
                    if data1 and os.path.exists(data1):
                        self.logger.info(f"Loading generated Stage 2 overlay data from: {data1}")
                        fm.load_stage2_overlay_data(data1)
                elif event_type == "stage3_progress_update":
                    self._handle_stage3_progress(data1)
                elif event_type == "stage3_results_success":
                    self._handle_stage3_results(data1, fs_proc)
                elif event_type == "stage3_status_update":
                    self.stage3_status_text = str(data1)
                    if data2 is not None:
                        self.stage3_overall_progress_label = str(data2)
                elif event_type == "stage3_completed":
                    self.stage3_final_elapsed_time_str = str(data1)
                    self.stage3_final_fps_str = str(data2)
                    self.stage3_status_text = "Completed"
                    self.stage3_overall_progress_value = 1.0
                elif event_type == "analysis_message":
                    self._handle_analysis_message(data1, data2)
                elif event_type == "refinement_completed":
                    payload = data1
                    chapter = payload.get('chapter')
                    new_actions = payload.get('new_actions')
                    if chapter and new_actions:
                        self.app.funscript_processor.apply_interactive_refinement(chapter, new_actions)
                else:
                    self.logger.warning(f"Unknown GUI event type received: {event_type}")
            except Exception as e:
                self.logger.error(f"Error processing GUI event in AppLogic's StageProcessor: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Per-event handler methods
    # ------------------------------------------------------------------

    def _handle_stage1_progress(self, prog_val, prog_data):
        if not isinstance(prog_data, dict):
            return
        self.stage1_progress_value = prog_val if prog_val != -1.0 else self.stage1_progress_value
        self.stage1_progress_label = str(prog_data.get("message", ""))

        t_el = prog_data.get("time_elapsed", 0.0)
        avg_fps = prog_data.get("avg_fps", 0.0)
        instant_fps = prog_data.get("instant_fps", 0.0)
        eta = prog_data.get("eta", 0.0)

        self.stage1_time_elapsed_str = f"{int(t_el // 3600):02d}:{int((t_el % 3600) // 60):02d}:{int(t_el % 60):02d}"
        self.stage1_processing_fps_str = f"{int(avg_fps)} FPS"
        self.stage1_instant_fps_str = f"{int(instant_fps)} FPS"

        if math.isnan(eta) or math.isinf(eta):
            self.stage1_eta_str = "Calculating..."
        elif eta > 0:
            self.stage1_eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
        else:
            self.stage1_eta_str = "Done"

        timing = prog_data.get("timing")
        if timing:
            self.stage1_decode_ms = timing.get('decode_ms', 0.0)
            self.stage1_unwarp_ms = timing.get('unwarp_ms', 0.0)
            self.stage1_yolo_det_ms = timing.get('yolo_det_ms', 0.0)
            self.stage1_yolo_pose_ms = timing.get('yolo_pose_ms', 0.0)

    def _handle_stage2_dual_progress(self, main_step_info, sub_step_info):
        if isinstance(main_step_info, tuple) and len(main_step_info) == 3:
            main_current, total_main, main_name = main_step_info
            self.stage2_main_progress_value = float(main_current) / total_main if total_main > 0 else 0.0
            self.stage2_main_progress_label = f"{main_name} ({int(main_current)}/{int(total_main)})"

        if isinstance(sub_step_info, dict):
            sub_current = sub_step_info.get("current", 0)
            sub_total = sub_step_info.get("total", 0)
            self.stage2_sub_progress_value = float(sub_current) / sub_total if sub_total > 0 else 0.0
            self.stage2_sub_progress_label = f"{sub_step_info.get('message', '')} ({sub_current}/{sub_total})"

            t_el = sub_step_info.get("time_elapsed", 0.0)
            fps = sub_step_info.get("fps", 0.0)
            eta = sub_step_info.get("eta", 0.0)

            self.stage2_sub_time_elapsed_str = f"{int(t_el // 3600):02d}:{int((t_el % 3600) // 60):02d}:{int(t_el % 60):02d}"
            self.stage2_sub_processing_fps_str = f"{int(fps)} FPS"
            if math.isnan(eta) or math.isinf(eta) or eta <= 0:
                self.stage2_sub_eta_str = "N/A"
            else:
                self.stage2_sub_eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
        elif isinstance(sub_step_info, tuple) and len(sub_step_info) == 3:
            sub_current, sub_total, sub_name = sub_step_info
            self.stage2_sub_progress_value = float(sub_current) / sub_total if sub_total > 0 else 0.0
            self.stage2_sub_progress_label = f"{sub_name} ({int(sub_current)}/{int(sub_total)})"
            self.stage2_sub_time_elapsed_str = ""
            self.stage2_sub_processing_fps_str = ""
            self.stage2_sub_eta_str = ""

    def _handle_stage2_results(self, packaged_data, s2_overlay_path_written, fs_proc):
        results_dict = packaged_data.get("results_dict", {})
        funscript_obj = results_dict.get("funscript")

        if funscript_obj:
            primary_actions = funscript_obj.primary_actions
            secondary_actions = funscript_obj.secondary_actions

            overwrite_chapters = self.app.app_settings.get("overwrite_chapters_on_analysis", False)

            if hasattr(funscript_obj, 'chapters') and funscript_obj.chapters:
                self._apply_chapters_from_funscript(funscript_obj, fs_proc, overwrite_chapters, "Stage 2")
        else:
            primary_actions = []
            secondary_actions = []
            self.app.logger.warning("No funscript object available from Stage 2 - using empty action lists")

        self._apply_actions_to_timelines(primary_actions, secondary_actions, fs_proc, "Stage 2")

        self.stage2_status_text = "S2 Completed. Results Processed."
        self.app.project_manager.project_dirty = True
        self.logger.info("Processed Stage 2 results.")

    def _handle_stage2_segments_only(self, video_segments_data, fs_proc):
        if self.force_rerun_stage2_segmentation:
            self.logger.info("Overwriting chapters with new segmentation results as requested.")
            fs_proc.video_chapters.clear()
            if isinstance(video_segments_data, list):
                for seg_data in video_segments_data:
                    if isinstance(seg_data, dict):
                        fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
        else:
            self.logger.info("Preserving existing chapters. S2 segmentation was not re-run.")

        self.stage2_status_text = "S2 Segmentation Processed."
        self.app.project_manager.project_dirty = True

    def _handle_stage3_progress(self, prog_data):
        if not isinstance(prog_data, dict):
            return

        chap_idx = prog_data.get('current_chapter_idx', 0)
        total_chaps = prog_data.get('total_chapters', 0)
        chap_name = prog_data.get('chapter_name', '')
        chunk_idx = prog_data.get('current_chunk_idx', 0)
        total_chunks = prog_data.get('total_chunks', 0)

        self.stage3_current_segment_label = f"Chapter: {chap_idx}/{total_chaps} ({chap_name})"
        self.stage3_overall_progress_label = f"Overall Task: Chunk {chunk_idx}/{total_chunks}"

        self.stage3_segment_progress_value = prog_data.get('segment_progress', 0.0)
        self.stage3_overall_progress_value = prog_data.get('overall_progress', 0.0)
        processed_overall = prog_data.get('total_frames_processed_overall', 0)
        to_process_overall = prog_data.get('total_frames_to_process_overall', 0)
        if to_process_overall > 0:
            self.stage3_overall_progress_label = f"Overall S3: {processed_overall}/{to_process_overall}"
        else:
            self.stage3_overall_progress_label = f"Overall S3: {self.stage3_overall_progress_value * 100:.0f}%"
        self.stage3_status_text = "Running Stage 3 (Optical Flow)..."

        t_el = prog_data.get("time_elapsed", 0.0)
        fps = prog_data.get("fps", 0.0)
        eta = prog_data.get("eta", 0.0)

        self.stage3_time_elapsed_str = (
            f"{int(t_el // 3600):02d}:{int((t_el % 3600) // 60):02d}:{int(t_el % 60):02d}"
            if not math.isnan(t_el) else "Calculating..."
        )
        self.stage3_processing_fps_str = f"{fps:.1f} FPS" if not math.isnan(fps) else "N/A FPS"

        is_done = (chunk_idx >= total_chunks and total_chunks > 0)
        if math.isnan(eta) or math.isinf(eta):
            self.stage3_eta_str = "Calculating..."
        elif eta > 1.0 and not is_done:
            self.stage3_eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
        else:
            self.stage3_eta_str = "Done"

    def _handle_stage3_results(self, packaged_data, fs_proc):
        if not isinstance(packaged_data, dict):
            self.logger.warning(f"stage3_results_success received non-dict data: {type(packaged_data)}")
            return
        results_dict = packaged_data.get("results_dict", {})
        if not isinstance(results_dict, dict):
            self.logger.error(f"Stage 3 results_dict is not a dictionary: {type(results_dict)} = {results_dict}")
            return

        funscript_obj = results_dict.get("funscript")
        if not funscript_obj:
            self.logger.warning("Stage 3 results missing funscript object - no actions applied")
            return

        self.logger.info("Processing Stage 3 results with funscript object")
        primary_actions = funscript_obj.primary_actions
        secondary_actions = funscript_obj.secondary_actions

        overwrite_chapters = self.app.app_settings.get("overwrite_chapters_on_analysis", False)
        if hasattr(funscript_obj, 'chapters') and funscript_obj.chapters:
            self._apply_chapters_from_funscript(funscript_obj, fs_proc, overwrite_chapters, "Stage 3")

        self._apply_actions_to_timelines(primary_actions, secondary_actions, fs_proc, "Stage 3")

        self.stage3_status_text = "S3 Completed. Results Processed."
        self.app.project_manager.project_dirty = True
        self.logger.info(f"Applied {len(primary_actions)} Stage 3 actions to funscript processor")

    def _handle_analysis_message(self, data1, data2):
        payload = data1 if isinstance(data1, dict) else {}
        status_override = payload.get("status", data2)
        log_msg = payload.get("message", str(data1))

        if status_override == "Completed":
            if log_msg:
                self.logger.info(log_msg, extra={'status_message': True})
            self.app.on_offline_analysis_completed(payload)

        elif status_override == "Aborted":
            if self.current_analysis_stage == 1 or self.stage1_status_text.startswith("Running"):
                self.stage1_status_text = "S1 Aborted."
            if self.current_analysis_stage == 2 or self.stage2_status_text.startswith("Running"):
                self.stage2_status_text = "S2 Aborted."
            if self.current_analysis_stage == 3 or self.stage3_status_text.startswith("Running"):
                self.stage3_status_text = "S3 Aborted."
            if self.app.is_batch_processing_active and hasattr(self.app, 'save_and_reset_complete_event'):
                self.logger.debug(f"Signaling batch loop to continue after handling '{status_override}' status.")
                self.app.save_and_reset_complete_event.set()

        elif status_override == "Failed":
            if self.current_analysis_stage == 1 or self.stage1_status_text.startswith("Running"):
                self.stage1_status_text = "S1 Failed."
            if self.current_analysis_stage == 2 or self.stage2_status_text.startswith("Running"):
                self.stage2_status_text = "S2 Failed."
            if self.current_analysis_stage == 3 or self.stage3_status_text.startswith("Running"):
                self.stage3_status_text = "S3 Failed."
            if self.app.is_batch_processing_active and hasattr(self.app, 'save_and_reset_complete_event'):
                self.logger.debug(f"Signaling batch loop to continue after handling '{status_override}' status.")
                self.app.save_and_reset_complete_event.set()

    # ------------------------------------------------------------------
    # Shared helpers for results handling
    # ------------------------------------------------------------------

    def _apply_chapters_from_funscript(self, funscript_obj, fs_proc, overwrite_chapters, stage_label):
        """Apply chapter data from a funscript object to the chapter list."""
        fps = (self.app.processor.video_info.get('fps', 30.0)
               if self.app.processor and self.app.processor.video_info else 30.0)

        if overwrite_chapters or len(fs_proc.video_chapters) == 0:
            if overwrite_chapters:
                self.logger.info(f"Overwriting chapters with {stage_label} analysis results (user setting enabled).")
            else:
                self.logger.info(f"Creating initial chapters from {stage_label} analysis results (no existing chapters).")

            fs_proc.video_chapters.clear()
            for chapter in funscript_obj.chapters:
                start_frame = int((chapter.get('start', 0) / 1000.0) * fps)
                end_frame = int((chapter.get('end', 0) / 1000.0) * fps)
                video_segment = VideoSegment(
                    start_frame_id=start_frame,
                    end_frame_id=end_frame,
                    class_id=None,
                    class_name=chapter.get('name', 'Unknown'),
                    segment_type="SexAct",
                    position_short_name=chapter.get('name', ''),
                    position_long_name=chapter.get('description', chapter.get('name', 'Unknown')),
                    source=f"{stage_label.lower().replace(' ', '')}_funscript"
                )
                fs_proc.video_chapters.append(video_segment)
            self.logger.info(f"Applied {len(funscript_obj.chapters)} chapters from {stage_label} funscript")
        else:
            self.logger.info(
                f"Preserving existing {len(fs_proc.video_chapters)} chapters "
                f"({stage_label} analysis results not applied to chapters)."
            )

    def _apply_actions_to_timelines(self, primary_actions, secondary_actions, fs_proc, stage_label):
        """Route primary/secondary actions to the correct timelines based on axis mode."""
        axis_mode = self.app.tracking_axis_mode
        target_timeline = self.app.single_axis_output_target

        self.app.logger.info(f"Applying {stage_label} results with axis mode: {axis_mode} and target: {target_timeline}")

        if axis_mode == "both":
            if primary_actions:
                fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, f"{stage_label} (Primary)")
                self.app.logger.info(f"Applied {len(primary_actions)} primary actions to Timeline 1")
            else:
                self.app.logger.warning(f"No primary actions from {stage_label} - Timeline 1 unchanged")

            if secondary_actions:
                fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, f"{stage_label} (Secondary)")
                self.app.logger.info(f"Applied {len(secondary_actions)} secondary actions to Timeline 2")
            else:
                self.app.logger.info(f"No secondary actions from {stage_label} - Timeline 2 unchanged")

        elif axis_mode == "vertical":
            if primary_actions:
                tl = 1 if target_timeline == "primary" else 2
                self.app.logger.info(f"Writing to Timeline {tl}, other timeline untouched.")
                fs_proc.clear_timeline_history_and_set_new_baseline(tl, primary_actions, f"{stage_label} (Vertical)")
            else:
                tl = 1 if target_timeline == "primary" else 2
                self.app.logger.warning(f"No vertical actions from {stage_label} - Timeline {tl} unchanged")

        elif axis_mode == "horizontal":
            if secondary_actions:
                tl = 1 if target_timeline == "primary" else 2
                self.app.logger.info(f"Writing horizontal data to Timeline {tl}, other timeline untouched.")
                fs_proc.clear_timeline_history_and_set_new_baseline(tl, secondary_actions, f"{stage_label} (Horizontal)")
            else:
                tl = 1 if target_timeline == "primary" else 2
                self.app.logger.warning(f"No horizontal actions from {stage_label} - Timeline {tl} unchanged")

        elif axis_mode in ["vertical", "horizontal"] and stage_label.startswith("Stage 3"):
            # Stage 3 fallback: write primary actions to target
            actions_to_use = primary_actions
            if actions_to_use:
                tl = 1 if target_timeline == "primary" else 2
                fs_proc.clear_timeline_history_and_set_new_baseline(tl, actions_to_use, stage_label)
                self.logger.info(f"Applied {len(actions_to_use)} actions to Timeline {tl}")
            else:
                tl = 1 if target_timeline == "primary" else 2
                self.logger.warning(f"No actions from {stage_label} - Timeline {tl} unchanged")
