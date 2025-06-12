from typing import Tuple


from application.utils.video_segment import VideoSegment
from application.utils.time_format import _format_time
from config.constants import DEFAULT_CHAPTER_FPS


class AppEventHandlers:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger

    def handle_playback_control(self, action_name: str):
        self.logger.debug(f"Playback control: {action_name}")
        if not self.app.file_manager.video_path or not self.app.processor or not self.app.processor.video_info:
            self.logger.info("No video loaded for playback control.", extra={'status_message': True})
            return

        total_frames = self.app.processor.video_info.get('total_frames', 0)
        current_frame = self.app.processor.current_frame_index
        fs_proc = self.app.funscript_processor
        app_state_ui = self.app.app_state_ui

        if action_name == "jump_start":
            self.app.processor.seek_video(0)
        elif action_name == "prev_frame":
            self.app.processor.seek_video(max(0, current_frame - 1))
        elif action_name == "play_pause":
            self.app.processor.set_tracker_processing_enabled(False)
            if self.app.processor.is_processing:
                self.app.processor.pause_processing()
                self.logger.info("Video paused.", extra={'status_message': True})
            else:
                start_f = fs_proc.scripting_start_frame if fs_proc.scripting_range_active else current_frame
                # If scripting_end_frame is -1, VideoProcessor's start_processing handles "to end"
                end_f = fs_proc.scripting_end_frame if fs_proc.scripting_range_active else -1
                self.app.processor.start_processing(start_frame=start_f, end_frame=end_f)
                self.logger.info("Video playing.", extra={'status_message': True})
        elif action_name == "stop":
            self.app.processor.reset()
            target_seek_frame = fs_proc.scripting_start_frame if fs_proc.scripting_range_active else 0
            self.app.processor.seek_video(target_seek_frame)
            self.logger.info("Video stopped.", extra={'status_message': True})
        elif action_name == "next_frame":
            self.app.processor.seek_video(min(total_frames - 1 if total_frames > 0 else 0, current_frame + 1))
        elif action_name == "jump_end":
            # Jump to end of scripting range if active, else video end
            target_end_frame = total_frames - 1 if total_frames > 0 else 0
            if fs_proc.scripting_range_active and fs_proc.scripting_end_frame != -1:
                target_end_frame = fs_proc.scripting_end_frame
            self.app.processor.seek_video(target_end_frame)

        # If not playing, ensure the current frame is displayed
        if action_name != "play_pause" or (action_name == "play_pause" and not self.app.processor.is_processing):
            if self.app.processor.is_processing:  # Should be paused by now if play_pause was pressed
                self.app.processor.pause_processing()
            self.app.processor.display_current_frame()  # Explicitly update display after seek/stop

        # Force timeline pan for seeking actions (except during active play/pause toggle)
        if action_name in ["jump_start", "prev_frame", "stop", "next_frame", "jump_end"]:
            app_state_ui.force_timeline_pan_to_current_frame = True

        self.app.energy_saver.reset_activity_timer()

    def handle_jump_to_point(self, direction: str):
        if not self.app.processor or not self.app.processor.is_video_open():
            self.logger.info("Cannot jump: No video loaded.", extra={'status_message': True})
            return

        fs = self.app.processor.tracker.funscript
        if not fs:
            self.logger.info("Cannot jump: Funscript object not available.", extra={'status_message': True})
            return

        current_frame = self.app.processor.current_frame_index
        fps = self.app.processor.fps

        target_frame = None
        if direction == 'next':
            target_frame = fs.find_next_jump_frame(current_frame, fps, 'primary')
        elif direction == 'prev':
            target_frame = fs.find_prev_jump_frame(current_frame, fps, 'primary')

        if target_frame is not None:
            total_frames = self.app.processor.total_frames
            if total_frames > 0:
                target_frame = min(target_frame, total_frames - 1)

            self.app.processor.seek_video(target_frame)
            self.app.app_state_ui.force_timeline_pan_to_current_frame = True
            self.app.energy_saver.reset_activity_timer()
        else:
            self.logger.info(f"No {direction} point found to jump to.", extra={'status_message': True})

    def handle_abort_process_click(self):
        stage_processor = self.app.stage_processor
        if stage_processor.full_analysis_active:
            stage_processor.abort_stage_processing()
            self.app.on_processing_stopped() # If aborting stage proc should also check pending app logic actions
        elif stage_processor.scene_detection_active:
            stage_processor.abort_stage_processing()
        elif self.app.processor and self.app.processor.is_processing:
            self.app.processor.stop_processing()
        elif self.app.is_setting_user_roi_mode:  # Abort ROI selection
            self.app.exit_set_user_roi_mode()
            self.logger.info("User ROI selection aborted.", extra={'status_message': True})
        else:
            self.logger.info("No process running to abort.", extra={'status_message': False})
        self.app.energy_saver.reset_activity_timer()

    def handle_start_ai_cv_analysis(self):  # New specific handler for AI CV
        if not self.app.tracker: self.logger.error("Tracker not initialized."); return
        self.app.tracker.set_tracking_mode("YOLO_ROI")  # Ensure correct mode
        self.app.stage_processor.start_full_analysis()
        self.app.energy_saver.reset_activity_timer()

    def handle_start_live_tracker_click(self):
        if not self.app.processor or not self.app.file_manager.video_path:
            self.logger.info("No video loaded for live tracking.", extra={'status_message': False})
            return
        if not self.app.tracker:
            self.logger.error("Tracker not initialized for live tracking.")
            return

        current_tracker_mode = self.app.tracker.tracking_mode

        if current_tracker_mode == "USER_FIXED_ROI":
            # FIX: Check for a global ROI OR a chapter-specific ROI at the current frame
            has_global_roi = bool(self.app.tracker.user_roi_fixed and self.app.tracker.user_roi_initial_point_relative)

            has_chapter_roi_at_current_frame = False
            if not has_global_roi:
                if self.app.processor and self.app.funscript_processor:
                    current_frame = self.app.processor.current_frame_index
                    chapter_at_cursor = self.app.funscript_processor.get_chapter_at_frame(current_frame)
                    if chapter_at_cursor and chapter_at_cursor.user_roi_fixed and chapter_at_cursor.user_roi_initial_point_relative:
                        has_chapter_roi_at_current_frame = True

            if not has_global_roi and not has_chapter_roi_at_current_frame:
                self.logger.info("User Defined ROI: Please set a global ROI or a chapter-specific ROI for the current position first.",
                                 extra={'status_message': True, 'duration': 5.0})
                return
            self.logger.info("Starting User Defined ROI tracking.")
        elif current_tracker_mode == "YOLO_ROI":
            self.logger.info("Starting Live Tracker (YOLO_ROI mode - if applicable).")
        else:
            self.logger.error(f"Unknown tracker mode for live start: {current_tracker_mode}");
            return

        self.app.processor.set_tracker_processing_enabled(True)
        fs_proc = self.app.funscript_processor
        start_frame = self.app.processor.current_frame_index
        end_frame = -1
        if fs_proc.scripting_range_active:
            start_frame = fs_proc.scripting_start_frame
            end_frame = fs_proc.scripting_end_frame
            self.app.processor.seek_video(start_frame)

        self.app.processor.start_processing(start_frame=start_frame, end_frame=end_frame)
        self.logger.info(
            f"Live tracker ({current_tracker_mode}) started. Range: {'scripting range' if fs_proc.scripting_range_active else 'full video from current'}",
            extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def handle_reset_live_tracker_click(self):
        if self.app.processor:
            self.app.processor.reset()  # Stops processing, resets tracker state, seeks
            # Reset specific tracker states
            if self.app.tracker:
                if self.app.tracker.tracking_mode == "USER_FIXED_ROI":
                    self.app.tracker.clear_user_defined_roi_and_point()
                    # Optionally switch back to default mode, or let UI combo handle it
                    # self.app.tracker.set_tracking_mode("YOLO_ROI")
                # For YOLO_ROI mode, tracker.reset() already handles its specific state.
            self.app.processor.set_tracker_processing_enabled(True)
        self.logger.info("Live Tracker reset.", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()


    def handle_scripting_range_active_toggle(self, new_active_state: bool):
        fs_proc = self.app.funscript_processor
        fs_proc.scripting_range_active = new_active_state
        if not new_active_state:
            fs_proc.selected_chapter_for_scripting = None  # Clear chapter selection if range deactivated
        self.app.project_manager.project_dirty = True
        self.logger.info(f"Scripting range {'enabled' if new_active_state else 'disabled'}.",
                         extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def handle_scripting_start_frame_input(self, new_start_val: int):
        fs_proc = self.app.funscript_processor
        video_total_frames = self.app.processor.total_frames if self.app.processor and self.app.processor.total_frames else 0

        fs_proc.scripting_start_frame = new_start_val
        if video_total_frames > 0:
            fs_proc.scripting_start_frame = min(max(0, fs_proc.scripting_start_frame), video_total_frames - 1)
        else:
            fs_proc.scripting_start_frame = max(0, fs_proc.scripting_start_frame)

        # If end frame is set (not -1) and start goes past it, adjust end frame
        if fs_proc.scripting_end_frame != -1 and fs_proc.scripting_start_frame > fs_proc.scripting_end_frame:
            fs_proc.scripting_end_frame = fs_proc.scripting_start_frame
        fs_proc.selected_chapter_for_scripting = None
        self.app.project_manager.project_dirty = True
        self.logger.debug(f"Scripting start frame updated to: {fs_proc.scripting_start_frame}")
        self.app.energy_saver.reset_activity_timer()

    def handle_scripting_end_frame_input(self, new_end_val: int):
        fs_proc = self.app.funscript_processor
        video_total_frames = self.app.processor.total_frames if self.app.processor and self.app.processor.total_frames else 0

        fs_proc.scripting_end_frame = new_end_val
        if fs_proc.scripting_end_frame != -1:
            if video_total_frames > 0:
                fs_proc.scripting_end_frame = min(max(0, fs_proc.scripting_end_frame), video_total_frames - 1)
            else:
                fs_proc.scripting_end_frame = max(0, fs_proc.scripting_end_frame)
            if fs_proc.scripting_start_frame > fs_proc.scripting_end_frame:
                fs_proc.scripting_start_frame = fs_proc.scripting_end_frame
        fs_proc.selected_chapter_for_scripting = None
        self.app.project_manager.project_dirty = True
        self.logger.debug(f"Scripting end frame updated to: {fs_proc.scripting_end_frame}")
        self.app.energy_saver.reset_activity_timer()

    def clear_scripting_range_selection(self):
        fs_proc = self.app.funscript_processor
        fs_proc.reset_scripting_range()
        self.app.project_manager.project_dirty = True
        self.logger.info("Scripting range cleared.", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def set_selected_axis_for_processing(self, axis: str):
        fs_proc = self.app.funscript_processor
        if axis in ['primary', 'secondary']:
            if fs_proc.selected_axis_for_processing != axis:
                fs_proc.selected_axis_for_processing = axis
                fs_proc.current_selection_indices.clear()
                self.logger.info(f"Target axis for operations set to: {axis.capitalize()}",
                                 extra={'status_message': True})
                self.app.energy_saver.reset_activity_timer()
        else:
            self.logger.warning(f"Attempt to set invalid axis for processing: {axis}")

    def update_sg_window_length(self, new_val: int):
        fs_proc = self.app.funscript_processor
        current_val = max(3, new_val + 1 if new_val % 2 == 0 else new_val)
        fs_proc.sg_window_length_input = min(99, current_val)
        if fs_proc.sg_polyorder_input >= fs_proc.sg_window_length_input:
            fs_proc.sg_polyorder_input = max(1, fs_proc.sg_window_length_input - 1)
        self.app.energy_saver.reset_activity_timer()
        # No status message unless it's a significant change or from a settings panel

    def handle_seek_bar_drag(self, frame_index: int):
        if self.app.processor:
            self.app.processor.seek_video(frame_index)
            # If not actively processing (playing/tracking), force timeline to sync
            if not self.app.processor.is_processing:
                self.app.app_state_ui.force_timeline_pan_to_current_frame = True
            self.app.project_manager.project_dirty = True  # Seeking can be considered a change
            self.app.energy_saver.reset_activity_timer()

    def handle_chapter_bar_segment_click(self, segment: VideoSegment, is_currently_selected: bool):
        fs_proc = self.app.funscript_processor
        app_state_ui = self.app.app_state_ui

        # Determine current FPS for time display, default if not available
        current_fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else DEFAULT_CHAPTER_FPS
        if is_currently_selected:
            fs_proc.scripting_range_active = False
            fs_proc.selected_chapter_for_scripting = None
            self.logger.info(f"Chapter range deselected: {segment.position_long_name}", extra={'status_message': True})
        else:
            fs_proc.scripting_range_active = True
            fs_proc.scripting_start_frame = segment.start_frame_id
            fs_proc.scripting_end_frame = segment.end_frame_id
            fs_proc.selected_chapter_for_scripting = segment

            start_t_str = _format_time(fs_proc.app, segment.start_frame_id / current_fps if current_fps > 0 else 0)
            end_t_str = _format_time(fs_proc.app, segment.end_frame_id / current_fps if current_fps > 0 else 0)
            self.logger.info(
                f"Scripting range set to chapter: {segment.position_long_name} [{start_t_str} - {end_t_str}]",
                extra={'status_message': True})

            # Seek video to start of chapter if video is loaded
            if self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0:
                self.app.processor.seek_video(segment.start_frame_id)
                app_state_ui.force_timeline_pan_to_current_frame = True
        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def get_native_fps_info_for_button(self) -> Tuple[str, float]:
        """Returns display string and value for a 'Set to Native FPS' button."""
        if self.app.file_manager.video_path and self.app.processor and \
                self.app.processor.video_info and self.app.processor.video_info.get('fps', 0) > 0:
            native_fps = self.app.processor.video_info['fps']
            return f"({native_fps:.2f})", native_fps
        return "", 0.0