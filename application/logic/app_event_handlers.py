from typing import Tuple


from application.utils import VideoSegment, _format_time
from config.constants import DEFAULT_CHAPTER_FPS


class AppEventHandlers:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger

    def handle_playback_control(self, action_name: str):
        self.logger.debug(f"Playback control: {action_name}")
        processor = self.app.processor
        if not self.app.file_manager.video_path or not processor or not processor.video_info:
            self.logger.info("No video loaded for playback control.", extra={'status_message': True})
            return

        total_frames = processor.video_info.get('total_frames', 0)
        current_frame = processor.current_frame_index
        fs_proc = self.app.funscript_processor
        app_state_ui = self.app.app_state_ui

        if action_name == "jump_start":
            processor.seek_video(0)
            return
        if action_name == "jump_end":
            processor.seek_video(total_frames - 1)
            return
        if action_name == "prev_frame":
            processor.seek_video(max(0, current_frame - 1))
            return
        if action_name == "next_frame":
            processor.seek_video(min(total_frames - 1, current_frame + 1))
            return
        if action_name == "play_pause":
            is_currently_playing = processor.is_processing and not processor.pause_event.is_set()
            if is_currently_playing:
                processor.pause_processing()
            else:
                # Only start regular video playback, never restart tracking sessions
                # Tracking sessions should only be started via the control panel
                processor.start_processing()
            return
        if action_name == "stop":
            processor.stop_processing()
            return

        # After any action other than starting playback, update the displayed frame.
        is_resuming_or_starting = action_name == "play_pause" and (
                    processor.is_processing and not processor.pause_event.is_set())
        if not is_resuming_or_starting:
            processor.display_current_frame()

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

        elif self.app.processor and self.app.processor.is_processing:
            self.app.processor.stop_processing()
        elif self.app.is_setting_user_roi_mode:  # Abort ROI selection
            self.app.exit_set_user_roi_mode()
            self.logger.info("User ROI selection aborted.", extra={'status_message': True})
        else:
            self.logger.info("No process running to abort.", extra={'status_message': False})
        self.app.energy_saver.reset_activity_timer()

    def handle_start_ai_cv_analysis(self):  # New specific handler for AI CV
        if not self.app._check_model_paths():
            return
        if not self.app.tracker: self.logger.error("Tracker not initialized."); return
        
        self.app.tracker.set_tracking_mode("YOLO_ROI")  # Ensure correct mode
        self.app.stage_processor.start_full_analysis(processing_mode=self.app.app_state_ui.selected_tracker_name)
        self.app.energy_saver.reset_activity_timer()

    def handle_start_live_tracker_click(self):
        if not self.app._check_model_paths():
            return
        if not self.app.processor or not self.app.file_manager.video_path:
            self.logger.info("No video loaded for live tracking.", extra={'status_message': True})
            return
        if not self.app.tracker:
            self.logger.error("Tracker not initialized for live tracking.")
            return

        selected_tracker_name = self.app.app_state_ui.selected_tracker_name
        
        # Set tracker using dynamic discovery
        if selected_tracker_name:
            self.app.tracker.set_tracking_mode(selected_tracker_name)

        # Use dynamic tracker discovery for logging and validation
        from application.gui_components.dynamic_tracker_ui import get_dynamic_tracker_ui
        tracker_ui = get_dynamic_tracker_ui()
        
        # Check if selected tracker is valid for live tracking
        if not selected_tracker_name or not tracker_ui.is_live_tracker(selected_tracker_name):
            self.logger.error(f"Invalid live tracker selected: {selected_tracker_name}")
            return

        # Handle user ROI tracker special case
        if tracker_ui.is_user_roi_tracker(selected_tracker_name):
            # Check for a global ROI OR a chapter-specific ROI at the current frame
            has_global_roi = bool(
                self.app.tracker.user_roi_fixed and (
                    self.app.tracker.user_roi_initial_point_relative or self.app.tracker.user_roi_tracked_point_relative
                )
            )
            
            # Debug logging for ROI validation
            self.logger.info(f"🔍 User ROI validation: user_roi_fixed={self.app.tracker.user_roi_fixed}, "
                           f"initial_point={self.app.tracker.user_roi_initial_point_relative}, "
                           f"tracked_point={self.app.tracker.user_roi_tracked_point_relative}, "
                           f"has_global_roi={has_global_roi}")

            has_chapter_roi_at_current_frame = False
            if not has_global_roi:
                if self.app.processor and self.app.funscript_processor:
                    current_frame = self.app.processor.current_frame_index
                    chapter_at_cursor = self.app.funscript_processor.get_chapter_at_frame(current_frame)
                    if chapter_at_cursor and chapter_at_cursor.user_roi_fixed and chapter_at_cursor.user_roi_initial_point_relative:
                        has_chapter_roi_at_current_frame = True

            if not has_global_roi and not has_chapter_roi_at_current_frame:
                self.logger.info("User Defined ROI: Please set a global ROI or a chapter-specific ROI for the current position first.", extra={'status_message': True, 'duration': 5.0})
                return

            if has_global_roi and self.app.tracker.user_roi_fixed and \
               not self.app.tracker.user_roi_initial_point_relative and \
               self.app.tracker.user_roi_tracked_point_relative:
                self.app.tracker.user_roi_initial_point_relative = self.app.tracker.user_roi_tracked_point_relative
            
            display_name = tracker_ui.get_tracker_display_name(selected_tracker_name)
            self.logger.info(f"Starting {display_name} tracking.")
        else:
            # For all other live trackers
            display_name = tracker_ui.get_tracker_display_name(selected_tracker_name)
            self.logger.info(f"Starting {display_name} tracking.")

        # Auto-set axis for axis projection trackers if not already set
        if "axis_projection" in selected_tracker_name:
            current_tracker = self.app.tracker.get_current_tracker()
            if current_tracker and hasattr(current_tracker, 'set_axis'):
                # Check if axis is already set
                axis_already_set = False
                if hasattr(current_tracker, 'axis_point_A') and hasattr(current_tracker, 'axis_point_B'):
                    axis_already_set = (current_tracker.axis_point_A is not None and 
                                      current_tracker.axis_point_B is not None)
                
                if not axis_already_set:
                    # Set default horizontal axis across middle of frame
                    margin = 50
                    width, height = 640, 640  # Processing frame size
                    axis_A = (margin, height // 2)  # Left side
                    axis_B = (width - margin, height // 2)  # Right side
                    result = current_tracker.set_axis(axis_A, axis_B)
                    self.logger.info(f"Auto-set axis for {selected_tracker_name}: A={axis_A}, B={axis_B}, result={result}")

        # Explicitly start the tracker before starting video processing
        self.app.tracker.start_tracking()
        self.app.processor.set_tracker_processing_enabled(True)

        fs_proc = self.app.funscript_processor
        start_frame = self.app.processor.current_frame_index
        end_frame = -1
        if fs_proc.scripting_range_active:
            start_frame = fs_proc.scripting_start_frame
            end_frame = fs_proc.scripting_end_frame
            self.app.processor.seek_video(start_frame)

        # Check if MAX_SPEED mode is selected - if so, use CLI pipeline for maximum performance
        from config.constants import ProcessingSpeedMode
        is_max_speed = (hasattr(self.app, 'app_state_ui') and 
                       hasattr(self.app.app_state_ui, 'selected_processing_speed_mode') and
                       self.app.app_state_ui.selected_processing_speed_mode == ProcessingSpeedMode.MAX_SPEED)
        
        # Normal processing for all modes - let the video processor handle optimizations
        self.app.processor.start_processing(start_frame=start_frame, end_frame=end_frame)
            
        display_name = tracker_ui.get_tracker_display_name(selected_tracker_name)
        self.logger.info(
            f"Live tracker ({display_name}) started. Range: {'scripting range' if fs_proc.scripting_range_active else 'full video from current'}",
            extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def handle_reset_live_tracker_click(self):
        if self.app.processor:
            # Stop processing and reset tracker state, but preserve the funscript data
            self.app.processor.stop_processing(join_thread=True)
            
            # Reset tracker state but preserve funscript
            if self.app.tracker:
                self.app.tracker.reset(reason="stop_preserve_funscript")
                
            # Reset processor frame position to current for potential restart
            # But don't seek to beginning since user might want to continue from current position
            self.app.processor.enable_tracker_processing = False
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
            if fs_proc.scripting_range_active:
                fs_proc.scripting_start_frame = segment.start_frame_id
                fs_proc.scripting_end_frame = segment.end_frame_id
                fs_proc.selected_chapter_for_scripting = segment

                start_t_str = _format_time(fs_proc.app, segment.start_frame_id / current_fps if current_fps > 0 else 0)
                end_t_str = _format_time(fs_proc.app, segment.end_frame_id / current_fps if current_fps > 0 else 0)
                self.logger.info(
                    f"Scripting range updated to chapter: {segment.position_long_name} [{start_t_str} - {end_t_str}]",
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

    def handle_interactive_refinement_click(self, chapter: VideoSegment, track_id: int):
        """
        This method is called by the UI. It saves the user's choice to the chapter
        object itself, making the highlight persistent.
        """
        if self.app.stage_processor:
            # Set the persistent attribute on the chapter
            chapter.refined_track_id = track_id
            self.app.project_manager.project_dirty = True

            # Start the backend analysis to update the funscript
            self.app.stage_processor.start_interactive_refinement_analysis(chapter, track_id)

