class AppCalibration:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger

        self.is_calibration_mode_active = False
        self.calibration_timeline_point_ms = 0.0
        self.calibration_reference_point_selected = False
        self.funscript_output_delay_frames = 0

    def start_latency_calibration(self):
        if not self.app.file_manager.video_path:
            self.logger.info("Please load a video first for calibration.", extra={'status_message': True})
            return
        funscript_processor = self.app.funscript_processor
        if not (self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript and \
                funscript_processor.get_actions('primary')):
            self.logger.info("Timeline 1 needs some points. Process some video or load a script.",
                             extra={'status_message': True})
            return

        self.is_calibration_mode_active = True
        self.calibration_reference_point_selected = False
        self.logger.info("Latency calibration started. Select a distinct point on Timeline 1.",
                         extra={'status_message': True, 'duration': 7.0})
        if self.app.processor and self.app.processor.is_processing:
            self.app.processor.pause_processing()
        self.app.energy_saver.reset_activity_timer()

    def handle_calibration_point_selection(self, selected_action_time_ms: float):
        """
        Called by the timeline. This method now ONLY handles the calibration state.
        All UI manipulation is handled by the timeline itself.
        """
        if not self.is_calibration_mode_active:
            return
        self.calibration_timeline_point_ms = selected_action_time_ms
        self.calibration_reference_point_selected = True
        self.logger.info(
            f"Reference point at {selected_action_time_ms:.0f}ms selected. Now navigate video to the visual match and confirm.",
            extra={'status_message': True, 'duration': 10.0})
        self.app.energy_saver.reset_activity_timer()

    def confirm_latency_calibration(self):
        if not (self.is_calibration_mode_active and self.calibration_reference_point_selected and \
                self.app.processor and self.app.processor.video_info and self.app.processor.fps > 0):
            self.logger.warning("Calibration not active, reference point not selected, or video/FPS info missing.",
                                extra={'status_message': True})
            return

        current_video_frame_index = self.app.processor.current_frame_index
        current_video_time_ms = (current_video_frame_index / self.app.processor.fps) * 1000.0

        delay_ms = self.calibration_timeline_point_ms - current_video_time_ms
        delay_frames = round((delay_ms / 1000.0) * self.app.processor.fps)

        clamped_delay_frames = max(0, min(delay_frames, 20))
        self.funscript_output_delay_frames = int(clamped_delay_frames)

        self.app.app_settings.set("funscript_output_delay_frames", self.funscript_output_delay_frames)
        self.update_tracker_delay_params()

        self.logger.info(
            f"Latency calibrated. Delay set to: {self.funscript_output_delay_frames} frames ({delay_ms:.0f} ms). Setting saved.",
            extra={'status_message': True, 'duration': 7.0})

        self.is_calibration_mode_active = False
        self.calibration_reference_point_selected = False
        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def update_tracker_delay_params(self):
        if self.app.tracker:
            self.app.tracker.output_delay_frames = self.funscript_output_delay_frames
            current_fps_for_delay = 30.0

            if self.app.processor:
                if self.app.processor.fps > 0:
                    current_fps_for_delay = self.app.processor.fps
                elif self.app.processor.video_info and self.app.processor.video_info.get('fps', 0) > 0:
                    current_fps_for_delay = self.app.processor.video_info['fps']

            self.app.tracker.current_video_fps_for_delay = current_fps_for_delay
            self.logger.debug(
                f"Tracker delay params updated: delay_frames={self.app.tracker.output_delay_frames}, fps_for_delay={self.app.tracker.current_video_fps_for_delay}")

    def update_settings_from_app(self):
        defaults = self.app.app_settings.get_default_settings()
        loaded_delay = self.app.app_settings.get(
            "funscript_output_delay_frames",
            defaults.get("funscript_output_delay_frames", 0)
        )
        self.funscript_output_delay_frames = max(0, min(loaded_delay, 20))
        self.update_tracker_delay_params()

    def save_settings_to_app(self):
        self.app.app_settings.set("funscript_output_delay_frames", self.funscript_output_delay_frames)
