import time

class AppEnergySaver:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.app_settings = self.app.app_settings

        # Initialize attributes with defaults, will be updated by update_settings_from_app
        self.energy_saver_enabled = True
        self.last_activity_time = time.time()
        self.energy_saver_active = False
        self.energy_saver_threshold_seconds = 60.0
        self.energy_saver_fps = 1
        self.main_loop_normal_fps_target = 60


    def reset_activity_timer(self):
        self.last_activity_time = time.time()
        if self.energy_saver_active:
            self.energy_saver_active = False
            self.logger.info("Energy saver mode deactivated due to activity.", extra={'status_message': True})

    def check_and_update_energy_saver(self):
        if not self.energy_saver_enabled:
            if self.energy_saver_active:
                self.energy_saver_active = False
                self.logger.info("Energy saver mode globally disabled by setting, deactivating.", extra={'status_message': True})
            return

        # Access stage_processor via self.app. Only treat the processor
        # as activity when it's actually PLAYING (not just paused) --
        # otherwise is_processing stays True after pause and the energy
        # saver + idle-render throttle never kick in.
        _proc = self.app.processor
        _proc_playing = bool(
            _proc
            and getattr(_proc, 'is_processing', False)
            and not (getattr(_proc, 'pause_event', None)
                     and _proc.pause_event.is_set()))
        # External mpv review counts as activity so the main loop keeps
        # painting at 60 Hz while the user is on another monitor.
        _mpv_ctrl = getattr(self.app, '_mpv_controller', None)
        _mpv_active = bool(_mpv_ctrl is not None and _mpv_ctrl.is_active)
        if self.app.stage_processor.full_analysis_active or \
           _proc_playing or \
           _mpv_active or \
           (self.app.stage_processor.stage_thread and self.app.stage_processor.stage_thread.is_alive()):
            self.reset_activity_timer()
            return

        if self.app.shortcut_manager and self.app.shortcut_manager.is_recording_shortcut_for:
            self.reset_activity_timer()
            return

        if time.time() - self.last_activity_time > self.energy_saver_threshold_seconds:
            if not self.energy_saver_active:
                self.energy_saver_active = True
                self.logger.info(
                    f"Energy saver mode activated after {self.energy_saver_threshold_seconds:.0f}s of inactivity.",
                    extra={'status_message': True}
                )

    def update_settings_from_app(self):
        """Called by AppLogic when settings are loaded or project is loaded."""
        es = self.app_settings.config.energy_saver
        self.energy_saver_enabled = es.enabled
        self.energy_saver_threshold_seconds = es.threshold_seconds
        self.energy_saver_fps = es.fps
        self.main_loop_normal_fps_target = self.app_settings.config.performance.main_loop_normal_fps_target
        # If energy_saver_enabled is now false, ensure energy_saver_active is also false
        if not self.energy_saver_enabled and self.energy_saver_active:
            self.energy_saver_active = False
            self.logger.info("Energy saver mode disabled by settings change, deactivating.", extra={'status_message': True})


    def save_settings_to_app(self):
        """Called by AppLogic when app settings are to be saved."""
        cfg = self.app_settings.config
        cfg.energy_saver.enabled = self.energy_saver_enabled
        cfg.energy_saver.threshold_seconds = self.energy_saver_threshold_seconds
        cfg.energy_saver.fps = self.energy_saver_fps
        cfg.performance.main_loop_normal_fps_target = self.main_loop_normal_fps_target
