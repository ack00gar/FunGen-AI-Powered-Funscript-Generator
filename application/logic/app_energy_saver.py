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

        # Access stage_processor via self.app
        if self.app.stage_processor.full_analysis_active or \
           (self.app.processor and self.app.processor.is_processing) or \
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
        defaults = self.app_settings.get_default_settings()
        self.energy_saver_enabled = self.app_settings.get(
            "energy_saver_enabled",
            defaults.get("energy_saver_enabled", True)
        )
        self.energy_saver_threshold_seconds = self.app_settings.get(
            "energy_saver_threshold_seconds",
            defaults.get("energy_saver_threshold_seconds", 60.0)
        )
        self.energy_saver_fps = self.app_settings.get(
            "energy_saver_fps",
            defaults.get("energy_saver_fps", 1)
        )
        self.main_loop_normal_fps_target = self.app_settings.get(
            "main_loop_normal_fps_target",
            defaults.get("main_loop_normal_fps_target", 60)
        )
        # If energy_saver_enabled is now false, ensure energy_saver_active is also false
        if not self.energy_saver_enabled and self.energy_saver_active:
            self.energy_saver_active = False
            self.logger.info("Energy saver mode disabled by settings change, deactivating.", extra={'status_message': True})


    def save_settings_to_app(self):
        """Called by AppLogic when app settings are to be saved."""
        self.app_settings.set("energy_saver_enabled", self.energy_saver_enabled)
        self.app_settings.set("energy_saver_threshold_seconds", self.energy_saver_threshold_seconds)
        self.app_settings.set("energy_saver_fps", self.energy_saver_fps)
        self.app_settings.set("main_loop_normal_fps_target", self.main_loop_normal_fps_target)
