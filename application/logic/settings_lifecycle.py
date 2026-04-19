"""Load/save of the scattered AppLogic settings across sub-modules."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic


class SettingsLifecycleController:
    __slots__ = ("app",)

    def __init__(self, app: "ApplicationLogic") -> None:
        self.app = app

    def apply_loaded(self) -> None:
        """Propagate loaded settings into AppLogic attrs + every sub-module."""
        app = self.app
        app.logger.debug("Applying loaded settings...")

        cfg = app.app_settings.config
        app.discarded_tracking_classes = cfg.tracking.discarded_classes or []

        # Logging level
        new_logging_level = cfg.logging.level
        if app.logging_level_setting != new_logging_level:
            app.set_application_logging_level(new_logging_level)

        # Hardware acceleration — validate against the cached hwaccels list.
        default_hw = "auto"
        if "auto" not in app.available_ffmpeg_hwaccels:
            default_hw = "none" if "none" in app.available_ffmpeg_hwaccels else (
                app.available_ffmpeg_hwaccels[0] if app.available_ffmpeg_hwaccels else "none")
        loaded_hw = cfg.performance.hardware_acceleration_method or default_hw
        if loaded_hw not in app.available_ffmpeg_hwaccels:
            app.logger.warning(
                f"Hardware acceleration method '{loaded_hw}' from settings is not currently "
                f"available ({app.available_ffmpeg_hwaccels}). Resetting to '{default_hw}'.")
            app.hardware_acceleration_method = default_hw
        else:
            app.hardware_acceleration_method = loaded_hw

        # YOLO model paths — update tracker refs if they changed.
        app.yolo_detection_model_path_setting = cfg.models.yolo_det_path
        app.yolo_pose_model_path_setting = cfg.models.yolo_pose_path
        if app.yolo_det_model_path != app.yolo_detection_model_path_setting:
            app.yolo_det_model_path = app.yolo_detection_model_path_setting or ""
            if app.tracker:
                app.tracker.det_model_path = app.yolo_det_model_path
            app.logger.info(
                f"Detection model path updated from settings: {os.path.basename(app.yolo_det_model_path or '')}")
        if app.yolo_pose_model_path != app.yolo_pose_model_path_setting:
            app.yolo_pose_model_path = app.yolo_pose_model_path_setting or ""
            if app.tracker:
                app.tracker.pose_model_path = app.yolo_pose_model_path
            app.logger.info(
                f"Pose model path updated from settings: {os.path.basename(app.yolo_pose_model_path or '')}")

        # Fan out to sub-modules.
        app.app_state_ui.update_settings_from_app()
        app.file_manager.update_settings_from_app()
        app.stage_processor.update_settings_from_app()
        app.energy_saver.update_settings_from_app()
        app.energy_saver.reset_activity_timer()

    def save(self) -> None:
        """Push AppLogic attrs back into AppSettings + ask each sub-module to save its own."""
        app = self.app
        app.logger.debug("Saving application settings...")

        cfg = app.app_settings.config
        cfg.performance.hardware_acceleration_method = app.hardware_acceleration_method
        cfg.models.yolo_det_path = app.yolo_detection_model_path_setting or ""
        cfg.models.yolo_pose_path = app.yolo_pose_model_path_setting or ""
        cfg.tracking.discarded_classes = app.discarded_tracking_classes

        app.app_state_ui.save_settings_to_app()
        app.file_manager.save_settings_to_app()
        app.stage_processor.save_settings_to_app()
        app.energy_saver.save_settings_to_app()
        app.app_settings.save_settings()
        app.logger.info("Application settings saved.", extra={'status_message': True})
        app.energy_saver.reset_activity_timer()
