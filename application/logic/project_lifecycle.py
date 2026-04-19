"""Project lifecycle: last-project auto-load, reset-for-new-project."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic


class ProjectLifecycleController:
    __slots__ = ("app",)

    def __init__(self, app: "ApplicationLogic") -> None:
        self.app = app

    def load_last_on_startup(self) -> None:
        """Load the most recently used project on application start, if any."""
        app = self.app
        app.logger.debug("Checking for last opened project...")

        last_path = app.app_settings.config.project.last_opened_path
        if not last_path:
            app.logger.debug("No last project found to load. Starting fresh.")
            return

        if os.path.exists(last_path):
            try:
                app.logger.debug(f"Loading last opened project: {last_path}")
                app.project_manager.load_project(last_path)
            except Exception as e:
                app.logger.error(f"Failed to load last project '{last_path}': {e}", exc_info=True)
                app.app_settings.config.project.last_opened_path = None
        else:
            app.logger.warning(f"Last project file not found: '{last_path}'. Clearing setting.")
            app.app_settings.config.project.last_opened_path = None

    def reset(self, for_new_project: bool = True) -> None:
        """Return the app to a clean state for a new or loaded project."""
        app = self.app
        app.logger.debug(f"Resetting project state ({'new project' if for_new_project else 'project load'})...")

        # Preserve bar visibility across the reset (user intent should persist).
        prev_show_heatmap = getattr(app.app_state_ui, 'show_heatmap', True)
        prev_show_fs_timeline = getattr(app.app_state_ui, 'show_funscript_timeline', True)

        # Stop active processing.
        if app.processor and app.processor.is_processing:
            app.processor.stop_processing()
        if app.stage_processor.full_analysis_active:
            app.stage_processor.abort_stage_processing()

        app.file_manager.close_video_action(
            clear_funscript_unconditionally=True,
            skip_tracker_reset=(not for_new_project))
        app.funscript_processor.reset_state_for_new_project()
        app.funscript_processor.update_funscript_stats_for_timeline(1, "Project Reset")
        app.funscript_processor.update_funscript_stats_for_timeline(2, "Project Reset")

        # Waveform
        with app._waveform_lock:
            app.audio_waveform_data = None
        app.app_state_ui.show_audio_waveform = False

        # UI defaults from typed config
        ui_cfg = app.app_settings.config.ui
        app.app_state_ui.timeline_pan_offset_ms = ui_cfg.timeline_pan_offset_ms
        app.app_state_ui.timeline_zoom_factor_ms_per_px = ui_cfg.timeline_zoom_factor_ms_per_px
        app.app_state_ui.show_funscript_interactive_timeline = ui_cfg.show_funscript_interactive_timeline
        app.app_state_ui.show_funscript_interactive_timeline2 = ui_cfg.show_funscript_interactive_timeline2
        app.app_state_ui.show_heatmap = ui_cfg.show_heatmap
        app.app_state_ui.show_stage2_overlay = ui_cfg.show_stage2_overlay
        app.app_state_ui.reset_video_zoom_pan()

        # Model paths — the project might have had different ones; restore from settings.
        app.yolo_detection_model_path_setting = app.app_settings.config.models.yolo_det_path
        app.yolo_det_model_path = app.yolo_detection_model_path_setting
        app.yolo_pose_model_path_setting = app.app_settings.config.models.yolo_pose_path
        app.yolo_pose_model_path = app.yolo_pose_model_path_setting
        if app.tracker:
            app.tracker.det_model_path = app.yolo_det_model_path
            app.tracker.pose_model_path = app.yolo_pose_model_path

        # Undo history + dirty flags
        app.undo_manager.clear()
        app.app_state_ui.heatmap_dirty = True
        app.app_state_ui.funscript_preview_dirty = True
        app.app_state_ui.force_timeline_pan_to_current_frame = True

        # Restore preserved bar visibility.
        if hasattr(app.app_state_ui, 'show_heatmap'):
            app.app_state_ui.show_heatmap = prev_show_heatmap
        if hasattr(app.app_state_ui, 'show_funscript_timeline'):
            app.app_state_ui.show_funscript_timeline = prev_show_fs_timeline

        if for_new_project:
            app.logger.info("New project state initialized.", extra={'status_message': True})
        app.energy_saver.reset_activity_timer()
