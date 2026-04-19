"""First-run model download + CoreML conversion.

Extracted from ApplicationLogic to keep the heavy `ultralytics.YOLO` import
off the startup path — it's only needed when the first-run setup thread
runs the CoreML export on macOS ARM.
"""
from __future__ import annotations

import os
import platform
import threading
from typing import TYPE_CHECKING

from config.constants import DEFAULT_MODELS_DIR, MODEL_DOWNLOAD_URLS

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic


class FirstRunSetupController:
    """Owns the first-run wizard's model download thread and progress state."""

    __slots__ = ("app",)

    def __init__(self, app: "ApplicationLogic") -> None:
        self.app = app

    def trigger(self) -> None:
        """Initiate first-run model download in a background thread."""
        app = self.app
        if app.first_run_thread and app.first_run_thread.is_alive():
            return
        app.show_first_run_setup_popup = True
        app.first_run_progress = 0
        app.first_run_status_message = "Starting setup..."
        app.first_run_thread = threading.Thread(
            target=self._run, daemon=True, name="FirstRunSetupThread")
        app.first_run_thread.start()

    def _run(self) -> None:
        """Background thread body — downloads models and optionally converts
        to CoreML on Apple Silicon."""
        app = self.app
        try:
            models_dir = DEFAULT_MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            app.first_run_status_message = f"Created directory: {models_dir}"
            app.logger.info(app.first_run_status_message)

            user_has_det = (app.yolo_detection_model_path_setting
                            and os.path.exists(app.yolo_detection_model_path_setting))
            user_has_pose = (app.yolo_pose_model_path_setting
                             and os.path.exists(app.yolo_pose_model_path_setting))

            is_mac_arm = platform.system() == "Darwin" and platform.machine() == 'arm64'

            # ---- Detection model ----
            if not user_has_det:
                det_url = MODEL_DOWNLOAD_URLS["detection_pt"]
                det_filename_pt = os.path.basename(det_url)
                det_path_pt = os.path.join(models_dir, det_filename_pt)
                app.first_run_status_message = f"Downloading Detection Model: {det_filename_pt}..."
                success = app.utility.download_file_with_progress(
                    det_url, det_path_pt, self._progress)
                if not success:
                    app.first_run_status_message = "Detection model download failed."
                    app.first_run_error = True
                    return

                final_det_path = det_path_pt
                if is_mac_arm:
                    app.first_run_status_message = "Converting detection model to CoreML format..."
                    app.logger.info(f"Running on macOS ARM. Converting {det_filename_pt} to .mlpackage")
                    try:
                        # Lazy import: ultralytics pulls torch (~2s cold), only
                        # needed on macOS ARM first-run.
                        from ultralytics import YOLO
                        YOLO(det_path_pt).export(format="coreml")
                        final_det_path = det_path_pt.replace('.pt', '.mlpackage')
                        app.logger.info(f"Successfully converted detection model to {final_det_path}")
                    except Exception as e:
                        app.logger.error(f"Failed to convert detection model to CoreML: {e}", exc_info=True)
                        app.first_run_status_message = "Detection model conversion to CoreML failed. Using .pt format."

                app.app_settings.config.models.yolo_det_path = final_det_path
                app.yolo_detection_model_path_setting = final_det_path
                app.yolo_det_model_path = final_det_path
                app.logger.info(f"Detection model set to: {final_det_path}")
            else:
                app.logger.info(f"User already has detection model selected: {app.yolo_detection_model_path_setting}")

            # ---- Pose model ----
            if not user_has_pose:
                app.first_run_progress = 0
                pose_url = MODEL_DOWNLOAD_URLS["pose_pt"]
                pose_filename_pt = os.path.basename(pose_url)
                pose_path_pt = os.path.join(models_dir, pose_filename_pt)
                app.first_run_status_message = f"Downloading Pose Model: {pose_filename_pt}..."
                success = app.utility.download_file_with_progress(
                    pose_url, pose_path_pt, self._progress)
                if not success:
                    app.first_run_status_message = "Pose model download failed."
                    app.first_run_error = True
                    return

                final_pose_path = pose_path_pt
                if is_mac_arm:
                    app.first_run_status_message = "Converting pose model to CoreML format..."
                    app.logger.info(f"Running on macOS ARM. Converting {pose_filename_pt} to .mlpackage")
                    try:
                        from ultralytics import YOLO
                        YOLO(pose_path_pt).export(format="coreml")
                        final_pose_path = pose_path_pt.replace('.pt', '.mlpackage')
                        app.logger.info(f"Successfully converted pose model to {final_pose_path}")
                    except Exception as e:
                        app.logger.error(f"Failed to convert pose model to CoreML: {e}", exc_info=True)
                        app.first_run_status_message = "Pose model conversion to CoreML failed. Using .pt format."

                app.app_settings.config.models.yolo_pose_path = final_pose_path
                app.yolo_pose_model_path_setting = final_pose_path
                app.yolo_pose_model_path = final_pose_path
                app.logger.info(f"Pose model set to: {final_pose_path}")
            else:
                app.logger.info(f"User already has pose model selected: {app.yolo_pose_model_path_setting}")

            app.first_run_status_message = "Setup complete! Please restart the application."
            app.logger.info("Default model setup complete.")
            app.first_run_progress = 100

        except Exception as e:
            app.first_run_status_message = f"An error occurred: {e}"
            app.logger.error(f"First run setup failed: {e}", exc_info=True)

    def _progress(self, percent, downloaded, total_size) -> None:
        """Download progress callback."""
        self.app.first_run_progress = percent
