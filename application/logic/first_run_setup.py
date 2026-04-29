"""First-run model download + CoreML conversion.

Extracted from ApplicationLogic to keep the heavy `ultralytics.YOLO` import
off the startup path — it's only needed when the first-run setup thread
runs the CoreML export on macOS ARM.
"""
from __future__ import annotations

import os
import platform
import threading
import zipfile
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
                if is_mac_arm:
                    final_det_path = self._download_mlpackage(
                        models_dir,
                        MODEL_DOWNLOAD_URLS["detection_mlpackage_zip"],
                        "Detection")
                else:
                    final_det_path = self._download_pt(
                        models_dir,
                        MODEL_DOWNLOAD_URLS["detection_pt"],
                        "Detection")

                if final_det_path is None:
                    app.first_run_error = True
                    return

                app.app_settings.config.models.yolo_det_path = final_det_path
                app.yolo_detection_model_path_setting = final_det_path
                app.yolo_det_model_path = final_det_path
                app.logger.info(f"Detection model set to: {final_det_path}")
            else:
                app.logger.info(f"User already has detection model selected: {app.yolo_detection_model_path_setting}")

            # ---- Pose model ----
            # Pose .mlpackage isn't hosted (ultralytics ships only .pt). On
            # Mac ARM the .pt works (slower than .mlpackage); CoreML conversion
            # would need coremltools which isn't a default dep.
            if not user_has_pose:
                app.first_run_progress = 0
                final_pose_path = self._download_pt(
                    models_dir,
                    MODEL_DOWNLOAD_URLS["pose_pt"],
                    "Pose")
                if final_pose_path is None:
                    app.first_run_error = True
                    return

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

    def _download_pt(self, models_dir: str, url: str, label: str):
        app = self.app
        fname = os.path.basename(url)
        target = os.path.join(models_dir, fname)
        app.first_run_status_message = f"Downloading {label} Model: {fname}..."
        if not app.utility.download_file_with_progress(url, target, self._progress):
            app.first_run_status_message = f"{label} model download failed."
            return None
        return target

    def _download_mlpackage(self, models_dir: str, zip_url: str, label: str):
        """Download a hosted .mlpackage.zip, unzip into models_dir, return the
        .mlpackage path. Skips the .pt + coremltools conversion path."""
        app = self.app
        zip_name = os.path.basename(zip_url)
        zip_path = os.path.join(models_dir, zip_name)
        app.first_run_status_message = f"Downloading {label} Model: {zip_name}..."
        if not app.utility.download_file_with_progress(zip_url, zip_path, self._progress):
            app.first_run_status_message = f"{label} model download failed."
            return None
        app.first_run_status_message = f"Extracting {zip_name}..."
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(models_dir)
        except Exception as e:
            app.logger.error(f"Failed to extract {zip_name}: {e}", exc_info=True)
            return None
        try:
            os.remove(zip_path)
        except OSError:
            pass
        mlp_name = zip_name[:-4] if zip_name.endswith('.zip') else zip_name
        return os.path.join(models_dir, mlp_name)
