"""AI model management functionality for ApplicationLogic."""

import os
import platform

from ultralytics import YOLO

from config.constants import DEFAULT_MODELS_DIR, MODEL_DOWNLOAD_URLS


class AppModelManager:
    """Handles AI model loading, unloading, and management."""

    def __init__(self, app_logic):
        self.app = app_logic

    def unload_model(self, model_type: str):
        """
        Clears the path for a given model type and releases it from the tracker.
        """
        # --- Invalidate cache when models change ---
        self.app.cached_class_names = None

        if model_type == 'detection':
            self.app.yolo_detection_model_path_setting = ""
            self.app.app_settings.set("yolo_det_model_path", "")
            if self.app.tracker:
                self.app.tracker.unload_detection_model()
            self.app.logger.info("YOLO Detection Model unloaded.", extra={'status_message': True})
        elif model_type == 'pose':
            self.app.yolo_pose_model_path_setting = ""
            self.app.app_settings.set("yolo_pose_model_path", "")
            if self.app.tracker:
                self.app.tracker.unload_pose_model()
            self.app.logger.info("YOLO Pose Model unloaded.", extra={'status_message': True})
        else:
            self.app.logger.warning(f"Unknown model type '{model_type}' for unload.")

        self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def _cache_tracking_classes(self):
        """
        Temporarily loads the detection model to get class names, then unloads it.
        This populates self.app.cached_class_names. It's a blocking operation.
        It will first try to get names from an already-loaded tracker model to be efficient.
        """
        # If cache is already populated, do nothing.
        if self.app.cached_class_names is not None:
            return

        # If a model is already loaded for active tracking, use its class names.
        if self.app.tracker and hasattr(self.app.tracker, '_current_tracker') and self.app.tracker._current_tracker:
            current_tracker = self.app.tracker._current_tracker
            if hasattr(current_tracker, 'yolo_model') and current_tracker.yolo_model and hasattr(current_tracker.yolo_model, 'names'):
                self.app.logger.info("Model already loaded for tracking, using its class names for cache.")
                model_names = current_tracker.yolo_model.names
                if isinstance(model_names, dict):
                    self.app.cached_class_names = sorted(list(model_names.values()))
                elif isinstance(model_names, list):
                    self.app.cached_class_names = sorted(model_names)
                else:
                    self.app.logger.warning("Tracker model names format not recognized while caching.")
                return

        model_path = self.app.yolo_det_model_path
        if not model_path or not os.path.exists(model_path):
            self.app.logger.info("Cannot cache tracking classes: Detection model path not set or invalid.")
            self.app.cached_class_names = []  # Cache as empty to prevent re-attempts.
            return

        try:
            self.app.logger.info(f"Temporarily loading model to cache class names: {os.path.basename(model_path)}")
            # This is the potentially slow operation that can freeze the UI.
            temp_model = YOLO(model_path)
            model_names = temp_model.names

            if isinstance(model_names, dict):
                self.app.cached_class_names = sorted(list(model_names.values()))
            elif isinstance(model_names, list):
                self.app.cached_class_names = sorted(model_names)
            else:
                self.app.logger.warning("Model loaded for caching, but names format not recognized.")
                self.app.cached_class_names = []  # Cache as empty

            self.app.logger.info("Class names cached successfully.")
            del temp_model  # Explicitly release the model object

        except Exception as e:
            self.app.logger.error(f"Failed to temporarily load model '{model_path}' to cache class names: {e}", exc_info=True)
            self.app.cached_class_names = []  # Cache as empty on failure to prevent retries.

    def get_available_tracking_classes(self):
        """
        Gets the list of class names from the model.
        It uses a cache to avoid reloading the model repeatedly.
        """
        # If cache is not populated, do it now.
        if self.app.cached_class_names is None:
            self._cache_tracking_classes()

        # The cache should be populated now (even if with an empty list on failure).
        return self.app.cached_class_names if self.app.cached_class_names is not None else []

    def _check_model_paths(self):
        """Checks essential model paths and auto-downloads if missing."""
        models_missing = False

        # Detection model remains essential
        if not self.app.yolo_det_model_path or not os.path.exists(self.app.yolo_det_model_path):
            self.app.logger.warning(
                f"YOLO Detection Model not found or path not set: '{self.app.yolo_det_model_path}'. Attempting auto-download...",
                extra={'status_message': True, 'duration': 5.0})
            models_missing = True

        # Pose model is now optional but we'll try to download it too
        if not self.app.yolo_pose_model_path or not os.path.exists(self.app.yolo_pose_model_path):
            self.app.logger.warning(
                f"YOLO Pose Model not found or path not set. Attempting auto-download...",
                extra={'status_message': True, 'duration': 5.0})
            models_missing = True

        # Auto-download missing models
        if models_missing:
            self.app.logger.info("Auto-downloading missing models...")
            self.download_default_models()

            # Re-check after download
            if not self.app.yolo_det_model_path or not os.path.exists(self.app.yolo_det_model_path):
                self.app.logger.error(
                    f"CRITICAL ERROR: Failed to auto-download or configure detection model.",
                    extra={'status_message': True, 'duration': 15.0})
                # GUI popup: Inform user auto-download failed
                if getattr(self.app, "gui_instance", None):
                    self.app.gui_instance.show_error_popup("Detection Model Missing", "Failed to auto-download detection model.\nPlease select a YOLO model file in the UI Configuration tab or check your internet connection.")
                return False
            else:
                self.app.logger.info("Detection model successfully configured!", extra={'status_message': True, 'duration': 3.0})

        return True

    def download_default_models(self):
        """Manually download default models if they don't exist."""
        try:
            # Create models directory
            models_dir = DEFAULT_MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            self.app.logger.info(f"Checking for default models in: {models_dir}")

            # Determine OS for model format
            is_mac_arm = platform.system() == "Darwin" and platform.machine() == 'arm64'
            self.app.logger.info(f"Platform detection: system={platform.system()}, machine={platform.machine()}, is_mac_arm={is_mac_arm}")
            downloaded_models = []

            # Check and download detection model
            det_url = MODEL_DOWNLOAD_URLS["detection_pt"]
            det_filename_pt = os.path.basename(det_url)
            det_model_path_pt = os.path.join(models_dir, det_filename_pt)
            det_model_path_mlpackage = det_model_path_pt.replace('.pt', '.mlpackage')

            # Check if either .pt or .mlpackage version exists
            if not os.path.exists(det_model_path_pt) and not os.path.exists(det_model_path_mlpackage):
                self.app.logger.info(f"Downloading detection model: {det_filename_pt}")
                success = self.app.utility.download_file_with_progress(det_url, det_model_path_pt, None)
                if success:
                    downloaded_models.append(f"Detection model: {det_filename_pt}")

                    # Convert to CoreML if on macOS ARM
                    if is_mac_arm:
                        self.app.logger.info(f"Attempting to convert detection model to CoreML (is_mac_arm={is_mac_arm})...")
                        try:
                            model = YOLO(det_model_path_pt)
                            self.app.logger.info(f"YOLO model loaded, starting export to CoreML format...")
                            model.export(format="coreml")
                            self.app.logger.info(f"Converted detection model to CoreML: {det_model_path_mlpackage}")
                            # Set the CoreML model path in settings
                            self.app.app_settings.set("yolo_det_model_path", det_model_path_mlpackage)
                            self.app.yolo_detection_model_path_setting = det_model_path_mlpackage
                            self.app.yolo_det_model_path = det_model_path_mlpackage
                        except Exception as e:
                            self.app.logger.error(f"Failed to convert detection model to CoreML: {e}")
                            # Fall back to PT model if CoreML conversion fails
                            self.app.app_settings.set("yolo_det_model_path", det_model_path_pt)
                            self.app.yolo_detection_model_path_setting = det_model_path_pt
                            self.app.yolo_det_model_path = det_model_path_pt
                    else:
                        # Set the PT model path in settings for non-macOS ARM
                        self.app.app_settings.set("yolo_det_model_path", det_model_path_pt)
                        self.app.yolo_detection_model_path_setting = det_model_path_pt
                        self.app.yolo_det_model_path = det_model_path_pt
                else:
                    self.app.logger.error("Failed to download detection model")
            else:
                self.app.logger.info("Detection model already exists")
                # Check if path is not set in settings and auto-configure
                current_setting = self.app.app_settings.get("yolo_det_model_path", "")
                if not current_setting or not os.path.exists(current_setting):
                    # Prefer .mlpackage on macOS ARM if it exists
                    if is_mac_arm and os.path.exists(det_model_path_mlpackage):
                        self.app.app_settings.set("yolo_det_model_path", det_model_path_mlpackage)
                        self.app.yolo_detection_model_path_setting = det_model_path_mlpackage
                        self.app.yolo_det_model_path = det_model_path_mlpackage
                        self.app.logger.info(f"Auto-configured detection model path to: {det_model_path_mlpackage}")
                    elif os.path.exists(det_model_path_pt):
                        self.app.app_settings.set("yolo_det_model_path", det_model_path_pt)
                        self.app.yolo_detection_model_path_setting = det_model_path_pt
                        self.app.yolo_det_model_path = det_model_path_pt
                        self.app.logger.info(f"Auto-configured detection model path to: {det_model_path_pt}")

            # Check and download pose model
            pose_url = MODEL_DOWNLOAD_URLS["pose_pt"]
            pose_filename_pt = os.path.basename(pose_url)
            pose_model_path_pt = os.path.join(models_dir, pose_filename_pt)
            pose_model_path_mlpackage = pose_model_path_pt.replace('.pt', '.mlpackage')

            # Check if either .pt or .mlpackage version exists
            if not os.path.exists(pose_model_path_pt) and not os.path.exists(pose_model_path_mlpackage):
                self.app.logger.info(f"Downloading pose model: {pose_filename_pt}")
                success = self.app.utility.download_file_with_progress(pose_url, pose_model_path_pt, None)
                if success:
                    downloaded_models.append(f"Pose model: {pose_filename_pt}")

                    # Convert to CoreML if on macOS ARM
                    if is_mac_arm:
                        self.app.logger.info(f"Attempting to convert pose model to CoreML (is_mac_arm={is_mac_arm})...")
                        try:
                            model = YOLO(pose_model_path_pt)
                            self.app.logger.info(f"YOLO pose model loaded, starting export to CoreML format...")
                            model.export(format="coreml")
                            self.app.logger.info(f"Converted pose model to CoreML: {pose_model_path_mlpackage}")
                            # Set the CoreML model path in settings
                            self.app.app_settings.set("yolo_pose_model_path", pose_model_path_mlpackage)
                            self.app.yolo_pose_model_path_setting = pose_model_path_mlpackage
                            self.app.yolo_pose_model_path = pose_model_path_mlpackage
                        except Exception as e:
                            self.app.logger.error(f"Failed to convert pose model to CoreML: {e}")
                            # Fall back to PT model if CoreML conversion fails
                            self.app.app_settings.set("yolo_pose_model_path", pose_model_path_pt)
                            self.app.yolo_pose_model_path_setting = pose_model_path_pt
                            self.app.yolo_pose_model_path = pose_model_path_pt
                    else:
                        # Set the PT model path in settings for non-macOS ARM
                        self.app.app_settings.set("yolo_pose_model_path", pose_model_path_pt)
                        self.app.yolo_pose_model_path_setting = pose_model_path_pt
                        self.app.yolo_pose_model_path = pose_model_path_pt
                else:
                    self.app.logger.error("Failed to download pose model")
            else:
                self.app.logger.info("Pose model already exists")
                # Check if path is not set in settings and auto-configure existing model
                current_setting = self.app.app_settings.get("yolo_pose_model_path", "")
                if not current_setting or not os.path.exists(current_setting):
                    if os.path.exists(pose_model_path_pt):
                        self.app.logger.info("Auto-configuring existing pose model path in settings")
                        self.app.app_settings.set("yolo_pose_model_path", pose_model_path_pt)
                        self.app.yolo_pose_model_path_setting = pose_model_path_pt
                        self.app.yolo_pose_model_path = pose_model_path_pt

            # Report results
            if downloaded_models:
                message = f"Downloaded models: {', '.join(downloaded_models)}"
                self.app.set_status_message(message, duration=5.0)
                self.app.logger.info(message)
            else:
                message = "All default models already exist"
                self.app.set_status_message(message, duration=3.0)
                self.app.logger.info(message)

        except Exception as e:
            error_msg = f"Error downloading models: {e}"
            self.app.set_status_message(error_msg, duration=5.0)
            self.app.logger.error(error_msg, exc_info=True)
