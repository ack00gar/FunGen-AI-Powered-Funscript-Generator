import numpy as np
import os
import shutil
import urllib.request
import zipfile
from typing import Dict, Tuple, TYPE_CHECKING
from config.constants import STATUS_DETECTED, STATUS_SMOOTHED
from config.constants_colors import RGBColors
from config.element_group_colors import BoxStyleColors

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic

class AppUtility:
    def __init__(self, app_instance=None):
        # app_instance might not be needed if all utility methods are static
        # or don't rely on application state.
        self.app = app_instance
        self.heatmap_colors_list = RGBColors.TIMELINE_HEATMAP
        self.step_val = RGBColors.TIMELINE_COLOR_SPEED_STEP
        self.alpha_val = RGBColors.TIMELINE_COLOR_ALPHA

        self.grey_rgb = RGBColors.GREY

    def _download_reporthook(self, block_num, block_size, total_size, progress_callback):
        """Callback for urllib.request.urlretrieve to report progress."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            if progress_callback:
                progress_callback(percent, downloaded, total_size)

    def download_file_with_progress(self, url: str, destination_path: str, progress_callback=None) -> bool:
        """Downloads a file and reports progress."""
        self.app.logger.info(f"Downloading from {url} to {destination_path}")
        try:
            reporthook = lambda bn, bs, ts: self._download_reporthook(bn, bs, ts, progress_callback)
            urllib.request.urlretrieve(url, destination_path, reporthook=reporthook)
            self.app.logger.info(f"Successfully downloaded {os.path.basename(destination_path)}.")
            if progress_callback:
                progress_callback(100, 0, 0)
            return True
        except Exception as e:
            self.app.logger.error(f"Failed to download {url}: {e}", exc_info=True)
            if os.path.exists(destination_path):
                os.remove(destination_path)
            return False

    def process_mac_model_archive(self, downloaded_path: str, destination_dir: str, original_filename: str) -> str | None:
        """
        Processes the downloaded file for a macOS .mlpackage model.
        It handles extraction if it's a zip, or renames it if it's an auto-unzipped package.
        Returns the final path to the .mlpackage.
        """
        self.app.logger.info(f"Processing macOS model: {os.path.basename(downloaded_path)}")

        if zipfile.is_zipfile(downloaded_path):
            self.app.logger.info("Archive is a valid zip file. Extracting...")
            try:
                with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                    mlpackage_name = next(
                        (name.split('/')[0] for name in zip_ref.namelist() if name.endswith('.mlpackage/')), None)
                    if not mlpackage_name:
                        self.app.logger.error("Could not find a .mlpackage directory inside the zip file.")
                        os.remove(downloaded_path)
                        return None
                    zip_ref.extractall(destination_dir)
                os.remove(downloaded_path)
                final_path = os.path.join(destination_dir, mlpackage_name)
                self.app.logger.info(f"Successfully extracted to: {final_path}")
                return final_path
            except Exception as e:
                self.app.logger.error(f"Failed to extract zip file: {e}", exc_info=True)
                return None
        else:
            self.app.logger.warning(
                "Downloaded item is not a zip archive. Assuming it is the model package and renaming.")
            final_name = original_filename.replace('.zip', '')
            final_path = os.path.join(destination_dir, final_name)
            try:
                if os.path.exists(final_path):
                    if os.path.isdir(final_path):
                        shutil.rmtree(final_path)
                    else:
                        os.remove(final_path)

                os.rename(downloaded_path, final_path)

                if os.path.exists(final_path):
                    self.app.logger.info(f"Successfully processed model package: {os.path.basename(final_path)}")
                    return final_path
                else:
                    self.app.logger.error(
                        f"Processed path '{final_path}' does not exist after rename. Model setup failed.")
                    return None
            except Exception as e:
                self.app.logger.error(f"Failed to process model file: {e}", exc_info=True)
                return None

    def get_box_style(self, box_data: Dict) -> Tuple[Tuple[float, float, float, float], float, bool]:
        role = box_data.get("role_in_frame", "general_detection")
        status = box_data.get("status", STATUS_DETECTED)
        class_name = box_data.get("class_name", "")
        color = BoxStyleColors.GENERAL
        thickness = 1.0
        is_dashed = False
        if role == "pref_penis":
            color = BoxStyleColors.PREF_PENIS
            thickness = 2.0
        elif role == "locked_penis_box":
            color = BoxStyleColors.LOCKED_PENIS
            thickness = 1.5
        elif role == "tracked_box":
            if class_name == "pussy":
                color = BoxStyleColors.PUSSY
            elif class_name == "butt":
                color = BoxStyleColors.BUTT
            else:
                color = BoxStyleColors.TRACKED
            thickness = 1.5
        elif role.startswith("tracked_box_"):
            color = BoxStyleColors.TRACKED_ALT
            thickness = 1.0
        elif role == "general_detection":
            color = BoxStyleColors.GENERAL_DETECTION
        if status not in [STATUS_DETECTED, STATUS_SMOOTHED]:
            is_dashed = True
            color = (color[0], color[1], color[2], max(0.4, color[3] * 0.6))
        if box_data.get("is_excluded", False):
            color = BoxStyleColors.EXCLUDED
            is_dashed = True
        return color, thickness, is_dashed

    def get_speed_color_from_map(self, speed_pps: float) -> tuple:
        if np.isnan(speed_pps):
            return (self.grey_rgb, self.alpha_val)  # gray for NaN

        max_index = len(self.heatmap_colors_list) - 1
        max_value = max_index * self.step_val

        if speed_pps <= 0:
            c = self.heatmap_colors_list[0]
        elif speed_pps >= max_value:
            c = self.heatmap_colors_list[max_index]
        else:
            index = int(speed_pps // self.step_val)
            t = (speed_pps % self.step_val) / self.step_val
            c1 = self.heatmap_colors_list[index]
            c2 = self.heatmap_colors_list[index + 1]
            c = [c1[i] + (c2[i] - c1[i]) * t for i in range(3)]

        r, g, b = (ch / 255.0 for ch in c)
        return (r, g, b, self.alpha_val)

