import numpy as np
import os
import shutil
import urllib.request
import zipfile
from typing import Dict, Tuple, TYPE_CHECKING


from config.constants import TIMELINE_HEATMAP_COLORS, TIMELINE_COLOR_SPEED_STEP, TIMELINE_COLOR_ALPHA, STATUS_DETECTED, STATUS_SMOOTHED

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic

class AppUtility:
    def __init__(self, app_instance=None):
        # app_instance might not be needed if all utility methods are static
        # or don't rely on application state.
        self.app = app_instance

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

    def process_mac_model_archive(self, downloaded_path: str, destination_dir: str,
                                  original_filename: str) -> str | None:
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
        color = (0.8, 0.8, 0.8, 0.7)
        thickness = 1.0
        is_dashed = False
        if role == "pref_penis":
            color = (0.1, 1.0, 0.1, 0.9)
            thickness = 2.0
        elif role == "locked_penis_box":
            color = (0.1, 0.9, 0.9, 0.8)
            thickness = 1.5
        elif role == "tracked_box":
            if class_name == "pussy":
                color = (1.0, 0.5, 0.8, 0.8)
            elif class_name == "butt":
                color = (0.9, 0.6, 0.2, 0.8)
            else:
                color = (1.0, 1.0, 0.2, 0.8)
            thickness = 1.5
        elif role.startswith("tracked_box_"):
            color = (0.7, 0.7, 0.7, 0.7)
            thickness = 1.0
        elif role == "general_detection":
            color = (0.2, 0.5, 1.0, 0.6)
        if status not in [STATUS_DETECTED, STATUS_SMOOTHED]:
            is_dashed = True
            color = (color[0], color[1], color[2], max(0.4, color[3] * 0.6))
        if box_data.get("is_excluded", False):
            color = (0.5, 0.1, 0.1, 0.5)
            is_dashed = True
        return color, thickness, is_dashed

    def get_speed_color_from_map(self, speed_pps: float) -> tuple:
        intensity = speed_pps
        heatmap_colors_list = TIMELINE_HEATMAP_COLORS
        step_val = TIMELINE_COLOR_SPEED_STEP
        alpha_val = TIMELINE_COLOR_ALPHA

        if np.isnan(intensity):
            return (128 / 255.0, 128 / 255.0, 128 / 255.0, alpha_val)
        if intensity <= 0:
            c = heatmap_colors_list[0]
            return (c[0] / 255, c[1] / 255, c[2] / 255, alpha_val)
        if intensity > (len(heatmap_colors_list) -1) * step_val:
            c = heatmap_colors_list[-1]
            return (c[0] / 255, c[1] / 255, c[2] / 255, alpha_val)

        index = int(intensity // step_val)
        index = max(0, min(index, len(heatmap_colors_list) - 2))

        t = max(0.0, min(1.0, (intensity - (index * step_val)) / step_val))
        c1, c2 = heatmap_colors_list[index], heatmap_colors_list[index + 1]
        r = (c1[0] + (c2[0] - c1[0]) * t) / 255.0
        g = (c1[1] + (c2[1] - c1[1]) * t) / 255.0
        b = (c1[2] + (c2[2] - c1[2]) * t) / 255.0
        return (r, g, b, alpha_val)
