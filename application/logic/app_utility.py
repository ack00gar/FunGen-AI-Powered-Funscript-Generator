import numpy as np
from typing import Dict, Tuple


from config.constants import HEATMAP_COLORS_TIMELINE, STEP_SIZE_TIMELINE, TIMELINE_COLOR_ALPHA, STATUS_DETECTED, STATUS_SMOOTHED


class AppUtility:
    def __init__(self, app_instance=None):
        # app_instance might not be needed if all utility methods are static
        # or don't rely on application state.
        self.app = app_instance

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
        heatmap_colors_list = HEATMAP_COLORS_TIMELINE
        step_val = STEP_SIZE_TIMELINE
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
