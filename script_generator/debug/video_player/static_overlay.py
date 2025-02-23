import cv2

from script_generator.constants import CLASS_COLORS
from script_generator.debug.video_player.controls import PROGRESS_BAR_HEIGHT
from script_generator.state.app_state import AppState
from script_generator.video.data_classes.video_info import get_cropped_dimensions


def draw_static_overlay(state: "AppState", metrics, overlay):
    width, height = get_cropped_dimensions(state.video_info)
    frames = state.video_info.total_frames
    px_frame = width / frames  # pixels per frame
    # overlay = np.zeros((height, width, 4), dtype=np.uint8)
    progress_bar_top = height - PROGRESS_BAR_HEIGHT

    if state.tracking_logic_version == 1:
        return overlay

    # Draw positions
    positions = metrics["positions"]
    for position in positions:
        start, end = position["start"], position["end"]
        px_start = int(px_frame * start)
        px_end = int(px_frame * end)
        color = (240, 240, 0)
        cv2.rectangle(
            overlay,
            (px_start, progress_bar_top - 1),
            (px_end, progress_bar_top + 1),
            color,
            -1
        )
        cv2.line(overlay, (px_start, height - PROGRESS_BAR_HEIGHT - 5), (px_start, height), color, 1)
        cv2.line(overlay, (px_end, height - PROGRESS_BAR_HEIGHT - 5), (px_end, height), color, 1)

        for sex_act in position["sex_acts"]:
            px_start = int(px_frame * sex_act["start"])
            px_end = int(px_frame * sex_act["end"])

            cv2.rectangle(
                overlay,
                (px_start, progress_bar_top - 4),
                (px_end, progress_bar_top - 2),
                CLASS_COLORS[sex_act["cls"]],
                -1
            )

    cuts = metrics.get("cuts", [])
    for cut in cuts:
        px_cut = int(px_frame * cut)
        cv2.line(overlay, (px_cut, height - PROGRESS_BAR_HEIGHT + 2), (px_cut, height), (0, 0, 255), 1)

    return overlay
