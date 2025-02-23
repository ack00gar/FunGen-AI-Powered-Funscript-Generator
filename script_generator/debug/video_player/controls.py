import cv2
import numpy as np

from script_generator.debug.video_player.video_player import VideoPlayer


def draw_media_controls(frame, video_player: VideoPlayer):
    """
    Draw a simple "progress bar" style media control at the bottom of the frame.
    """
    height, width, _ = frame.shape
    bar_height = 10
    bar_y_start = height - bar_height

    # Background of the progress bar
    cv2.rectangle(frame, (0, bar_y_start), (width, height), (50, 50, 50), -1)

    # Progress indicator
    progress_x = int((video_player.current_frame / video_player.total_frames) * width)
    cv2.rectangle(frame, (0, bar_y_start), (progress_x, height), (0, 255, 0), -1)

def draw_media_controls_static_overlay(metrics, first_frame):
    height, width, _ = first_frame.shape
    overlay_height = 14
    y_start = height - overlay_height
    frames = len(metrics)
    px_frame = width / frames

    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    prev_sex_position = None
    for frame_pos, key in enumerate(metrics.keys()):
        metric = metrics[key]
        variables =  metric["variables"]
        sex_position = variables["sex_position"]
        if prev_sex_position != sex_position:
            prev_sex_position = sex_position

            x_pos = int(min(640, px_frame * frame_pos))
            cv2.line(overlay, (x_pos, y_start), (x_pos, height), (0, 0, 255), 1)

    return overlay