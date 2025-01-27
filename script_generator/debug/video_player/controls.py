import cv2

from script_generator.debug.video_player.state import VideoPlayer


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