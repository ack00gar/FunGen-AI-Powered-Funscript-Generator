import cv2

from script_generator.debug.video_player.constants import BAR_HEIGHT
from script_generator.debug.logger import logger


def mouse_callback(event, x, y, flags, param):
    """
    Called when the user clicks on the OpenCV window.
    We pass in the video player via param for seeking.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        video_player = param
        height = int(video_player.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bar_y_start = height - BAR_HEIGHT

        # If clicked in the progress bar area, adjust the seek
        if y >= bar_y_start:
            width = int(video_player.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            target_frame = int((x / width) * video_player.total_frames)
            video_player.set_frame(target_frame)
            logger.info(f"Seek to frame: {target_frame}")