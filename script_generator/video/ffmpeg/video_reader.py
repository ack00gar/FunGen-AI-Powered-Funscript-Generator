import subprocess
import cv2
import numpy as np
from script_generator.debug.logger import logger
from script_generator.state.app_state import AppState
from script_generator.video.ffmpeg.commands import get_ffmpeg_read_cmd

class VideoReaderFFmpeg:
    def __init__(self, state, start_frame=0):
        self.state: AppState = state
        self.video_path = state.video_path
        self.start_frame = start_frame
        self.current_frame_number = 0
        self.current_time = 0
        self.process = None
        self.frame_size = None
        self.width = None
        self.height = None

    def _start_process(self, start_frame=0):
        self.current_frame_number = start_frame

        # Kill the process if already running
        if self.process:
            self.process.terminate()

        cmd, self.frame_size, self.width, self.height = get_ffmpeg_read_cmd(
            self.state.video_info,
            self.state.video_reader,
            self.state.ffmpeg_hwaccel,
            start_frame,
            disable_opengl=True
        )
        logger.info(f"Starting FFmpeg reader with command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        """Read the next frame from the video."""
        if not self.process:
            self._start_process(start_frame=self.start_frame)

        try:
            in_bytes = self.process.stdout.read(self.frame_size)
            if not in_bytes:
                return False, None  # End of video

            frame = np.frombuffer(in_bytes, np.uint8).reshape((self.height, self.width, 3))
            self.current_frame_number += 1
            self.current_time = (self.current_frame_number / self.state.video_info.fps) * 1000

            return True, frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def set(self, prop_id, value):
        """Set properties like frame position (mimics OpenCV's cap.set())."""
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self.start_frame = int(value)
            self._start_process(start_frame=self.start_frame)
        else:
            logger.error(f"Unsupported property ID: {prop_id}")

    def get(self, prop_id):
        """Get properties like FPS, width, height (mimics OpenCV's cap.get())."""
        props = {
            cv2.CAP_PROP_FPS: self.state.video_info.fps,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FRAME_COUNT: self.state.video_info.total_frames,
            cv2.CAP_PROP_POS_FRAMES: self.current_frame_number,
        }
        return props.get(prop_id, logger.error(f"Unsupported property ID: {prop_id}"))

    def release(self):
        """Release resources and terminate the FFmpeg process."""
        if self.process:
            self.process.stdout.close()
            self.process.terminate()
            self.process = None
