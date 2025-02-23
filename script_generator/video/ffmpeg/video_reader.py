import os
import subprocess
import cv2
import imageio
import numpy as np
from script_generator.debug.logger import log
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
            self.state,
            start_frame,
            disable_opengl=True
        )
        log.info(f"Starting FFmpeg reader with command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        """Read the next frame from the video."""
        if not self.process:
            self._start_process(start_frame=self.start_frame)

        try:
            in_bytes = self.process.stdout.read(self.frame_size)
            if not in_bytes:
                log.warn("FFmpeg video reader could not read frame / end of file")
                return False, None  # End of video

            frame = np.frombuffer(in_bytes, np.uint8).reshape((self.height, self.width, 3))

            # output_path = os.path.join("C:/cvr/funscript-generator/tmp_output", f"frame_{self.current_frame_number:05d}.png")
            # imageio.imwrite(output_path, frame)

            self.current_frame_number += 1
            self.current_time = (self.current_frame_number / self.state.video_info.fps) * 1000

            return True, frame
        except Exception as e:
            log.error(f"Error reading frame: {e}")
            return False, None

    def set_frame(self, frame_id):
        self.start_frame = int(frame_id)
        self._start_process(start_frame=frame_id)

    def release(self):
        """Release resources and terminate the FFmpeg process."""
        if self.process:
            self.process.stdout.close()
            self.process.terminate()
            self.process = None
