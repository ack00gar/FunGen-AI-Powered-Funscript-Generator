import cv2
import numpy as np

from script_generator.constants import FUNSCRIPT_BUFFER_SIZE, GAUGE_WIDTH, GAUGE_HEIGHT
from script_generator.state.app_state import AppState
from script_generator.video.data_classes.video_info import get_cropped_dimensions
from script_generator.video.ffmpeg.video_reader import VideoReaderFFmpeg


class VideoPlayer:
    def __init__(self, state: AppState, start_frame, end_frame):
        self.video_path = state.video_path
        self.video_info = state.video_info
        self.total_frames = state.video_info.total_frames
        self.start_frame = start_frame
        self.end_frame = state.video_info.total_frames if not end_frame else end_frame
        self.current_frame = 0

        # for rolling funscripts
        # self.funscript_graph = FunscriptGraph(state)

        # gauge position
        width, height = get_cropped_dimensions(self.video_info)
        self.gauge_position = (width - GAUGE_WIDTH), (height - GAUGE_HEIGHT)

        # a buffer to show outliers longer then 1 frame and slowly fade them out
        self.outlier_buffer = []

        self.reader = VideoReaderFFmpeg(state, start_frame)
        self.paused = False

        if start_frame != 0:
            self.set_frame(start_frame)

    def release(self):
        if self.reader:
            self.reader.release()

    def set_frame(self, frame_id):
        if frame_id < 0:
            frame_id = 0
        if frame_id >= self.total_frames:
            frame_id = self.total_frames - 1

        self.current_frame = frame_id
        self.reader.set_frame(self.current_frame)

    def read_frame(self):
        ret, frame = self.reader.read()
        if ret and not self.paused:
            self.current_frame += 1
        return ret, frame