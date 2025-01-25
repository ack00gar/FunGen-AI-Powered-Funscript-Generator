import cv2

from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg


class VideoPlayer:
    def __init__(self, video_path, video_info, start_frame, end_frame):
        self.video_path = video_path
        self.video_info = video_info
        self.total_frames = video_info.total_frames
        self.start_frame = start_frame
        self.end_frame = video_info.total_frames if not end_frame else end_frame
        self.current_frame = 0

        self.cap = VideoReaderFFmpeg(self.video_path, is_vr=video_info.is_vr)
        self.paused = False

    def release(self):
        if self.cap:
            self.cap.release()

    def set_frame(self, frame_id):
        if frame_id < 0:
            frame_id = 0
        if frame_id >= self.total_frames:
            frame_id = self.total_frames - 1

        self.current_frame = frame_id
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret and not self.paused:
            self.current_frame += 1
        return ret, frame