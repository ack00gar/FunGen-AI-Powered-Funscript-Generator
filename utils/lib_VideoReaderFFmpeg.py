import subprocess
import cv2
import numpy as np
from utils.config import ffmpeg_path, ffprobe_path


class VideoReaderFFmpeg:
    def __init__(self, video_path, is_VR=False, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path):
        """
        Initialize the VideoReaderFFmpeg class.
        :param video_path: Path to the video file.
        :param ffmpeg_path: Path to the FFmpeg binary (default: "ffmpeg").
        :param ffprobe_path: Path to the FFprobe binary (default: "ffprobe").
        """
        self.video_path = video_path
        self.is_VR = is_VR
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._initialize_video_info()  # Initialize video metadata
        self.start_frame = 0
        self.current_frame_number = 0
        self.current_time = 0
        self.process = None
        self.frame_size = None
        self.type = ""
        self.iv_fov = 0
        self.ih_fov = 0
        self.d_fov = 0

    def _initialize_video_info(self):
        """
        Retrieve video metadata (fps, resolution, codec, etc.) using FFprobe.
        """
        try:
            # Run FFprobe to get video metadata
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,width,height,codec_name,nb_frames,duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.video_path,
            ]
            output = subprocess.check_output(cmd).decode("utf-8").splitlines()

            # Ensure the output has the correct number of fields
            if len(output) < 6:
                raise ValueError("FFprobe output is missing required fields.")

            # Parse metadata
            codec_name = output[0]  # codec_name
            width = int(output[1])  # width
            height = int(output[2])  # height
            r_frame_rate = output[3]  # r_frame_rate in "num/den" format
            duration = float(output[4])  # duration in seconds
            total_frames = int(output[5])  # nb_frames

            # Calculate FPS
            num, den = map(int, r_frame_rate.split('/'))  # Split numerator and denominator
            fps = num / den  # Calculate FPS

            # Store metadata
            self.fps = fps
            self.width = width
            if self.is_VR:
                self.width //= 2
            self.height = height
            self.codec = codec_name
            self.total_frames = total_frames
            self.duration = duration * 1000  # Convert duration to milliseconds

            print(f"FPS: {self.fps}, Resolution: {self.width}x{self.height}, "
                  f"Codec: {self.codec}, Total Frames: {self.total_frames}, Duration: {self.duration:.2f} ms")
        except Exception as e:
            print(f"Error initializing video info: {e}")
            raise

    def _start_process(self, start_frame=0):
        """
        Start the FFmpeg process to read frames.
        :param start_frame: Frame number to start reading from.
        """
        start_time = (start_frame / self.fps) * 1000  # Convert to milliseconds
        self.current_frame_number = start_frame

        if self.is_VR:
            # FFmpeg command to read frames with VR reprojection
            # if 'FISHEYE' is present in video_path then type = "fisheye"
            if 'FISHEYE' in self.video_path:
                self.type = "fisheye"
                self.iv_fov = 190  # 120
                self.ih_fov = 190
                self.v_fov = 90
                self.h_fov = 90
                self.d_fov = 180  # 110
            else:
                type = "he"  # [0:v]v360=input=he:in_stereo=sbs:pitch=-35:v_fov=90:h_fov=90:output=sg:w=2048:h=2048
                self.iv_fov = 90
                self.ih_fov = 90
                self.d_fov = 100
            #"""
            cmd = [
                self.ffmpeg_path,
                #"-hwaccel", "videotoolbox",  # Use hardware acceleration on macOS
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", self.video_path,
                "-an",  # Disable audio processing
                #"-filter_complex",  # Apply the v360 filter for VR reprojection
                #f"[0:v]v360=input=he:in_stereo=sbs:pitch=-35:yaw=-0.75:roll=0:output=sg:v_fov=90:h_fov=90:d_fov=180:w={self.width}:h={self.height}",
                "-map", "0:v:0",
                "-vf", f"crop=w=iw/2:h=ih:x=0:y=0,v360={self.type}:sg:iv_fov={self.iv_fov}:ih_fov={self.ih_fov}:d_fov={self.d_fov}:v_fov={self.v_fov}:h_fov={self.h_fov}:pitch=-20:yaw=0:roll=0:w={self.width}:h={self.height}:interp=lanczos:reset_rot=1",
                "-f", "rawvideo",  # Output raw video data
                "-pix_fmt", "bgr24",  # Pixel format (BGR for OpenCV)
                "-vsync", "0",  # Disable frame rate synchronization
                "-threads", "0",  # Use maximum threads available
                "-",  # Output to stdout
            ]
            """
            cmd = [
                self.ffmpeg_path,
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", self.video_path,  # Input video file
                "-vf", "crop=in_w/2:in_h:0:0",  # Crop to the left half (width/2, full height, starting from top-left)
                "-an",  # Disable audio processing
                "-f", "rawvideo",  # Output raw video data
                "-pix_fmt", "bgr24",  # Pixel format (BGR for OpenCV)
                "-vsync", "0",  # Disable frame rate synchronization
                "-",  # Output to stdout
            ]
            """
        else:
            # FFmpeg command to read frames
            cmd = [
                self.ffmpeg_path,
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", self.video_path,
                "-f", "rawvideo",  # Output raw video data
                "-pix_fmt", "bgr24",  # Pixel format (BGR for OpenCV)
                "-vsync", "0",  # Disable frame rate synchronization
                "-",  # Output to stdout
            ]

        # kill the process if already running
        if self.process:
            self.process.terminate()
            # self.process.wait()

        # Start FFmpeg process
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frame_size = self.width * self.height * 3  # Size of one frame in bytes

    def read(self):
        """
        Read the next frame from the video. Mimics OpenCV's cap.read().
        :return: (True, frame) if successful, (False, None) if end of video.
        """
        if self.process is None:
            self._start_process(start_frame=self.start_frame)

        try:
            in_bytes = self.process.stdout.read(self.frame_size)
            if not in_bytes:
                return False, None  # End of video
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            self.current_frame_number += 1
            self.current_time = (self.current_frame_number / self.fps) * 1000
            return True, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None

    def set(self, prop_id, value):
        """
        Mimics OpenCV's cap.set().
        :param prop_id: Property ID (e.g., cv2.CAP_PROP_POS_FRAMES).
        :param value: Value to set.
        """
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self.start_frame = int(value)
            self._start_process(start_frame=self.start_frame)
        elif prop_id == cv2.CAP_PROP_FPS:
            self.fps = value
        else:
            print(f"Property {prop_id} not supported.")

    def get(self, prop_id):
        """
        Mimics OpenCV's cap.get().
        :param prop_id: Property ID (e.g., cv2.CAP_PROP_FPS).
        :return: Property value.
        """
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.current_frame_number
        else:
            print(f"Property {prop_id} not supported.")
            return None

    def release(self):
        """
        Release the resources and close the FFmpeg process.
        """
        if self.process:
            self.process.stdout.close()
            # self.process.wait()
            self.process = None
