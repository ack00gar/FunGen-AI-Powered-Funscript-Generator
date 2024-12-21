import ffmpeg
import numpy as np
import cv2  # For OpenCV constants


class FFmpegVideoCapture:
    def __init__(self, video_path):
        """
        Initialize the FFmpegVideoCapture with a video file.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.probe = ffmpeg.probe(video_path)  # Use the correct probe function
        video_stream = next((stream for stream in self.probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise ValueError("No video stream found in the input file.")

        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])
        self.fps = int(video_stream['r_frame_rate'].split('/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
        self.total_frames = int(video_stream.get('nb_frames', -1))
        self.frame_count = 0

        # Initialize FFmpeg process
        self.process = None
        self.start_process()

    def start_process(self):
        """
        Start the FFmpeg process to read frames.
        """
        try:
            self.process = (
                ffmpeg
                .input(self.video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to start FFmpeg process: {e}")

    def read(self):
        """
        Read the next frame from the video.

        Returns:
            bool: True if the frame was read successfully, False otherwise.
            np.ndarray: The frame as a NumPy array (or None if no frame was read).
        """
        if not self.process:
            return False, None  # Process not initialized or already released

        in_bytes = self.process.stdout.read(self.width * self.height * 3)
        if not in_bytes:
            return False, None

        frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))
        self.frame_count += 1
        return True, frame

    def get(self, prop_id):
        """
        Get a property of the video.

        Args:
            prop_id (int): Property ID (e.g., cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_HEIGHT, etc.).

        Returns:
            float: The value of the property.
        """
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.frame_count
        else:
            return 0.0  # Unsupported property

    def set(self, prop_id, value):
        """
        Set a property of the video.

        Args:
            prop_id (int): Property ID (e.g., cv2.CAP_PROP_POS_FRAMES).
            value (float): The value to set.

        Returns:
            bool: True if the property was set successfully, False otherwise.
        """
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            # Seek to the specified frame
            if value < 0 or (self.total_frames > 0 and value >= self.total_frames):
                return False  # Invalid frame number

            # Calculate the time in seconds to seek to
            seek_time = value / self.fps

            # Restart the FFmpeg process with the seek time
            self.release()
            try:
                self.process = (
                    ffmpeg
                    .input(self.video_path, ss=seek_time)  # Seek to the specified time
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run_async(pipe_stdout=True)
                )
                self.frame_count = value
                return True
            except ffmpeg.Error as e:
                raise RuntimeError(f"Failed to start FFmpeg process after seek: {e}")
        else:
            return False  # Unsupported property

    def release(self):
        """
        Release the FFmpeg process and clean up resources.
        """
        if self.process:
            self.process.stdout.close()
            self.process.wait()
            self.process = None

    def __del__(self):
        """
        Ensure the FFmpeg process is released when the object is deleted.
        """
        self.release()