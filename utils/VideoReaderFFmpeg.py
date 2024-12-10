import subprocess
import cv2
import numpy as np
import ffmpeg
import time


class VideoReaderFFmpeg:
    def __init__(self, video_path):
        self.video_path = video_path
        self._initialize_video_info()
        self.playback_speed = 1.0  # Default is 1.0 (real-time)
        self.paused = False
        self.start_frame = 0
        self.current_frame_number = 0
        self.current_time = 0
        self.process = None
        self.frame_size = None

    def _initialize_video_info(self):
        """ Retrieves video metadata like fps, total frames, resolution, etc. using ffprobe. """
        try:
            probe = ffmpeg.probe(self.video_path)
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            video_stream = video_streams[0]  # First video stream

            self.fps = eval(video_stream['r_frame_rate'])
            self.width = int(video_stream['width'])
            self.height = int(video_stream['height'])
            self.codec = video_stream['codec_name']
            self.total_frames = int(video_stream['nb_frames']) if 'nb_frames' in video_stream else None
            self.duration = float(video_stream['duration']) * 1000  # Duration in milliseconds

            print(f"FPS: {self.fps}, Resolution: {self.width}x{self.height}, "
                  f"Codec: {self.codec}, Total Frames: {self.total_frames}, Duration: {self.duration:.2f} ms")
        except Exception as e:
            print(f"Error initializing video info: {e}")

    def _start_process(self, start_frame=0):
        """ Start the FFmpeg process to read frames. """
        start_time = (start_frame / self.fps) * 1000  # Convert to milliseconds
        self.current_frame_number = start_frame
        self.process = (
            ffmpeg.input(self.video_path, ss=start_time / 1000)  # Convert back to seconds for FFmpeg
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
        )
        self.frame_size = self.width * self.height * 3  # For 'bgr24', 3 bytes per pixel

    def prev_initialize_video_info(self):
        """ Retrieves video metadata like fps, total frames, resolution, etc. using ffprobe. """
        # Retrieve FPS, frame count, codec, width, and height
        try:
            probe = ffmpeg.probe(self.video_path)
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            if not video_streams:
                raise ValueError("No video streams found in the input file.")
            video_stream = video_streams[0]  # First video stream

            self.fps = eval(video_stream['r_frame_rate'])
            self.width = int(video_stream['width'])
            self.height = int(video_stream['height'])
            self.codec = video_stream['codec_name']
            self.total_frames = int(video_stream['nb_frames']) if 'nb_frames' in video_stream else None
            self.duration = float(video_stream['duration']) * 1000  # Duration in milliseconds

            print(f"FPS: {self.fps}, Resolution: {self.width}x{self.height}, "
                  f"Codec: {self.codec}, Total Frames: {self.total_frames}, Duration: {self.duration:.2f} ms")

        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode('utf8')}")
            raise

    def get_frame(self, frame_number: int, crop = None, vr_roi= None) -> np.ndarray:
        """ Retrieve a specific frame by number, optionally crop or resize. """
        return self._get_frame_by_time_or_number(frame_number=frame_number, crop=crop, vr_roi=vr_roi)

    def get_frame_at_time(self, time_milliseconds, crop=None, vr_roi=None):
        """ Retrieve a frame at a specific time in the video (in milliseconds), with crop and resize options. """
        return self._get_frame_by_time_or_number(time_milliseconds=time_milliseconds, crop=crop, vr_roi=vr_roi)

    def _get_frame_by_time_or_number(self, frame_number=None, time_milliseconds=None, crop=None, vr_roi=None):
        """ Internal function to retrieve a frame either by frame number or time, with optional crop/resize. """
        try:
            if frame_number is not None:
                time_milliseconds = (frame_number / self.fps) * 1000  # Convert frame number to milliseconds
            elif time_milliseconds is None:
                raise ValueError("Either frame_number or time_milliseconds must be provided.")

            # FFmpeg cropping command if needed
            ffmpeg_input = ffmpeg.input(self.video_path, ss=time_milliseconds / 1000)  # Convert back to seconds for FFmpeg

            cmd = (
                ffmpeg_input
                .output('pipe:', vframes=1, format='image2', vcodec='rawvideo', pix_fmt='bgr24')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            frame_size = self.width * self.height * 3  # For 'bgr24', 3 bytes per pixel
            raw_frame = cmd.stdout.read(frame_size)
            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))

            if crop:
                if crop == "Left":
                    frame = frame[:, :frame.shape[1] // 2, :]
                elif crop == "Right":
                    frame = frame[:, frame.shape[1] // 2:, :]
                elif crop == "Top":
                    frame = frame[:frame.shape[0] // 2, :, :]
                elif crop == "Bottom":
                    frame = frame[frame.shape[0] // 2:, :, :]
            if vr_roi:
                panel_height, panel_width = frame.shape[:2]
                frame = frame[2 * (panel_height // 5):, (panel_width // 4):(3 * panel_width // 4)]

            # Close FFmpeg process
            cmd.stdout.close()
            cmd.wait()

            return frame

        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return None

    def read(self, crop=None, vr_roi=None, resize=None):
        """ Read the next frame from the video. Returns (ret, frame). """
        if self.process is None:
            self._start_process(start_frame=self.start_frame)

        try:
            in_bytes = self.process.stdout.read(self.frame_size)
            if not in_bytes:
                return False, None  # End of video
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            if crop:
                if crop == "Left":
                    frame = frame[:, :frame.shape[1] // 2, :]
                elif crop == "Right":
                    frame = frame[:, frame.shape[1] // 2:, :]
                elif crop == "Top":
                    frame = frame[:frame.shape[0] // 2, :, :]
                elif crop == "Bottom":
                    frame = frame[frame.shape[0] // 2:, :, :]
            if vr_roi:
                panel_height, panel_width = frame.shape[:2]
                frame = frame[2 * (panel_height // 5):, (panel_width // 4):(3 * panel_width // 4)]

            self.current_frame_number += 1
            self.current_time = (self.current_frame_number / self.fps) * 1000
            return True, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None

    def read_frames(self, start_frame=0, crop=None, resize=None):
        """
        Generator function that yields frames starting from a specific frame.
        You can crop and resize frames on the fly.
        """
        # Seek to the starting frame by calculating the time in milliseconds
        start_time = (start_frame / self.fps) * 1000  # Convert to milliseconds
        process = (
            ffmpeg.input(self.video_path, ss=start_time / 1000)  # Convert back to seconds for FFmpeg
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
        )

        frame_size = self.width * self.height * 3  # For 'bgr24', 3 bytes per pixel
        frame_number = start_frame

        try:
            while True:
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    break  # End of video
                frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])

                # Apply crop
                if crop:
                    x, y, w, h = crop
                    frame = frame[y:y + h, x:x + w]

                # Apply resize
                if resize:
                    new_w, new_h = resize
                    frame = cv2.resize(frame, (new_w, new_h))

                self.current_frame_number = frame_number
                yield frame, frame_number, frame_number * 1000 / self.fps  # Return frame, frame number, and timestamp in ms

                # Adjust for playback speed
                if not self.paused:
                    time.sleep(1 / (self.fps * self.playback_speed))
                    frame_number += 1
        finally:
            process.stdout.close()
            process.wait()



    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_playback_speed(self, speed):
        """ Set playback speed (1.0 = normal speed, 2.0 = 2x, 0.5 = 0.5x, etc.). """
        self.playback_speed = speed

    def seek_and_read(self, seek_frame, crop=None, resize=None):
        """ Seeks to a specific time and returns the next frame with crop and resize options. """
        frame = self.get_frame(seek_frame) #, crop=crop, resize=resize)
        return frame

    def get_metadata(self):
        """ Return the video's metadata as a dictionary. """
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'codec': self.codec,
            'total_frames': self.total_frames,
            'duration': self.duration,
        }

    def release(self):
        """ Release the resources and close the FFmpeg process. """
        if self.process:
            self.process.stdout.close()
            self.process.wait()
            self.process = None

"""
# Example Usage
if __name__ == '__main__':
    video_path = 'your_video.mp4'
    reader = VideoReaderFFmpeg(video_path)

    # Set playback speed to 2x
    reader.set_playback_speed(2.0)

    # Read and display frames at 2x speed
    for frame, frame_num, timestamp in reader.read_frames(start_frame=0, crop=(0, 0, reader.width // 2, reader.height)):
        print(f"Frame: {frame_num}, Time: {timestamp:.2f} ms")
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
"""