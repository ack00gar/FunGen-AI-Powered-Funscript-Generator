import subprocess
import cv2
import numpy as np
import argparse
from params.config import ffmpeg_path, ffprobe_path


class VideoReaderFFmpeg:
    def __init__(self, video_path, is_VR=False, unwarp=True, projection=None, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path):
        """
        Initialize the VideoReaderFFmpeg class.
        :param video_path: Path to the video file.
        :param unwarp: Will try and unwarp VR videos.
        :param projection: FISHEYE, EQUIRECTANGULAR. Will also look for 'FISHEYE' in the video path.
        :param ffmpeg_path: Path to the FFmpeg binary (default: "ffmpeg").
        :param ffprobe_path: Path to the FFprobe binary (default: "ffprobe").
        """
        self.video_path = video_path
        self.is_VR = is_VR
        if not self.is_VR:
            self.unwarp = False
        else:
            self.unwarp = unwarp
        self.projection = projection
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
            #arg_line = "crop=w=iw/2:h=ih:x=0:y=0"
            arg_line = ""
            if self.unwarp:
                if self.projection == "FISHEYE" or (self.projection == None and "FISHEYE" in self.video_path.upper()):
                    print("Proceeding with fisheye projection correction")
                    self.type = "fisheye"
                    self.iv_fov = 190
                    self.ih_fov = 190
                    self.v_fov = 90
                    self.h_fov = 90
                    self.d_fov = 180
                else:  # Assuming Equirectangular
                    print("Assuming Equirectangular projection")
                    self.type = "he"
                    self.iv_fov = 250
                    self.ih_fov = 120
                    self.v_fov = 90
                    self.h_fov = 90
                    self.d_fov = 180
                arg_line = arg_line + f"v360={self.type}:in_stereo=sbs:output=sg"
                arg_line = arg_line + f":iv_fov={self.iv_fov}:ih_fov={self.ih_fov}"
                arg_line = arg_line + f":d_fov={self.d_fov}:v_fov={self.v_fov}:h_fov={self.h_fov}"
                arg_line = arg_line + f":pitch=-25:yaw=0:roll=0"
                arg_line = arg_line + f":w={self.width}:h={self.height}"
                arg_line = arg_line + f":interp=lanczos:reset_rot=1"
                arg_line = arg_line + f",lutyuv=y=gammaval(0.7)"
                #arg_line = arg_line + f",eq=brightness=0.1:contrast=1.5"
                #arg_line = arg_line + f",format=gray"
                #arg_line = arg_line + f",histeq"
            else:
                arg_line = "crop=w=iw/2:h=ih:x=0:y=0"
            # Add scale filter with height and auto-width (-1)
            #arg_line += f",scale=-1:{target_height}"
            #arg_line += f",scale=-1:{1080}"
            # perf for on the fly downscale to 1080p were terrible

            cmd = [
                self.ffmpeg_path,
                '-nostats',  # Disable progress statistics
                '-loglevel', 'warning',
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", self.video_path,
                "-an",  # Disable audio processing
                "-map", "0:v:0",
                "-vf", arg_line,
                "-f", "rawvideo",  # Output raw video data
                "-pix_fmt", "bgr24",  # Pixel format (BGR for OpenCV)
                "-vsync", "0",  # Disable frame rate synchronization
                "-threads", "0",  # Use maximum threads available
                "-",  # Output to stdout
            ]
        else:
            # FFmpeg command to read frames
            cmd = [
                self.ffmpeg_path,
                '-nostats',  # Disable progress statistics
                '-loglevel', 'warning',
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", self.video_path,
                "-an",  # Disable audio processing
                "-f", "rawvideo",  # Output raw video data
                "-pix_fmt", "bgr24",  # Pixel format (BGR for OpenCV)
                "-vsync", "0",  # Disable frame rate synchronization
                "-",  # Output to stdout
            ]

        # Kill the process if already running
        if self.process:
            self.process.terminate()

        # Start FFmpeg process
        #self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
            self.process = None


def main(video_path, is_VR=False):
    """
    Display a video using the VideoReaderFFmpeg class.
    :param video_path: Path to the video file.
    :param is_VR: Whether the video is a VR video (default: False).
    """
    # Initialize the video reader
    video_reader = VideoReaderFFmpeg(video_path, is_VR=is_VR)

    # Display the video
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # Release resources
    video_reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Display a video using FFmpeg.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("--is_vr", action="store_true", help="Enable VR mode for processing VR videos.")
    args = parser.parse_args()

    # Run the main function
    main(args.video_path, args.is_vr)