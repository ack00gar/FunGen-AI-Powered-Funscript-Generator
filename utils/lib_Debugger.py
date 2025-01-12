import json
import time
import cv2
import numpy as np
from params.config import class_colors
from scipy.interpolate import interp1d
from utils.lib_Visualizer import Visualizer
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg


class Debugger:
    """
    A class for debugging video frames by logging variables, bounding boxes, and visualizing them.
    """

    def __init__(self, video_path, isVR=False, video_reader=None, output_dir=None):
        """
        Initialize the Debugger.
        :param video_path: Path to the video file.
        :param output_dir: Directory to save debug logs.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.logs = {}  # Dictionary to store logs for each frame
        self.cap = None  # Video capture object
        self.current_frame = 0  # Current frame being processed
        self.total_frames = 0  # Total number of frames in the video
        self.fps = 0  # Frames per second of the video
        self.bar_y_start = 0  # Y-coordinate of the progress bar
        self.isVR = isVR
        self.video_reader = video_reader

    def log_frame(self, frame_id, variables=None, bounding_boxes=None):
        """
        Log or update the state of variables and bounding boxes for a specific frame.
        :param frame_id: The frame ID to log.
        :param variables: Dictionary of variables to log.
        :param bounding_boxes: List of bounding boxes to log.
        """
        if frame_id not in self.logs:
            self.logs[frame_id] = {"variables": {}, "bounding_boxes": []}

        if variables:
            self.logs[frame_id]["variables"].update(variables)

        if bounding_boxes:
            self.logs[frame_id]["bounding_boxes"].extend(bounding_boxes)

    def save_logs(self):
        """
        Save the logs to a JSON file.
        """
        log_file = f"{self.output_dir}_debug_logs.json"

        def default(obj):
            """Convert NumPy types to native Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(log_file, "w") as f:
            json.dump(self.logs, f, indent=4, default=default)
        print(f"Logs saved to {log_file}")

    def load_logs(self):
        """
        Load existing logs from a JSON file.
        """
        log_file = f"{self.output_dir}_debug_logs.json"
        try:
            with open(log_file, "r") as f:
                self.logs = json.load(f)
            print(f"Logs loaded from {log_file}")
        except FileNotFoundError:
            print(f"Log file {log_file} not found. Starting with empty logs.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {log_file}. Starting with empty logs.")

    def display_frame(self, frame_id):
        """
        Display the logged frame with bounding boxes and variable states.
        :param frame_id: The frame ID to display.
        """
        self.play_video(frame_id, duration=-1)

    def play_video(self, start_frame=0, duration=0, rolling_window_size=100, record=False, downsize_ratio=1):
        """
        Play the video from a specified frame, displaying variables, bounding boxes, and rolling window curves.
        :param start_frame: Frame to start playback from.
        :param duration: Duration of playback in seconds.
        :param rolling_window_size: Size of the rolling window for distance and funscript data.
        :param record: Whether to record the debug video.
        :param downsize_ratio: Ratio to downsize the recorded video.
        """
        visualizer = Visualizer()

        # Load the video
        if self.video_reader == "FFmpeg":
            self.cap = VideoReaderFFmpeg(self.video_path, is_VR=self.isVR)
        else:
            self.cap = cv2.VideoCapture(self.video_path)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {self.total_frames}, FPS: {self.fps}")

        # Initialize video writer if recording
        if record:
            ret, frame = self.cap.read()
            if self.video_reader == "OpenCV" and self.isVR:
                frame = frame[:, :frame.shape[1] // 2, :]  # only half left of the frame, for VR half
            #if self.cap.is_VR:
            #    frame_copy = frame[:, frame.shape[1] // 3 : 2 * frame.shape[1] // 3, :]
            output_path = self.video_path.replace(".mp4", "_debug.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path, fourcc, self.fps, (frame.shape[1] // downsize_ratio, frame.shape[0] // downsize_ratio)
            )
            if not out.isOpened():
                print(f"Error: Could not open video writer for {output_path}")
                self.cap.release()
                return

        # Set the starting frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Load the funscript file
        funscript_path = self.video_path.replace(".mp4", ".funscript")
        try:
            with open(funscript_path, "r") as f:
                funscript_data = json.load(f)
            actions = funscript_data.get("actions", [])
            funscript_times = [action["at"] for action in actions]
            funscript_positions = [action["pos"] for action in actions]
            funscript_interpolator = interp1d(funscript_times, funscript_positions, kind="linear", fill_value="extrapolate")
        except FileNotFoundError:
            print(f"Funscript file not found at {funscript_path}")
            funscript_interpolator = None

        # Initialize rolling window buffers
        half_window = rolling_window_size // 2
        distance_buffer = np.zeros(rolling_window_size)
        funscript_buffer = np.zeros(rolling_window_size)

        # Calculate end frame
        self.current_frame = start_frame
        end_frame = start_frame + int(duration * self.fps) if duration > 0 else self.total_frames

        # Set up mouse callback for progress bar
        if self.video_reader == "OpenCV" and self.isVR:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
        else:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        cv2.namedWindow("Debug Video")
        cv2.setMouseCallback("Debug Video", self._mouse_callback, param=width)

        # Play the video
        while self.current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.video_reader == "OpenCV" and self.isVR:
                frame = frame[:, :frame.shape[1] // 2, :]  # only half left of the frame, for VR half
            frame_copy = frame.copy()  # make a copy of the frame to make it writeable, useful for ffmpeg library here
            # Display variables and bounding boxes
            str_frame_id = str(self.current_frame)
            if str_frame_id in self.logs:
                variables = self.logs[str_frame_id]["variables"]
                bounding_boxes = self.logs[str_frame_id]["bounding_boxes"]

                # Draw bounding boxes
                for box in bounding_boxes:
                    x1, y1, x2, y2 = box["box"]
                    class_name = box["class_name"]
                    position = box["position"]
                    color = class_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_copy, f"{class_name} {position}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display variables
                y_offset = frame.shape[0] // 3   # 30
                for key, value in variables.items():
                    cv2.putText(frame_copy, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    y_offset += 20

                # Draw the locked_penis_box if it exists
                locked_penis_box = variables.get("locked_penis_box")
                #print(f"locked_penis_box: {locked_penis_box}")
                if locked_penis_box['active']:
                    x1, y1, x2, y2 = locked_penis_box['box']
                    color = class_colors.get("penis", (0, 255, 0))
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_copy, "Locked Penis", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            try:
                # Update rolling window buffers
                # Update rolling window buffers
                distance = variables.get("distance", 0)
                funscript_value = self._get_funscript_value(funscript_interpolator, self.current_frame,
                                                            self.fps) if funscript_interpolator else 0
                visualizer.draw_gauge(frame_copy, funscript_value)
                # Shift buffers to the left and add new values at the center
                distance_buffer = np.roll(distance_buffer, -1)
                distance_buffer[-1] = distance
                funscript_buffer = np.roll(funscript_buffer, -1)
                funscript_buffer[-1] = funscript_value

                # Draw rolling window curves
                graph_height = int(frame_copy.shape[0] * 0.2)
                graph_y_start = y_offset + 10
                self._draw_rolling_window_curve(frame_copy, distance_buffer, (0, 255, 0), 0.5, graph_height, graph_y_start)
                self._draw_rolling_window_curve(frame_copy, funscript_buffer, (255, 0, 0), 0.5, graph_height, graph_y_start)
            except:
                # no variables logged at this frame
                pass
            # Draw progress bar
            self._draw_progress_bar(frame_copy, frame.shape[1], frame.shape[0])

            # Show the frame
            cv2.imshow("Debug Video", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.waitKey(1) & 0xFF == 32:  # Pause on spacebar
                # press a key to resume
                cv2.waitKey(0)
            if duration == -1:
                cv2.imwrite(f"{self.video_path[:-4]}_frame_{self.current_frame}.png", frame_copy)
                break

            # Record the frame if enabled
            if record:
                frame_copy = cv2.resize(frame_copy, (frame_copy.shape[1] // downsize_ratio, frame_copy.shape[0] // downsize_ratio))
                out.write(frame_copy)

            self.current_frame += 1

        # Release resources
        self.cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()

    def _get_funscript_value(self, interpolator, frame_id, fps):
        """
        Get the interpolated funscript value for a given frame.
        :param interpolator: Interpolation function for funscript data.
        :param frame_id: The frame ID.
        :param fps: Frames per second of the video.
        :return: Interpolated funscript value.
        """
        time_in_milliseconds = int((frame_id / fps) * 1000)
        return interpolator(time_in_milliseconds)

    def _draw_rolling_window_curve(self, frame, buffer, color, alpha, graph_height, graph_y_start):
        """
        Draw a rolling window curve on the frame as a transparent overlay.
        :param frame: The frame to draw on.
        :param buffer: The buffer of values to plot.
        :param color: The color of the curve.
        :param alpha: Transparency of the overlay.
        :param graph_height: Height of the graph.
        :param graph_y_start: Y-coordinate to start the graph.
        """
        overlay = frame.copy()
        height, width, _ = frame.shape
        buffer_clipped = np.clip(buffer, 0, 100)

        for i in range(len(buffer_clipped) - 1):
            x1 = int(width * (i / len(buffer_clipped)))
            x2 = int(width * ((i + 1) / len(buffer_clipped)))
            y1 = graph_y_start + graph_height - int((buffer_clipped[i] / 100) * graph_height)
            y2 = graph_y_start + graph_height - int((buffer_clipped[i + 1] / 100) * graph_height)
            cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for video navigation.
        :param event: The mouse event.
        :param x: X-coordinate of the mouse.
        :param y: Y-coordinate of the mouse.
        :param flags: Additional flags.
        :param param: Additional parameters (width of the frame).
        """
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.bar_y_start:
            self._update_frame_from_mouse(x, param)

    def _update_frame_from_mouse(self, x, width):
        """
        Update the current frame based on the mouse's X position.
        :param x: X-coordinate of the mouse.
        :param width: Width of the frame.
        """
        self.current_frame = int((x / width) * self.total_frames)
        print(f"Target frame: {self.current_frame}")
        if self.video_reader == "FFmpeg":
            self.cap.release()
            self.cap = VideoReaderFFmpeg(self.video_path, self.isVR)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        print("Done resetting and jumping to target frame")

    def _draw_progress_bar(self, frame, width, height):
        """
        Draw a progress bar on the frame indicating the current playback position.
        :param frame: The frame to draw on.
        :param width: Width of the frame.
        :param height: Height of the frame.
        """
        bar_height = 20
        bar_x_start = 0
        bar_x_end = width
        bar_y_start = height - bar_height
        self.bar_y_start = bar_y_start
        bar_y_end = height

        # Background of the progress bar
        cv2.rectangle(frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), (50, 50, 50), -1)

        # Progress indicator
        progress_x = int((self.current_frame / self.total_frames) * width)
        cv2.rectangle(frame, (bar_x_start, bar_y_start), (progress_x, bar_y_end), (0, 255, 0), -1)
