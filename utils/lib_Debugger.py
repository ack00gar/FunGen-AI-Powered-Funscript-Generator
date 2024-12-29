import json
import time

import cv2
import numpy as np
from utils.config import class_colors
from scipy.interpolate import interp1d
from utils.lib_Visualizer import Visualizer
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg

class Debugger:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.logs = {}

        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0

    def log_frame(self, frame_id, variables=None, bounding_boxes=None):
        """
        Log or update the state of variables and bounding boxes for a specific frame.
        """
        if frame_id not in self.logs:
            self.logs[frame_id] = {
                'variables': {},
                'bounding_boxes': []
            }

        if variables:
            self.logs[frame_id]['variables'].update(variables)

        if bounding_boxes:
            self.logs[frame_id]['bounding_boxes'].extend(bounding_boxes)

    def save_logs(self):
        """
        Save the logs to a JSON file.
        """
        log_file = f"{self.output_dir}_debug_logs.json"

        # Custom function to convert NumPy types to native Python types
        def default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Save logs with the custom default function
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=4, default=default)
        print(f"Logs saved to {log_file}")

    def load_logs(self):
        """
        Load existing logs from a JSON file.
        """
        log_file = f"{self.output_dir}_debug_logs.json"
        try:
            with open(log_file, 'r') as f:
                self.logs = json.load(f)
            print(f"Logs loaded from {log_file}")
        except FileNotFoundError:
            print(f"Log file {log_file} not found. Starting with empty logs.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {log_file}. Starting with empty logs.")

    def display_frame(self, frame_id):
        """
        Display the logged frame with bounding boxes and variable states.
        """
        str_frame_id = str(frame_id)
        if str_frame_id not in self.logs:
            print(f"No logs found for frame {frame_id}")
            return

        # Load the frame from the video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Failed to read frame {frame_id}")
            return

        frame = frame[:, frame.shape[1] // 6:frame.shape[1] // 3, :]

        # Draw bounding boxes with color coding
        for box in self.logs[str_frame_id]['bounding_boxes']:
            x1, y1, x2, y2 = box['box']
            class_name = box['class_name']
            conf = box['conf']

            print(f"{class_name}: conf {conf}, box: [{x1}, {y1}, {x2}, {y2}]")

            # Get the color for the class from the class_colors dictionary
            color = class_colors.get(class_name, (0, 255, 0))  # Default to green if class not found

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add the class name and confidence score as text
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display variables line by line
        variables = self.logs[str_frame_id]['variables']
        y_offset = 30  # Initial vertical offset for the first variable
        for key, value in variables.items():
            variable_text = f"{key}: {value}"
            print(variable_text)
            cv2.putText(frame, variable_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 20  # Move down by 20 pixels for the next variable

        # Draw the locked_penis_box if it exists
        locked_penis_box = variables.get('locked_penis_box')
        if locked_penis_box:
            x1, y1, x2, y2 = locked_penis_box
            color = class_colors.get('penis', (0, 255, 0))  # Use the color for "penis" or default to green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Locked Penis", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frame
        cv2.imshow(f"Debug Frame {frame_id}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def debug_frame(self, frame_id, variables, bounding_boxes):
        """
        Log and display the frame for debugging.
        """
        self.log_frame(frame_id, variables, bounding_boxes)
        self.display_frame(frame_id)

    def play_video(self, start_frame=0, duration=0, rolling_window_size=100, record=False, downsize_ratio=1):
        """
        Play the video from a specified frame or timestamp, displaying variables, bounding boxes,
        and rolling window curves for distance and funscript data.
        """

        visualizer = Visualizer()

        # Load the video
        #cap = cv2.VideoCapture(self.video_path)
        self.cap = VideoReaderFFmpeg(self.video_path, is_VR=True)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {self.total_frames}, FPS: {self.fps}")

        if record:
            ret, frame = self.cap.read()
            if self.cap.is_VR:
                frame_copy = frame.copy()
                frame_copy = frame_copy[:, frame.shape[1] // 3:2 * frame.shape[1] // 3, :]
            # Define the output video writer
            output_path = self.video_path.replace(".mp4", "_debug.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (frame_copy.shape[1] // downsize_ratio, frame_copy.shape[0] // downsize_ratio))

            if not out.isOpened():
                print(f"Error: Could not open video writer for {output_path}")
                self.cap.release()
                return

        # Set the starting frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Load the funscript file
        funscript_path = self.video_path.replace(".mp4", ".funscript")
        try:
            with open(funscript_path, 'r') as f:
                funscript_data = json.load(f)
        except FileNotFoundError:
            print(f"Funscript file not found at {funscript_path}")
            funscript_data = None

        # Initialize rolling window buffers
        distance_buffer = np.zeros(rolling_window_size)
        funscript_buffer = np.zeros(rolling_window_size)

        # Prepare funscript interpolation
        if funscript_data:
            actions = funscript_data.get('actions', [])
            funscript_times = [action['at'] for action in actions]
            funscript_positions = [action['pos'] for action in actions]
            print(f"len of funscript_times: {len(funscript_times)}")
            funscript_interpolator = interp1d(funscript_times, funscript_positions, kind='linear',
                                              fill_value="extrapolate")

        self.current_frame = start_frame
        if duration > 0:
            end_frame = start_frame + int(duration * self.fps)  # Calculate the end frame based on duration
        else:
            end_frame = self.total_frames

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        if self.cap.is_VR:
            width = width // 6

        cv2.namedWindow("Debug Video")
        cv2.setMouseCallback("Debug Video", self._mouse_callback,
                             param=width)

        while self.current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Crop the frame (as done in display_frame)
            #frame = frame[:, frame.shape[1] // 6:frame.shape[1] // 3, :]
            if self.cap.is_VR:
                frame_copy = frame.copy()
                frame_copy = frame_copy[:, frame.shape[1] // 3:2 * frame.shape[1] // 3, :]

            # Display variables and bounding boxes
            str_frame_id = str(self.current_frame)
            if str_frame_id in self.logs:
                variables = self.logs[str_frame_id]['variables']
                bounding_boxes = self.logs[str_frame_id]['bounding_boxes']

                # Draw bounding boxes
                for box in bounding_boxes:
                    x1, y1, x2, y2 = box['box']
                    class_name = box['class_name']
                    conf = box['conf']
                    track_id = box['track_id']
                    position = box['position']
                    color = class_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_copy, f"{class_name} {position}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display variables
                y_offset = 30
                for key, value in variables.items():
                    variable_text = f"{key}: {value}"
                    cv2.putText(frame_copy, variable_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    y_offset += 20

                # Draw the locked_penis_box if it exists
                locked_penis_box = variables.get('locked_penis_box')
                if locked_penis_box:
                    x1, y1, x2, y2 = locked_penis_box
                    color = class_colors.get('penis', (0, 255, 0))
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_copy, "Locked Penis", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update rolling window buffers
            distance = variables.get('distance', 0)
            funscript_value = self._get_funscript_value(funscript_interpolator, self.current_frame, self.fps) if funscript_data else 0
            #print(f"Funscript value: {funscript_value}")
            visualizer.draw_gauge(frame_copy, funscript_value)
            distance_buffer = np.roll(distance_buffer, -1)
            distance_buffer[-1] = distance
            funscript_buffer = np.roll(funscript_buffer, -1)
            funscript_buffer[-1] = funscript_value

            # Draw rolling window curves
            graph_height = int(frame_copy.shape[0] * 0.2)  # 20% of the frame height
            graph_y_start = y_offset + 10  # Start the graph right under the variables
            self._draw_rolling_window_curve(frame_copy, distance_buffer, (0, 255, 0), 0.5, graph_height, graph_y_start)
            self._draw_rolling_window_curve(frame_copy, funscript_buffer, (255, 0, 0), 0.5, graph_height, graph_y_start)

            self._draw_progress_bar(frame_copy, frame.shape[1], frame.shape[0])

            # Show the frame
            cv2.imshow(f"Debug Video", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if spacebar is pressed, pause
            if cv2.waitKey(1) & 0xFF == 32:
                time.sleep(10)

            if record:
                # resize the frame, divide dimensions by 2
                frame_copy = cv2.resize(frame_copy, (frame_copy.shape[1] // downsize_ratio, frame_copy.shape[0] // downsize_ratio))
                out.write(frame_copy)

            self.current_frame += 1

        self.cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()

    def _get_funscript_value(self, interpolator, frame_id, fps):
        """
        Get the interpolated funscript value for a given frame.
        """
        time_in_milliseconds = int((frame_id / fps) * 1000)
        return interpolator(time_in_milliseconds)

    def _draw_rolling_window_curve(self, frame, buffer, color, alpha, graph_height, graph_y_start):
        """
        Draw a rolling window curve on the frame as a transparent overlay with a fixed Y-axis range of 0 to 100.
        """
        overlay = frame.copy()
        height, width, _ = frame.shape

        # Ensure the buffer values are within the range 0 to 100
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
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # Start dragging
            if y >= self.bar_y_start:  # Clicked on the progress bar
                self._update_frame_from_mouse(x, param)
        """
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:  # Dragging
            self._update_frame_from_mouse(x, param)

        elif event == cv2.EVENT_LBUTTONUP:  # End dragging
            self.is_dragging = False
            self._update_frame_from_mouse(x, param)
        """

    def _update_frame_from_mouse(self, x, width):
        """
        Update the current frame based on the mouse's X position.
        """
        self.current_frame = int((x / width) * self.total_frames)
        print(f"Target frame: {self.current_frame}")
        self.cap.release()
        print("Release cap")
        #self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
        self.cap = VideoReaderFFmpeg(self.video_path, is_VR=True)
        print("Set up new cap")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        print("Done resetting and jumping to target frame")

    def _draw_progress_bar(self, frame, width, height):
        """
        Draw a progress bar on the frame indicating the current playback position.
        """
        # Ensure the frame is writable
        # frame = np.ascontiguousarray(frame)

        bar_height = 10
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


