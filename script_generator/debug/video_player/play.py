import json
import tempfile

import cv2
import numpy as np
from scipy.interpolate import interp1d

from script_generator.debug.debug_data import load_logs
from script_generator.debug.video_player.controls import draw_media_controls
from script_generator.debug.video_player.debug_overlay import draw_overlay
from script_generator.debug.video_player.interaction import mouse_callback
from script_generator.debug.video_player.state import VideoPlayer
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger
from script_generator.video.info.video_info import get_cropped_dimensions
from utils.lib_Visualizer import Visualizer


def play_debug_video(state, start_frame=0, end_frame=None, rolling_window_size=100, save_video_mode=False):
    visualizer = Visualizer()
    video_info = state.video_info
    width, height = get_cropped_dimensions(video_info)

    video_player = VideoPlayer(
        video_path=state.video_path,
        video_info=video_info,
        start_frame=start_frame,
        end_frame=end_frame
    )

    # Prepare funscript interpolation if available
    funscript_path, _ = get_output_file_path(state.video_path, ".funscript")
    try:
        with open(funscript_path, "r") as f:
            funscript_data = json.load(f)
        actions = funscript_data.get("actions", [])
        funscript_times = [action["at"] for action in actions]
        funscript_positions = [action["pos"] for action in actions]
        funscript_interpolator = interp1d(
            funscript_times,
            funscript_positions,
            kind="linear",
            fill_value="extrapolate"
        )
    except FileNotFoundError:
        logger.error(f"Funscript file not found at {funscript_path}")
        funscript_interpolator = None

    logs = load_logs(state)

    # Initialize rolling window buffers
    distance_buffer = np.zeros(rolling_window_size)
    funscript_buffer = np.zeros(rolling_window_size)

    # Initialize video writer if recording
    if save_video_mode:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video_path = temp_file.name
        temp_file.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_fps = video_info.fps
        frame_size = (width, height)
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, output_fps, frame_size)
        logger.info(f"Recording debug video to temporary file: {temp_video_path}")
    else:
        # Setup OpenCV window
        window_name = "Debug Video"
        cv2.namedWindow(window_name)

        # Attach mouse callback for seeking
        cv2.setMouseCallback(window_name, mouse_callback, param=video_player)

    last_frame = video_player.current_frame

    while True:
        # Handle user input
        if not save_video_mode and not handle_user_input(window_name, video_player):
            break

        if video_player.current_frame >= video_player.end_frame:
            break

        # Check if paused and the frame hasn't changed (allows seeking while being paused)
        if video_player.paused and video_player.current_frame == last_frame:
            continue

        last_frame = video_player.current_frame

        # Read the next frame
        ret, frame_bgr = video_player.read_frame()
        if not ret:
            # End of video or read error
            break

        # Convert to RGB if needed for your pipeline
        frame = frame_bgr.copy()

        # Call your overlay function
        distance_buffer, funscript_buffer = draw_overlay(
            frame=frame,
            frame_id=video_player.current_frame,
            logs=logs,
            funscript_interpolator=funscript_interpolator,
            distance_buffer=distance_buffer,
            funscript_buffer=funscript_buffer,
            visualizer=visualizer,
            fps=video_info.fps,
            y_offset_start=frame.shape[0] // 3
        )

        if save_video_mode:
            # Save the processed frame
            video_writer.write(frame)
        else:
            # Display the frame
            draw_media_controls(frame, video_player)
            cv2.imshow(window_name, frame)

    if save_video_mode:
        video_player.release()
        return temp_video_path
    else:
        cv2.destroyAllWindows()


def handle_user_input(window_name, video_player):
    key = cv2.waitKey(1) & 0xFF

    # Check if the window has been closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False

    if key == ord("q"):
        return False
    elif key == ord(" "):  # Toggle pause on space bar
        video_player.paused = not video_player.paused

    return True