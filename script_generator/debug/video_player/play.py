import math
import os
import tempfile
import time

import cv2
import numpy as np
from scipy.interpolate import interp1d

from script_generator.debug.debug_data import load_debug_metrics
from script_generator.debug.logger import log
from script_generator.debug.video_player.controls import draw_media_controls
from script_generator.debug.video_player.debug_overlay import draw_overlay
from script_generator.debug.video_player.interaction import mouse_callback
from script_generator.debug.video_player.state import VideoPlayer
from script_generator.funscript.util.util import load_funscript_json
from script_generator.utils.file import get_output_file_path
from script_generator.video.data_classes.video_info import get_cropped_dimensions


def play_debug_video(state, start_frame=0, end_frame=None, rolling_window_size=100, save_video_mode=False):
    video_info = state.video_info
    _, metrics, _, _ = load_debug_metrics(state)
    width, height = get_cropped_dimensions(video_info)

    video_player = VideoPlayer(
        state=state,
        start_frame=start_frame,
        end_frame=end_frame
    )

    # Prepare funscript interpolation if available
    funscript_path, _ = get_output_file_path(state.video_path, ".funscript")
    funscript_times, funscript_positions = load_funscript_json(funscript_path)
    funscript_interpolator = interp1d(
        funscript_times,
        funscript_positions,
        kind="linear",
        fill_value="extrapolate"
    )
    funscript_interpolator_ref = None
    if state.reference_script and os.path.exists(state.reference_script):
        funscript_times_ref, funscript_positions_ref = load_funscript_json(state.reference_script)
        funscript_interpolator_ref = interp1d(
            funscript_times_ref,
            funscript_positions_ref,
            kind="linear",
            fill_value="extrapolate"
        )

    # Initialize rolling window buffers
    distance_buffer = np.zeros(rolling_window_size)
    funscript_buffer = np.zeros(rolling_window_size)
    funscript_buffer_ref = np.zeros(rolling_window_size)

    # Initialize video writer if recording
    if save_video_mode:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video_path = temp_file.name
        temp_file.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_fps = video_info.fps
        frame_size = (width, height)
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, output_fps, frame_size)
        log.info(f"Recording debug video to temporary file: {temp_video_path}")
    else:
        # Setup OpenCV window
        window_name = "Debug Video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # zoom in the image is only 640x640 now
        cv2.resizeWindow(window_name, int(width * 2), int(height * 2))

        # Attach mouse callback for seeking
        cv2.setMouseCallback(window_name, mouse_callback, param=video_player)

    last_frame = video_player.current_frame

    def get_ceiled_fps(value):
        try:
            return math.ceil(float(value))
        except (ValueError, TypeError):
            return 60

    while True:
        frame_interval = 1.0 / get_ceiled_fps(state.max_preview_fps)
        loop_start_time = time.time()

        # Handle user input
        if not save_video_mode and not handle_user_input(window_name, video_player):
            break

        if video_player.current_frame >= video_player.end_frame:
            break

        # Check if paused and the frame hasn't changed (allows seeking while being paused)
        if video_player.paused and video_player.current_frame == last_frame:
            # Avoid a tight loop hogging CPU if paused
            time.sleep(0.01)
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
        distance_buffer, funscript_buffer, funscript_buffer_ref = draw_overlay(
            frame=frame,
            frame_id=video_player.current_frame,
            logs=metrics,
            funscript_interpolator=funscript_interpolator,
            funscript_interpolator_ref=funscript_interpolator_ref,
            distance_buffer=distance_buffer,
            funscript_buffer=funscript_buffer,
            funscript_buffer_ref=funscript_buffer_ref,
            fps=video_info.fps
        )

        if save_video_mode:
            # Save the processed frame
            video_writer.write(frame)
        else:
            # Display the frame
            draw_media_controls(frame, video_player)
            cv2.imshow("Debug Video", frame)

            # Throttle the loop to maintain DESIRED_FPS if processing is faster
            elapsed = time.time() - loop_start_time
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

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