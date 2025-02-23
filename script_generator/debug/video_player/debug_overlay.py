import cv2
import numpy as np

from script_generator.constants import CLASS_COLORS
from script_generator.debug.video_player.overlay_widgets import OverlayWidgets
from script_generator.state.app_state import AppState


def get_funscript_value(interpolator, frame_id, fps):
    """
    Returns the interpolated funscript value for a given frame (frame_id) at the given fps.
    """
    time_in_milliseconds = int((frame_id / fps) * 1000)
    return interpolator(time_in_milliseconds)

FONT_SIZE = 0.3

def draw_overlay(
    state: "AppState",
    frame,
    frame_id,
    logs,
    funscript_interpolator,
    funscript_interpolator_ref,
    distance_buffer,
    funscript_buffer,
    funscript_buffer_ref,
    fps
):
    """
    Draws bounding boxes, text overlays, rolling window graphs, etc. on top of `frame`.

    :param frame:                       The current video
    :param frame_id:                    The integer ID of the frame.
    :param logs:                        A dict containing per-frame logs.
    :param funscript_interpolator:      If present, used to compute funscript values.
    :param funscript_interpolator_ref:  If present, used to compute funscript values.
    :param distance_buffer:             Rolling buffer for 'distance' data.
    :param funscript_buffer:            Rolling buffer for funscript values.
    :param funscript_buffer_ref:        Rolling buffer for reference funscript values.
    :param fps:                         The video frames per second.
    """
    # Pull the logs for this frame
    variables, bounding_boxes = {}, []
    if frame_id in logs:
        variables = logs[frame_id].get("variables", {})
        bounding_boxes = logs[frame_id].get("bounding_boxes", [])

    # Display variables
    y_offset_stats = 20
    for key, value in variables.items():
        cv2.putText(
            frame,
            f"{key}: {value}",
            (10, y_offset_stats),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            (0, 0, 255),
            1
        )
        y_offset_stats += 12

    # Draw bounding boxes
    for box in bounding_boxes:
        x1, y1, x2, y2 = box["box"]
        class_name = box["class_name"]
        position = box["position"]
        box_excluded = position is None and state.debug_mode == "funscript"
        color = CLASS_COLORS.get(class_name, (0, 255, 0)) if not box_excluded else (70, 70, 70)
        alpha = 0.5 if box_excluded else 0.8

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

        text = f"{class_name} {position}" if state.debug_mode == "funscript" else f"{class_name} c:{box['conf']} id:{box['track_id']}"
        cv2.putText(
            overlay,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            color,
            1
        )

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if state.debug_mode == "funscript":
        # Draw locked penis box if present
        locked_penis_box = variables.get("locked_penis_box")
        if locked_penis_box and locked_penis_box.get("active", False):
            overlay = frame.copy()
            x1, y1, x2, y2 = locked_penis_box["box"]
            alpha = 0.4
            color = CLASS_COLORS.get("locked_penis", (255, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame,
                "Locked Penis",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                color,
                1
            )
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Compute distance and funscript values for rolling buffer
    distance = variables.get("distance", 0)
    funscript_value, funscript_value_ref = 0, 0
    if funscript_interpolator:
        funscript_value = get_funscript_value(funscript_interpolator, frame_id, fps)
    if funscript_interpolator_ref:
        funscript_value_ref = get_funscript_value(funscript_interpolator_ref, frame_id, fps)

    # Shift rolling window buffers left and insert new values
    distance_buffer = np.roll(distance_buffer, -1)
    distance_buffer[-1] = distance
    funscript_buffer = np.roll(funscript_buffer, -1)
    funscript_buffer[-1] = funscript_value

    if funscript_interpolator_ref:
        funscript_buffer_ref = np.roll(funscript_buffer_ref, -1)
        funscript_buffer_ref[-1] = funscript_value_ref

    # Draw rolling window curves
    graph_height = 100
    graph_y_start = frame.shape[0] - graph_height - 15  # B, G, R | R, G, B
    if state.debug_mode == "funscript":
        # Draw gauge (example usage)
        OverlayWidgets.draw_gauge(frame, funscript_value)

        draw_rolling_window_curve(frame, distance_buffer, (0, 255, 0), 0.5, graph_height, graph_y_start)
        draw_rolling_window_curve(frame, funscript_buffer, (255, 0, 0), 0.5, graph_height, graph_y_start)
    if funscript_interpolator_ref:
        draw_rolling_window_curve(frame, funscript_buffer_ref, (0, 0, 255), 0.5, graph_height, graph_y_start)

    # Return updated buffers in case they need to be maintained externally
    return distance_buffer, funscript_buffer, funscript_buffer_ref


def draw_rolling_window_curve(frame, buffer, color, alpha, graph_height, graph_y_start):
    """
    Draw a rolling window curve on the frame as a transparent overlay.
    """
    overlay = frame.copy()
    height, width, _ = frame.shape
    buffer_clipped = np.clip(buffer, 0, 100)

    for i in range(len(buffer_clipped) - 1):
        x1 = int(width * (i / len(buffer_clipped)))
        x2 = int(width * ((i + 1) / len(buffer_clipped)))
        y1 = graph_y_start + graph_height - int((buffer_clipped[i] / 100) * graph_height)
        y2 = graph_y_start + graph_height - int((buffer_clipped[i + 1] / 100) * graph_height)
        cv2.line(overlay, (x1, y1), (x2, y2), color, 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
