import cv2
import numpy as np

from script_generator.constants import CLASS_COLORS
from script_generator.debug.video_player.overlay_widgets import OverlayWidgets


def get_funscript_value(interpolator, frame_id, fps):
    """
    Returns the interpolated funscript value for a given frame (frame_id) at the given fps.
    """
    time_in_milliseconds = int((frame_id / fps) * 1000)
    return interpolator(time_in_milliseconds)

FONT_SIZE = 0.3

def draw_overlay(
    frame,
    frame_id,
    logs,
    funscript_interpolator,
    distance_buffer,
    funscript_buffer,
    fps,
    y_offset_start=0,
):
    """
    Draws bounding boxes, text overlays, rolling window graphs, etc. on top of `frame`.

    :param frame:                   The current video
    :param frame_id:                The integer ID of the frame.
    :param logs:                    A dict containing per-frame logs.
    :param funscript_interpolator:  If present, used to compute funscript values.
    :param distance_buffer:         Rolling buffer for 'distance' data.
    :param funscript_buffer:        Rolling buffer for funscript values.
    :param fps:                     The video frames per second.
    :param y_offset_start:          The initial Y offset for drawing text.
    """
    # Pull the logs for this frame
    str_frame_id = str(frame_id)
    variables, bounding_boxes = {}, []
    if str_frame_id in logs:
        variables = logs[str_frame_id].get("variables", {})
        bounding_boxes = logs[str_frame_id].get("bounding_boxes", [])

    # Draw bounding boxes
    for box in bounding_boxes:
        x1, y1, x2, y2 = box["box"]
        class_name = box["class_name"]
        position = box["position"]
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame,
            f"{class_name} {position}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            color,
            1
        )

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

    # Draw locked penis box if present
    locked_penis_box = variables.get("locked_penis_box")
    if locked_penis_box and locked_penis_box.get("active", False):
        x1, y1, x2, y2 = locked_penis_box["box"]
        color = CLASS_COLORS.get("penis", (0, 255, 0))
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

    # Compute distance and funscript values for rolling buffer
    distance = variables.get("distance", 0)
    funscript_value = 0
    if funscript_interpolator:
        funscript_value = get_funscript_value(funscript_interpolator, frame_id, fps)

    # Draw gauge (example usage)
    OverlayWidgets.draw_gauge(frame, funscript_value)

    # Shift rolling window buffers left and insert new values
    distance_buffer = np.roll(distance_buffer, -1)
    distance_buffer[-1] = distance
    funscript_buffer = np.roll(funscript_buffer, -1)
    funscript_buffer[-1] = funscript_value

    # Draw rolling window curves
    graph_height = int(frame.shape[0] * 0.2)
    graph_y_start = y_offset_start
    draw_rolling_window_curve(frame, distance_buffer, (0, 255, 0), 0.5, graph_height, graph_y_start)
    draw_rolling_window_curve(frame, funscript_buffer, (255, 0, 0), 0.5, graph_height, graph_y_start)

    # Return updated buffers in case they need to be maintained externally
    return distance_buffer, funscript_buffer


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
        cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


