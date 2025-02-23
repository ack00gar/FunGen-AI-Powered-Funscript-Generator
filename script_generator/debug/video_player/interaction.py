from typing import TYPE_CHECKING, Tuple

import cv2

from script_generator.constants import LEFT, RIGHT, SPACE, Q, COMMA, PERIOD, LEFT_BRACKET, RIGHT_BRACKET
from script_generator.debug.logger import log
from script_generator.gui.messages.messages import UpdateGUIState
from script_generator.video.data_classes.video_info import get_cropped_dimensions

if TYPE_CHECKING:
    from script_generator.state.app_state import AppState
    from script_generator.debug.video_player.video_player import VideoPlayer


def mouse_callback(event, x, y, flags, params: Tuple["AppState", "VideoPlayer"]):
    """
    Called when the user clicks on the OpenCV window.
    We pass in the video player via param for seeking.
    """
    state, video_player = params
    video_info = state.video_info

    if event == cv2.EVENT_LBUTTONDOWN:
        width, height = get_cropped_dimensions(video_info)
        bar_y_start = height - 10

        # If clicked in the progress bar area, adjust the seek
        if y >= bar_y_start:

            # allow seeking when a static debug frame is set but pause the video
            if state.static_debug_frame:
                state.static_debug_frame = None
                if state.update_ui:
                    state.update_ui(UpdateGUIState(attr="static_debug_frame", value=None))
                video_player.paused = True

            # video_player.funscript_graph.reset()

            target_frame = int((x / width) * video_player.total_frames)
            video_player.set_frame(target_frame)
            log.info(f"Seek to frame: {target_frame}")


def handle_user_input(window_name, state: "AppState", video_player: "VideoPlayer", metrics):
    key = cv2.waitKeyEx(1)

    if key == -1:
        return True

    log.info(f"Key pressed: {key}")

    # Check if the window has been closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False

    if key == Q:
        return False

    elif key == SPACE:  # Toggle pause on space bar
        video_player.paused = not video_player.paused

        if not video_player.paused and state.static_debug_frame:
            state.static_debug_frame = None
            if state.update_ui:
                state.update_ui(UpdateGUIState(attr="static_debug_frame", value=None))

    elif key == COMMA:
        video_player.set_frame(video_player.current_frame - int(video_player.video_info.fps * 2))

    elif key == PERIOD:
        video_player.set_frame(video_player.current_frame + int(video_player.video_info.fps * 2))

    elif key == RIGHT:
        seek_to_position(metrics, video_player, True)

    elif key == LEFT:
        seek_to_position(metrics, video_player, False)

    elif key == LEFT_BRACKET:
        video_player.set_frame(max(video_player.current_frame - 1, 0))

    elif key == RIGHT_BRACKET:
        video_player.set_frame(min(video_player.current_frame + 1, video_player.total_frames - 1))

    return True


def seek_to_position(metrics, video_player: "VideoPlayer", is_next: bool = True):
    positions = metrics.get("positions", [])
    tolerance = int(video_player.video_info.fps) if not video_player.paused else 2
    seek_frame = None

    if is_next:
        curr_position = next((p for p in positions if int(p.get("start", 0)) <= video_player.current_frame < int(p.get("end", 0))), None)
    else:
        curr_position = next((p for p in positions if int(p.get("start", 0)) + tolerance * 2 <= video_player.current_frame + (tolerance if not video_player.paused else -1) < int(p.get("end", 0))), None)


    # Currently inside a position?
    if curr_position:
        seek_frame = curr_position['end'] if is_next else curr_position['start']

    # Not inside position?
    else:
        # Find the next or previous position
        sorted_positions = sorted(positions, key=lambda x: int(x['start']))
        if is_next:
            next_position = next((p for p in sorted_positions if int(p['start']) > video_player.current_frame), None)
            if next_position:
                seek_frame = next_position['start']
        else:
            prev_position = next((p for p in reversed(sorted_positions) if int(p['end']) < video_player.current_frame), None)
            if prev_position:
                tol = tolerance if not video_player.paused else -1
                if prev_position['end'] + tol >= video_player.current_frame:
                    seek_frame = prev_position['start']
                else:
                    seek_frame = prev_position['end']

    if seek_frame is not None:
        video_player.set_frame(seek_frame)
