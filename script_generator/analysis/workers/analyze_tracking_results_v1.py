import json
import time
from datetime import timedelta

import cv2
from tqdm import tqdm

from script_generator.constants import UPDATE_PROGRESS_INTERVAL
from script_generator.constants import CLASS_COLORS
from script_generator.debug.video_player.overlay_widgets import OverlayWidgets
from script_generator.gui.messages.messages import ProgressMessage, UpdateGUIState
from script_generator.object_detection.util.data import load_yolo_data
from script_generator.object_detection.util.object_detection import make_data_boxes, parse_yolo_data_looking_for_penis
from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from script_generator.debug.logger import log, log_tr
from script_generator.video.ffmpeg.video_reader import VideoReaderFFmpeg
from script_generator.video.data_classes.video_info import get_cropped_dimensions
from utils.lib_ObjectTracker import ObjectTracker


def analyze_tracking_results_v1(state: AppState):
    exists, yolo_data, raw_yolo_path, _ = load_yolo_data(state)
    results = make_data_boxes(yolo_data)
    width, height = get_cropped_dimensions(state.video_info)
    list_of_frames = results.get_all_frame_ids()  # Get all frame IDs with detections

    # Looking for the first instance of penis within the YOLO results
    first_penis_frame = parse_yolo_data_looking_for_penis(yolo_data, 0)
    no_penis_frame = False
    if first_penis_frame is None:
        log_tr.error(f"No penis instance found in video. Further tracking is not possible.")
        first_penis_frame = 0
        no_penis_frame = True

    # Deciding whether we start from there or from a user-specified later frame
    state.frame_start_track = max(max(first_penis_frame - int(state.video_info.fps), state.frame_start - int(state.video_info.fps)), 0)
    # state.frame_end = state.video_info.total_frames if not state.frame_end else state.frame_end
    frame_end = state.video_info.total_frames if not state.frame_end else state.frame_end
    log_tr.info(f"Frame Start adjusted to first frame with a penis or penis tip: {state.frame_start_track}")

    video_info = state.video_info
    fps = video_info.fps
    reader = None

    state.frame_area = width * height
    debug_window_open = False
    cuts = []

    """ discarding the scene detection for now
    # Load scene cuts if the file exists
    cuts_path, _ = get_output_file_path(state.video_path, "_cuts.json")
    if os.path.exists(cuts_path):
        logger.info(f"Loading cuts from {cuts_path}")
        with open(cuts_path, 'r') as f:
            cuts = json.load(f)
        logger.info(f"Loaded {len(cuts)} cuts : {cuts}")

        if state.update_ui:
            state.update_ui(ProgressMessage(
                process="SCENE_DETECTION",
                frames_processed=state.video_info.total_frames,
                total_frames=state.frame_end,
                eta="Done"
            ))
    else:
        # Detect scene changes if the cuts file does not exist
        scene_list = detect_scene_changes(state, video_info.is_vr, 0.9, state.frame_start, state.frame_end)
        logger.info(f"Analyzing frames {state.frame_start} to {state.frame_end}")
        cuts = [scene[1] for scene in scene_list]
        cuts = cuts[:-1]  # Remove the last entry
        # Save the cuts to a file
        with open(cuts_path, 'w') as f:
            json.dump(cuts, f)
    """

    state.funscript_frames = []  # List to store Funscript frames
    state.funscript_distances = []
    state.funscript_data = []

    # tracker = ObjectTracker(fps, state.frame_start, image_area, video_info.is_vr)  # Initialize the object tracker
    tracker = ObjectTracker(state)

    # Start time for ETA calculation
    start_time = time.time()

    last_ui_update_time = time.time()
    live_preview_mode_prev = state.live_preview_mode

    for frame_pos in tqdm(
            # range(state.frame_start, state.frame_end), unit="f", desc="Analyzing tracking data", position=0,
            # range(state.frame_start_track, state.frame_end),
            range(state.frame_start_track, frame_end),
            unit="f",
            desc="Analyzing tracking data", position=0,
            unit_scale=False,
            unit_divisor=1
    ):
        state.current_frame_id = frame_pos
        if frame_pos in cuts:
            # Reinitialize the tracker at scene cuts
            log_tr.info(f"Reaching cut at frame {frame_pos}")
            previous_distances = tracker.previous_distances
            log_tr.info(f"Reinitializing tracker with previous distances: {previous_distances}")
            # tracker = ObjectTracker(fps, frame_pos, image_area, video_info.is_vr)
            tracker = ObjectTracker(state)
            tracker.previous_distances = previous_distances

        if frame_pos in list_of_frames:
            # Get sorted boxes based on class priority for the current frame
            sorted_boxes = results.get_boxes(frame_pos)

            # Perform tracking logic
            #if state.frame_start_track >= frame_pos and not no_penis_frame:
            if state.frame_start_track <= frame_pos and not no_penis_frame:
                tracker.tracking_logic(state, sorted_boxes)  # Apply tracking logic

            if tracker and tracker.distance:
                # Append Funscript data if distance is available
                state.funscript_frames.append(frame_pos)
                state.funscript_distances.append(int(tracker.distance))

            if state.save_debug_file:
                # Log debugging information
                bounding_boxes = []
                for box in sorted_boxes:
                    if box[4] in tracker.normalized_absolute_tracked_positions:
                        if box[4] == 0:  # generic track_id for 'hips center'
                            str_dist_penis = 'None'
                        else:
                            if box[4] in tracker.normalized_distance_to_penis:
                                str_dist_penis = str(int(tracker.normalized_distance_to_penis[box[4]][-1]))
                            else:
                                str_dist_penis = 'None'
                        str_abs_pos = str(int(tracker.normalized_absolute_tracked_positions[box[4]][-1]))
                        position = 'p: ' + str_dist_penis + ' | ' + 'a: ' + str_abs_pos
                        if box[4] in tracker.pct_weights and len(tracker.pct_weights[box[4]]) > 0:
                            weight = tracker.pct_weights[box[4]][-1]
                            position += ' | w: ' + str(weight)
                    else:
                        position = None
                    bounding_boxes.append({
                        'box': box[0],
                        'conf': box[1],
                        'class_name': box[3],
                        'track_id': box[4],
                        'position': position,
                    })

                # Add debug information to the debugger class so it can be saved later
                state.debug_data.add_frame(
                    frame_pos,
                    bounding_boxes=bounding_boxes,
                    variables={
                        'frame': frame_pos,
                        # time of the frame hh:mm:ss
                        'time': str(timedelta(seconds=int(frame_pos / fps))),
                        'distance': tracker.distance,
                        'Penetration': tracker.penetration,
                        'sex_position': tracker.sex_position,
                        'sex_position_reason': tracker.sex_position_reason,
                        'tracked_body_part': tracker.tracked_body_part,
                        'locked_penis_box': tracker.locked_penis_box.to_dict(),
                        'glans_detected': tracker.glans_detected,
                        'cons._glans_detections': tracker.consecutive_detections['glans'],
                        'cons._glans_non_detections': tracker.consecutive_non_detections['glans'],
                        'cons._penis_detections': tracker.consecutive_detections['penis'],
                        'cons._penis_non_detections': tracker.consecutive_non_detections['penis'],
                        'breast_tracking': tracker.breast_tracking,
                    }
                )

        # Display object detection tracking results in a live preview window
        window_name = "Tracking analysis preview"

        # Close window and release reader. Also, we don't want to call cv2.getWindowProperty every iteration
        if debug_window_open and not state.live_preview_mode:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_name)
                debug_window_open = False
            if reader:
                reader.release()
                reader = None

        if state.live_preview_mode:
            if not reader:
                reader = VideoReaderFFmpeg(state, frame_pos)
                reader.set_frame(frame_pos)

            ret, frame = reader.read()
            frame = frame.copy()

            for box in tracker.tracked_boxes:
                frame = OverlayWidgets.draw_bounding_box(
                    frame,
                    box[0],
                    str(box[2]) + ": " + box[1],
                    CLASS_COLORS[str(box[1])],
                    state.offset_x
                )

            if tracker.locked_penis_box is not None and tracker.locked_penis_box.is_active():
                frame = OverlayWidgets.draw_bounding_box(
                    frame,
                    tracker.locked_penis_box.box,
                    "Locked_Penis",
                    CLASS_COLORS['penis'],
                    state.offset_x
                )
            else:
                log_tr.debug("No active locked penis box to draw.")

            if tracker.glans_detected:
                frame = OverlayWidgets.draw_bounding_box(
                    frame,
                    tracker.boxes['glans'],
                    "Glans",
                    CLASS_COLORS['glans'],
                    state.offset_x
                )
            if state.funscript_distances:
                frame = OverlayWidgets.draw_gauge(frame, state.funscript_distances[-1])

            # Reinitialize the window if needed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 and state.live_preview_mode:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, int(width * 2), int(height * 2))
                debug_window_open = True
            cv2.imshow(window_name, frame)

            if not handle_user_input(window_name) or not state.live_preview_mode:
                if state.update_ui and state.live_preview_mode:
                    state.update_ui(UpdateGUIState(attr="live_preview_mode", value=False))
                state.live_preview_mode = False

        # Update progress periodically
        if state.update_ui:
            current_time = time.time()
            if current_time - last_ui_update_time >= UPDATE_PROGRESS_INTERVAL:
                last_ui_update_time = current_time
                elapsed_time = current_time - start_time
                frames_processed = frame_pos - state.frame_start + 1
                # frames_remaining = state.frame_end - frame_pos - 1
                frames_remaining = frame_end - frame_pos - 1
                eta = (elapsed_time / frames_processed) * frames_remaining if frames_processed > 0 else 0

                state.update_ui(ProgressMessage(
                    process="TRACKING_ANALYSIS",
                    frames_processed=frames_processed,
                    # total_frames=state.frame_end,
                    total_frames=frame_end,
                    eta=time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float('inf') else "Calculating..."
                ))

    # stop processing when the task is force closed
    if state.analyze_task and state.analyze_task.is_stopped:
        return

    # Final UI update: we're done
    if state.update_ui:
        state.update_ui(ProgressMessage(
            process="TRACKING_ANALYSIS",
            frames_processed=state.video_info.total_frames,
            total_frames=state.video_info.total_frames,
            eta="Done"
        ))

    # Prepare Funscript data
    state.funscript_data = list(zip(state.funscript_frames, state.funscript_distances))

    # Save the raw funscript data to JSON
    raw_funscript_path, _ = get_output_file_path(state.video_path, ".json", "rawfunscript")
    with open(raw_funscript_path, 'w') as f:
        json.dump(state.funscript_data, f)

    return state.funscript_data

def handle_user_input(window_name):
    key = cv2.waitKey(1) & 0xFF

    # Check if the window has been closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False

    if key == ord("q"):
        return False
    return True