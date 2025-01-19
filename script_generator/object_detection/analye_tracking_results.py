import json
import os
import time
from datetime import datetime

import cv2
from tqdm import tqdm
from datetime import timedelta

from config import CLASS_COLORS
from script_generator.gui.messages.messages import ProgressMessage
from script_generator.utils.file import get_output_file_path
from script_generator.video.video_info import get_cropped_dimensions
from utils.lib_ObjectTracker import ObjectTracker
from utils.lib_SceneCutsDetect import detect_scene_changes
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg
from utils.lib_Visualizer import Visualizer


def analyze_tracking_results(state, results, progress_callback=None):
    width, height = get_cropped_dimensions(state.video_info)
    list_of_frames = results.get_all_frame_ids()  # Get all frame IDs with detections
    visualizer = Visualizer()  # Initialize the visualizer

    video_info = state.video_info
    fps = video_info.fps
    nb_frames = video_info.total_frames
    if state.frame_start and state.frame_end:
        nb_frames = state.frame_end - state.frame_start
    frame_width, frame_height = get_cropped_dimensions(video_info)
    reader = VideoReaderFFmpeg(state.video_path, is_vr=video_info.is_vr)

    image_area = video_info.width * video_info.height

    cuts = []

    if state.life_display_mode:
        reader.set(cv2.CAP_PROP_POS_FRAMES, state.frame_start)
    else:
        reader.release()

    # Load scene cuts if the file exists
    cuts_path, _ = get_output_file_path(state.video_path, "_cuts.json")
    if os.path.exists(cuts_path):
        print(f"Loading cuts from {cuts_path}")
        with open(cuts_path, 'r') as f:
            cuts = json.load(f)
        print(f"Loaded {len(cuts)} cuts : {cuts}")
    else:
        # Detect scene changes if the cuts file does not exist
        scene_list = detect_scene_changes(state.video_path, video_info.is_vr, 0.9, state.frame_start, state.frame_end)
        print(f"Analyzing frames {state.frame_start} to {state.frame_end}")
        cuts = [scene[1] for scene in scene_list]
        cuts = cuts[:-1]  # Remove the last entry
        # Save the cuts to a file
        with open(cuts_path, 'w') as f:
            json.dump(cuts, f)

    state.funscript_frames = []  # List to store Funscript frames
    tracker = ObjectTracker(fps, state.frame_start, image_area, video_info.is_vr)  # Initialize the object tracker

    # Start time for ETA calculation
    start_time = time.time()

    for frame_pos in tqdm(range(state.frame_start, state.frame_end), unit="f"):
        if frame_pos in cuts:
            # Reinitialize the tracker at scene cuts
            print(f"Reaching cut at frame {frame_pos}")
            previous_distances = tracker.previous_distances
            print(f"Reinitializing tracker with previous distances: {previous_distances}")
            tracker = ObjectTracker(fps, frame_pos, image_area, video_info.is_vr)
            tracker.previous_distances = previous_distances

        if frame_pos in list_of_frames:
            # Get sorted boxes for the current frame
            sorted_boxes = results.get_boxes(frame_pos)
            tracker.tracking_logic(sorted_boxes, frame_pos, height)  # Apply tracking logic

            if tracker.distance:
                # Append Funscript data if distance is available
                state.funscript_frames.append(frame_pos)
                state.funscript_distances.append(int(tracker.distance))

            if state.debug_mode:
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
                        if box[4] in tracker.pct_weights:
                            if len(tracker.pct_weights[box[4]]) > 0:
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
                state.debugger.log_frame(
                    frame_pos,
                    bounding_boxes=bounding_boxes,
                    variables={
                        'frame': frame_pos,
                        # time of the frame hh:mm:ss
                        'time': datetime.fromtimestamp(frame_pos / fps).strftime('%H:%M:%S'),
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

        if state.life_display_mode:
            # Display the tracking results for testing
            ret, frame = reader.read()

            if state.video_reader == "OpenCV" and video_info.is_vr:
                frame_display = frame[:, :frame.shape[1] // 2, :]  # only half left of the frame, for VR half
            else:
                frame_display = frame.copy()

            for box in tracker.tracked_boxes:
                frame_display = visualizer.draw_bounding_box(
                    frame_display,
                    box[0],
                    str(box[2]) + ": " + box[1],
                    CLASS_COLORS[str(box[1])],
                    state.offset_x
                )
            if tracker.locked_penis_box is not None and tracker.locked_penis_box.is_active():
                frame_display = visualizer.draw_bounding_box(
                    frame_display, tracker.locked_penis_box.box,
                    "Locked_Penis",
                    CLASS_COLORS['penis'],
                    state.offset_x
                )
            else:
                print("No active locked penis box to draw.")

            if tracker.glans_detected:
                frame_display = visualizer.draw_bounding_box(
                    frame_display, tracker.boxes['glans'],
                    "Glans",
                    CLASS_COLORS['glans'],
                    state.offset_x
                )
            if state.funscript_distances:
                frame_display = visualizer.draw_gauge(frame_display, state.funscript_distances[-1])

            cv2.imshow("Combined Results", frame_display)
            cv2.waitKey(1)

        # Update progress
        if progress_callback:
            elapsed_time = time.time() - start_time
            frames_processed = frame_pos - state.frame_start + 1
            frames_remaining = state.frame_end - frame_pos - 1
            eta = (elapsed_time / frames_processed) * frames_remaining if frames_processed > 0 else 0

            state.update_ui(ProgressMessage(
                process="TRACKING_ANALYSIS",
                frames_processed=frames_processed,
                total_frames=state.frame_end,
                eta=time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float('inf') else "Calculating..."
            ))

            # progress_callback(frame_pos, frame_end, time.strftime("%H:%M:%S", time.gmtime(eta)))

    # Prepare Funscript data
    state.funscript_data = list(zip(state.funscript_frames, state.funscript_distances))

    points = "["
    for i in range(len(state.funscript_frames)):
        if i != 0:
            points += ","
        points += f"[{state.funscript_frames[i]}, {state.funscript_distances[i]}]"
    points += "]"
    # Write the raw Funscript data to a JSON file
    raw_funscript_path, _ = get_output_file_path(state.video_path, "_rawfunscript.json")
    with open(raw_funscript_path, 'w') as f:
        json.dump(state.funscript_data, f)
    return state.funscript_data