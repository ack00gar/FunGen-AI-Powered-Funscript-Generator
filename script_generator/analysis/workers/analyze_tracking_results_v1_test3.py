import numpy as np
import cv2
from tqdm import tqdm
from typing import Tuple, List, Dict
from datetime import timedelta
from scipy.signal import savgol_filter, find_peaks
from simplification.cutil import simplify_coords_vw  #, simplify_coords
from copy import deepcopy

from script_generator.state.app_state import AppState
from script_generator.object_detection.util.data import load_yolo_data
from script_generator.object_detection.util.object_detection import make_data_boxes
from script_generator.video.data_classes.video_info import get_cropped_dimensions
from script_generator.debug.logger import log, log_tr
# from utils.lib_ObjectTracker import ObjectTracker
from script_generator.utils.file import get_output_file_path
from script_generator.funscript.util.util import write_funscript


def _assign_frame_position(contacts: List[Dict]) -> str:
    """
    Assign a major position to a frame based on the detected body parts in contact.

    Args:
        contacts: List of contacts for the frame, each containing a "class_name".

    Returns:
        The most likely major position for the frame.
    """
    if not contacts:
        return "Not relevant"

    # Count the occurrences of each detected body part
    detected_parts = [contact["class_name"] for contact in contacts]

    most_likely_position = 'Not relevant'

    if 'pussy' in detected_parts:
        most_likely_position = 'Cowgirl / Missionary'
    elif 'butt' in detected_parts:
        most_likely_position = 'Rev. Cowgirl / Doggy'
    elif 'face' in detected_parts and 'hand' in detected_parts:
        #most_likely_position = 'Blowjob'
        most_likely_position = 'Handjob / Blowjob'
    elif 'face' in detected_parts:
        #most_likely_position = 'Blowjob'
        most_likely_position = 'Handjob / Blowjob'
    elif 'hand' in detected_parts and 'breast' in detected_parts:
        most_likely_position = 'Boobjob'
    elif 'hand' in detected_parts:
        #most_likely_position = 'Handjob'
        most_likely_position = 'Handjob / Blowjob'
    elif 'foot' in detected_parts:
        most_likely_position = 'Footjob'
    elif 'breast' in detected_parts:
        most_likely_position = 'Boobjob'

    return most_likely_position

def _aggregate_segments(frame_positions: List[str], fps: float, min_segment_duration: float = 10.0) -> List[Dict]:
    """
    Aggregate frames into segments based on consistent positions. A new segment is created when the position
    changes consistently. Each segment must be at least `min_segment_duration` seconds long.

    Args:
        frame_positions: List of positions for each frame.
        fps: Frames per second of the video.
        min_segment_duration: Minimum duration of a segment in seconds.

    Returns:
        A list of segments, each with:
        - start_frame: Start frame of the segment.
        - end_frame: End frame of the segment.
        - start_time: Start time of the segment.
        - end_time: End time of the segment.
        - major_position: Major position of the segment.
    """
    segments = []
    current_segment = None
    min_frames_per_segment = int(min_segment_duration * fps)

    for frame_pos, position in enumerate(frame_positions):
        if current_segment is None:
            # Start a new segment
            current_segment = {
                "start_frame": frame_pos,
                "end_frame": frame_pos,
                "positions": [position]
            }
        else:
            # Check if the current frame's position matches the segment's major position
            segment_major_position = max(set(current_segment["positions"]),
                                         key=current_segment["positions"].count)
            if position == segment_major_position:
                # Extend the current segment
                current_segment["end_frame"] = frame_pos
                current_segment["positions"].append(position)
            else:
                # Check if the position change is consistent over the next few frames
                # (e.g., at least 1 second of consistent change)
                consistent_change_frames = int(fps)  # 1 second of consistent change
                consistent_change = True

                # Check the next `consistent_change_frames` frames
                for i in range(1, consistent_change_frames + 1):
                    if frame_pos + i >= len(frame_positions) or frame_positions[frame_pos + i] != position:
                        consistent_change = False
                        break

                if consistent_change:
                    # Finalize the current segment if it meets the minimum duration
                    if current_segment["end_frame"] - current_segment["start_frame"] + 1 >= min_frames_per_segment:
                        segments.append({
                            "start_frame": current_segment["start_frame"],
                            "end_frame": current_segment["end_frame"],
                            "start_time": current_segment["start_frame"] / fps,
                            "end_time": current_segment["end_frame"] / fps,
                            "major_position": segment_major_position
                        })
                    # Start a new segment
                    current_segment = {
                        "start_frame": frame_pos,
                        "end_frame": frame_pos,
                        "positions": [position]
                    }
                else:
                    # Continue the current segment
                    current_segment["end_frame"] = frame_pos
                    current_segment["positions"].append(position)

    # Finalize the last segment if it meets the minimum duration
    if current_segment is not None and current_segment["end_frame"] - current_segment[
        "start_frame"] + 1 >= min_frames_per_segment:
        segment_major_position = max(set(current_segment["positions"]), key=current_segment["positions"].count)
        segments.append({
            "start_frame": current_segment["start_frame"],
            "end_frame": current_segment["end_frame"],
            "start_time": current_segment["start_frame"] / fps,
            "end_time": current_segment["end_frame"] / fps,
            "major_position": segment_major_position
        })

    # Merge consecutive segments of the same position
    merged_segments = []
    for segment in segments:
        if merged_segments and segment["major_position"] == merged_segments[-1]["major_position"]:
            # Merge with the previous segment
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["end_time"] = segment["end_time"]
        else:
            # Add as a new segment
            merged_segments.append(segment)

    # Remove segments shorter than min_segment_duration
    final_segments = []
    for segment in merged_segments:
        segment_duration = segment["end_time"] - segment["start_time"]
        if segment_duration >= min_segment_duration:
            final_segments.append(segment)
        elif final_segments:
            # Merge short segment with the previous segment
            final_segments[-1]["end_frame"] = segment["end_frame"]
            final_segments[-1]["end_time"] = segment["end_time"]

    return final_segments

def _calculate_speed(distances):
    """Calculate speed based on absolute deltas between consecutive distances."""
    if len(distances) < 2:
        return 0

    total_delta = 0
    for i in range(1, len(distances)):
        total_delta += abs(distances[i] - distances[i - 1])

    # Average delta as a simple speed measure
    return round(total_delta / (len(distances) - 1), 2)

def _frame_to_timecode(frame_number, fps):
    """Convert frame number to timecode format."""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def _initialize_kalman_filter() -> cv2.KalmanFilter:
    """
    Initialize a Kalman filter for tracking the height of the penis box.
    """
    kalman_filter = cv2.KalmanFilter(2, 1)  # 2 state variables (height, d_height), 1 measurement variable (height)
    kalman_filter.measurementMatrix = np.array([[1, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
    kalman_filter.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
    kalman_filter.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.1
    kalman_filter.errorCovPost = np.eye(2, dtype=np.float32)
    return kalman_filter


def _get_central_third(frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """
    Get the coordinates of the central third of the frame.
    """
    x1 = frame_width // 3
    x2 = 2 * frame_width // 3
    y1 = frame_height // 3
    y2 = 2 * frame_height // 3
    return x1, y1, x2, y2

def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def _calculate_normalized_distance(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> float:
    """
    Calculate the normalized Euclidean distance between the centers of two bounding boxes.
    """
    # Calculate box centers
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

    # Calculate Euclidean distance
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    # Normalize by the frame diagonal
    frame_diagonal = np.sqrt(frame_width ** 2 + frame_height ** 2)
    normalized_distance = distance / frame_diagonal

    return normalized_distance

def _calculate_normalized_distance_to_base(locked_penis_box: Tuple[int, int, int, int],
                                           class_name: str,
                                           class_box: Tuple[int, int, int, int],
                                           max_distance: int) -> float:
    """
    Calculate the normalized distance (0 to 100) between the base of the locked penis and a relevant class box.

    Args:
        locked_penis_box: The locked penis box (x1, y1, x2, y2).
        class_box: The bounding box of the relevant class (x1, y1, x2, y2).

    Returns:
        Normalized distance (0 to 100), where 0 is at the base of the penis and 100 is farthest away.
    """
    # Base of the locked penis is at y2 (bottom of the box)
    penis_base_y = locked_penis_box[3]

    # if class_name in ['hand', 'foot']:
    #     # top y
    #     box_y = class_box[1]
    # else:
    #     # bottom y
    #     box_y = class_box[3]

    # Determine which y-coordinate to use based on class
    # box_y = (class_box[3] + class_box[1]) // 2 if class_name in ('hand', 'foot') else class_box[3] if class_name not in ['butt', 'pussy'] else (9 * class_box[3] + class_box[1]) // 10

    if class_name in ['face']:
        box_y = class_box[3]
    elif class_name in ['hand']:
        box_y = class_box[1]
    elif class_name in ['butt']:
        box_y = (9 * class_box[3] + class_box[1]) // 10
    else:
        box_y = (class_box[3] + class_box[1]) // 2
    #box_y = (class_box[3] + class_box[1]) // 2 if class_name != 'butt' else (9 * class_box[3] + class_box[1]) // 10

    # Distance from the base of the penis to the class bottom
    # distance = abs(box_y - penis_base_y)
    distance = penis_base_y - box_y
    # distance = max(penis_base_y - box_y, 0)

    normalized_distance = int((distance / max_distance) * 100) if max_distance > 0 else 100 # min((distance / max_distance) * 100 if max_distance > 0 else 100, 100)

    return normalized_distance
    #return int((distance / max_distance) * 100 if max_distance > 0 else 100)

def prev_normalize_funscript_sparse(funscript_data, segments):
    result = deepcopy(funscript_data)

    for seg_counter, segment in enumerate(segments):
        position_type = segment['major_position']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']

        # Extract only frames within this segment
        segment_indices = [i for i, (f, _) in enumerate(result) if start_frame <= f <= end_frame]
        if not segment_indices:
            continue  # Skip empty segments

        segment_data = [result[i] for i in segment_indices]
        values = [x[1] for x in segment_data]

        p01 = np.percentile(values, 5)
        p99 = np.percentile(values, 95)

        scale_range = p99 - p01

        # Adjust the values properly
        for i in segment_indices:
            frame, val = result[i]

            if position_type in ['Not relevant', 'Close up']:
                val = 100  # Directly set to 100
            else:
                if val <= p01:
                    val = 0  # Ensure lowest values are mapped to 0
                elif val >= p99:
                    val = 100  # Ensure highest values are mapped to 100
                else:
                    if scale_range < 1e-6:
                        val = 50  # If no variation, set to mid-range
                    else:
                        # Rescale values within p01 - p99 range
                        val = ((val - p01) / scale_range) * 100

                        # Apply specific scaling for certain positions
                        # if position_type in ['Handjob / Blowjob', 'Footjob', 'Boobjob']:
                        #     val = val * 0.7 + 30  # Keeps within ~30-95 range

            val = int(max(0, min(100, round(val))))  # Ensure final value stays in bounds
            result[i] = (frame, val)

    return result


def _normalize_funscript_sparse(funscript_data, segments):
    result = deepcopy(funscript_data)
    for seg_counter, segment in enumerate(segments):
        position_type = segment['major_position']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        # Extract only frames within this segment
        segment_indices = [i for i, (f, _) in enumerate(result) if start_frame <= f <= end_frame]
        if not segment_indices:
            continue  # Skip empty segments

        segment_data = [result[i] for i in segment_indices]
        values = [x[1] for x in segment_data]

        # Still use percentiles for outlier detection
        p01 = np.percentile(values, 5)
        p99 = np.percentile(values, 95)

        # But also track the min and max of non-outlier values
        filtered_values = [v for v in values if p01 <= v <= p99]
        if not filtered_values:
            # If all values are outliers, use the full range
            filtered_min = min(values)
            filtered_max = max(values)
        else:
            filtered_min = min(filtered_values)
            filtered_max = max(filtered_values)

        scale_range = filtered_max - filtered_min

        # Adjust the values properly
        for i in segment_indices:
            frame, val = result[i]
            if position_type in ['Not relevant', 'Close up']:
                val = 100  # Directly set to 100
            else:
                if val <= p01:
                    # For values below p01, preserve relative differences
                    # Map them to the range 0-5 based on their position relative to min
                    min_val = min(values)
                    if val == min_val and min_val == p01:
                        val = 0
                    elif min_val == p01:
                        val = 0  # Edge case
                    else:
                        val = ((val - min_val) / (p01 - min_val)) * 5

                elif val >= p99:
                    # For values above p99, preserve relative differences
                    # Map them to the range 95-100 based on their position relative to max
                    max_val = max(values)
                    if val == max_val and max_val == p99:
                        val = 100
                    elif max_val == p99:
                        val = 100  # Edge case
                    else:
                        val = 95 + ((val - p99) / (max_val - p99)) * 5

                else:
                    # For non-outlier values, scale them to range 5-95
                    if scale_range < 1e-6:
                        val = 50  # If no variation, set to mid-range
                    else:
                        val = 5 + ((val - filtered_min) / scale_range) * 90

            val = int(max(0, min(100, round(val))))  # Ensure final value stays in bounds
            result[i] = (frame, val)
    return result

def interpolate_box(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], t: float) -> Tuple[int, int, int, int]:
    """
    Interpolate between two bounding boxes.
    """
    x1 = int(box1[0] + t * (box2[0] - box1[0]))
    y1 = int(box1[1] + t * (box2[1] - box1[1]))
    x2 = int(box1[2] + t * (box2[2] - box1[2]))
    y2 = int(box1[3] + t * (box2[3] - box1[3]))
    return (x1, y1, x2, y2)

def analyze_tracking_results_v1(state: AppState):
    """
    Analyze tracking results, focusing on the penis box and other detected classes.
    """
    video_info = state.video_info
    fps = video_info.fps
    int_fps = int(fps)
    exists, yolo_data, raw_yolo_path, _ = load_yolo_data(state)
    results = make_data_boxes(yolo_data)
    width, height = get_cropped_dimensions(state.video_info)
    frame_area = width * height
    list_of_frames = results.get_all_frame_ids()  # Get all frame IDs with detections
    total_frames = state.video_info.total_frames

    # Get central third of the frame
    x1, y1, x2, y2 = _get_central_third(width, height)


    # region PASS 1 - Interpolate the position of boxes when there are gaps (flickering) for the same track_id
    # PASS 1: Interpolate the position of boxes when there are gaps (flickering) for the same track_id.

    # Dictionary to store the last seen frame and box for each track_id
    track_history = {}  # Format: {track_id: {"last_frame": int, "last_box": Tuple[int, int, int, int]}}

    # Get sorted boxes for all frames

    sorted_boxes = results.get_all_boxes(total_frames)

    # List to store the final output after interpolation
    interpolated_frames = [[] for _ in range(total_frames)]
    total_interpolated = 0

    MAX_GAP = 10  # Limit interpolation to 10-frame gaps

    for frame_pos in tqdm(range(total_frames), desc="Pass 1 - Interpolating boxes (linear)", unit="f"):
        frame_boxes = sorted_boxes[frame_pos] if frame_pos < len(sorted_boxes) else []

        for box in frame_boxes:
            box_coords, confidence, class_id, class_name, track_id = box

            if track_id in track_history:
                last_frame = track_history[track_id]["last_frame"]
                last_box = track_history[track_id]["last_box"]

                # Ensure the gap is reasonable
                if 1 < (frame_pos - last_frame) <= MAX_GAP:
                    for i in range(last_frame + 1, frame_pos):
                        total_interpolated += 1
                        t = (i - last_frame) / (frame_pos - last_frame)

                        interpolated_box = interpolate_box(last_box, box_coords, t)

                        # Ensure the interpolated box is not too far from expected motion
                        if np.linalg.norm(np.array(interpolated_box[:2]) - np.array(last_box[:2])) < 50:
                            interpolated_frames[i].append(
                                (interpolated_box, confidence, class_id, class_name, track_id)
                            )

            # Update track history
            track_history[track_id] = {"last_frame": frame_pos, "last_box": box_coords}
            interpolated_frames[frame_pos].append(box)

    print(f"Interpolated {total_interpolated} detections")
    #endregion


    # region PASS 2 : Tracking

    # PASS 2 : Kalman every detected boxes for fps duration, and if they match a newly detected box of same class, merge their track_ids
    # Dictionary to store last known locations and track_ids of disappeared objects
    disappeared_objects = {}  # Format: {class_name: {track_id: {"last_box": Tuple[int, int, int, int], "last_frame": int}}}

    # Dictionary to store active track_ids and their last known frame
    active_tracks = {}  # Format: {class_name: {track_id: last_frame}}

    # Dictionary to store mappings between matched track_ids
    track_id_mapping = {}  # Format: {new_track_id: original_track_id}

    # Set to store track_ids that have been checked and did not match any disappeared track_id
    unmatched_track_ids = set()

    # List to store the final output after merging
    merged_frames = []

    total_merged = 0

    # Weights for IOU and distance
    w1 = 0.7  # Weight for IOU
    w2 = 0.3  # Weight for distance

    for frame_pos, frame_boxes in tqdm(enumerate(interpolated_frames), desc="Pass 2 - Merging track_ids", unit="f"):
        merged_frame_boxes = []

        # Step 1: Suppress overlapping boxes of the same class
        suppressed_boxes = []
        suppressed_track_ids = set()  # Track IDs of suppressed boxes

        for i, box1 in enumerate(frame_boxes):
            suppress = False
            for j, box2 in enumerate(frame_boxes):
                if i != j and box1[3] == box2[3]:  # Same class
                    iou = _calculate_iou(box1[0], box2[0])
                    if iou > 0.3:  # High overlap
                        # Suppress the box with lower confidence
                        if box1[1] < box2[1]:
                            suppress = True
                            suppressed_track_ids.add(box1[4])  # Add suppressed track_id
                            break
            if not suppress:
                suppressed_boxes.append(box1)

        # Step 2: Add suppressed track_ids to disappeared_objects
        for track_id in suppressed_track_ids:
            class_name = None
            last_box = None

            # Find the class and last box of the suppressed track_id
            for box in frame_boxes:
                if box[4] == track_id:
                    class_name = box[3]
                    last_box = box[0]
                    break

            if class_name and last_box:
                if class_name not in disappeared_objects:
                    disappeared_objects[class_name] = {}
                disappeared_objects[class_name][track_id] = {
                    "last_box": last_box,
                    "last_frame": frame_pos
                }

        # Step 3: Process each box in the current frame
        for box in suppressed_boxes:
            box_coords = box[0]
            confidence = box[1]
            class_id = box[2]
            class_name = box[3]
            track_id = box[4]

            # Check if this track_id has been mapped to another track_id
            if track_id in track_id_mapping:
                # Reassign the track_id to the mapped value
                track_id = track_id_mapping[track_id]

            # Check if this is the first appearance of the track_id and it hasn't been matched before
            if track_id not in unmatched_track_ids and track_id not in track_id_mapping:
                # Check if this class has disappeared objects
                if class_name in disappeared_objects:
                    # Find the best match based on combined IOU and distance
                    best_match = None
                    best_score = -1

                    for disappeared_track_id, disappeared_data in disappeared_objects[class_name].items():
                        # Skip if the disappeared track_id has already been mapped
                        if disappeared_track_id in track_id_mapping.values():
                            continue

                        # Calculate IOU and normalized distance
                        iou = _calculate_iou(box_coords, disappeared_data["last_box"])
                        normalized_distance = _calculate_normalized_distance(box_coords, disappeared_data["last_box"], width, height)

                        # Calculate combined score
                        score = (w1 * iou) + (w2 * (1 - normalized_distance))

                        if score > best_score:
                            best_score = score
                            best_match = disappeared_track_id

                    if best_score > 0.1:  # Minimum score threshold for matching
                        # Reallocate the track_id of the disappeared object
                        track_id_mapping[track_id] = best_match  # Map the new track_id to the original one
                        track_id = best_match
                        total_merged += 1
                        del disappeared_objects[class_name][best_match]  # Remove from disappeared objects
                    else:
                        # No match found, add to unmatched_track_ids
                        unmatched_track_ids.add(track_id)

            # Add the box to the merged frame
            merged_frame_boxes.append((box_coords, confidence, class_id, class_name, track_id))

            # Update active tracks
            if class_name not in active_tracks:
                active_tracks[class_name] = {}
            active_tracks[class_name][track_id] = frame_pos

        # Step 4: Update disappeared objects
        for class_name in list(active_tracks.keys()):
            for track_id in list(active_tracks[class_name].keys()):
                if frame_pos - active_tracks[class_name][track_id] > 3 * fps:  # Increased threshold (3xFPS)
                    # Add to disappeared_objects
                    if class_name not in disappeared_objects:
                        disappeared_objects[class_name] = {}
                    disappeared_objects[class_name][track_id] = {
                        "last_box": box_coords,  # Last known box coordinates
                        "last_frame": active_tracks[class_name][track_id]  # Last frame where the box was seen
                    }
                    del active_tracks[class_name][track_id]  # Remove from active tracks

        # Step 5: Remove disappeared objects that have been missing for more than 3xFPS frames
        for class_name in list(disappeared_objects.keys()):
            for track_id in list(disappeared_objects[class_name].keys()):
                if frame_pos - disappeared_objects[class_name][track_id]["last_frame"] > 3 * fps:
                    del disappeared_objects[class_name][track_id]

        # Add the merged frame to the output
        merged_frames.append(merged_frame_boxes)

    print(f"Merged {total_merged} boxes' track_ids")
    # endregion


    # region PASS 3 - Building the locked_penis box

    # Initialize tracking data
    tracking_data = []  # List to store tracking data for each frame
    penetration = False

    # Initialize locked_penis variables
    locked_penis = {
        "active": False,  # Whether the lock is active
        "last_box": None,  # Last known full box (x1, y1, x2, y2)
        "max_height": 0,  # Maximum height of the penis box
        'max_penetration_height': 0, # Maximum penetration height of the penis box
        'area': 0,  # Area of the penis box
        "visible_part": 100,
        "consecutive_detections": 0,  # Number of consecutive frames with penis detection
        "consecutive_non_detections": 0,  # Number of consecutive frames without penis detection
        "glans_detected": False,  # Whether the glans is detected
    }

    # Initialize Kalman filter for height prediction
    kalman_filter = _initialize_kalman_filter()

    # TODO  : add if is_Vr ?

    # Precompute central boxes for all frames
    central_boxes_all_frames = []
    for frame_pos in range(state.video_info.total_frames):
        if frame_pos < len(merged_frames):
            central_boxes = [
                box for box in merged_frames[frame_pos]
                if x1 <= (box[0][0] + box[0][2]) // 2 <= x2  # and y1 <= (box[0][1] + box[0][3]) // 2 <= y2
            ]
        else:
            central_boxes = []
        central_boxes_all_frames.append(central_boxes)

    last_penis_box = None
    last_locked_penis_box = None
    last_active = False
    last_max_height = 0
    last_area = 0
    last_max_penetration_height = 0
    last_visible_part = 100
    last_glans_detected = False

    # Process each frame
    for frame_pos in tqdm(
            range(state.video_info.total_frames),
            unit="f",
            desc="Pass 3 - Building the locked penis box",
            position=0,
            unit_scale=False,
            unit_divisor=1
    ):

        frame_data = {
            "frame_id": frame_pos,
            "penis_box": None,
            "locked_penis": {
                'box': [0, 0, 0, 0] if not last_locked_penis_box else last_locked_penis_box,  # Default box
                'active': last_active,  # Default state
                'max_height': last_max_height,
                'max_penetration_height': last_max_penetration_height,
                'area' : last_area,
                'consecutive_detections': 0,
                'consecutive_non_detections': 0,
                'visible_part': last_visible_part,
                'glans_detected': last_glans_detected
            },
            "detected_boxes": [],
            "distances_to_penis": []
        }

        central_boxes = central_boxes_all_frames[frame_pos]

        for box in central_boxes:
            if box[3] != "penis":
                frame_data["detected_boxes"].append(box)

        # Locate the penis box (if multiple, select the one with the max y2)
        penis_boxes = [box for box in central_boxes if box[3] == "penis"]
        if penis_boxes:
            selected_penis_box = max(penis_boxes, key=lambda box: box[0][3])  # Select box with max y2
            frame_data["penis_box"] = selected_penis_box[0]  # Store the box coordinates (x1, y1, x2, y2)

            frame_data["detected_boxes"].append(selected_penis_box)

            # Update locked_penis variables
            locked_penis["consecutive_detections"] += 1
            locked_penis["consecutive_non_detections"] = 0

            # Update last known box
            last_penis_box = selected_penis_box[0]

            # Update max height (only increase, never decrease unless glans is detected)
            current_height = selected_penis_box[0][3] - selected_penis_box[0][1]
            if current_height > locked_penis["max_height"]:
                locked_penis["max_height"] = current_height
                locked_penis["max_penetration_height"] = int(0.65 * current_height)


            locked_penis["glans_detected"] = False
            # Check for glans within the penis box
            glans_boxes = [box for box in central_boxes if box[3] == "glans"]
            for glans_box in glans_boxes:
                if _calculate_iou(selected_penis_box[0], glans_box[0]) > 0:
                    locked_penis["glans_detected"] = True
                    # Adjust height if glans is detected within the penis box
                    glans_bottom_height = selected_penis_box[0][3] - glans_box[0][3]
                    glans_top_height = selected_penis_box[0][3] - glans_box[0][1]
                    locked_penis["max_penetration_height"] = max(0, min(locked_penis["max_height"], glans_bottom_height))
                    locked_penis["max_height"] = max(0, min(locked_penis["max_height"], current_height), glans_top_height)
            #if not locked_penis["glans_detected"] and current_height > locked_penis["max_penetration_height"]:
            #    locked_penis["max_penetration_height"] = current_height

            # Check if navel box is within the penis box, then this is a "giant penis" type error, we need to limit the size
            navel_boxes = [box for box in central_boxes if box[3] == "navel"]
            if frame_data["locked_penis"]["active"]:
                for navel_box in navel_boxes:
                    if _calculate_iou(frame_data["locked_penis"]["box"], navel_box[0]) > 0:
                        navel_height = selected_penis_box[0][3] - navel_box[0][3]
                        locked_penis["max_height"] = max(0, min(locked_penis["max_height"], navel_height))


            # Activate locked_penis after fps / 5 frames with consistent detection
            if not locked_penis["active"] and locked_penis["consecutive_detections"] >= fps / 5: #10:
                locked_penis["active"] = True

            # Update Kalman filter with the observed height
            kalman_filter.correct(np.array([[current_height]], dtype=np.float32))
        else:
            # No penis detection in this frame
            locked_penis["consecutive_detections"] = 0
            locked_penis["consecutive_non_detections"] += 1

            # Deactivate locked_penis after 3 * fps frames without detection
            if locked_penis["active"] and not penetration and locked_penis["consecutive_non_detections"] >= 3 * fps:
                locked_penis["active"] = False

        # If locked_penis is active, use the locked box and predicted height
        if locked_penis["active"]:
            # Predict height using the Kalman filter
            predicted_height = int(kalman_filter.predict()[0][0])

            # Constrain the predicted height within the bounds of the locked penis box
            predicted_height = min(max(predicted_height, 0), locked_penis["max_height"])

            frame_data["penis_box"] = (
                last_penis_box[0],  # x1
                last_penis_box[3] - predicted_height,  # y1
                last_penis_box[2],  # x2
                last_penis_box[3]  # y2
            )

            # Create the locked penis box
            locked_penis["last_box"] = (
                last_penis_box[0],  # x1
                last_penis_box[3] - locked_penis["max_height"],  # y1
                last_penis_box[2],  # x2
                last_penis_box[3]  # y2
            )

            if locked_penis["max_height"] == 0:
                locked_penis["visible_part"] = 0
                locked_penis["area"] = 0
            else:
                locked_penis["visible_part"] = int((predicted_height / locked_penis["max_height"]) * 100)
                locked_penis["area"] = int((locked_penis['last_box'][2] - locked_penis['last_box'][0]) * (locked_penis['last_box'][3] - locked_penis['last_box'][1]))

        last_locked_penis_box = locked_penis["last_box"]
        last_active = locked_penis["active"]
        last_max_height = locked_penis["max_height"]
        last_max_penetration_height = locked_penis["max_penetration_height"]
        last_visible_part = locked_penis["visible_part"]
        last_glans_detected = locked_penis["glans_detected"]
        last_area = locked_penis["area"]

        if last_penis_box:
            frame_data["locked_penis"] = {
                'box' : (
                    last_penis_box[0],  # x1
                    last_penis_box[3] - locked_penis["max_height"],  # y1
                    last_penis_box[2],  # x2
                    last_penis_box[3]  # y2
                    ),
                'active': locked_penis["active"],
                'consecutive_detections': locked_penis["consecutive_detections"],
                'consecutive_non_detections': locked_penis["consecutive_non_detections"],
                'max_height': locked_penis["max_height"],
                'max_penetration_height': locked_penis["max_penetration_height"],
                'area': locked_penis["area"],
                'visible_part': locked_penis["visible_part"],
                'glans_detected': locked_penis["glans_detected"]
            }

        # Append frame data to tracking data
        tracking_data.append(frame_data)
        #locked_penis_history.append(locked_penis)

    # endregion


    # region PASS 4 - Identify all boxes, classes, track_id in contact with the locked penis box
    # PASS 4 - Identify all boxes, classes, track_id in contact with the locked penis box
    contact_data = []  # List to store contact data for each frame

    # Step 1: Assign a position to each frame
    frame_positions = []
    for frame_data in tqdm(tracking_data, desc="Pass 4 - Assigning positions", unit="f"):
        position = "Not relevant"
        if frame_data["locked_penis"]["active"]:
            # Identify contacts and assign a position
            contacts = []
            for box in frame_data["detected_boxes"]:
                box_coords = box[0]
                class_name = box[3]
                iou = _calculate_iou(frame_data["locked_penis"]["box"], box_coords)
                if iou > 0.05:
                    box_area = (box[0][2] - box[0][0]) * (box[0][3] - box[0][1])
                    if class_name not in ["hand", "foot"] or (
                            class_name in ["hand", "foot"] and box_area > 0.7 * locked_penis["area"] and box_area < 2 * locked_penis["area"]):
                        contacts.append({"class_name": class_name})
            position = _assign_frame_position(contacts)
        elif not frame_data["locked_penis"]["active"]:
            for box in frame_data["detected_boxes"]:
                box_coords = box[0]
                class_name = box[3]
                # if pussy present and pussy box area > 10% of the frame area then, closeup
                if class_name in ["pussy", "butt"]:
                    box_area = (box_coords[2] - box_coords[0]) * (box_coords[3] - box_coords[1])
                    if box_area > 0.07 * frame_area:
                        position = "Close up"
                        break


            if position != "Close up":
                position = "Not relevant"
        else:
            position = "Not relevant"
        frame_positions.append(position)

    # Step 2: Aggregate frames into segments
    segments = _aggregate_segments(frame_positions, fps, min_segment_duration=1.0)

    keep_going = True
    # Step 3: Clean and merge segments
    while keep_going:
        keep_going = False
        for i in range(len(segments) - 1):
            if i < len(segments) - 2:
                if segments[i]["major_position"] == segments[i + 1]:
                    segments[i]["end_frame"] = segments[i + 1]["end_frame"]
                    segments.pop(i + 1)
                    keep_going = True
                    break
                if segments[i]["major_position"] == segments[i + 2]["major_position"] != "Not relevant" and (segments[i + 1]["end_frame"] - segments[i+1]["start_frame"] < 10 * fps):
                    segments[i]["end_frame"] = segments[i + 2]["end_frame"]
                    segments.pop(i + 2)
                    segments.pop(i + 1)
                    keep_going = True
                    break
                if  i>0 and segments[i]["major_position"] != segments[i + 1]["major_position"] and (segments[i]["end_frame"] - segments[i]["start_frame"] < 5 * fps):
                    segments[i-1]["end_frame"] = segments[i]["end_frame"]
                    segments.pop(i)
                    keep_going = True
                    break


    # # Print segment one by one
    # log_tr.info("Technical segments:")
    # for segment in segments:
    #     start_frame = segment["start_frame"]
    #     end_frame = segment["end_frame"]
    #     position = segment["major_position"]
    #     start_time = _frame_to_timecode(start_frame, fps)
    #     end_time = _frame_to_timecode(end_frame, fps)
    #     log_tr.info(f"From {start_frame} to {end_frame} - {start_time} to {end_time} - duration: {int((end_frame - start_frame) / fps)} s -  {position}")

    timestamps = []
    keep_going = True
    iteration = 1

    i = 0

    # Initial pass to create timestamps
    for segment in segments:
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        start_time = _frame_to_timecode(start_frame, fps)
        end_time = _frame_to_timecode(end_frame, fps)
        position = segment["major_position"]
        timestamps.append([start_frame, end_frame, start_time, end_time, position])


    while keep_going:
        keep_going = False
        for i in range(len(timestamps)-1):
            if i < len(timestamps)-2 and timestamps[i][4] == timestamps[i + 2][4] and timestamps[i + 1][4] == 'Not relevant' and (timestamps[i+1][1] - timestamps[i+1][0] < 10 * fps):
                # Merge with the previous timestamp
                timestamps[i][1] = timestamps[i + 2][1]  # Update end_frame
                timestamps[i][3] = timestamps[i + 2][3]
                # pop line i+2 and i+1
                timestamps.pop(i + 2)
                timestamps.pop(i + 1)
                keep_going = True
                break
            if i < len(timestamps)-1 and timestamps[i][4] == timestamps[i + 1][4]:
                # Merge with the next timestamp
                timestamps[i][1] = timestamps[i + 1][1]  # Update end_frame
                timestamps[i][3] = timestamps[i + 1][3]
                # pop line i+1
                timestamps.pop(i + 1)
                keep_going = True
                break
            if i < len(timestamps)-1 and timestamps[i][4] != timestamps[i + 1][4] and timestamps[i + 1][4] == 'Not relevant' and (timestamps[i+1][1] - timestamps[i+1][0] < 15 * fps):
                # Merge with the next timestamp
                timestamps[i][1] = timestamps[i + 1][1]  # Update end_frame
                timestamps[i][3] = timestamps[i + 1][3]
                # pop line i+1
                timestamps.pop(i + 1)
                keep_going = True
                break
            if i < len(timestamps)-1 and timestamps[i][4] in ['Handjob', 'Blowjob', 'Boobjob'] and timestamps[i + 1][4] in ['Handjob', 'Blowjob', 'Boobjob']  and ((timestamps[i][1] - timestamps[i][0] < 15 * fps) or (timestamps[i+1][1] - timestamps[i+1][0] < 15 * fps)):
                # Merge with the next timestamp
                timestamps[i][4] = 'Blowjob'
                timestamps[i][1] = timestamps[i + 1][1]  # Update end_frame
                timestamps[i][3] = timestamps[i + 1][3]
                # pop line i+1
                timestamps.pop(i + 1)
                keep_going = True
                break
            if i >= 1 and (timestamps[i][1] - timestamps[i][0] < 15 * fps):
                timestamps[i][4] = 'Not relevant'
                keep_going = True
                break



    # Print the final timestamps
    log_tr.info("Final timestamps:")
    for timestamp in timestamps:
        start_frame, end_frame, start_time, end_time, position = timestamp
        log_tr.info(f"From {start_frame} to {end_frame} - {start_time} to {end_time} - duration : {(end_frame - start_frame) // fps}s - {position}")

    # endregion


    # region PASS 5 - Determine the final distance for each frame, depending on the classes relevant to the position of the segment the frame is in
    # PASS 5 - Determine the final distance for each frame, depending on the classes relevant to the position of the segment the frame is in

    distance = []
    relevant_classes = []
    prev_nb_touching = 0
    prev_fastest_track_id = 0

    transition_active = False
    transition_frames = 2 * int(fps)
    remaining_transition_frames = transition_frames

    # Dictionary to maintain track history {track_id: [last_10_distances]}
    track_history = {}

    # Dictionary to maintain previous frame's distance for each track {track_id: previous_distance}
    previous_distances = {}

    # kalman filter for distance estimation
    # kf_distance = _initialize_kalman_filter()

    # # Update Kalman filter with the observed distance
    # kalman_filter.correct(np.array([[comp_distance]], dtype=np.float32))
    #
    # predicted_distance = int(kalman_filter.predict()[0][0])
    # # Constrain the predicted distance between 0 and 100
    # predicted_distance = min(max(predicted_distance, 0), 100)

    for frame_pos in tqdm(range(total_frames), desc="Pass 5 - Determining distance", unit="f"):
        # Determine the position the frame belongs to in the segments

        comp_distance = 0
        nb_touching = 0
        track_speeds = {}
        fastest_track_id = 0

        position = "Not relevant"
        bounding_boxes = []
        counter = 0

        if not tracking_data[frame_pos]["locked_penis"]:
            comp_distance = 100
            counter += 1
            print(f"Frame {frame_pos} has no locked penis - skipped ({counter}/{total_frames})")
            penis_box = [0,0,0,0]
            penis_active = 'Undetected'
            penis_consecutive_detections = 0
            penis_consecutive_non_detections = 0
            penis_max_height = 0
            penis_max_penetration_height = 0
            penis_visible_part = 0
            penis_glans_detected = False
            penis_box_area = 0
        else:
            penis = tracking_data[frame_pos]["locked_penis"]
            penis_box = tuple(penis['box'])
            penis_active = penis['active']
            penis_consecutive_detections = penis['consecutive_detections']
            penis_consecutive_non_detections = penis['consecutive_non_detections']
            penis_max_height = penis['max_height']
            penis_max_penetration_height = penis['max_penetration_height']
            penis_visible_part = penis['visible_part']
            penis_glans_detected = penis['glans_detected']
            penis_box_area = penis['area']

            for segment in segments:
                if segment["start_frame"] <= frame_pos <= segment["end_frame"]:
                    position = segment["major_position"]
                    break

            relevant_classes = []
            penetration = False

            if position == "Not relevant":
                relevant_classes = []
                comp_distance = 100
            elif position == "Cowgirl / Missionary":
                relevant_classes = ["pussy"]
                if not penis_glans_detected:
                    penetration = True
            elif position == "Rev. Cowgirl / Doggy":
                relevant_classes = ["butt"]
                if not penis_glans_detected:
                    penetration = True
            elif position == "Handjob / Blowjob":
                relevant_classes = ["face", "hand"]
            elif position == "Boobjob":
                relevant_classes = ["breast"]
                # relevant_classes = ["breast", "hand"]
            #elif position == "Handjob":
            #    relevant_classes = ["hand"]
            elif position == "Footjob":
                relevant_classes = ["foot"]
            else:
                relevant_classes = []
                comp_distance = 100

            # Check in the detections the relevant classes boxes that are touching the locked penis
            bounding_boxes = []
            #if locked_penis_box and frame_pos < len(tracking_data):
            for box in merged_frames[frame_pos]:  #
            # for box in tracking_data[frame_pos]["detected_boxes"]:
                box_coords = box[0]
                class_name = box[3]
                track_id = box[4]
                box_area = (box[0][2] - box[0][0]) * (box[0][3] - box[0][1])
                relative_p_area = round(box_area / penis_box_area, 1) if penis_box_area != 0 else 0
                relative_f_area = round(box_area / frame_area, 2) if frame_area != 0 else 0
                iou = _calculate_iou(penis_box, box_coords)

                # Update track history
                if track_id not in track_history:
                    track_history[track_id] = []
                pos = (box_coords[1] + box_coords[3]) // 2
                track_history[track_id].append(pos)
                # Keep only last "fps" distances
                if len(track_history[track_id]) > int_fps:
                    track_history[track_id] = track_history[track_id][-int_fps:]
                # Calculate speed
                speed = _calculate_speed(track_history[track_id])

                if class_name in relevant_classes:
                    # iou = _calculate_iou(penis_box, box_coords)
                    if iou > 0.01:
                        if class_name not in ["hand", "foot"] or (class_name in ["hand", "foot"] and box_area > 0.5 * penis_box_area and box_area < 3 * penis_box_area):
                            if penetration:
                                # max_height = penis_max_penetration_height
                                max_height = penis_max_height
                            else:
                                max_height = penis_max_height

                            dist = _calculate_normalized_distance_to_base(penis_box, class_name, box_coords, max_height)

                            comp_distance += dist
                            nb_touching += 1
                            pos_info = 'p: ' + str(int(dist))
                            int_position = int(dist)
                        else:
                            pos_info = 'Area excluded'
                            int_position = 0
                    else:
                        pos_info = 'IOU excluded'
                        int_position = 0
                else:
                    pos_info = 'Non relevant'
                    int_position = 0

                bounding_boxes.append({
                    'box': box[0],
                    'conf': box[1],
                    'class_name': box[3],
                    'track_id': box[4],
                     'position': '' if box[3] in ['penis', 'glans'] else pos_info + ' | s: ' + str(speed) + ' | i: ' + str(round(iou, 2)) + ' | fa: ' + str(
                         relative_f_area) + ' | pa: ' + str(relative_p_area),
                    'int_position': int_position,
                    'speed': speed,
                    'iou': iou,
                })

            # Clean up track history - remove tracks not present in current frame
            current_track_ids = {box[4] for box in tracking_data[frame_pos]["detected_boxes"]}
            for track_id in list(track_history.keys()):
                if track_id not in current_track_ids:
                    del track_history[track_id]

            if track_history:
                # Calculate speed for each track and find the maximum
                fastest_track_id = max(track_history.keys(),
                                       key=lambda x: _calculate_speed(track_history[x]))
            else:
                fastest_track_id = None  # or 0 if you prefer




            if penis_box and position in ["Cowgirl / Missionary", "Rev. Cowgirl / Doggy"]:  #and nb_touching == 0:
                    comp_distance += penis_visible_part
                    nb_touching += 1

            if nb_touching == 0:
                comp_distance = 100
            elif position == "Handjob / Blowjob":

                if nb_touching != prev_nb_touching or fastest_track_id != prev_fastest_track_id:
                    transition_active = True
                    remaining_transition_frames = transition_frames

                prev_nb_touching = nb_touching
                prev_fastest_track_id = fastest_track_id

                total_speeds = sum(box['speed'] for box in bounding_boxes if box['class_name'] in relevant_classes)
                weighted_distance = (
                    sum(box['int_position'] * box['speed'] for box in bounding_boxes if box['class_name'] in relevant_classes)
                    / total_speeds if total_speeds != 0 else 0
                )
                if total_speeds != 0:
                    comp_distance =  int(weighted_distance)  #int(max(0, min(100, weighted_distance)))
                else:
                    comp_distance = distance[-1]

            else:
                comp_distance = int(comp_distance / nb_touching)  # int(max(0, min(100, comp_distance / nb_touching)))




        if transition_active:
            if remaining_transition_frames > 0:
                if comp_distance - distance[-1] == 0:
                    transition_distance = distance[-1]
                    log_tr.info(f"Used {transition_frames - remaining_transition_frames} transition frames before meeting at {comp_distance}")
                    transition_active = False
                    remaining_transition_frames = transition_frames
                else:
                    # previous distance + 1 or -1 depending on the sign of the previous distance - comp_distance
                    transition_distance = distance[-1] + (1 if (comp_distance - distance[-1] > 0) else -1)
                    # comp_distance = transition_distance
                    remaining_transition_frames -= 1
                distance.append(transition_distance)
            else:
                transition_active = False
                remaining_transition_frames = transition_frames
                distance.append(comp_distance)
        else:
            distance.append(comp_distance)

        # Add frame data to debug_data
        state.debug_data.add_frame(
            frame_pos,
            bounding_boxes=bounding_boxes,
            variables={
                'frame': frame_pos,
                'time': str(timedelta(seconds=int(frame_pos / fps))),
                'distance': distance[-1], #comp_distance,
                'transition': str(transition_active) + (" | " + str(remaining_transition_frames) + " frames left" if transition_active else ""),
                'raw_distance': comp_distance,
                'penetration': penetration,
                'sex_position': position,
                'relevant_classes': relevant_classes,
                #'sex_position_reason': 'NA',
                #'tracked_body_part': 'NA',
                'locked_penis_box': {
                    'box': penis_box,
                    'active': penis_active,
                    'visible': penis_visible_part,
                    'height': penis_max_height,
                    'penetration_height': penis_max_penetration_height,
                },
                'glans_detected': False,
                #'cons._glans_detections': 0,
                #'cons._glans_non_detections': 0,
                'cons.penis_detections': penis_consecutive_detections,
                'cons.penis_non_detections': penis_consecutive_non_detections,
                #'breast_tracking': False,
            }
        )
    # endregion


    # region PASS 6 - Re-parse multi-contact sections : 'Handjob / Blowjob',  'Boobjob', 'Footjob'
    # PASS 6 - Re-parse multi-contact sections : 'Handjob / Blowjob',  'Boobjob', 'Footjob'

    # for i, segment in tqdm(enumerate(segments), desc="Pass 6 - Re-parse multi-contact sections", unit="section"):
    #     if segment["major_position"] in ["Handjob / Blowjob"]:
    #         current_leader = None
    #         leader_history = defaultdict(lambda: {
    #             'total_speed': 0, 'count': 0, 'last_positions': [],
    #             'consistency': 0, 'position_history': [], 'frames_present': 0
    #         })
    #
    #         transition_frames = int(fps)  # 1 second transition
    #         remaining_transition = 0
    #         transition_start_value = 0
    #         transition_target_value = 0
    #         leader_stability_threshold = fps * 0.5  # Leader must be stable for at least 0.5 seconds
    #         leader_stability_counter = 0
    #
    #         for s in range(segment["start_frame"], segment["end_frame"] + 1):
    #             frame_data = state.debug_data.get_metrics_at_frame(s)
    #             variables = frame_data['variables']
    #             relevant_classes = variables.get('relevant_classes', [])
    #
    #             # Organize bounding boxes for quick lookup
    #             relevant_boxes_by_track_id = {
    #                 box['track_id']: box for box in frame_data['bounding_boxes']
    #                 if box['class_name'] in relevant_classes and box['int_position'] > 0
    #             }
    #
    #             if not relevant_boxes_by_track_id:
    #                 # No relevant contacts, maintain last known distance
    #                 distance[s] = distance[s - 1] if s > 0 else 0
    #                 continue
    #
    #             # Compute leader scores
    #             potential_leaders = {}
    #             for track_id, box in relevant_boxes_by_track_id.items():
    #                 speed, position = box['speed'], box['int_position']
    #                 history = leader_history[track_id]
    #
    #                 # Update statistics
    #                 history['total_speed'] += speed
    #                 history['count'] += 1
    #                 history['frames_present'] += 1
    #                 history['position_history'].append(position)
    #
    #                 # Maintain fixed-size position history
    #                 if len(history['position_history']) > 5:
    #                     history['position_history'].pop(0)
    #
    #                 # Compute movement consistency
    #                 if len(history['position_history']) >= 2:
    #                     movement = position - history['position_history'][-2]
    #                     last_movement = history['position_history'][-2] - history['last_positions'][-1] if history[
    #                         'last_positions'] else 0
    #
    #                     if movement * last_movement >= 0:  # Same direction
    #                         history['consistency'] = min(history['consistency'] + 1, 10)
    #                     else:  # Direction change
    #                         history['consistency'] = max(history['consistency'] - 2, 0)
    #
    #                 # Maintain short-term movement history
    #                 history['last_positions'].append(position)
    #                 if len(history['last_positions']) > 3:
    #                     history['last_positions'].pop(0)
    #
    #                 # Compute stability score
    #                 avg_speed = history['total_speed'] / max(1, history['count'])
    #                 consistency = history['consistency']
    #                 presence_ratio = history['frames_present'] / (s - segment["start_frame"] + 1)
    #
    #                 # Penalize tracks that appear sporadically
    #                 if presence_ratio < 0.7:
    #                     consistency *= 0.5
    #
    #                 # Score potential leaders
    #                 potential_leaders[track_id] = avg_speed * (1 + consistency * 0.1) * (0.5 + presence_ratio * 0.5)
    #
    #             # Determine new leader
    #             current_score = potential_leaders.get(current_leader, 0)
    #             potential_new_leaders = {
    #                 k: v for k, v in potential_leaders.items() if v > current_score * 1.3
    #             }
    #
    #             if potential_new_leaders:
    #                 new_leader_candidate = max(potential_new_leaders, key=potential_new_leaders.get)
    #                 leader_stability_counter += 1
    #
    #                 # Confirm switch only after stability threshold
    #                 if leader_stability_counter >= leader_stability_threshold:
    #                     new_leader = new_leader_candidate
    #                     leader_stability_counter = 0
    #                 else:
    #                     new_leader = current_leader
    #             else:
    #                 new_leader = current_leader
    #                 leader_stability_counter = max(0, leader_stability_counter - 1)
    #
    #             # Initialize leader if none exists
    #             if current_leader is None and potential_leaders:
    #                 new_leader = max(potential_leaders, key=potential_leaders.get)
    #                 leader_stability_counter = 0
    #
    #             # Handle leader transitions smoothly
    #             if new_leader != current_leader and new_leader is not None:
    #                 if current_leader is not None and leader_stability_counter >= leader_stability_threshold:
    #                     remaining_transition = transition_frames
    #                     transition_start_value = distance[s - 1] if s > 0 else 0
    #                     transition_target_value = relevant_boxes_by_track_id[new_leader]['int_position']
    #                 else:
    #                     remaining_transition = 0
    #
    #                 current_leader = new_leader
    #
    #             # Compute distance using transitions
    #             if remaining_transition > 0:
    #                 progress = 1 - (remaining_transition / transition_frames)
    #                 distance[s] = int(
    #                     transition_start_value + (transition_target_value - transition_start_value) * progress)
    #                 remaining_transition -= 1
    #             elif current_leader:
    #                 distance[s] = relevant_boxes_by_track_id.get(current_leader, {}).get('int_position',
    #                                                                                      distance[s - 1])
    #             elif s > 0:
    #                 distance[s] = distance[s - 1]
    #
    #             # Reduce jitter from low-speed movements
    #             avg_speed = sum(box['speed'] for box in relevant_boxes_by_track_id.values()) / len(
    #                 relevant_boxes_by_track_id)
    #             if avg_speed < 2 and s > 0:
    #                 distance[s] = int((distance[s - 1] + distance[s]) / 2)
    #
    #             # Update debug data efficiently
    #             frame_data['variables'].update({
    #                 'distance': distance[s],
    #                 'leader_track_id': current_leader,
    #                 'transition_active': remaining_transition > 0
    #             })
    #
    #             if s % 10 == 0:  # Reduce redundant debug data updates
    #                 debug_info = state.debug_data.get_variables_at_frame(s)
    #                 debug_info.update({
    #                     "distance": int(distance[s]),
    #                     "raw_distance": int(distance[s]),
    #                     "tracked_body_part": relevant_boxes_by_track_id.get(current_leader, {}).get('class_name', None)
    #                 })
    # # endregion



    # for i, segment in tqdm(enumerate(segments), desc="Pass 6 - Remapping sections 0 - 100", unit="section") :
    #     if segment["major_position"] in ["Not relevant", "Close up"]:
    #         continue
    #
    #     # retrieve min and max of the distance list in that portion of the video segment["start_frame"] to segment["end_frame"] and remap it 0 - 100
    #     min_distance = min(distance[segment["start_frame"]:segment["end_frame"]])
    #     max_distance = max(distance[segment["start_frame"]:segment["end_frame"]])
    #
    #     if min_distance != max_distance:
    #         for i in range(segment["start_frame"], segment["end_frame"]):
    #             distance[i] = int((distance[i] - min_distance) / (max_distance - min_distance) * 100)
    #         # save actuated remapped min and max
    #
    #     new_min = min(distance[segment["start_frame"]:segment["end_frame"]])
    #     new_max = max(distance[segment["start_frame"]:segment["end_frame"]])
    #
    #     log_tr.info(f"Remapped segment {i} - {segment['major_position']} from {min_distance} to {max_distance} to {new_min} to {new_max}")

    # endregion


    # region PASS 6 - Processing signal - Denoising
    # PASS 6 - Processing signal - Denoising

    log_tr.info("Pass 6 - Processing signal - Denoising with Savitzky-Golay algorithm")
    # Run savitzky golay algorithm
    distance = savgol_filter(distance, 11, 2)

    for frame_pos in tqdm(range(len(distance)), desc="Pass 7 - Processing signal - Applying edits", unit="frame"):

        debug_info = state.debug_data.get_variables_at_frame(frame_pos)
        debug_info["distance"] = int(distance[frame_pos])

    # endregion


    # region PASS 7 - Simplifying signal
    # PASS 7 - Simplifying signal

    log_tr.info("Pass 8 - Simplifying signal - Identifying peaks and valleys")

    # zip frame_pos and distance
    funscript_data = list(zip(range(len(distance)), distance))

    # test_funscript_data = _normalize_funscript(funscript_data, segments)  #, int_fps)
    test_funscript_data = funscript_data

    # SUB PASS 1

    frames, positions = zip(*test_funscript_data)  # Unzip the frame and position
    positions = np.array(positions)  # Convert positions to a numpy array

    # Find peaks (local maxima)
    peaks, _ = find_peaks(positions, prominence=0.1)

    # Find valleys (local minima) by inverting the signal
    valleys, _ = find_peaks(-positions, prominence=0.1)

    # Combine the indices of peaks and valleys
    keep_indices = set(peaks).union(valleys)
    log_tr.info(f"Pass 8 - Simplifying signal - Found {len(keep_indices)} peaks and valleys")

    # SUB PASS 2

    data_len = len(test_funscript_data)
    removed = 0
    to_remove = set()

    for i in keep_indices:
        if i <= 0 or i >= data_len - 1:
            continue  # Skip first and last frame to avoid out-of-range

        current = test_funscript_data[i][1]
        prev = test_funscript_data[i - 1][1]
        next_ = test_funscript_data[i + 1][1]

        if prev < current < next_ or prev > current > next_:
            to_remove.add(i)
            removed += 1

    # Apply safe filtering
    keep_indices.difference_update(to_remove)

    log_tr.info(f"Pass 8 - Simplifying signal - Removed {removed} intermediate points, kept {len(keep_indices)} peaks and valleys")

    temp_funscript_data = [test_funscript_data[i] for i in keep_indices]

    vw_funscript_data = simplify_coords_vw(temp_funscript_data, 2.0)

    log_tr.info(f"Number of positions before simplifying: {len(funscript_data)}")
    log_tr.info(f"Number of positions after 1st pass of simplifying: {len(temp_funscript_data)}")
    log_tr.info(f"Number of positions after 2nd pass of simplifying: {len(vw_funscript_data)}")

    # endregion

    # region PASS 9 - Amplifying signal
    # PASS 9 - Amplifying signal
    log_tr.info("Pass 9 - Amplifying signal + Simplifying pass 2")

    amplification = True

    if amplification:
        amplified_funscript_data = _normalize_funscript_sparse(vw_funscript_data, segments)
        log_tr.info("Pass 9 - Amplification activated")
    else:
        log_tr.info("Pass 9 - Amplification deactivated")
        amplified_funscript_data = vw_funscript_data

    cleaned_data = amplified_funscript_data[:]  # Copy data
    pass_counter = 0
    max_passes = 10
    variation_threshold = 10

    while pass_counter < max_passes:
        deleted_counter = 0
        new_data = [cleaned_data[0]]  # Always keep the first point

        for i in range(1, len(cleaned_data) - 1):
            prev, current, next_ = cleaned_data[i - 1][1], cleaned_data[i][1], cleaned_data[i + 1][1]

            # Compute local variation
            diff_prev = abs(current - prev)
            diff_next = abs(current - next_)

            # Keep only meaningful variations
            if diff_prev >= variation_threshold or diff_next >= variation_threshold:
                new_data.append(cleaned_data[i])
            else:
                deleted_counter += 1

        new_data.append(cleaned_data[-1])  # Always keep the last point

        # Stop if no changes
        if deleted_counter == 0:
            break

        cleaned_data = new_data  # Update dataset
        pass_counter += 1

    vw_funscript_data = cleaned_data

    log_tr.info(f"Pass 9 - Amplification and Simplification complete - Kept {len(vw_funscript_data)} peaks and valleys")

    # amplified_funscript_data = vw_funscript_data
    # endregion

    # region PASS 10 - Generating funscript
    # PASS 10 - Generating funscript

    log_tr.info("Pass 10 - Generating funscript")

    output_path, _ = get_output_file_path(state.video_path, ".funscript")

    write_funscript(amplified_funscript_data, output_path, state.video_info.fps, timestamps)

    # endregion
