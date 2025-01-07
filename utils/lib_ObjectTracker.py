from utils import lib_KalmanFilter as KF
from collections import deque
import numpy as np
from params.config import class_names
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

class LockedPenisBox:
    """
    A class to manage the locked penis box, which represents the detected penis area.
    It stores the box coordinates, height, and active status.
    """
    def __init__(self):
        self.box = None  # Coordinates of the locked penis box
        self.height = 0  # Height of the locked penis box
        self.active = False  # Whether the locked penis box is active
        self.visible = 0 # Visibility of the locked penis box 0-100

    def update(self, new_box, new_height, visible=None):
        """Update the locked penis box with new coordinates and height."""
        self.box = new_box
        self.height = new_height
        self.active = True
        if visible is not None:
            self.visible = visible

    def deactivate(self):
        """Deactivate the locked penis box."""
        self.box = None
        self.height = 0
        self.active = False

    def get_box(self):
        """Return the coordinates of the locked penis box."""
        return self.box

    def get_height(self):
        """Return the height of the locked penis box."""
        return self.height

    def is_active(self):
        """Check if the locked penis box is active."""
        return self.active

    def to_dict(self):
        """
        Convert the LockedPenisBox object into a dictionary for JSON serialization.
        """
        return {
            "box": self.box,
            "height": self.height,
            "active": self.active
        }

    @classmethod
    def from_dict(cls, data):
        """
        Reconstruct a LockedPenisBox object from a dictionary.
        """
        locked_box = cls()
        if data["box"] is not None:
            locked_box.update(data["box"], data["height"])
        return locked_box

class ObjectTracker:
    """
    A class to track objects (e.g., penis, glans, etc.) in a video frame and determine their positions and interactions.
    """

    def __init__(self, fps, frame_pos, image_area):
        """
        Initialize the ObjectTracker.

        Args:
            fps (int): Frames per second of the video.
            frame_pos (int): Current frame position.
            image_area (int): Area of the video frame.
        """
        self.class_names = class_names  # List of class names to track
        self.active_tracks = {}  # Active tracks: {track_id: Track object}
        self.inactive_tracks = []  # Inactive tracks (lost tracks)
        self.distance_kf = KF.KalmanFilter()  # Kalman filter for distance smoothing

        self.frame = None  # Current video frame
        self.current_frame_id = frame_pos  # Current frame ID
        self.image_y_size = 0  # Height of the video frame
        self.image_area = image_area  # Area of the video frame
        self.fps = fps  # Frames per second of the video

        # Speed and distance thresholds
        self.max_speed = 100 / (0.18 * self.fps)  # Maximum allowed speed for distance changes
        self.max_allowed = 100  # Maximum allowed distance

        # Tracking state
        self.boxes = {class_name: None for class_name in self.class_names}  # Detected boxes for each class
        self.tracked_boxes = []  # List of tracked boxes
        self.penis_box = None  # Detected penis box
        self.locked_penis_box = LockedPenisBox()  # Locked penis box
        self.glans_detected = False  # Whether the glans is detected
        self.breast_tracking = False  # Whether breast tracking is active
        self.distance = 100  # Current distance
        self.previous_distances = [100, 100, 100]  # Previous distances for smoothing
        self.tracked_body_part = "Nothing"  # Currently tracked body part

        # Sex position tracking
        self.sex_position = "Not relevant"  # Current sex position
        self.sex_position_reason = ""  # Reason for the current sex position
        max_history = int(self.fps) * 10  # Maximum history for sex position tracking
        self.sex_position_history = deque(maxlen=max_history)  # History of sex positions

        # Position and distance tracking
        self.tracked_positions = {}  # Tracked positions for each track_id
        self.normalized_absolute_tracked_positions = {}  # Normalized absolute positions for each track_id
        self.normalized_distance_to_penis = {}  # Normalized distance to penis for each track_id

        # Initialize normalized positions and distances
        self.normalized_positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.normalized_distances = {class_name: deque(maxlen=200) for class_name in class_names}

        # Detection thresholds
        self.consecutive_detections = {class_name: 0 for class_name in class_names}
        self.consecutive_non_detections = {class_name: 0 for class_name in class_names}
        self.detections_threshold = 3  # Threshold for considering a detection valid

        # Penetration state
        self.penetration = False  # Whether penetration is detected
        self.close_up = False  # Whether a close-up is detected

    def update_distance(self, raw_distance):
        """
        Update the tracked distance using a Kalman filter and speed constraints.

        Args:
            raw_distance (int): Raw distance measurement.

        Returns:
            int: Filtered distance.
        """
        if raw_distance is None:
            # Predict distance using exponential moving average (EMA)
            filtered_distance = (
                0.1 * self.previous_distances[-3]
                + 0.2 * self.previous_distances[-2]
                + 0.7 * self.previous_distances[-1]
            )
        else:
            # Apply EMA to raw distance
            ema_distance = (
                0.1 * self.previous_distances[-2]
                + 0.2 * self.previous_distances[-1]
                + 0.7 * raw_distance
            )

            # Limit speed of distance change
            if abs(self.distance - ema_distance) > self.max_speed:
                ema_distance = self.distance + np.sign(ema_distance - self.distance) * self.max_speed

            filtered_distance = ema_distance

        # Ensure distance is within bounds (0-100)
        filtered_distance = int(max(0, min(100, filtered_distance)))
        self.distance = filtered_distance

        # Update previous distances
        self.previous_distances.pop(0)
        self.previous_distances.append(self.distance)

        return filtered_distance

    def boxes_overlap(self, box1, box2):
        """
        Check if two bounding boxes overlap.

        Args:
            box1 (tuple): Coordinates of the first box.
            box2 (tuple): Coordinates of the second box.

        Returns:
            bool: True if boxes overlap, False otherwise.
        """
        if box1 is None or box2 is None:
            return False
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

    def calculate_distance(self, penis_box, other_box):
        """
        Calculate the distance between the penis box and another box.

        Args:
            penis_box (tuple): Coordinates of the penis box.
            other_box (tuple): Coordinates of the other box.

        Returns:
            float: Distance between the boxes, or None if invalid.
        """
        if other_box is None:
            return None
        if penis_box is None:
            logging.warning(f"Penis box is None for frame {self.current_frame_id}, cannot compute distance")
            return None
        ox1, oy1, ox2, oy2 = other_box
        y_pos = (oy1 + 2 * oy2) // 3
        x_pos = (ox1 + ox2) // 2
        px1, py1, px2, py2 = penis_box
        return math.sqrt((x_pos - px2) ** 2 + (y_pos - py2) ** 2)

    def detect_sex_position_change(self, sex_position, reason):
        """
        Detect and log changes in the sex position.

        Args:
            sex_position (str): New sex position.
            reason (str): Reason for the change.
        """
        self.sex_position_history.append(sex_position)
        position_counts = {position: self.sex_position_history.count(position) for position in self.sex_position_history}
        most_frequent_position = max(position_counts, key=position_counts.get, default="Not relevant")
        if most_frequent_position != self.sex_position:
            logging.info(f"@{self.current_frame_id} - Sex position switched to: {most_frequent_position}")
            self.sex_position = most_frequent_position
            self.sex_position_reason = reason

    def tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        """
        Main tracking logic to process detected boxes and update tracking state.

        Args:
            sorted_boxes (list): List of detected boxes with confidence, class, and track ID.
            current_frame_id (int): Current frame ID.
            image_y_size (int): Height of the video frame.
        """
        self.current_frame_id = current_frame_id
        self.image_y_size = image_y_size

        close_up_detected = False

        # Initialize tracking state
        self.glans_detected = False
        self.boxes = {class_name: None for class_name in self.class_names}
        self.tracked_boxes = []
        all_detections = {class_name: [] for class_name in self.class_names}

        classes_touching_penis = []
        breast_tracked = None
        self.breast_tracking = False

        # Collect all detections by class name
        for box, conf, cls, class_name, track_id in sorted_boxes:
            if conf > 0.3:
                all_detections[class_name].append([conf, box, track_id])

        # Find the best box for specific classes
        found_box = {class_name: [] for class_name in self.class_names}
        for check_class_first in ['glans', 'penis', 'navel', 'breast']:
            prev_conf = 0
            for conf, box, track_id in all_detections[check_class_first]:
                if conf > prev_conf:
                    found_box[check_class_first] = [box, conf, track_id]
                    prev_conf = conf

        # Update tracking for specific classes
        for check_class_first in ['glans', 'penis', 'navel']:
            if found_box[check_class_first]:
                self.boxes[check_class_first] = found_box[check_class_first][0]
                conf = found_box[check_class_first][1]
                self.consecutive_detections[check_class_first] += 1
                self.consecutive_non_detections[check_class_first] = 0
                self.handle_class_first(check_class_first, self.boxes[check_class_first], conf)
            else:
                self.consecutive_detections[check_class_first] = 0
                self.consecutive_non_detections[check_class_first] += 1

        if self.consecutive_non_detections[check_class_first] > self.detections_threshold and check_class_first == 'penis':
            if self.locked_penis_box.is_active():
                self.locked_penis_box.deactivate()
                logging.info(f"@{self.current_frame_id} - Deactivated locked_penis_box")

        # Check if pussy boxes are inside butt boxes
        if 'pussy' in all_detections and 'butt' in all_detections:
            for pussy_detection in all_detections['pussy']:
                p_conf, p_box, p_track_id = pussy_detection

                for butt_detection in all_detections['butt']:
                    b_conf, b_box, b_track_id = butt_detection
                    b_x1, b_y1, b_x2, b_y2 = b_box

                    if self.boxes_overlap(p_box, b_box):
                        # Calculate the area of the butt box
                        butt_box_area = (b_x2 - b_x1) * (b_y2 - b_y1)

                        # Check if the butt box is unusually large (close-up)
                        if butt_box_area > 0.15 * self.image_area:
                            close_up_detected = True
                            self.detect_sex_position_change('Close up', 'Butt box size beyond threshold')
                            if not self.close_up:
                                logging.info(
                                    f"@{self.current_frame_id} - Close up detected - butt size beyond threshold: {int((butt_box_area / self.image_area) * 100)}%"
                                )
                                self.close_up = True
                            distance = 100
                            self.update_distance(distance)
                            return

        sum_pos = 0
        sum_weight_pos = 0

        for box, conf, cls, class_name, track_id in sorted_boxes:
            if class_name in ['glans', 'penis', 'navel', 'hips center']:
                continue
            elif self.locked_penis_box.is_active() and class_name == 'breast' and not self.boxes_overlap(box, self.locked_penis_box.get_box()):
                x1, y1, x2, y2 = box
                mid_y = (y1 + y2) // 2
                normalized_y = self.update_tracked_positions(track_id, mid_y)
                breast_tracked = normalized_y

            elif self.locked_penis_box.is_active() and self.boxes_overlap(box, self.locked_penis_box.get_box()):
                if class_name not in classes_touching_penis:
                    classes_touching_penis.append(class_name)

                self.tracked_boxes.append([box, class_name, track_id])
                x1, y1, x2, y2 = box
                mid_y = y2

                normalized_y = self.update_tracked_positions(track_id, mid_y)

                if class_name != 'hips center':
                    low_y = y2 if class_name != 'butt' else (0.8 * (y2 - y1)) + y1
                    dist_to_penis_base = int(
                        ((self.locked_penis_box.get_box()[3] - low_y) / self.locked_penis_box.get_height()) * 100
                    )
                    normalized_dist_to_penis_base = min(max(0, dist_to_penis_base), 100)
                    self.update_normalized_distance_to_penis(track_id, normalized_dist_to_penis_base)

                    # attempt : as a weight, we use the absolute position
                    # weight_pos_track_id = sum(
                    #    abs(self.tracked_positions[track_id][i] - self.tracked_positions[track_id][i - 1])
                    #    for i in range(1, len(self.tracked_positions[track_id])))

                    weight_pos_track_id = sum(
                        abs(self.normalized_distance_to_penis[track_id][i] - self.normalized_distance_to_penis[track_id][i - 1])
                        for i in range(1, len(self.normalized_distance_to_penis[track_id])))
                    sum_pos += ((self.normalized_distance_to_penis[track_id][-1] + normalized_y) // 2) * weight_pos_track_id

                    # attempt : use only the distance to penis value, discarding the absolute position
                    # sum_pos += self.normalized_distance_to_penis[track_id][-1] * weight_pos_track_id
                    sum_weight_pos += weight_pos_track_id

        if len(classes_touching_penis) == 0 or not self.locked_penis_box.is_active():
            self.penetration = False
            distance = 100
            self.detect_sex_position_change('Not relevant', "no part touching penis / no penis")
            self.tracked_body_part = 'Nothing'
        elif 'pussy' in classes_touching_penis and not self.glans_detected:
            if self.sex_position == 'Missionnary / Cowgirl':
                self.penetration = True
            self.detect_sex_position_change('Missionnary / Cowgirl', "pussy visible and touching")
            self.tracked_body_part = 'pussy'
        elif 'ass' in classes_touching_penis and not self.glans_detected:
            if self.sex_position == 'Doggy / Rev. Cowgirl':
                self.penetration = True
            self.detect_sex_position_change('Doggy / Rev. Cowgirl', "ass visible and touching")
            self.tracked_body_part = 'ass'
        elif ('hand' in classes_touching_penis or 'face' in classes_touching_penis) and not self.penetration:
            self.detect_sex_position_change('Handjob / Blowjob', "hand or face visible and touching")
            if 'face' not in classes_touching_penis:
                self.penetration = False
                self.tracked_body_part = 'hand'
        elif 'foot' in classes_touching_penis and not self.penetration:
            self.detect_sex_position_change('Footjob', "foot visible and touching")
        elif 'breast' in classes_touching_penis and not self.penetration:
            self.detect_sex_position_change('Boobjob', "breast visible and touching")
            self.tracked_body_part = 'breast'

        if sum_weight_pos > 0 and self.sex_position not in ['Not relevant', 'Close up']:
            distance = int(sum_pos / sum_weight_pos)
        elif sum_weight_pos == 0 and self.sex_position == 'Missionnary / Cowgirl' and 'pussy' not in classes_touching_penis and breast_tracked:
            distance = breast_tracked
            self.breast_tracking = True
        else:
            distance = 100

        self.update_distance(distance)

    def handle_class_first(self, class_name, box, conf):
        """
        Handle tracking for specific classes (e.g., penis, glans, navel).

        Args:
            class_name (str): Name of the class.
            box (tuple): Coordinates of the box.
            conf (float): Confidence of the detection.
        """
        if class_name == 'penis':
            if box is not None and self.penis_box is None:
                logging.info(f"@{self.current_frame_id} - Penis detected with confidence {conf}")
            self.penis_box = box

            if self.penis_box:
                if self.consecutive_detections['penis'] >= self.detections_threshold:
                    px1, py1, px2, py2 = self.penis_box
                    current_height = py2 - py1
                    if self.locked_penis_box.is_active():
                        if current_height > self.locked_penis_box.get_height():
                            self.locked_penis_box.update(self.penis_box, current_height)

                        # Move locked penis box towards current penis box
                        max_move = max(1, int(self.image_y_size / 960))

                        if abs(self.penis_box[0] - self.locked_penis_box.get_box()[0]) > max_move:
                            px1 = self.penis_box[0] + np.sign(self.penis_box[0] - self.locked_penis_box.get_box()[0]) * max_move
                        if abs(self.penis_box[2] - self.locked_penis_box.get_box()[2]) > max_move:
                            px2 = self.penis_box[2] + np.sign(self.penis_box[2] - self.locked_penis_box.get_box()[2]) * max_move
                        if abs(self.penis_box[3] - self.locked_penis_box.get_box()[3]) > max_move:
                            py2 = self.locked_penis_box.get_box()[3] + np.sign(self.penis_box[3] - self.locked_penis_box.get_box()[3]) * max_move

                        self.locked_penis_box.update((px1, py2 - self.locked_penis_box.get_height(), px2, py2), self.locked_penis_box.get_height())
                    else:
                        self.locked_penis_box.update(self.penis_box, current_height)
        elif class_name == 'glans' and box:
            if self.consecutive_detections['glans'] >= self.detections_threshold:
                self.boxes['glans'] = box
                self.glans_detected = True
                if self.penis_box:
                    self.locked_penis_box.update(self.penis_box, self.penis_box[3] - self.penis_box[1])
                if self.penetration:
                    self.penetration = False
                    logging.info(
                        f"@{self.current_frame_id} - Penetration ended after {self.consecutive_detections['glans']} detections of glans"
                    )
                    if self.tracked_body_part != 'Nothing':
                        self.normalized_distances[self.tracked_body_part].clear()
                        self.normalized_distances[self.tracked_body_part].append(100)

    def update_tracked_positions(self, track_id, mid_y):
        """
        Update tracked positions and normalized positions for a given track_id.

        Args:
            track_id (int): ID of the track.
            mid_y (int): Midpoint y-coordinate of the box.

        Returns:
            int: Normalized y-coordinate.
        """
        if track_id not in self.tracked_positions:
            self.tracked_positions[track_id] = []
        if len(self.tracked_positions[track_id]) > 2:
            mid_y = (
                0.2 * self.tracked_positions[track_id][-2]
                + 0.3 * self.tracked_positions[track_id][-1]
                + 0.5 * mid_y
            )
        self.tracked_positions[track_id].append(int(mid_y))
        if len(self.tracked_positions[track_id]) > 600:
            self.tracked_positions[track_id].pop(0)

        if track_id not in self.normalized_absolute_tracked_positions:
            self.normalized_absolute_tracked_positions[track_id] = []
        min_y, max_y = min(self.tracked_positions[track_id]), max(self.tracked_positions[track_id])
        normalized_y = 100 if max_y - min_y == 0 else int(min(max(0, 100 - (((mid_y - min_y) / (max_y - min_y)) * 100)), 100))
        self.normalized_absolute_tracked_positions[track_id].append(normalized_y)
        if len(self.normalized_absolute_tracked_positions[track_id]) > 60:
            self.normalized_absolute_tracked_positions[track_id].pop(0)

        return normalized_y

    def update_normalized_distance_to_penis(self, track_id, normalized_dist_to_penis_base):
        """
        Update normalized distance to penis for a given track_id.

        Args:
            track_id (int): ID of the track.
            normalized_dist_to_penis_base (int): Normalized distance to the base of the penis.
        """
        if track_id not in self.normalized_distance_to_penis:
            self.normalized_distance_to_penis[track_id] = []
        if len(self.normalized_distance_to_penis[track_id]) > 2:
            normalized_dist_to_penis_base = (
                0.2 * self.normalized_distance_to_penis[track_id][-2]
                + 0.3 * self.normalized_distance_to_penis[track_id][-1]
                + 0.5 * normalized_dist_to_penis_base
            )
        self.normalized_distance_to_penis[track_id].append(normalized_dist_to_penis_base)
        if len(self.normalized_distance_to_penis[track_id]) > 60:
            self.normalized_distance_to_penis[track_id].pop(0)

    def normalize_box_area(self, box, frame_width, frame_height):
        """
        Normalize the area of a bounding box relative to the frame size.

        Args:
            box (tuple): Coordinates of the box.
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.

        Returns:
            int: Normalized area (0-100).
        """
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return 0
        box_area = (x2 - x1) * (y2 - y1)
        max_area = frame_width * frame_height
        normalized_area = 100 * (box_area / max_area)
        return max(0, min(100, normalized_area))
