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
        # Initialize class attributes
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
        self.max_speed = 100 / (.18 * self.fps)  # Maximum allowed speed for distance changes
        # TEST - trying to fix the speed issue by temporarily doubling it up
        #self.max_speed = int(self.max_speed * 3)

        self.max_allowed = 100  # Maximum allowed distance

        # Tracking state
        self.boxes = {}  # Detected boxes for each class
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
        self.areas = {class_name: deque(maxlen=200) for class_name in class_names}  # Areas for each class

        # Normalized positions and distances
        self.normalized_positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.normalized_distances = {class_name: deque(maxlen=200) for class_name in class_names}

        # Tracked positions and normalized tracked positions
        self.tracked_positions = {}
        self.normalized_absolute_tracked_positions = {}
        self.normalized_distance_to_penis = {}

        # Initialize normalized distances and positions
        for class_name in class_names:
            self.normalized_distances[class_name].append(100)
            self.normalized_positions[class_name].append(100)

        # Detection thresholds
        self.consecutive_detections = {class_name: 0 for class_name in class_names}  # Consecutive detections for each class
        self.consecutive_non_detections = {class_name: 0 for class_name in class_names}  # Consecutive non-detections for each class
        self.detections_threshold = 3  # Threshold for considering a detection valid

        # Penetration state
        self.penetration = False  # Whether penetration is detected

        self.close_up = False


    def update_distance(self, raw_distance):
        """
        Update the tracked distance using a Kalman filter and speed constraints.
        """
        if raw_distance is None:
            #filtered_distance = self.distance_kf.predict()
            # simple predict using ema, waiting for kalman filter fix
            filtered_distance = 0.1 * self.previous_distances[-3] + 0.2 * self.previous_distances[-2] + 0.7 * self.previous_distances[-1]
            #print(f"predicted distance: {filtered_distance}")
        else:
            #print(f"before fix, raw distance is: {raw_distance}")
            # applying ema
            ema_distance = 0.1 * self.previous_distances[-2] + 0.2 * self.previous_distances[-1] + 0.7 * raw_distance
            #print(f"self distance: {self.distance}, ema_distance: {ema_distance}, max speed: {self.max_speed}, delta distances: {self.distance-ema_distance}")
            if abs(self.distance - ema_distance) > self.max_speed:
                # print(f"Limited speed: {self.max_speed}, was {int(self.distance - ema_distance)}")
                ema_distance = self.distance + np.sign(ema_distance - self.distance) * self.max_speed
                #print(f"adjusted ema distance: {ema_distance}")
            #raw_distance_2d = (raw_distance, 0)
            # Update the Kalman filter with the new measurement
            #self.distance_kf.update(raw_distance_2d)

            # Get the filtered distance from the Kalman filter
            #filtered_k_distance = self.distance_kf.position[0]

            #print(f"Kalman filtered distance: {filtered_k_distance} vs {raw_distance}")
            filtered_distance = ema_distance
            #filtered_distance = raw_distance

        filtered_distance = int(max(0, min(100, filtered_distance)))
        self.distance = filtered_distance
        self.previous_distances.pop(0)
        self.previous_distances.append(self.distance)

        return filtered_distance

    def _predict_distance(self):
        """Predict the distance using the Kalman filter."""
        return int(float(self.distance_kf.predict()[0]) / 5) * 5

    def boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap."""
        if box1 is None or box2 is None:
            return False
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

    def calculate_distance(self, penis_box, other_box):
        """Calculate the distance between the penis box and another box."""
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
        """Detect and log changes in the sex position."""
        self.sex_position_history.append(sex_position)
        position_counts = {position: self.sex_position_history.count(position) for position in
                           self.sex_position_history}
        most_frequent_position = max(position_counts, key=position_counts.get, default="Not relevant")
        if most_frequent_position != self.sex_position:
            logging.info(f"@{self.current_frame_id} - Sex position switched to: {most_frequent_position}")
            self.sex_position = most_frequent_position
            self.sex_position_reason = reason

    def tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        """
        Main tracking logic to process detected boxes and update tracking state.
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

        if self.consecutive_non_detections[
            check_class_first] > self.detections_threshold and check_class_first == 'penis':
            if self.locked_penis_box.is_active():
                self.locked_penis_box.deactivate()
                logging.info(f"@{self.current_frame_id} - Deactivated locked_penis_box")


        """
        # checking for close up in case butt or pussy box are unusually large
        for class_name in ['butt', 'pussy']:
            for detection in all_detections[class_name]:
                conf, box, track_id = detection
                # if box[3] - box[1] > 0.15 * image_y_size:
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                #if box_area > 0:
                #    print(f"{class_name} area: {box_area}, size: {int((box_area/self.image_area)*100)}%")
                if box_area > 0.12 * self.image_area:
                    close_up_detected = True
                    self.detect_sex_position_change('Close up', f'{class_name} size beyond threshold')
                    if not self.close_up:
                        logging.info(f"@{self.current_frame_id} - Close up detected - {class_name} size beyond threshold: {int((box_area/self.image_area)*100)}%")
                        #self.detect_sex_position_change('Close up', f'{class_name} size beyond threshold')
                        self.close_up = True
                    distance = 100
                    self.update_distance(distance)
                    return
        """

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
                                    f"@{self.current_frame_id} - Close up detected - butt size beyond threshold: {int((butt_box_area / self.image_area) * 100)}%")
                                # self.detect_sex_position_change('Close up', f'{class_name} size beyond threshold')
                                self.close_up = True
                            distance = 100
                            self.update_distance(distance)
                            return

        sum_pos = 0
        sum_weight_pos = 0

        for box, conf, cls, class_name, track_id in sorted_boxes:
            if class_name in ['glans', 'penis', 'navel', 'hips center']:
            #if class_name in ['glans', 'navel']:
                continue
            elif self.locked_penis_box.is_active() and class_name == 'breast' and not self.boxes_overlap(box, self.locked_penis_box.get_box()):
                x1, y1, x2, y2 = box
                mid_y = (y1 + y2) // 2
                # will need to refactor all that...
                # Update tracked positions within the frame
                if track_id not in self.tracked_positions:
                    self.tracked_positions[track_id] = []
                # let's apply a moving average to try and filter outliers
                if len(self.tracked_positions[track_id]) > 2:
                    mid_y = 0.2 * self.tracked_positions[track_id][-2] + 0.3 * \
                            self.tracked_positions[track_id][
                                -1] + 0.5 * mid_y
                self.tracked_positions[track_id].append(int(mid_y))
                # Maintain a fixed-size history
                if len(self.tracked_positions[track_id]) > 600:
                    self.tracked_positions[track_id].pop(0)

                # Normalize absolute position 0-100 based on historic min and max
                if track_id not in self.normalized_absolute_tracked_positions:
                    self.normalized_absolute_tracked_positions[track_id] = []
                # let's retrieve min and max from self.tracked_positions
                min_y, max_y = min(self.tracked_positions[track_id]), max(self.tracked_positions[track_id])
                # now, normalize
                if max_y - min_y == 0:
                    normalized_y = 100
                else:
                    normalized_y = int(min(max(0, 100 - (((mid_y - min_y) / (max_y - min_y)) * 100)), 100))
                self.normalized_absolute_tracked_positions[track_id].append(normalized_y)
                # Maintain a fixed-size history
                if len(self.normalized_absolute_tracked_positions[track_id]) > 60:
                    self.normalized_absolute_tracked_positions[track_id].pop(0)
                breast_tracked = normalized_y

            elif self.locked_penis_box.is_active() and self.boxes_overlap(box, self.locked_penis_box.get_box()):
                #or class_name == 'hips center'):
                if class_name not in classes_touching_penis:
                    classes_touching_penis.append(class_name)

                self.tracked_boxes.append([box, class_name, track_id])
                x1, y1, x2, y2 = box
                # mid_y = (y1 + y2) // 2  # not happy with this, trying alternative
                mid_y = y2

                # Update tracked positions within the frame
                if track_id not in self.tracked_positions:
                    self.tracked_positions[track_id] = []
                # let's apply a moving average to try and filter outliers
                if len(self.tracked_positions[track_id]) > 2:
                    mid_y = 0.1 * self.tracked_positions[track_id][-2] + 0.2 * \
                                                    self.tracked_positions[track_id][
                                                        -1] + 0.7 * mid_y
                self.tracked_positions[track_id].append(int(mid_y))
                # Maintain a fixed-size history
                if len(self.tracked_positions[track_id]) > 600:
                    self.tracked_positions[track_id].pop(0)

                # Normalize absolute position 0-100 based on historic min and max
                if track_id not in self.normalized_absolute_tracked_positions:
                    self.normalized_absolute_tracked_positions[track_id] = []
                # let's retrieve min and max from self.tracked_positions
                min_y, max_y = min(self.tracked_positions[track_id]), max(self.tracked_positions[track_id])
                # now, normalize
                if max_y - min_y == 0:
                    normalized_y = 100
                else:
                    normalized_y = int(min(max(0, 100 - (((mid_y - min_y) / (max_y - min_y)) * 100)), 100))
                self.normalized_absolute_tracked_positions[track_id].append(normalized_y)
                # Maintain a fixed-size history
                if len(self.normalized_absolute_tracked_positions[track_id]) > 60:
                    self.normalized_absolute_tracked_positions[track_id].pop(0)

                # Now, we compute the distance to penis base
                if track_id not in self.normalized_distance_to_penis:
                    self.normalized_distance_to_penis[track_id] = []
                if class_name == 'butt':  # 80% of (y2 - y1) + y1
                    low_y = (.8 * (y2 - y1)) + y1
                elif class_name != 'hips center':
                    low_y = y2

                # Now, we compute the distance to penis base for classes that are not 'hips center'
                # And computing the weighted position based on recent moves
                if class_name != 'hips center':
                    # compute the distance from low_y to penis base
                    dist_to_penis_base = int(
                        ((self.locked_penis_box.get_box()[3] - low_y) / self.locked_penis_box.get_height()) * 100)
                    # maintain the distance 0-100
                    normalized_dist_to_penis_base = min(max(0, dist_to_penis_base), 100)
                    # let's apply a moving average to try and filter outliers
                    if len(self.normalized_distance_to_penis[track_id]) > 2:
                        normalized_dist_to_penis_base = 0.2 * self.normalized_distance_to_penis[track_id][-2] + 0.3 * \
                                self.normalized_distance_to_penis[track_id][
                                    -1] + 0.5 * normalized_dist_to_penis_base

                    # making sure the distance remains within 0-100 before appending
                    normalized_dist_to_penis_base = min(max(0, normalized_dist_to_penis_base), 100)
                    # Update normalized tracked positions
                    #if track_id not in self.normalized_distance_to_penis:
                    #    self.normalized_distance_to_penis[track_id] = []
                    self.normalized_distance_to_penis[track_id].append(normalized_dist_to_penis_base)
                    # Maintain a fixed-size history
                    if len(self.normalized_distance_to_penis[track_id]) > 60:
                        self.normalized_distance_to_penis[track_id].pop(0)

                    # Sum delta positions of all touching items, weighted by the length of their history
                    weight_pos_track_id = sum(
                        abs(self.normalized_distance_to_penis[track_id][i] - self.normalized_distance_to_penis[track_id][
                            i - 1]) for i in
                        range(1, len(self.normalized_distance_to_penis[track_id])))
                    #sum_pos += max(0, (normalized_dist_to_penis_base - (100 - self.max_allowed))) * weight_pos_track_id
                    sum_pos += ((self.normalized_distance_to_penis[track_id][-1] +
                                self.normalized_absolute_tracked_positions[track_id][-1]) // 2) * weight_pos_track_id
                    sum_weight_pos += weight_pos_track_id
                else:  # hips center
                    pass
                    # Need to fix this...
                    #if self.penetration:  # only consider them during penetration
                    #    weight_pos_track_id = sum(
                    #        abs(self.normalized_absolute_tracked_positions[track_id][i] -
                    #            self.normalized_absolute_tracked_positions[track_id][
                    #                i - 1]) for i in
                    #        range(1, len(self.normalized_absolute_tracked_positions[track_id])))
                    #    sum_pos += self.normalized_absolute_tracked_positions[track_id][-1] * weight_pos_track_id
                    #    sum_weight_pos += weight_pos_track_id

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
        # attempt, in case we lose the pussy during grinding for instance
        elif sum_weight_pos == 0 and self.sex_position == 'Missionnary / Cowgirl' and 'pussy' not in classes_touching_penis and breast_tracked:
            distance = breast_tracked
            self.breast_tracking = True
        else:
            distance = 100

        self.update_distance(distance)

    def handle_class_first(self, class_name, box, conf):
        """
        Handle tracking for specific classes (e.g., penis, glans, navel).
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
                    else: # activating locked penis box
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
                        f"@{self.current_frame_id} - Penetration ended after {self.consecutive_detections['glans']} detections of glans")
                    if self.tracked_body_part != 'Nothing':
                        self.normalized_distances[self.tracked_body_part].clear()
                        self.normalized_distances[self.tracked_body_part].append(100)
        elif class_name == 'navel' and box:
            if (self.image_y_size - box[1]) / self.image_y_size < 0.15 and not self.breast_tracking:
                logging.info("@{self.current_frame_id} - Breast tracking mode activated given navel position in lower 15th of frame")
                self.breast_tracking = True

    def normalize_box_area(self, box, frame_width, frame_height):
        """Normalize the area of a bounding box relative to the frame size."""
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return 0
        box_area = (x2 - x1) * (y2 - y1)
        max_area = frame_width * frame_height
        normalized_area = 100 * (box_area / max_area)
        return max(0, min(100, normalized_area))

    """def log_and_normalize_pos(self, box, track_id):
        self.boxes[class_name] = box
        _, y1, _, y2 = box
        mid_y = (y1 + y2) / 2
        self.positions[class_name].append(mid_y)

        min_y, max_y = min(self.positions[class_name]), max(self.positions[class_name])
        normalized_y = (100 - int(100 * ((mid_y - min_y) / (max_y - min_y)))) if min_y != max_y else 100

        # In case of breast, which is a fallback class to pussy, we compute not only the normalized position
        # but also the normalized breast area which shows back and forth in case of grinding
        if class_name == "breast" and self.frame:
            normalized_breast_area = self.normalize_box_area(box, self.frame.shape[1], self.frame.shape[0])
            normalized_y = ((0.75 * normalized_breast_area) + normalized_y) / 1.75

        # self.normalized_positions[class_name][current_frame_id] = normalized_y
        self.normalized_positions[class_name].append(normalized_y)
        return normalized_y
    """
