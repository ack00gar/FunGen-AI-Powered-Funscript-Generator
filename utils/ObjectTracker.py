from utils import KalmanFilter as KF
from collections import deque
import numpy as np
from utils.config import class_names
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ObjectTracker:
    def __init__(self, fps, frame_pos, state=None):
        self.class_names = class_names
        self.active_tracks = {}  # Key: track_id, Value: Track object
        self.inactive_tracks = []  # List of Track objects for lost tracks
        self.distance_kf = KF.KalmanFilter()

        self.frame = None
        self.current_frame_id = frame_pos
        self.image_y_size = 0
        self.fps = fps

        # if trying to limit speed to accomodate Handy device : 40cm/s on 11cm range
        # Handy : 11 / 40 = 0.275
        # Assuming devices like 0SR2 can go double speed
        # OSR2 : 11 / 80 = 0.13
        # Settling down to an average of 60cm/s => 11 / 60 = 0.18
        # @60fps = 0.275 * 60 = 16.8, max_movement on 1 frame: 100 / 16.8 = 5.9
        # @30fps = 0.275 * 30 = 8.1, max_movement on 1 frame: 100 / 8.1 = 12.5
        self.max_speed = 100 / (.18 * self.fps)

        self.max_allowed = 100

        self.boxes = {}
        self.tracked_boxes = []
        self.avg_move = {}

        self.penis_box, self.locked_penis_box = None, None
        self.glans_detected = False
        self.locked_penis_height = 0
        self.breast_tracking = False
        self.distance = 100
        self.raw_distance = 100
        self.previous_distances = [100, 100, 100]
        self.tracked_body_part = "Nothing"

        self.sex_position = "Not relevant"
        self.prev_sex_position = "Not relevant"
        self.sex_position_reason = ""
        self.sex_position_history = deque(maxlen=30)
        self.sub_sex_position = "Not relevant"

        self.positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.distances = {class_name: deque(maxlen=200) for class_name in class_names}
        self.areas = {class_name: deque(maxlen=200) for class_name in class_names}

        self.normalized_positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.normalized_distances = {class_name: deque(maxlen=200) for class_name in class_names}
        self.activated_kalman = {class_name: False for class_name in class_names}

        self.tracked_positions = {}
        self.normalized_tracked_positions = {}

        self.offsets = {class_name: 0 for class_name in class_names}

        for class_name in class_names:
            self.normalized_distances[class_name].append(100)
            self.normalized_positions[class_name].append(100)

        self.normalized_areas = {class_name: deque(maxlen=200) for class_name in class_names}

        self.moving_average_window = 5
        self.consecutive_detections = {class_name: 0 for class_name in class_names}
        self.consecutive_non_detections = {class_name: 0 for class_name in class_names}
        self.detections_threshold = 3

        self.consecutive_grinding_suspicions = {class_name: 0 for class_name in ['pussy', 'butt']}

        self.penetration = False
        self.grinding = False
        self.rubbing = False

        self.max_predictions = int(self.fps)
        self.max_inactive_frames = int(fps * 2)  # Adjust based on expected occlusion duration
        self.distance_threshold = 50  # Define an appropriate distance threshold

    def update_distance(self, raw_distance):
        if raw_distance is None:
            filtered_distance = self._predict_distance()
        else:
            if abs (self.distance - raw_distance) > self.max_speed:
                raw_distance = self.distance + np.sign(raw_distance - self.distance) * self.max_speed
            filtered_distance = raw_distance
            #rounded_distance = round(raw_distance / 5) * 5
            #ema_distance = self._calculate_ema_distance(rounded_distance)
            #self._update_previous_distances(ema_distance)
            #self.distance_kf.update((ema_distance, 0))
            #filtered_distance = ema_distance

        filtered_distance = int(max(0, min(100, filtered_distance)))
        self.distance = filtered_distance
        return filtered_distance

    def _predict_distance(self):
        return int(float(self.distance_kf.predict()[0]) / 5) * 5

    def _calculate_ema_distance(self, rounded_distance):
        return int(0.7 * rounded_distance + 0.1 * self.previous_distances[1] + 0.2 * self.previous_distances[2])

    def _update_previous_distances(self, ema_distance):
        if abs(ema_distance - self.previous_distances[-1]) > 15:
            ema_distance = self.previous_distances[-1] + np.sign(ema_distance - self.previous_distances[-1]) * 15
        elif abs(ema_distance - self.previous_distances[-1]) < 2:
            ema_distance = self.previous_distances[-1]
        self.previous_distances.pop(0)
        self.previous_distances.append(ema_distance)

    def boxes_overlap(self, box1, box2):
        if box1 is None or box2 is None:
            return False
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

    def calculate_distance(self, penis_box, other_box):
        print(f"penis box argument in calculate_distance: {penis_box}")
        if other_box is None:
            return None
        if penis_box is None:
            print(f"penis box: {penis_box}")
            print(f"Penis box is None for frame {self.current_frame_id}, cannot compute distance")
            return None
        ox1, oy1, ox2, oy2 = other_box
        y_pos = (oy1 + 2 * oy2) // 3
        x_pos = (ox1 + ox2) // 2
        px1, py1, px2, py2 = penis_box
        return math.sqrt((x_pos - px2) ** 2 + (y_pos - py2) ** 2)

    def detect_sex_position_change(self, sex_position, reason):
        self.sex_position_history.append(sex_position)
        position_counts = {position: self.sex_position_history.count(position) for position in
                           self.sex_position_history}
        most_frequent_position = max(position_counts, key=position_counts.get, default="Not relevant")
        if most_frequent_position != self.sex_position:
            logging.info(f"@{self.current_frame_id} - Sex position switched to: {most_frequent_position}")
            self.sex_position = most_frequent_position
            self.sex_position_reason = reason

    def tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        self.current_frame_id = current_frame_id
        self.image_y_size = image_y_size

        # Collect all detections
        detections = [(box, conf, class_name, track_id) for box, conf, cls, class_name, track_id in sorted_boxes]

        # Initialize tracking state
        self.glans_detected = False
        self.boxes = {class_name: None for class_name in self.class_names}
        self.tracked_boxes = []
        #self.avg_move = {}
        all_detections = {class_name: [] for class_name in self.class_names}

        classes_touching_penis = []

        # Collect all detections by class name
        for box, conf, cls, class_name, track_id in sorted_boxes:
            if class_name == "penis":
                print(f"penis detected @1")
            if conf > 0.3:
                all_detections[class_name].append([conf, box, track_id])

        # Find the best box for specific classes
        found_box = {class_name: [] for class_name in self.class_names}
        for check_class_first in ['glans', 'penis', 'navel']:
            prev_conf = 0
            for conf, box, track_id in all_detections[check_class_first]:
                if conf > prev_conf:
                    if check_class_first == 'penis':
                        print("penis detected @2")
                    found_box[check_class_first] = [box, conf, track_id]
                    prev_conf = conf

        # Update tracking for specific classes
        for check_class_first in ['glans', 'penis', 'navel']:
            if found_box[check_class_first]:
                self.boxes[check_class_first] = found_box[check_class_first][0]
                conf = found_box[check_class_first][1]
                self.consecutive_detections[check_class_first] += 1
                self.consecutive_non_detections[check_class_first] = 0
                if check_class_first == 'penis':
                    print("sending to handle penis")
                self.handle_class_first(check_class_first, self.boxes[check_class_first], conf)
            else:
                self.consecutive_detections[check_class_first] = 0
                self.consecutive_non_detections[check_class_first] += 1

        if self.consecutive_non_detections[
            check_class_first] > self.detections_threshold and check_class_first == 'penis':
            if self.locked_penis_box:
                self.locked_penis_box = None
                logging.info(f"@{self.current_frame_id} - Deactivated locked_penis_box")
            #self.handle_class_first(check_class_first, box, conf)


        # First, how much of the penis can we see, 0 to 100 as a baseline value

        #if self.locked_penis_box and found_box['penis'] and self.boxes_overlap(self.penis_box, found_box['penis'][0]):
        #    self.visible_penis_height = int(((self.locked_penis_box[3] - found_box['penis'][0][1]) / self.locked_penis_height) * 100)
        #    self.max_allowed = self.visible_penis_height
        #else:
        #    self.visible_penis_height = None
        #    self.max_allowed = 100

        sum_pos = 0
        weight_pos = 0

        computation_method = 1

        target_pos = None
        for box, conf, cls, class_name, track_id in sorted_boxes:
            if class_name in ['glans', 'penis', 'navel']:
                continue
            if self.boxes_overlap(box, self.locked_penis_box):
                if class_name not in classes_touching_penis:
                    classes_touching_penis.append(class_name)
                self.tracked_boxes.append([box, class_name, track_id])
                x1, y1, x2, y2 = box
                mid_y = (y1 + 4 * y2) // 5
                dist_to_penis_base = int(((self.locked_penis_box[3] - mid_y) / (self.locked_penis_height * .8)) * 100)
                # print(f"locked_penis_height: {self.locked_penis_height}, locked_penis_box[3]: {self.locked_penis_box[3]}, mid_y: {mid_y}, dist_to_penis_base: {dist_to_penis_base}")
                # print(f"For class {class_name}, mid_y: {mid_y}, dist_to_penis_base: {dist_to_penis_base}")
                normalized_y = min(max(0, dist_to_penis_base), 100)

                # Update tracked positions
                if track_id not in self.tracked_positions:
                    self.tracked_positions[track_id] = []
                # self.tracked_positions[track_id].append(mid_y)
                self.tracked_positions[track_id].append(normalized_y)

                # Maintain a fixed-size history
                if len(self.tracked_positions[track_id]) > 600:
                    self.tracked_positions[track_id].pop(0)

                # Normalize the y position
                # min_y, max_y = min(self.tracked_positions[track_id]), max(self.tracked_positions[track_id])
                # normalized_y = (100 - int(100 * ((mid_y - min_y) / (max_y - min_y)))) if min_y != max_y else 100

                # Update normalized tracked positions
                if track_id not in self.normalized_tracked_positions:
                    self.normalized_tracked_positions[track_id] = []
                self.normalized_tracked_positions[track_id].append(normalized_y)

                # Maintain a fixed-size history
                if len(self.normalized_tracked_positions[track_id]) > 60:
                    self.normalized_tracked_positions[track_id].pop(0)

                # sum delta positions of all touching items, ponderated by the length of their history
                weight_pos_track_id = sum(
                    abs(self.normalized_tracked_positions[track_id][i] - self.normalized_tracked_positions[track_id][
                        i - 1]) for i in
                    range(1, len(self.normalized_tracked_positions[track_id])))
                sum_pos += max(0, (normalized_y - (100 - self.max_allowed))) * weight_pos_track_id
                weight_pos += weight_pos_track_id

        if len(classes_touching_penis) == 0 or self.locked_penis_box is None:
            self.penetration = False
            distance = 100
            self.detect_sex_position_change('Not relevant', "no part touching penis / no penis")
        elif 'pussy' in classes_touching_penis and not self.glans_detected:
            self.penetration = True
            self.detect_sex_position_change('Missionnary / Cowgirl', "pussy visible and touching")
        elif 'ass' in classes_touching_penis and not self.glans_detected:
            self.penetration = True
            self.detect_sex_position_change('Doggy / Rev. Cowgirl', "ass visible and touching")
        elif 'hand' in classes_touching_penis or 'face' in classes_touching_penis:
            self.detect_sex_position_change('Handjob / Blowjob', "hand or face visible and touching")
        elif 'foot' in classes_touching_penis:
            self.detect_sex_position_change('Footjob', "foot visible and touching")
        elif 'breast' in classes_touching_penis:
            self.detect_sex_position_change('Boobjob', "breast visible and touching")


        if weight_pos > 0:
            distance = int(sum_pos / weight_pos)
        else:
            distance = 100

        """if self.penetration:
            if self.penis_box:
                # visible part of the penis
                weighted_distance = distance
                penis_height = self.penis_box[3] - self.penis_box[1]
                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
                normalized_penis_height = int((penis_height / locked_penis_height) * 100)
                distance_real = normalized_penis_height
                distance = int((distance_real + weighted_distance) / 2)
                #print(f"distance_real: {distance_real} vs distance: {weighted_distance} => {distance}")
            else:
                distance = 100
        """


        self.update_distance(distance)

        """

        if self.penetration and class_name == 'breast' and not self.boxes_overlap(box, self.locked_penis_box):
            x1, y1, x2, y2 = box
            mid_y = (y1 + y2) // 2

            # Update tracked positions
            if track_id not in self.tracked_positions:
                self.tracked_positions[track_id] = []
            self.tracked_positions[track_id].append(mid_y)

            # Maintain a fixed-size history
            if len(self.tracked_positions[track_id]) > 600:
                self.tracked_positions[track_id].pop(0)

            if max(self.tracked_positions[track_id]) != min(self.tracked_positions[track_id]):
                normalized_y = mid_y / (max(self.tracked_positions[track_id]) - min(self.tracked_positions[track_id]))
                # Update normalized tracked positions
                if track_id not in self.normalized_tracked_positions:
                    self.normalized_tracked_positions[track_id] = []
                self.normalized_tracked_positions[track_id].append(normalized_y)
                weight_pos_track_id = sum(
                    abs(self.normalized_tracked_positions[track_id][i] - self.normalized_tracked_positions[track_id][
                        i - 1]) for i in
                    range(1, len(self.normalized_tracked_positions[track_id])))
                sum_pos += max(0, (normalized_y - (100 - self.max_allowed))) * weight_pos_track_id
                weight_pos += weight_pos_track_id
        """
        # Check if the box overlaps with the locked penis box
        #el



        # Calculate average movement over the last 30 frames
        """
        if track_id not in self.avg_move:
            self.avg_move[track_id] = 0
        else:
            if len(self.tracked_positions[track_id]) > 30:
                total_move = sum(
                    abs(self.tracked_positions[track_id][i] - self.tracked_positions[track_id][i - 1]) for i in
                    range(1, 31))
                self.avg_move[track_id] = total_move #/ 30
            else:
                self.avg_move[track_id] = 0

        nb_items_touching_penis += 1
        """

        # Determine the target track ID based on movement
        """
        if nb_items_touching_penis > 0:
            # Filter out tracks with minimal movement (idle state)
            active_tracks = {track_id: move for track_id, move in self.avg_move.items() if
                             move > 5}  # Threshold for movement
            if active_tracks:
                print("active tracks")
                target_track_id = max(active_tracks, key=active_tracks.get)
                distance = self.normalized_tracked_positions[target_track_id][-1]
                print(f"@{self.current_frame_id} : Tracking {target_track_id} at {distance}")
            else:
                # Fallback to the previously tracked ID if no active movement
                if hasattr(self, 'last_tracked_id') and self.last_tracked_id in self.normalized_tracked_positions:
                    target_track_id = self.last_tracked_id
                    distance = self.normalized_tracked_positions[target_track_id][-1]
                    print(f"@{self.current_frame_id} : Fallback to {target_track_id} at {distance}")
                else:
                    distance = None
        else:
            distance = None

        # Update the last tracked ID
        if distance is not None:
            self.last_tracked_id = target_track_id
        """

    def handle_class_first(self, class_name, box, conf):
        if class_name == 'penis':
            if box is not None and self.penis_box is None:
                logging.info(f"Penis detected at frame {self.current_frame_id} with confidence {conf}")
            self.penis_box = box
            #print(f"self.penis_box: {self.penis_box}")

            #if self.consecutive_non_detections['penis'] >= self.detections_threshold:
            #    if self.locked_penis_box:
            #        self.locked_penis_box = None
            #        self.locked_penis_height = 0
            #        self.penetration = False
            #        logging.info(f"@{self.current_frame_id} - Locked penis box deactivated")
            if self.penis_box:
                if self.consecutive_detections['penis'] >= self.detections_threshold:
                    px1, py1, px2, py2 = self.penis_box
                    current_height = py2 - py1
                    # move locked penis box to current penis box
                    if self.locked_penis_box:
                        if current_height > self.locked_penis_height:
                            self.locked_penis_height = current_height

                        # moving locked penis box towards current penis box
                        max_move = max(1, int(self.image_y_size / 960))

                        if abs(self.penis_box[0] - self.locked_penis_box[0]) > max_move:
                            px1 = self.penis_box[0] + np.sign(self.penis_box[0] - self.locked_penis_box[0]) * max_move
                        if abs(self.penis_box[2] - self.locked_penis_box[2]) > max_move:
                            px2 = self.penis_box[2] + np.sign(self.penis_box[2] - self.locked_penis_box[2]) * max_move
                        if abs(self.penis_box[3] - self.locked_penis_box[3]) > max_move:
                            print("adjusting now")
                            py2 = self.locked_penis_box[3] + np.sign(self.penis_box[3] - self.locked_penis_box[3]) * max_move

                        self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
                    else:
                        self.locked_penis_box = self.penis_box
                        self.locked_penis_height = current_height
                    """    
                    
                    current_height = py2 - py1
                    if self.locked_penis_box is None or self.glans_detected: # or current_height > self.locked_penis_height:
                        if self.locked_penis_box is None:
                            logging.info(f"@{self.current_frame_id} - Locked penis box activated")
                        self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
                        self.locked_penis_height = current_height
                    """
                #if self.locked_penis_box and self.glans_detected:
                #    if current_height > self.locked_penis_height:
                #        self.locked_penis_height = current_height
                #        self.locked_penis_box = self.penis_box
                #    if self.penis_box[3] != self.locked_penis_box[3]:
                #        self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
            #if self.locked_penis_box and self.penetration:
            #    penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]
            #    locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
            #    if locked_penis_height == 0:
            #        scale = 100
            #    else:
            #        scale = min(int((penis_height / locked_penis_height) * 100), 100)
            #    distance = 100 - scale
            #    self.positions['penis'].append(distance)
            #    min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
            #    normalized_distance = (100 - int(100 * ((distance - min_distance) / (
            #                max_distance - min_distance)))) if min_distance != max_distance else 100
            #    self.normalized_positions['penis'].append(normalized_distance)
            #if (self.image_y_size - py1) / self.image_y_size < 0.1 and not self.breast_tracking and \
            #        self.tracked_boxes['breast']['detected']:
            #        #self.tracked_objects['breast']['detected']:
            #    self.breast_tracking = True
        elif class_name == 'glans' and box:
            if self.consecutive_detections['glans'] >= self.detections_threshold:
                self.boxes['glans'] = box
                self.glans_detected = True
                if self.penis_box:
                    self.locked_penis_box = self.penis_box
                    self.locked_penis_height = self.penis_box[3] - self.penis_box[1]
                if self.penetration:
                    self.penetration = False
                    logging.info(
                        f"@{self.current_frame_id} - Penetration ended after {self.consecutive_detections['glans']} detections of glans")
                    if self.tracked_body_part != 'Nothing':
                        self.normalized_distances[self.tracked_body_part].clear()
                        self.normalized_distances[self.tracked_body_part].append(100)
        elif class_name == 'navel' and box:
            if (self.image_y_size - box[1]) / self.image_y_size < 0.15 and not self.breast_tracking:
                logging.info("Breast tracking mode activated given navel position in lower 15th of frame")
                self.breast_tracking = True

    def normalize_box_area(self, box, frame_width, frame_height):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return 0
        box_area = (x2 - x1) * (y2 - y1)
        max_area = frame_width * frame_height
        normalized_area = 100 * (box_area / max_area)
        return max(0, min(100, normalized_area))

    def log_and_normalize_pos(self, box, class_name):
        self.boxes[class_name] = box
        _, y1, _, y2 = box
        mid_y = (y1 + y2) / 2
        self.positions[class_name].append(mid_y)

        min_y, max_y = min(self.positions[class_name]), max(self.positions[class_name])
        normalized_y = (100 - int(100 * ((mid_y - min_y) / (max_y - min_y)))) if min_y != max_y else 100

        if class_name == "breast" and self.frame:
            normalized_breast_area = self.normalize_box_area(box, self.frame.shape[1], self.frame.shape[0])
            normalized_y = ((0.75 * normalized_breast_area) + normalized_y) / 1.75

        self.normalized_positions[class_name].append(normalized_y)
        return normalized_y

    def log_and_normalize_distance(self, box, class_name):
        locked_penis_box = self.locked_penis_box or (0, 0, 0, 0)
        distance = self.calculate_distance(locked_penis_box, box)
        self.distances[class_name].append(distance)
        min_distance, max_distance = min(self.distances[class_name]), max(self.distances[class_name])
        normalized_distance = (100 - int(
            100 * ((distance - min_distance) / (max_distance - min_distance)))) if min_distance != max_distance else 100
        self.normalized_distances[class_name].append(normalized_distance)