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
        self.tracked_objects = {
            class_name: {
                'kf': KF.KalmanFilter(),
                'position': None,
                'detected': False,
                'touching': False,
                'prediction_count': 0
            }
            for class_name in class_names
        }

        self.distance_kf = KF.KalmanFilter()

        self.frame = None
        self.current_frame_id = frame_pos
        self.image_y_size = 0
        self.fps = fps

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
        self.sex_position_history = deque(maxlen=10)
        self.sub_sex_position = "Not relevant"

        self.positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.distances = {class_name: deque(maxlen=200) for class_name in class_names}
        self.areas = {class_name: deque(maxlen=200) for class_name in class_names}

        self.normalized_positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.normalized_distances = {class_name: deque(maxlen=200) for class_name in class_names}
        self.activated_kalman = {class_name: False for class_name in class_names}

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

    def update_distance(self, raw_distance):
        if raw_distance is None:
            filtered_distance = self._predict_distance()
        else:
            rounded_distance = round(raw_distance / 5) * 5
            ema_distance = self._calculate_ema_distance(rounded_distance)
            self._update_previous_distances(ema_distance)
            self.distance_kf.update((ema_distance, 0))
            filtered_distance = ema_distance

        filtered_distance = max(0, min(100, filtered_distance))
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

    def update_tracking(self, class_name, box, fallback_class, tracked_item):
        tracked = self.tracked_objects[class_name]
        if box is not None:
            self._handle_detection(tracked, box, class_name)
        elif class_name == tracked_item and box is None:
            self._handle_prediction(tracked, class_name, tracked_item)

        if tracked['touching']:
            self.log_and_normalize_pos(tracked['position'], class_name)
            self.log_and_normalize_distance(tracked['position'], class_name)
        else:
            self._reset_normalized_data(class_name)

        return tracked['position']

    def _handle_detection(self, tracked, box, class_name):
        tracked['kf'].update([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        tracked['position'] = box
        tracked['detected'] = True
        tracked['half_height'] = (box[3] - box[1]) // 2
        tracked['half_width'] = (box[2] - box[0]) // 2
        tracked['touching'] = self.boxes_overlap(self.locked_penis_box, box)
        tracked['prediction_count'] = 0
        if class_name == self.tracked_body_part and self.activated_kalman[class_name]:
            self.activated_kalman[class_name] = False
            logging.info(f"@{self.current_frame_id} - Stopped Kalman prediction for {class_name} as it was found back")

    def _handle_prediction(self, tracked, class_name, tracked_item):
        tracked['kf'].predict()
        if not self.activated_kalman[class_name]:
            self.activated_kalman[class_name] = True
            logging.info(f"@{self.current_frame_id} - Activating Kalman prediction for {class_name}")
        logging.info(
            f"@{self.current_frame_id} - Kalman prediction for {class_name}: {tracked['prediction_count'] + 1} / {self.max_predictions}")
        tracked['detected'] = False
        tracked['position'] = tracked['kf'].position
        tracked['touching'] = self.boxes_overlap(self.locked_penis_box, tracked['position'])

        if tracked['prediction_count'] >= self.max_predictions:
            logging.info(f"No fallback available for {class_name}, deactivating tracking")
            tracked['position'] = None
            tracked['detected'] = False
            tracked['touching'] = False
            tracked['prediction_count'] = 0
            self.tracked_body_part = 'Nothing'

    def _reset_normalized_data(self, class_name):
        self.normalized_positions[class_name].clear()
        self.normalized_distances[class_name].clear()
        self.normalized_positions[class_name].append(100)
        self.normalized_distances[class_name].append(100)

    def boxes_overlap(self, box1, box2):
        if box1 is None or box2 is None:
            return False
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

    def calculate_distance(self, penis_box, other_box):
        if other_box is None:
            return None
        ox1, oy1, ox2, oy2 = other_box
        y_pos = (oy1 + 2 * oy2) // 3
        x_pos = (ox1 + ox2) // 2
        px1, py1, px2, py2 = penis_box
        return math.sqrt((x_pos - px2) ** 2 + (y_pos - py2) ** 2)

    def detect_sex_position_change(self, sex_position, reason):
        if self.sex_position_history and self.sex_position_history[-1] == "Blowjob" and self.sub_sex_position == "Handjob":
            sex_position = "Blowjob"
        self.sex_position_history.append(sex_position)
        position_counts = {position: self.sex_position_history.count(position) for position in self.sex_position_history}
        most_frequent_position = max(position_counts, key=position_counts.get, default="Not relevant")
        if most_frequent_position != self.sex_position:
            logging.info(f"@{self.current_frame_id} - Sex position switched to: {most_frequent_position}")
            self.sex_position = most_frequent_position
            self.sex_position_reason = reason

    def tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        self.image_y_size = image_y_size
        self.current_frame_id = current_frame_id

        self.glans_detected = False
        self.boxes = {class_name: None for class_name in class_names}
        all_detections = {class_name: [] for class_name in class_names}

        for box, conf, cls, class_name, track_id in sorted_boxes:
            all_detections[class_name].append([conf, box, track_id])

        found_box = {class_name: [] for class_name in class_names}
        classes_touching_penis = {class_name: None for class_name in class_names}
        list_of_touching_classes = []

        for check_class_first in ['glans', 'penis', 'navel']:
            prev_conf = 0
            for conf, box, track_id in all_detections[check_class_first]:
                if conf > prev_conf:
                    found_box[check_class_first] = [box, conf, track_id]
                    prev_conf = conf

        for check_class_first in ['glans', 'penis', 'navel']:
            box = None
            if found_box[check_class_first]:
                self.boxes[check_class_first] = found_box[check_class_first][0]
                box = self.boxes[check_class_first]
                conf = found_box[check_class_first][1]
                self.consecutive_detections[check_class_first] += 1
                self.consecutive_non_detections[check_class_first] = 0
                self.handle_class_first(check_class_first, self.boxes[check_class_first], conf)
            else:
                self.consecutive_detections[check_class_first] = 0
                self.consecutive_non_detections[check_class_first] += 1

        prev_class = ''
        nb_items_touching_penis = 0

        for check_class_second in class_names:
            if check_class_second in ['glans', 'penis', 'navel']:
                continue
            prev_conf = 0
            for conf, box, track_id in all_detections[check_class_second]:
                if self.locked_penis_box and self.boxes_overlap(box, self.locked_penis_box) and conf > prev_conf:
                    classes_touching_penis[check_class_second] = box
                    prev_conf = conf
                    if check_class_second != prev_class:
                        nb_items_touching_penis += 1
                        list_of_touching_classes.append(check_class_second)
                        prev_class = check_class_second
            if prev_conf != 0:
                self.update_tracking(check_class_second, classes_touching_penis[check_class_second], None,
                                     self.tracked_body_part)

        if self.tracked_body_part != "Nothing" and self.tracked_body_part not in list_of_touching_classes:
            self.update_tracking(self.tracked_body_part, None, None, self.tracked_body_part)

        if nb_items_touching_penis == 0:
            self.handle_closeup('No body parts touching')
        else:
            if classes_touching_penis['butt']:
                if all_detections['pussy'] and self.boxes_overlap(all_detections['pussy'][0][1],
                                                                  classes_touching_penis['butt']):
                    self.handle_closeup('Presence of butt and overlapping pussy')
                elif self.boxes['penis'] and not self.boxes['glans']:
                    self.tracked_body_part = 'butt'
                    self.penetration = True
                    self.detect_sex_position_change('Doggy', 'butt touching penis')
                    self.sub_sex_position = "Not relevant"
            elif classes_touching_penis['pussy'] and self.boxes['penis'] and not self.boxes['glans']:
                self.penetration = True
                self.tracked_body_part = 'pussy'
                self.detect_sex_position_change('Cowgirl', 'pussy touching penis')
                self.sub_sex_position = "Not relevant"
            elif classes_touching_penis['face']:
                self.tracked_body_part = 'face'
                self.detect_sex_position_change('Blowjob', 'face touching penis')
                self.sub_sex_position = "Not relevant"
            #elif classes_touching_penis['left hand'] or classes_touching_penis['right hand']:
            elif classes_touching_penis['left hand']:
                self.detect_sex_position_change('Handjob', 'hand touching penis')
                self.sub_sex_position = "Handjob"

            if self.boxes['penis'] is not None:
                penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]
                if self.locked_penis_box:
                    locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]

                    for class_name in list_of_touching_classes:
                        y2 = classes_touching_penis[class_name][3]
                        self.offsets[class_name] = y2 - self.boxes['penis'][1]
                    if locked_penis_height > 0:
                        scale = min(int((penis_height / (0.8 * locked_penis_height)) * 100), 100)
                    else:
                        scale = 0
                    distance = 100 - scale

                    #self.positions['penis'].append(distance)
                    #min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                    #normalized_distance = (100 - int(100 * ((distance - min_distance) / (max_distance - min_distance)))) if min_distance != max_distance else 100
                    #self.normalized_positions['penis'].append(normalized_distance)
                    normalized_distance = distance
                    self.update_distance(normalized_distance)
            elif self.sex_position == "Cowgirl" and classes_touching_penis['pussy']:
                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]

                y2 = classes_touching_penis['pussy'][3]
                y2 -= self.offsets['pussy']

                guessed_penis_height = self.locked_penis_box[3] - y2

                if guessed_penis_height < 0 :
                    self.offsets['pussy'] = self.locked_penis_box[3] - y2
                    guessed_penis_height = 0

                scale = min(int((guessed_penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale

                normalized_distance = distance
                self.update_distance(normalized_distance)
                #self.update_distance(self.normalized_distances['pussy'][-1])
            elif self.sex_position == "Doggy" and classes_touching_penis['butt']:
                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]

                y2 = classes_touching_penis['butt'][3]
                y2 -= self.offsets['butt']

                guessed_penis_height = self.locked_penis_box[3] - y2

                if guessed_penis_height < 0:
                    self.offsets['butt'] = self.locked_penis_box[3] - y2
                    guessed_penis_height = 0

                scale = min(int((guessed_penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale

                normalized_distance = distance
                self.update_distance(normalized_distance)
                #self.update_distance(self.normalized_distances['butt'][-1])

    def handle_class_first(self, class_name, box, conf):
        if class_name == 'penis':
            if self.penis_box is None:
                logging.info(f"Penis detected at frame {self.current_frame_id} with confidence {conf}")
            self.penis_box = box
            px1, py1, px2, py2 = self.penis_box
            current_height = py2 - py1
            if self.consecutive_detections['penis'] >= self.detections_threshold:
                if self.locked_penis_box is None or self.glans_detected or current_height > self.locked_penis_height:
                    if self.locked_penis_box is None:
                        logging.info(f"Locked penis box activated at frame {self.current_frame_id}")
                    self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
                    self.locked_penis_height = current_height
            if self.locked_penis_box:
                if current_height > self.locked_penis_height:
                    self.locked_penis_height = current_height
                    self.locked_penis_box = self.penis_box
                if self.penis_box[3] != self.locked_penis_box[3]:
                    self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
            if self.locked_penis_box and self.penetration:
                penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]
                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
                scale = min(int((penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale
                self.positions['penis'].append(distance)
                min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                normalized_distance = (100 - int(100 * ((distance - min_distance) / (max_distance - min_distance)))) if min_distance != max_distance else 100
                self.normalized_positions['penis'].append(normalized_distance)
            if (self.image_y_size - py1) / self.image_y_size < 0.1 and not self.breast_tracking and \
                    self.tracked_objects['breast']['detected']:
                self.breast_tracking = True
        elif class_name == 'glans':
            if self.consecutive_detections['glans'] >= self.detections_threshold:
                self.boxes['glans'] = box
                self.glans_detected = True
                if self.penetration:
                    self.penetration = False
                    logging.info(
                        f"@{self.current_frame_id} - Penetration ended after {self.consecutive_detections['glans']} detections of glans")
                    if self.tracked_body_part != 'Nothing':
                        self.normalized_distances[self.tracked_body_part].clear()
                        self.normalized_distances[self.tracked_body_part].append(100)
        elif class_name == 'navel':
            if (self.image_y_size - box[1]) / self.image_y_size < 0.15 and not self.breast_tracking:
                logging.info("Breast tracking mode activated given navel position in lower 15th of frame")
                self.breast_tracking = True

    def handle_closeup(self, reason):
        if self.tracked_body_part != 'Nothing':
            self.normalized_distances[self.tracked_body_part].clear()
            self.normalized_distances[self.tracked_body_part].append(100)
        self.tracked_body_part = 'Nothing'
        self.detect_sex_position_change('Not relevant', reason)
        self.sub_sex_position = "Not relevant"
        self.breast_tracking = False
        self.penetration = False
        self.grinding = False
        self.rubbing = False
        self.distance = 100
        self.update_distance(100)

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