from utils import KalmanFilter as KF
from collections import deque
import numpy as np
#import cv2
from utils.config import class_names
import math

class ObjectTracker:
    def __init__(self, fps, frame_pos, state=None):
        self.class_names = class_names
        self.tracked_objects = {
            class_name: {'kf': KF.KalmanFilter(),
                         'position': None, 'detected': False, 'touching': False, 'prediction_count': 0}
            for class_name in class_names
        }
        # self.detection_interval = 10
        # self.frame_count = 0
        # self.trackers = {}

        self.distance_kf = KF.KalmanFilter()

        self.frame = None
        self.current_frame_id = frame_pos
        self.image_y_size = 0
        self.fps = fps

        self.occlusion = False

        self.penis_box, self.locked_penis_box = None, None
        self.glans_detected = False
        self.locked_penis_height = 0
        self.breast_tracking = False
        self.distance = 100
        self.raw_distance = 100
        self.previous_distances = [100, 100, 100]
        self.tracked_body_part = "Nothing"
        #self.activated_kalman = False

        self.face_distance = 100
        self.hand_distance = 100
        self.previous_hand_distance = 100
        self.previous_face_distance = 100
        self.smoothed_distance = 100
        self.smoothed_foot_distance = 100

        # Handjob / Blowjob tweaking and stabilization parameters
        self.switch_threshold_multiplier = 1.5  # Movement must be 1.5x higher to switch
        self.required_frames_to_switch = 5  # 3 Number of consecutive frames needed to switch
        self.cooldown_frames = 10  # 5 Frames to wait before allowing another switch
        self.last_tracked_body_part = None
        self.switch_cooldown = 0  # Frames left in cooldown
        # Movement tracking
        self.face_hands_movements = {}
        self.previous_hand_distance = 0
        self.face_distances = []  # List to store recent face distances
        self.right_hand_distances = []  # List to store recent right hand distances
        self.left_hand_distances = []  # List to store recent left hand distances

        # Footjob tweaking and stabilization parameters
        # Stabilization parameters for feet
        self.foot_switch_threshold_multiplier = 1.5  # Movement must be 1.5x higher to switch
        self.required_foot_frames_to_switch = 3  # Number of consecutive frames needed to switch
        self.foot_cooldown_frames = 5  # Cooldown period for switching foot tracking
        self.last_tracked_foot = None
        self.foot_switch_cooldown = 0  # Frames left in cooldown
        # Movement tracking for feet
        self.foot_movements = {}
        self.right_foot_distances = []  # List to store recent right foot distances
        self.left_foot_distances = []  # List to store recent left foot distances

        self.sex_position = "Not relevant"
        self.prev_sex_position = "Not relevant"
        self.sex_position_reason = ""
        self.sex_position_history = deque(maxlen=10)
        self.sub_sex_position = "Not relevant"

        #self.positions = {class_name: [] for class_name in class_names}
        self.positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.distances = {class_name: deque(maxlen=200) for class_name in class_names}
        self.areas = {class_name: deque(maxlen=200) for class_name in class_names}

        # Initialize normalized_positions as a dictionary of dictionaries
        self.normalized_positions = {class_name: deque(maxlen=200) for class_name in class_names}
        self.normalized_distances = {class_name: deque(maxlen=200) for class_name in class_names}
        self.activated_kalman = {class_name: False for class_name in class_names}

        for class_name in class_names:
            self.normalized_distances[class_name].append(100)
            self.normalized_positions[class_name].append(100)


        self.normalized_areas = {class_name: deque(maxlen=200) for class_name in class_names}
        #self.normalized_positions['navel'].append(100)
        #self.normalized_positions['breast'].append(100)
        #self.normalized_positions['butt'].append(100)

        #    'breast': deque(maxlen=10),
        #    'navel': deque(maxlen=10),
            # add other classes as needed
        #}

        #self.normalized_positions = []

        self.moving_average_window = 5  # Number of frames to consider for moving average

        # Counters for consecutive detections and non-detections
        self.consecutive_detections = {class_name: 0 for class_name in class_names}
        self.consecutive_non_detections = {class_name: 0 for class_name in class_names}
        self.detections_threshold = 3

        self.consecutive_grinding_supiscions = {class_name: 0 for class_name in ['pussy', 'butt']}

        self.penetration = False
        self.grinding = False
        self.rubbing = False
        self.penetration_mode = 'No penetration'
        self.hand_blow_footjob = False

        self.max_predictions = self.fps #600


    def update_distance(self, raw_distance):

        #self.raw_distance = raw_distance
        if raw_distance is not None:

            rounded_distance = round(raw_distance / 5) * 5
            # self.previous_distances.pop(0)
            # self.previous_distances.append(raw_distance)
            ema_distance = int(
                0.7 * rounded_distance + 0.1 * self.previous_distances[1] + 0.2 * self.previous_distances[2])
            # cap the change in distance to a maximum of +5 or -5 vs previous distance
            if abs(ema_distance - self.previous_distances[-1]) > 15:
                ema_distance = self.previous_distances[-1] + np.sign(ema_distance - self.previous_distances[-1]) * 15
                # print(f"Capping distance change to {ema_distance} vs {raw_distance}")
            elif abs(ema_distance - self.previous_distances[-1]) < 2:
                # if the change is less than 2, use the previous distance to eliminate jitter
                # print(f"Skipping distance change to {ema_distance} vs {self.previous_distances[-1]} because change is less than 4")
                ema_distance = self.previous_distances[-1]
            self.previous_distances.pop(0)
            self.previous_distances.append(ema_distance)

            # keep the distance between 0 and 100
            #raw_distance = max(0, min(100, int(raw_distance)))
            # Update the Kalman filter with the measurement
            #raw_distance_2d = (raw_distance, 0)
            ema_distance_2d = (ema_distance, 0)
            #filtered_distance = self.distance_kf.update(raw_distance_2d)
            #filtered_distance = self.distance_kf.update(raw_distance_2d)
            self.distance_kf.update(ema_distance_2d)
            #filtered_distance = float(self.distance_kf.predict()[0])
            filtered_distance = ema_distance
        else:
            # Predict without updating
            #filtered_distance = self.distance_kf.predict()
            filtered_distance = int(float(self.distance_kf.predict()[0]) / 5) * 5
            # keep the distance between 0 and 100

        filtered_distance = max(0, min(100, filtered_distance))

        self.distance = filtered_distance
        # print(f"Raw distance vs filtered distance: {raw_distance} vs {filtered_distance}")
        return filtered_distance

    def update_tracking(self, class_name, box, fallback_class, tracked_item):
        tracked = self.tracked_objects[class_name]
        if box is not None:
            # Object detected: Update Kalman filter
            tracked['kf'].update([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            tracked['position'] = box
            tracked['detected'] = True
            tracked['half_height'] = (box[3] - box[1]) // 2
            tracked['half_width'] = (box[2] - box[0]) // 2
            tracked['touching'] = self.boxes_overlap(self.locked_penis_box, box)
            tracked['prediction_count'] = 0  # Reset prediction count when object is detected
            if class_name == tracked_item and self.activated_kalman[class_name]:
                self.activated_kalman[class_name] = False
                print(f"@{self.current_frame_id} - Stopped Kalman prediction for {class_name} as it was found back")
        # Only perform prediction for the tracked item
        elif class_name == tracked_item and box is None:
            # Object not detected: Predict position based on the Kalman filter
            # print(f"Kalman prediction for {class_name}: {tracked['prediction_count'] + 1} / {int(self.max_predictions)}")
            tracked['kf'].predict()
            if self.activated_kalman[class_name] == False:
                self.activated_kalman[class_name] = True
                print(f"@{self.current_frame_id} - Activating Kalman prediction for {class_name}")
            print(f"@{self.current_frame_id} - Kalman prediction for {class_name}: {tracked['prediction_count'] + 1} / {int(self.max_predictions)}")
            tracked['detected'] = False
            tracked['position'] = tracked['kf'].position
            tracked['touching'] = self.boxes_overlap(self.locked_penis_box, tracked['position'])

            # Use fallback if primary tracked object is lost and prediction count exceeds 60
            if tracked['prediction_count'] >= self.max_predictions:
                #if fallback_class and self.tracked_objects[fallback_class]['detected']:
                #    print(f"Handling occlusion of {class_name}, fallback to {fallback_class}")
                #    tracked['position'] = self.tracked_objects[fallback_class]['position']
                #else:
                print(f"No fallback available for {class_name}, deactivating tracking")
                tracked['position'] = None
                tracked['detected'] = False
                tracked['touching'] = False
                tracked['prediction_count'] = 0
                self.tracked_body_part = 'Nothing'
        if tracked['touching']:
            self.log_and_normalize_pos(tracked['position'], class_name)
            self.log_and_normalize_distance(tracked['position'], class_name)
        else:
            self.normalized_positions[class_name].clear()
            self.normalized_distances[class_name].clear()
            self.normalized_positions[class_name].append(100)
            self.normalized_distances[class_name].append(100)

                    # print(f"Fallback to {fallback_class}")
                    #draw_bounding_box(image, tracked['position'], fallback_class)
                #else:
                    # print(f"No fallback available for {class_name}")

        return tracked['position']

    def boxes_overlap(self, box1, box2):
        if box1 is None or box2 is None:
            return False
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
            return False
        return True

    def calculate_distance(self, penis_box, other_box):
        # compute middle of other_box
        if other_box is None:
            return None
        ox1, oy1, ox2, oy2 = other_box
        y_pos = (oy1 + 2 * oy2) // 3
        x_pos = (ox1 + ox2) // 2
        # compute middle of lower side of penis_box
        px1, py1, px2, py2 = penis_box
        # compute distance between (x_pos, y_pos) and (px2, py2)
        distance = math.sqrt((x_pos - px2) ** 2 + (y_pos - py2) ** 2)
        return distance



    def detect_sex_position_change(self, sex_position, reason):
        if len(self.sex_position_history) > 0 and self.sex_position_history[-1] == "Blowjob" and self.sub_sex_position == "Handjob":
            sex_position = "Blowjob"
        self.sex_position_history.append(sex_position)
        # Count the occurrences of each sex position in the history
        position_counts = {}
        for position in self.sex_position_history:
            if position in position_counts:
                position_counts[position] += 1
            else:
                position_counts[position] = 1

        # Find the most frequent sex position
        if position_counts:
            most_frequent_position = max(position_counts, key=position_counts.get)
        else:
            most_frequent_position = "Not relevant"  # No positions in history
        if most_frequent_position != self.sex_position:
            print(f"@{self.current_frame_id} - Sex position switched to: {most_frequent_position}")
            self.sex_position = most_frequent_position

    def prev_detect_sex_position_change(self, sex_position, reason):
        if sex_position != self.sex_position and sex_position == self.prev_sex_position:
            #print(f"Sex position changed from {self.sex_position} to {sex_position} given {reason}")
            print(f"\n@{self.current_frame_id} - {sex_position} given {reason}")
            self.sex_position = sex_position
            self.sex_position_reason = reason
        else:
            self.prev_sex_position = sex_position

    def tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        self.image_y_size = image_y_size
        self.current_frame_id = current_frame_id

        # initialize values
        self.glans_detected = False
        self.boxes = {class_name: None for class_name in class_names}
        all_detections = {class_name: [] for class_name in class_names}

        # detected_classes = []
        list_of_classes = class_names.copy()

        for box, conf, cls, class_name in sorted_boxes:
            all_detections[class_name].append([conf, box])

        found_box = {class_name: [] for class_name in class_names}
        classes_touching_penis = {class_name: None for class_name in class_names}
        list_of_touching_classes = []

        # Checking priority classes, and keeping the ones with best confidence level
        for check_class_first in ['glans', 'penis', 'navel']:
            prev_conf = 0
            for conf, box in all_detections[check_class_first]:
                if conf > prev_conf:
                    found_box[check_class_first] = [box, conf]
                    prev_conf = conf
        # Logging priority classes detection and non detections
        for check_class_first in ['glans', 'penis', 'navel']:
            box = None
            if len(found_box[check_class_first]) > 0:
                list_of_classes.pop(check_class_first)
                self.boxes[check_class_first] = found_box[check_class_first][0]
                box = self.boxes[check_class_first]
                conf = found_box[check_class_first][1]
                self.consecutive_detections[check_class_first] += 1
                self.consecutive_non_detections[check_class_first] = 0
                self.handle_class_first(check_class_first, self.boxes[check_class_first], conf)
            else:
                self.consecutive_detections[check_class_first] = 0
                self.consecutive_non_detections[check_class_first] += 1
            #if check_class_first == 'penis':
            #    self.update_tracking(check_class_first, box, None, 'penis')

        prev_class = ''
        nb_items_touching_penis = 0

        # Listing detected classes touching the locked penis box
        for check_class_second in list_of_classes:
            prev_conf = 0
            for conf, box in all_detections[check_class_second]:
                if self.locked_penis_box and self.boxes_overlap(box, self.locked_penis_box) and conf > prev_conf:
                    classes_touching_penis[check_class_second] = box
                    prev_conf = conf
                    if check_class_second != prev_class:
                        nb_items_touching_penis += 1
                        list_of_touching_classes.append(check_class_second)
                        prev_class = check_class_second
            if prev_conf != 0:  # we have one item of that class touching
                #self.log_and_normalize_distance(classes_touching_penis[check_class_second], check_class_second)
                #self.log_and_normalize_pos(classes_touching_penis[check_class_second], check_class_second)
                self.update_tracking(check_class_second, classes_touching_penis[check_class_second], None, self.tracked_body_part)

        if self.tracked_body_part != "Nothing" and self.tracked_body_part not in list_of_touching_classes:
            # try kalman filter prediction
            self.update_tracking(self.tracked_body_part, None, None, self.tracked_body_part)


        if nb_items_touching_penis == 0:
            self.handle_closeup('No body parts touching')
        else:
            # Determine action type
            if classes_touching_penis['butt']:
                # Could still be closeup if pussy can be seen also
                if len(all_detections['pussy']) > 0 and self.boxes_overlap(all_detections['pussy'][0][1], classes_touching_penis['butt']):
                    self.handle_closeup('Presence of butt and overlapping pussy')
                elif self.boxes['penis'] and not self.boxes['glans']:
                    self.tracked_body_part = 'butt'
                    self.penetration = True
                    #self.log_and_normalize_distance(classes_touching_penis['butt'], 'butt')
                    self.detect_sex_position_change('Doggy', 'butt touching penis')
                    self.sub_sex_position = "Not relevant"
            elif classes_touching_penis['pussy'] and self.boxes['penis'] and not self.boxes['glans']:
                self.penetration = True
                self.tracked_body_part = 'pussy'
                #self.log_and_normalize_distance(classes_touching_penis['pussy'], 'pussy')
                self.detect_sex_position_change('Cowgirl', 'pussy touching penis')
                self.sub_sex_position = "Not relevant"
            elif classes_touching_penis['face']:
                self.tracked_body_part = 'face'
                #self.log_and_normalize_distance(classes_touching_penis['face'], 'face')
                self.detect_sex_position_change('Blowjob', 'face touching penis')
                self.sub_sex_position = "Not relevant"
            elif classes_touching_penis['left hand'] or classes_touching_penis['right hand']:
                self.detect_sex_position_change('Handjob', 'hand touching penis')
                self.sub_sex_position = "Handjob"
                #if classes_touching_penis['left hand']:
                #    self.log_and_normalize_distance(classes_touching_penis['left hand'], 'left hand')
                #else:
                #    self.normalized_distances['left hand'].clear()
                #if classes_touching_penis['right hand']:
                #    self.log_and_normalize_distance(classes_touching_penis['right hand'], 'right hand')
                #else:
                #    self.normalized_distances['right hand'].clear()


            if self.boxes['penis'] is not None:
                # Compare the height of the two boxes and scale it to 0 - 100
                penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]

                if self.locked_penis_box:
                    if self.sub_sex_position == 'Handjob':
                        if classes_touching_penis['left hand'] and not classes_touching_penis['right hand']:
                            #normalized_distance = self.normalized_distances['left hand'][-1]
                            normalized_distance = self.normalized_positions['left hand'][-1]
                            self.tracked_body_part = "left hand"
                        elif classes_touching_penis['right hand'] and not classes_touching_penis['left hand']:
                            #normalized_distance = self.normalized_distances['right hand'][-1]
                            normalized_distance = self.normalized_positions['right hand'][-1]
                            self.tracked_body_part = "right hand"
                        else:
                            if len(self.normalized_distances['left hand']) > 3 and len(self.normalized_distances['right hand']) > 3:
                                #avg_left = self.normalized_distances['left hand'][-1] - self.normalized_distances['left hand'][-2] +\
                                #            self.normalized_distances['left hand'][-2] - self.normalized_distances['left hand'][-3]
                                #avg_right = self.normalized_distances['right hand'][-1] - self.normalized_distances['right hand'][-2] +\
                                #            self.normalized_distances['right hand'][-2] - self.normalized_distances['right hand'][-3]
                                avg_left = self.normalized_positions['left hand'][-1] - \
                                           self.normalized_positions['left hand'][-2] + \
                                           self.normalized_positions['left hand'][-2] - \
                                           self.normalized_positions['left hand'][-3]
                                avg_right = self.normalized_positions['right hand'][-1] - \
                                            self.normalized_positions['right hand'][-2] + \
                                            self.normalized_positions['right hand'][-2] - \
                                            self.normalized_positions['right hand'][-3]

                                if abs(avg_left) > abs(avg_right):
                                    #normalized_distance = self.normalized_distances['left hand'][-1]
                                    normalized_distance = self.normalized_positions['left hand'][-1]
                                    self.tracked_body_part = "left hand"
                                else:
                                    #normalized_distance = self.normalized_distances['right hand'][-1]
                                    normalized_distance = self.normalized_positions['right hand'][-1]
                                    self.tracked_body_part = "right hand"
                            else:
                                # arbitrary choice
                                #if len(self.normalized_distances['right hand']) > 0 and len(self.normalized_distances['right hand']) > len(self.normalized_distances['left hand']):
                                #    normalized_distance = self.normalized_distances['right hand'][-1]
                                #    self.tracked_body_part = "right hand"
                                #elif len(self.normalized_distances['left hand']) > 0:
                                #    normalized_distance = self.normalized_distances['left hand'][-1]
                                #    self.tracked_body_part = "left hand"
                                if len(self.normalized_positions['right hand']) > 0 and len(self.normalized_positions['right hand']) > len(self.normalized_distances['left hand']):
                                    normalized_distance = self.normalized_positions['right hand'][-1]
                                    self.tracked_body_part = "right hand"
                                elif len(self.normalized_positions['left hand']) > 0:
                                    normalized_distance = self.normalized_positions['left hand'][-1]
                                    self.tracked_body_part = "left hand"

                    else:  # other cases than handjob
                        locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
                        if locked_penis_height == 0:
                            locked_penis_height = 0.0001
                        scale = min(int((penis_height / locked_penis_height) * 100), 100)
                        distance = 100 - scale

                        self.positions['penis'].append(distance)

                        min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                        normalized_distance = (100 - int(100 * ((distance - min_distance) / (
                                max_distance - min_distance)))) if min_distance != max_distance else 100

                        self.normalized_positions['penis'].append(normalized_distance)

                        if self.sex_position == 'Blowjob':
                            delta = 0
                            if classes_touching_penis['left hand'] and len(self.normalized_distances['left hand']) > 3:
                                delta += self.normalized_distances['left hand'][-1] - self.normalized_distances['left hand'][-4]
                            if classes_touching_penis['right hand']and len(self.normalized_distances['right hand']) > 3:
                                delta += self.normalized_distances['right hand'][-1] - self.normalized_distances['right hand'][-4]
                            if not self.penetration and classes_touching_penis['face']and len(self.normalized_distances['face']) > 3:
                                delta += self.normalized_distances['face'][-1] - self.normalized_distances['face'][-4]

                            renormalized_distance = max(0, min(100, (normalized_distance + delta)))
                            #print(f"Renormalized distance during BJ/HJ to {renormalized_distance} vs {normalized_distance}")
                            normalized_distance = renormalized_distance
                    self.update_distance(normalized_distance)
            elif self.sex_position == "Cowgirl":
                self.update_distance(self.normalized_distances['pussy'][-1])
            elif self.sex_position == "Doggy":
                self.update_distance(self.normalized_distances['butt'][-1])

    def handle_class_first(self, class_name, box, conf):
        if class_name == 'penis':
            if self.penis_box is None:
                print(f"Penis detected at frame {self.current_frame_id} with confidence {conf}")
            self.penis_box = box
            px1, py1, px2, py2 = self.penis_box
            current_height = py2 - py1
            if self.consecutive_detections['penis'] >= self.detections_threshold:
                if self.locked_penis_box is None or self.glans_detected or current_height > self.locked_penis_height:
                    if self.locked_penis_box is None:
                        print(f"Locked penis box activated at frame {self.current_frame_id}")
                    self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
                    self.locked_penis_height = current_height
            if self.locked_penis_box:
                if current_height > self.locked_penis_height:
                    self.locked_penis_height = current_height
                    self.locked_penis_box = self.penis_box
                if self.penis_box[3] != self.locked_penis_box[3]:
                    self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
            if self.locked_penis_box and self.penetration:
                # Compare the height of the two boxes and scale it to 0 - 100
                penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]
                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
                scale = min(int((penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale
                self.positions['penis'].append(distance)
                min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                normalized_distance = (100 - int(100 * ((distance - min_distance) / (
                        max_distance - min_distance)))) if min_distance != max_distance else 100
                self.normalized_positions['penis'].append(normalized_distance)
            if (self.image_y_size - py1) / self.image_y_size < 0.1 and not self.breast_tracking and \
                    self.tracked_objects['breast']['detected']:
                # print("Breast tracking mode activated given penis position in lower 10th of frame")
                self.breast_tracking = True
        elif class_name == 'glans':
            if self.consecutive_detections['glans'] >= self.detections_threshold:
                self.boxes['glans'] = box
                self.glans_detected = True
                if self.penetration:
                    self.penetration = False
                    print(
                        f"@{self.current_frame_id} - Penetration ended after {self.consecutive_detections['glans']} detections of glans")
        elif class_name == 'navel':
            if (self.image_y_size - box[1]) / self.image_y_size < 0.15 and not self.breast_tracking:
                print("Breast tracking mode activated given navel position in lower 15th of frame")
                self.breast_tracking = True

    def handle_closeup(self, reason):
        self.tracked_body_part = 'Nothing'
        self.detect_sex_position_change('Not relevant', reason)
        self.sub_sex_position = "Not relevant"
        self.penetration = False
        self.grinding = False
        self.rubbing = False
        self.distance = 100
        self.update_distance(100)

    def prev3_tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        # self.frame = frame
        self.image_y_size = image_y_size
        self.current_frame_id = current_frame_id
        # self.frame_count += 1

        # initialize values
        self.foot_detected = False
        self.glans_detected = False
        self.tracking_states = {class_name: False for class_name in class_names}
        self.boxes = {class_name: None for class_name in class_names}
        all_detections = {class_name: [] for class_name in class_names}

        detected_classes = []
        list_of_classes = class_names.copy()


        for box, conf, cls, class_name in sorted_boxes:

            all_detections[class_name].append([conf, box])

            if class_name not in detected_classes:
                detected_classes.append(class_name)

        # First, let's handle the penis and glans
        # count the number of detected glans in all_detections, keep the one with the highest confidence

        found_box = {class_name: [] for class_name in ['glans', 'penis']}

        for check_class_first in ['glans', 'penis']:
            prev_conf = 0
            for conf, box in all_detections[check_class_first]:
                if conf > prev_conf:
                    found_box[check_class_first] = [box, conf]
                    # print(f"{found_box}")
                    prev_conf = conf

        if len(found_box['glans']) > 0:
            # print(f"Found glans at frame {current_frame_id}")
            # remove glans from list_of_classes
            list_of_classes.pop('glans')
            self.boxes['glans'] = found_box['glans'][0]
            self.glans_handling(found_box['glans'][0], found_box['glans'][1], current_frame_id)

        if len(found_box['penis']) > 0:
            # remove penis from list_of_classes
            list_of_classes.pop('penis')
            self.boxes['penis'] = found_box['penis'][0]
            self.penis_handling(found_box['penis'][0], found_box['penis'][1], current_frame_id)

        if self.tracked_body_part != "Nothing" and self.tracked_body_part not in detected_classes:
            # Performing Kalman prediction
            if not self.activated_kalman:
                print(f"Kalman prediction for tracked body part {self.tracked_body_part}")
            self.update_tracking(self.tracked_body_part, None, None, self.tracked_body_part)

        # Check if we have multiple instances of breast, navel, butt etc.

        for checking_class in ["breast", "navel", "pussy", "butt", "left foot", "right foot", "left hand",
                               "right hand", "face"]:
            temp_boxes = []
            # count the number of instances of the class in detections
            count = 0
            for class_name, detections in all_detections.items():
                for conf, box in detections:
                    if checking_class == class_name and box is not None:
                        temp_boxes.append(box)
                        count += 1
            if count >= 1:
                # print(f"Multiple instances of {checking_class} detected in frame {current_frame_id}")
                # check if one is touching the penis, else keep the box that is lowest in the frame
                lowest_box = temp_boxes[0]
                for box in temp_boxes:
                    if box[3] > lowest_box[3]:
                        lowest_box = box
                    if self.locked_penis_box and self.boxes_overlap(box, self.locked_penis_box):
                        lowest_box = box
                        # exit the for loop, no need to look further
                        break
                self.boxes[checking_class] = lowest_box
                if checking_class in ["navel", "breast"]:
                    self.log_and_normalize_pos(lowest_box, checking_class)
                    self.log_and_normalize_distance(lowest_box, checking_class)
                    if checking_class == "navel":
                        self.navel_handling(lowest_box)
                        # remove the class from list_of_classes
                    list_of_classes.pop(checking_class)

                elif checking_class in ["left foot", "right foot", "left hand",
                                      "right hand", "butt", "breast", "face", "pussy"] and \
                            self.boxes_overlap(lowest_box, self.locked_penis_box):
                    # remove the class from list_of_classes
                    list_of_classes.pop(checking_class)
                    self.breast_tracking = False
                    self.log_and_normalize_distance(lowest_box, checking_class)
                    self.boxes[checking_class] = lowest_box
                    if checking_class == 'pussy':
                        fallback_class = 'breast'
                    else:
                        fallback_class = None
                    self.update_tracking(checking_class, lowest_box, fallback_class, self.tracked_body_part)
                    if checking_class in ["butt", "face", "pussy"] and self.tracked_objects[checking_class]['touching']:
                        if not self.penetration and len(found_box['glans']) == 0:
                            self.penetration = True
                            print(
                                f"Penetration started at frame {current_frame_id} as {checking_class} is detected and touching")
                    self.tracking_states[checking_class] = True

        for unwanted_class in list_of_classes:
            self.tracking_states[unwanted_class] = False

        if self.boxes['penis'] is not None:
            # Compare the height of the two boxes and scale it to 0 - 100
            penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]

            if self.locked_penis_box:
                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
                if locked_penis_height == 0:
                    locked_penis_height = 0.0001
                scale = min(int((penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale

                self.positions['penis'].append(distance)

                min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                normalized_distance = (100 - int(100 * ((distance - min_distance) / (
                            max_distance - min_distance)))) if min_distance != max_distance else 100

                self.normalized_positions['penis'].append(normalized_distance)

                self.update_distance(self.normalized_positions['penis'][-1])


        # Handling irrelevant closeups first

        if self.tracking_states['pussy'] and self.tracking_states['butt']:
            # position likely to be Closeup
            self.handle_closeup_position("presence of pussy and butt")
            self.breast_tracking = False
        elif (self.tracking_states['pussy'] and self.boxes['pussy'] is not None and (
                self.boxes['pussy'][3] - self.boxes['pussy'][1]) >
              image_y_size * 0.1):
            # position likely to be Closeup
            self.handle_closeup_position("size of pussy in frame")

        # Handling relevant closeups
        elif self.breast_tracking and not self.glans_detected:
            if self.boxes['breast'] is not None and not self.boxes_overlap(self.locked_penis_box, self.boxes['breast']):
                # if self.boxes['breast'] is not None and not self.tracked_objects['breast']['touching']:
                # experimenting to handle delay in breast tracking when boobs reach peak or low position
                # projected_frame_id = current_frame_id  # + int(fps//5)
                # distances[projected_frame_id] = normalized_positions['breast'].get(current_frame_id, 100)
                distance = self.normalized_positions['breast'][-1]
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "breast"
                self.detect_sex_position_change("Closeup missionary", "breast tracking activated")
                if not self.penetration:
                    self.penetration = True
                    print(f"Penetration started at frame {current_frame_id} as we are tracking closeup breasts")

        # Handling tagged scenes
        elif self.sex_position == "Handjob / Blowjob":
            # if no right hand, no left hand, no face touching, then cancel the position
            if not self.tracked_objects['left hand']['touching'] and \
                    not self.tracked_objects['right hand']['touching'] and \
                    not self.tracked_objects['face']['touching']:
                # deactivating position
                self.detect_sex_position_change("Not relevant", "No hand no face")
            else:
                self.handle_Blowjob_Handjob_position("Hands or Face detected", current_frame_id)

        elif self.sex_position == "Doggy / Reverse Cowgirl / Pronebone":
            if not self.boxes['butt']:
                # deactivating position
                self.detect_sex_position_change("Not relevant", "No butt detected")
            else:
                self.handle_RevCowGirl_Doggy_Pronebone_position("Doggy / Reverse Cowgirl / Pronebone", "Butt detected", current_frame_id)

        elif self.sex_position == "Cowgirl / Missionary":
            if not self.boxes['pussy']:
                # deactivating position
                self.detect_sex_position_change("Not relevant", "No pussy detected")
            else:
                self.handle_CowGirl_Missionary_position("Pussy detected", current_frame_id, "pussy")

        # Now handling visible penetrations scenarii
        elif self.locked_penis_box and self.penetration:
            if self.boxes['penis'] is not None:
                # Compare the height of the two boxes and scale it to 0 - 100
                penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]

                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]

                scale = min(int((penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale

                self.positions['penis'].append(distance)

                min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                normalized_distance = (100 - int(100 * ((distance - min_distance) / (
                            max_distance - min_distance)))) if min_distance != max_distance else 100

                self.normalized_positions['penis'].append(normalized_distance)

                self.update_distance(self.normalized_positions['penis'][-1])

            else:
                # Still handling penetration, but penis is not visible, could be a grinding scene
                if self.boxes['pussy'] is not None and self.tracking_states['pussy'] and self.boxes[
                    'breast'] is not None:
                    # position likely to be CowGirl / Missionary
                    # defaulting to navel tracking or breast tracking
                    # if self.tracking_states['navel']:
                    #    self.handle_CowGirl_Missionary_position("presence of navel", current_frame_id, 'navel')
                    # el
                    if self.tracking_states['breast']:
                        self.handle_CowGirl_Missionary_position("presence of breast", current_frame_id, 'breast')
                    # self.handle_CowGirl_Missionary_position("presence of pussy and breast", current_frame_id, 'pussy')
                    if self.tracked_objects['pussy']['touching'] and not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id} during pussy grinding")
                elif self.boxes['butt'] is not None and self.tracking_states['butt']:
                    # position likely to be Reverse Cowgirl or Doggy / Pronebone
                    self.handle_RevCowGirl_Doggy_Pronebone_position(
                        "Doggy / Reverse Cowgirl / Pronebone", "presence of foot", current_frame_id)
                    if self.tracked_objects['butt']['touching'] and not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id} during butt grinding")
                elif self.boxes['navel'] is not None and not self.boxes_overlap(self.locked_penis_box,
                                                                                self.boxes[
                                                                                    'navel']) and self.penetration:
                    # position likely to be CowGirl / Missionary, with occluded pussy
                    self.handle_CowGirl_Missionary_position("presence of navel during penetration",
                                                            current_frame_id,
                                                            'navel')
                elif self.boxes['breast'] is not None and self.tracking_states[
                    'breast'] and not self.penetration and \
                        self.tracked_objects["breast"]['touching']:
                    # position likely to be Boobjob
                    self.handle_Boobjob_position("presence of breast", current_frame_id)
                elif ((self.boxes['face'] is not None and self.tracking_states['face']) or
                      ((self.boxes['left hand'] is not None and self.tracking_states['left hand']) or
                       (self.boxes['right hand'] is not None and self.tracking_states[
                           'right hand']))) and not self.penetration:
                    self.handle_Blowjob_Handjob_position("presence of face or hand", current_frame_id)
        elif self.locked_penis_box and not self.penetration:
            # Now handling handjob / blowjob scenarii without mouth penetration
            if ((self.boxes['face'] is not None and self.tracking_states['face']) or
                ((self.boxes['left hand'] is not None and self.tracking_states['left hand']) or
                 (self.boxes['right hand'] is not None and self.tracking_states[
                     'right hand']))) and not self.penetration:

                self.handle_Blowjob_Handjob_position("presence of face or hand", current_frame_id)

            elif ((self.boxes['left foot'] is not None and self.tracking_states['left foot']) or (
                    self.boxes['right foot'] is not None and self.tracking_states[
                'right foot'])) and not self.penetration:
                # position likely to be Footjob
                self.handle_Footjob_position("presence of foot", current_frame_id)

        # Check for non-detection of penis
        if self.boxes['penis'] is None:
            self.consecutive_non_detections['penis'] += 1
            self.consecutive_detections['penis'] = 0  # Reset the detection counter

            # Deactivate locked_penis_box after x consecutive non-detections
            if self.consecutive_non_detections['penis'] >= self.max_predictions:
                if self.locked_penis_box is not None:
                    self.locked_penis_box = None
                    print(
                        f"Penis tracking deactivated at frame {current_frame_id} after {self.consecutive_non_detections['penis']} consecutive non-detections")
                    # print("Reinitializing tracker")
                    # self.__init__(self.class_names)
                    self.penis_box = None

        # Check for non-detection of glans
        if self.boxes['glans'] is None:
            self.consecutive_non_detections['glans'] += 1
            self.consecutive_detections['glans'] = 0  # Reset the detection counter



    def prev2_tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        # self.frame = frame
        self.image_y_size = image_y_size
        # self.frame_count += 1

        # initialize values
        self.foot_detected = False
        self.glans_detected = False
        self.tracking_states = {class_name: False for class_name in class_names}
        self.boxes = {class_name: None for class_name in class_names}
        all_detections = {class_name: [] for class_name in class_names}

        """
        for box, conf, cls, class_name in sorted_boxes:
            
            if class_name == "glans" and conf > 0.5:
                self.glans_handling(box, conf, current_frame_id)

            elif class_name == "penis" and conf > 0.5:
                self.boxes['penis'] = box
                self.penis_handling(box, conf, current_frame_id)

            elif class_name == "navel" and conf > 0.5:
                self.navel_handling(box)

            if class_name in ["breast", "navel"] and conf > 0.5:
                self.log_and_normalize_pos(box, class_name)
                self.boxes[class_name] = box

            if self.locked_penis_box:
                self.update_tracking(class_name, box, None, self.tracked_body_part)

                if class_name in ["left foot", "right foot", "left hand",
                                  "right hand", "butt", "face", "pussy"] and self.tracked_objects[class_name]['touching']:
                    #self.log_and_normalize_pos(box, class_name)
                    self.log_and_normalize_distance(box, class_name)
                    self.boxes[class_name] = box

                if class_name in ["butt", "face", "pussy"] and self.tracked_objects[class_name]['touching']:
                    if not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id} as {class_name} is detected and touching")
                #self.boxes[class_name] = box
                self.tracking_states[class_name] = True
                
        """

        for box, conf, cls, class_name in sorted_boxes:

            all_detections[class_name].append([conf, box])

            if class_name == "glans" and conf >= 0.5:
                self.glans_handling(box, conf, current_frame_id)

            elif class_name == "penis" and conf >= 0.5:
                self.boxes['penis'] = box
                self.penis_handling(box, conf, current_frame_id)

        # for each class, check if it touches the locked penis box
        for class_name, detections in all_detections.items():
            if self.locked_penis_box:
                for conf, box in detections:
                    if class_name in ["left foot", "right foot", "left hand",
                                      "right hand", "butt", "face", "pussy"] and \
                            self.boxes_overlap(box, self.locked_penis_box):
                        self.log_and_normalize_distance(box, class_name)
                        self.boxes[class_name] = box
                        self.update_tracking(class_name, box, None, self.tracked_body_part)
                        if class_name in ["butt", "face", "pussy"] and self.tracked_objects[class_name]['touching']:
                            if not self.penetration:
                                self.penetration = True
                                print(
                                    f"Penetration started at frame {current_frame_id} as {class_name} is detected and touching")
                        self.tracking_states[class_name] = True

        # build a list of classes that have no self.boxes[class_name]
        no_box_classes = [class_name for class_name in class_names if self.boxes[class_name] is None]

        # if breast, navel, pussy, butt are missing, check if they are present in detections and keep the ones
        # that are present and lowest in the frame if multiple instances of them

        for checking_class in ["breast", "navel", "pussy", "butt"]:
            temp_boxes = []
            if checking_class in no_box_classes:
                # count the number of instances of the missing class in detections
                count = 0
                for class_name, detections in all_detections.items():
                    for conf, box in detections:
                        if checking_class == class_name and box is not None:
                            temp_boxes.append(box)
                            count += 1
                if count > 0:
                    no_box_classes.remove(checking_class)
                if count >= 1:
                    # print(f"Multiple instances of {checking_class} detected in frame {current_frame_id}")
                    # keep the box that is lowest in the frame
                    lowest_box = temp_boxes[0]
                    for box in temp_boxes:
                        if box[3] > lowest_box[3]:
                            lowest_box = box
                    self.boxes[checking_class] = lowest_box
                    if checking_class in ["navel", "breast"]:
                        self.log_and_normalize_pos(lowest_box, checking_class)
                        if checking_class == "navel":
                            self.navel_handling(lowest_box)
                    self.tracking_states[checking_class] = True

        # Handling irrelevant closeups first

        if self.tracking_states['pussy'] and self.tracking_states['butt']:
            # position likely to be Closeup
            self.handle_closeup_position("presence of pussy and butt")
        elif (self.tracking_states['pussy'] and self.boxes['pussy'] is not None and (
                self.boxes['pussy'][3] - self.boxes['pussy'][1]) >
              image_y_size * 0.1):
            # position likely to be Closeup
            self.handle_closeup_position("size of pussy in frame")

        # Handling relevant closeups
        elif self.breast_tracking and not self.glans_detected:
            if self.boxes['breast'] is not None and not self.boxes_overlap(self.locked_penis_box, self.boxes['breast']):
                # if self.boxes['breast'] is not None and not self.tracked_objects['breast']['touching']:
                # experimenting to handle delay in breast tracking when boobs reach peak or low position
                # projected_frame_id = current_frame_id  # + int(fps//5)
                # distances[projected_frame_id] = normalized_positions['breast'].get(current_frame_id, 100)
                distance = self.normalized_positions['breast'][-1]
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "breast"
                self.detect_sex_position_change("Closeup missionary", "breast tracking activated")
                if not self.penetration:
                    self.penetration = True
                    print(f"Penetration started at frame {current_frame_id} as we are tracking closeup breasts")

        # Now handling visible penetrations scenarii
        elif self.locked_penis_box and self.penetration:
            if self.boxes['penis'] is not None:
                # Compare the height of the two boxes and scale it to 0 - 100
                penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]

                locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]

                scale = min(int((penis_height / locked_penis_height) * 100), 100)
                distance = 100 - scale

                self.positions['penis'].append(distance)

                min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
                normalized_distance = (100 - int(100 * ((distance - min_distance) / (max_distance - min_distance)))) if min_distance != max_distance else 100

                self.normalized_positions['penis'].append(normalized_distance)

                self.update_distance(self.normalized_positions['penis'][-1])
            else:
                # Still handling penetration, but penis is not visible, could be a grinding scene
                if self.boxes['pussy'] is not None and self.tracking_states['pussy'] and self.boxes[
                    'breast'] is not None:
                    # position likely to be CowGirl / Missionary
                    # defaulting to navel tracking or breast tracking
                    #if self.tracking_states['navel']:
                    #    self.handle_CowGirl_Missionary_position("presence of navel", current_frame_id, 'navel')
                    #el
                    if self.tracking_states['breast']:
                        self.handle_CowGirl_Missionary_position("presence of breast", current_frame_id, 'breast')
                    # self.handle_CowGirl_Missionary_position("presence of pussy and breast", current_frame_id, 'pussy')
                    if self.tracked_objects['pussy']['touching'] and not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id} during pussy grinding")
                elif self.boxes['butt'] is not None and self.tracking_states['butt']:
                    # position likely to be Reverse Cowgirl or Doggy / Pronebone
                    self.handle_RevCowGirl_Doggy_Pronebone_position(
                        "Doggy / Reverse Cowgirl / Pronebone", "presence of foot", current_frame_id)
                    """
                    if self.tracking_states['left foot'] or self.tracking_states['right foot']:
                        # position likely to be Reverse Cowgirl
                        self.handle_RevCowGirl_Doggy_Pronebone_position(
                            "Reverse Cowgirl", "presence of foot", current_frame_id)
                    elif self.tracking_states['anus']:
                        self.handle_RevCowGirl_Doggy_Pronebone_position(
                            "Doggy / Pronebone", "presence of anus", current_frame_id)
                    else:
                        self.handle_RevCowGirl_Doggy_Pronebone_position(
                            "Doggy / Pronebone", "absence of foot", current_frame_id)
                    """
                    if self.tracked_objects['butt']['touching'] and not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id} during butt grinding")
                elif self.boxes['navel'] is not None and not self.boxes_overlap(self.locked_penis_box,
                                                                                self.boxes[
                                                                                    'navel']) and self.penetration:
                    # position likely to be CowGirl / Missionary, with occluded pussy
                    self.handle_CowGirl_Missionary_position("presence of navel during penetration",
                                                            current_frame_id,
                                                            'navel')
                elif self.boxes['breast'] is not None and self.tracking_states[
                    'breast'] and not self.penetration and \
                        self.tracked_objects["breast"]['touching']:
                    # position likely to be Boobjob
                    self.handle_Boobjob_position("presence of breast", current_frame_id)
                elif ((self.boxes['face'] is not None and self.tracking_states['face']) or
                      ((self.boxes['left hand'] is not None and self.tracking_states['left hand']) or
                       (self.boxes['right hand'] is not None and self.tracking_states[
                           'right hand']))) and not self.penetration:

                    self.handle_Blowjob_Handjob_position("presence of face or hand", current_frame_id)
        elif self.locked_penis_box and not self.penetration:
            # Now handling handjob / blowjob scenarii without mouth penetration
            if ((self.boxes['face'] is not None and self.tracking_states['face']) or
                  ((self.boxes['left hand'] is not None and self.tracking_states['left hand']) or
                   (self.boxes['right hand'] is not None and self.tracking_states[
                       'right hand']))) and not self.penetration:

                self.handle_Blowjob_Handjob_position("presence of face or hand", current_frame_id)

            elif ((self.boxes['left foot'] is not None and self.tracking_states['left foot']) or (
                    self.boxes['right foot'] is not None and self.tracking_states[
                'right foot'])) and not self.penetration:
                # position likely to be Footjob
                self.handle_Footjob_position("presence of foot", current_frame_id)

        # Check for non-detection of penis
        if self.boxes['penis'] is None:
            self.consecutive_non_detections['penis'] += 1
            self.consecutive_detections['penis'] = 0  # Reset the detection counter

            # Deactivate locked_penis_box after x consecutive non-detections
            if self.consecutive_non_detections['penis'] >= self.max_predictions:
                if self.locked_penis_box is not None:
                    self.locked_penis_box = None
                    print(
                        f"Penis tracking deactivated at frame {current_frame_id} after {self.consecutive_no_penis_detections} consecutive non-detections")
                    #print("Reinitializing tracker")
                    #self.__init__(self.class_names)
                    self.penis_box = None

        # Check for non-detection of glans
        if self.boxes['glans'] is None:
            self.consecutive_non_detections['glans'] += 1
            self.consecutive_detections['glans'] = 0  # Reset the detection counter


    def prev_tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
        #self.frame = frame
        self.image_y_size = image_y_size
        self.frame_count += 1

        #if self.frame_count % self.detection_interval == 0:
        #    self.initialize_trackers(sorted_boxes)
        #else:
        #    self.update_trackers()

        # initialize values
        self.foot_detected = False
        self.glans_detected = False
        self.tracking_states = {class_name: False for class_name in class_names}
        self.boxes = {class_name: None for class_name in class_names}

        #for class_name in class_names:
        #    # check if class_name is in sorted_boxes
        #    if any(box[3] == class_name for box in sorted_boxes):
        #        self.tracking_states[class_name] = True

        for box, conf, cls, class_name in sorted_boxes:
            if class_name == "glans" and conf > 0.7:
                self.glans_detected = True
                if conf > 0.8 and self.penetration:
                    self.penetration = False
                    # print(f"Penetration ended at frame {current_frame_id}")

            elif class_name == "penis" and conf > 0.7:
                self.boxes['penis'] = box
                self.penis_handling(box, conf, current_frame_id)

            elif class_name == "navel" and conf > 0.7:
                self.navel_handling(box)

            if class_name in ["breast", "navel"] and conf > 0.7:
                self.log_and_normalize_pos(box, class_name)

            elif self.locked_penis_box and class_name not in ["penis", "glans"]:
                if conf > 0.5:
                    self.update_tracking(class_name, box, None, self.tracked_body_part)
                else:
                    fallback_class = {'pussy': 'breast', 'breast': 'navel'}.get(class_name)
                    self.update_tracking(class_name, None, fallback_class, self.tracked_body_part)

                #if class_name not in ["breast", "navel"] and not self.boxes_overlap(locked_penis_box, box):
                #    continue

                if class_name not in ["breast", "navel"] and not self.tracked_objects[class_name]['touching']:
                    continue

                if class_name in ["left foot", "right foot", "left hand", "right hand", "butt", "face", "pussy"] and self.tracked_objects[class_name]['touching']:
                    self.log_and_normalize_pos(box, class_name)

                self.boxes[class_name] = box
                self.tracking_states[class_name] = True
        if self.tracked_body_part == "Nothing":
            self.update_distance(100)

        if self.tracked_body_part and self.tracked_body_part != "Nothing":
            if not self.tracking_states[self.tracked_body_part]:
                if not self.occlusion:
                    #print(f"Lost tracking of {self.tracked_body_part} at frame {current_frame_id}, suspecting occlusion")
                    self.occlusion = True
                self.update_tracking(self.tracked_body_part, None, None, self.tracked_body_part)
            else:
                if self.occlusion:
                    #print(f"Found back {self.tracked_body_part} at frame {current_frame_id}")
                    self.occlusion = False

        if self.breast_tracking:
            if self.boxes['breast'] is not None and not self.boxes_overlap(self.locked_penis_box, self.boxes['breast']):
            #if self.boxes['breast'] is not None and not self.tracked_objects['breast']['touching']:
                # experimenting to handle delay in breast tracking when boobs reach peak or low position
                #projected_frame_id = current_frame_id  # + int(fps//5)
                #distances[projected_frame_id] = normalized_positions['breast'].get(current_frame_id, 100)
                distance = self.normalized_positions['breast'][-1]
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "breast"
                self.detect_sex_position_change("Closeup missionary", "breast tracking activated")
                if not self.penetration:
                    self.penetration = True
                    print(f"Penetration started at frame {current_frame_id}")
        else:
            if self.penis_box is None and self.penetration and self.tracked_body_part == "pussy":
                # trying to handle pussy occlusion
                if self.tracking_states['navel']:
                    print(f"Handling pussy occlusion at frame {current_frame_id}")
                    #self.update_distance(self.normalized_positions['navel'].get(current_frame_id, 100))
                    self.update_distance(self.normalized_positions['navel'][-1])

            #elif self.penis_box is not None and self.locked_penis_box is not None:
            elif self.locked_penis_box:
                # Dealing with irrelevant positions
                if self.tracking_states['pussy'] and self.tracking_states['butt']:
                    # position likely to be Closeup
                    self.handle_closeup_position("presence of pussy and butt")

                elif (self.tracking_states['pussy'] and (self.boxes['pussy'][3] - self.boxes['pussy'][1]) >
                      image_y_size * 0.1):
                    # position likely to be Closeup
                    self.handle_closeup_position("size of pussy in frame")

                # Now dealing with relevant positions
                elif self.boxes['pussy'] is not None and self.tracking_states['pussy'] and self.boxes['breast'] is not None:
                    # position likely to be CowGirl / Missionary
                    self.handle_CowGirl_Missionary_position("presence of pussy and breast", current_frame_id, 'pussy')
                    if self.tracked_objects['pussy']['touching'] and not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id}")

                elif self.boxes['butt'] is not None and self.tracking_states['butt']:
                    # position likely to be Reverse Cowgirl or Doggy / Pronebone
                    if self.tracking_states['left foot'] or self.tracking_states['right foot']:
                        # position likely to be Reverse Cowgirl
                        self.handle_RevCowGirl_Doggy_Pronebone_position(
                            "Reverse Cowgirl","presence of foot", current_frame_id)
                    elif self.tracking_states['anus']:
                        self.handle_RevCowGirl_Doggy_Pronebone_position(
                            "Doggy / Pronebone", "presence of anus", current_frame_id)
                    else:
                        self.handle_RevCowGirl_Doggy_Pronebone_position(
                            "Doggy / Pronebone", "absence of foot",current_frame_id)
                    if self.tracked_objects['butt']['touching'] and not self.penetration:
                        self.penetration = True
                        print(f"Penetration started at frame {current_frame_id}")

                elif ((self.boxes['face'] is not None and self.tracking_states['face']) or
                      ((self.boxes['left hand'] is not None and self.tracking_states['left hand']) or
                       (self.boxes['right hand'] is not None and self.tracking_states['right hand']))) and not self.penetration:

                    self.handle_Blowjob_Handjob_position("presence of face or hand", current_frame_id)

                elif self.boxes['navel'] is not None and not self.boxes_overlap(self.locked_penis_box, self.boxes['navel']) and self.penetration:
                    # position likely to be CowGirl / Missionary, with occluded pussy
                    self.handle_CowGirl_Missionary_position("presence of navel during penetration", current_frame_id, 'navel')

                elif self.boxes['breast'] is not None and self.tracking_states['breast'] and not self.penetration and self.tracked_objects["breast"]['touching']:
                    # position likely to be Boobjob
                    self.handle_Boobjob_position("presence of breast", current_frame_id)

                elif ((self.boxes['left foot'] is not None and self.tracking_states['left foot']) or (
                        self.boxes['right foot'] is not None and self.tracking_states['right foot'])) and not self.penetration:
                    # position likely to be Footjob
                    self.handle_Footjob_position("presence of foot", current_frame_id)
                #else:
                #    # Kalman ?
                #    self.distance = self.previous_distance

        # Check for non-detection of penis
        if self.boxes['penis'] is None:
            self.consecutive_no_penis_detections += 1
            self.consecutive_penis_detections = 0  # Reset the detection counter

            # Deactivate locked_penis_box after x consecutive non-detections
            if self.consecutive_no_penis_detections >= self.max_predictions:
                if self.locked_penis_box is not None:
                    self.locked_penis_box = None
                    print(
                        f"Penis tracking deactivated at frame {current_frame_id} after {self.consecutive_no_penis_detections} consecutive non-detections")
                    #print("Reinitializing tracker")
                    #self.__init__(self.class_names)
                    self.penis_box = None

    def normalize_box_area(self, box, frame_width, frame_height):
        """
        Normalize the area of a bounding box to a 0-100 scale.

        Parameters:
            box (tuple): Bounding box in the format (x1, y1, x2, y2).
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.

        Returns:
            float: Normalized area of the box (0 to 100 scale).
        """
        x1, y1, x2, y2 = box
        # Ensure valid dimensions
        if x2 <= x1 or y2 <= y1:
            return 0  # Invalid box

        # Calculate the area of the box
        box_area = (x2 - x1) * (y2 - y1)

        # Define the maximum possible area (entire frame)
        max_area = frame_width * frame_height

        # Normalize the area to 0-100
        normalized_area = 100 * (box_area / max_area)

        # Clamp to the range [0, 100]
        return max(0, min(100, normalized_area))

    def penis_handling(self, box, conf, current_frame_id):
        if self.penis_box is None:
            print(f"Penis detected at frame {current_frame_id} with confidence {conf}")

        self.penis_box = box

        self.consecutive_detections['penis'] += 1
        self.consecutive_non_detections['penis'] = 0

        px1, py1, px2, py2 = self.penis_box
        current_height = py2 - py1

        #if self.glans_detected:
        #    self.locked_penis_height = current_height
        #el
        #if current_height > self.locked_penis_height:
        #    self.locked_penis_height = current_height

        #if self.locked_penis_box:
        #    self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)

        if self.consecutive_detections['penis'] >= self.detections_threshold:
            if self.locked_penis_box is None or self.glans_detected or current_height > self.locked_penis_height:
                if self.locked_penis_box is None:
                    print(f"@{self.current_frame_id} - Locked penis box activated")
                self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
                #self.locked_penis_box = self.penis_box
                self.locked_penis_height = current_height


        if self.locked_penis_box:
            if current_height > self.locked_penis_height:
                self.locked_penis_height = current_height
                self.locked_penis_box = self.penis_box
            if self.penis_box[3] != self.locked_penis_box[3]:
                self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
        """
        if self.consecutive_detections['penis'] >= self.detections_threshold and self.glans_detected:
            # self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
            self.locked_penis_box = self.penis_box
            # print(f"Actuating penis box at frame {current_frame_id}")
        """

        if self.locked_penis_box and self.penetration:
            # Compare the height of the two boxes and scale it to 0 - 100
            penis_height = self.boxes['penis'][3] - self.boxes['penis'][1]
            locked_penis_height = self.locked_penis_box[3] - self.locked_penis_box[1]
            scale = min(int((penis_height / locked_penis_height) * 100), 100)
            distance = 100 - scale
            self.positions['penis'].append(distance)
            min_distance, max_distance = min(self.positions['penis']), max(self.positions['penis'])
            normalized_distance = (100 - int(100 * ((distance - min_distance) / (
                    max_distance - min_distance)))) if min_distance != max_distance else 100
            self.normalized_positions['penis'].append(normalized_distance)

        if (self.image_y_size - py1) / self.image_y_size < 0.1 and not self.breast_tracking and self.tracked_objects['breast']['detected']:
            # print("Breast tracking mode activated given penis position in lower 10th of frame")
            self.breast_tracking = True

        return self.penis_box, self.locked_penis_box, self.breast_tracking

    def glans_handling(self, box, conf, current_frame_id):
        self.consecutive_detections['glans'] += 1
        self.consecutive_non_detections['glans'] = 0
        #print(f"Glans consecutive detections: {self.consecutive_detections['glans']}")
        if self.consecutive_detections['glans'] >= self.detections_threshold:
            self.boxes['glans'] = box
            self.glans_detected = True
            #print(f"Glans box: {self.boxes['glans']}")
            if self.penetration:
                self.penetration = False
                print(f"Penetration ended at frame {current_frame_id} after {self.consecutive_detections['glans']} detections of glans")
            #if not self.penetration:
            #    self.penetration = True
            #    print(
            #        f"Penetration started at frame {current_frame_id} after {self.consecutive_non_detections['glans']} consecutive non-detections")


    def navel_handling(self, box):
        if (self.image_y_size - box[1]) / self.image_y_size < 0.15 and not self.breast_tracking:
            print("Breast tracking mode activated given navel position in lower 15th of frame")
            self.breast_tracking = True

    def log_and_normalize_pos(self, box, class_name):
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

        #self.normalized_positions[class_name][current_frame_id] = normalized_y
        self.normalized_positions[class_name].append(normalized_y)
        return normalized_y

    def log_and_normalize_distance(self, box, class_name):
        if self.locked_penis_box is None:
            locked_penis_box = (0,0,0,0)
        else:
            locked_penis_box = self.locked_penis_box
        distance = self.calculate_distance(locked_penis_box, box)
        self.distances[class_name].append(distance)
        min_distance, max_distance = min(self.distances[class_name]), max(self.distances[class_name])
        normalized_distance = (100 - int(100 * ((distance - min_distance) / (max_distance - min_distance)))) if min_distance != max_distance else 100
        self.normalized_distances[class_name].append(normalized_distance)


    def handle_closeup_position(self, reason):
        self.distance = 100
        self.update_distance(self.distance)
        self.tracked_body_part = "Nothing"
        self.detect_sex_position_change("Closeup", reason)

    def handle_CowGirl_Missionary_position(self, reason, current_frame_id, body_part):
        distance = None
        if body_part == "pussy":
            if self.penis_box:
                x1, y1, x2, y2 = self.penis_box
                if self.boxes['pussy'] is not None:
                    y_p = self.boxes['pussy'][3]
                    # if the bottom of pussy box remains in the low 30% of the penis_box
                    if y_p < y2 and y_p > 0.7 * (y2 - y1):
                        self.consecutive_grinding_supiscions['pussy'] += 1
                    else:
                        self.consecutive_grinding_supiscions['pussy'] = 0
                        self.grinding = False
            else:
                self.consecutive_grinding_supiscions['pussy'] += 1
            if self.consecutive_grinding_supiscions['pussy'] > self.fps:
                self.grinding = True

        if self.grinding:
            # default to a combination of navel and breast
            #nb_items = 0
            #distance = 0
            if self.boxes['navel']:
                #distance = self.normalized_positions['navel'][-1]
                distance = self.normalized_distances['navel'][-1]
                #nb_items += 1
            #if self.boxes['breast']:
            #    distance += self.normalized_positions['breast'][-1]
            #    nb_items += 1
            #distance = distance // nb_items
        else:
            #distance = self.normalized_positions[body_part][-1]
            distance = self.normalized_distances[body_part][-1]

        self.distance = self.update_distance(distance)
        self.tracked_body_part = body_part
        self.detect_sex_position_change("Cowgirl / Missionary", reason)

    def handle_RevCowGirl_Doggy_Pronebone_position(self, position, reason, current_frame_id):
        distance = None
        #if self.boxes['butt'][3] > self.locked_penis_box[3]:
        #    # print(f"Actuating locked penis box to fit butt lower position at frame {current_frame_id}")
        #    self.locked_penis_box = (
        #        self.locked_penis_box[0], self.locked_penis_box[1],
        #        self.locked_penis_box[2], self.boxes['butt'][3])
        # distance = self.calculate_distance(self.locked_penis_box, self.boxes['butt'], 'butt')
        if self.normalized_positions['penis'][-1] is not None:
            distance = self.normalized_positions['penis'][-1]
        else:
            #distance = self.calculate_distance(self.locked_penis_box, self.boxes['butt'], 'butt')
            #distance = self.normalized_distances['butt'][-1]
            distance = (self.normalized_positions['butt'][-1] + self.normalized_distances['butt'][-1]) // 2
        self.distance = self.update_distance(distance)
        self.tracked_body_part = "butt"
        self.detect_sex_position_change(position, reason)

    def handle_Blowjob_Handjob_position(self, reason, current_frame_id):
        distance = None

        if self.tracking_states['face'] and self.tracked_objects['face']['touching']:
            if len(self.face_distances) > 3:
                self.face_distances.pop(0)
            #distance = self.calculate_distance(self.locked_penis_box, self.boxes['face'])
            distance = self.normalized_distances['face'][-1]
            #print(f"Face distance: {distance}")
            #self.face_distances.append(distance)
            self.face_distances.append(self.normalized_distances['face'][-1])
            self.face_hands_movements['face'] = abs(max(self.face_distances) - min(self.face_distances))
            # if no left hand or no right hand, return
            if not self.tracked_objects['left hand']['touching'] and not self.tracked_objects['right hand']['touching']:
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "face"
                self.detect_sex_position_change("Blowjob / Handjob", "presence of face only")
                return
        else:
            self.face_hands_movements['face'] = 0

        if self.tracking_states['left hand'] and self.tracked_objects['left hand']['touching'] and self.boxes['left hand'] is not None:
            if len(self.left_hand_distances) > 3:
                self.left_hand_distances.pop(0)
            #distance = self.calculate_distance(self.locked_penis_box, self.boxes['left hand'])
            distance = self.normalized_distances['left hand'][-1]
            #print(f"Left hand distance: {distance}")
            #self.left_hand_distances.append(distance)
            self.left_hand_distances.append(self.normalized_distances['left hand'][-1])
            self.face_hands_movements['left hand'] = abs(max(self.left_hand_distances) - min(self.left_hand_distances))
            if not self.tracked_objects['right hand']['touching'] and not self.tracked_objects['face']['touching']:
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "left hand"
                self.detect_sex_position_change("Blowjob / Handjob", "presence of left hand only")
                return
        else:
            self.face_hands_movements['left hand'] = 0

        if self.tracking_states['right hand'] and self.tracked_objects['right hand']['touching'] and self.boxes['right hand'] is not None:
            if len(self.right_hand_distances) > 3:
                self.right_hand_distances.pop(0)
            #distance = self.calculate_distance(self.locked_penis_box, self.boxes['right hand'])
            distance = self.normalized_distances['right hand'][-1]
            #print(f"Right hand distance: {distance}, self.locked_penis_box: {self.locked_penis_box}, self.boxes['right hand']: {self.boxes['right hand']}")
            #self.right_hand_distances.append(distance)
            self.right_hand_distances.append(self.normalized_distances['right hand'][-1])
            self.face_hands_movements['right hand'] = abs(
                max(self.right_hand_distances) - min(self.right_hand_distances))
            if not self.tracked_objects['left hand']['touching'] and not self.tracked_objects['face']['touching']:
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "right hand"
                self.detect_sex_position_change("Blowjob / Handjob", "presence of right hand only")
                return
        else:
            self.face_hands_movements['right hand'] = 0

        # Determine the most moving body part
        most_moving_part = max(self.face_hands_movements, key=self.face_hands_movements.get, default=None)
        #print(f"Most moving part: {most_moving_part}")
        max_movement = self.face_hands_movements.get(most_moving_part, 0)
        current_tracked_movement = self.face_hands_movements.get(self.tracked_body_part, 0)

        if most_moving_part and most_moving_part != self.tracked_body_part:
            if (
                    max_movement > current_tracked_movement * self.switch_threshold_multiplier and
                    self.switch_cooldown == 0
            ):
                if most_moving_part == self.last_tracked_body_part:
                    self.consecutive_switch_frames += 1
                else:
                    self.consecutive_switch_frames = 1  # Reset counter if different part
                self.last_tracked_body_part = most_moving_part

                if self.consecutive_switch_frames >= self.required_frames_to_switch:
                    self.tracked_body_part = most_moving_part
                    self.consecutive_switch_frames = 0
                    self.switch_cooldown = self.cooldown_frames  # Set cooldown
                    #print(f"Switching to: {self.tracked_body_part} due to sustained movement")
            else:
                self.consecutive_switch_frames = 0  # Reset counter if condition not met
        else:
            self.consecutive_switch_frames = 0  # Reset if no significant movement detected

        # Cooldown logic
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1

        # Update position and distance for the tracked body part
        if self.tracked_body_part in ['face', 'left hand', 'right hand']:
            self.detect_sex_position_change("Blowjob / Handjob", reason)
            if self.tracked_body_part == "face":
                distance = self.face_distances[-1]
            elif self.tracked_body_part == "left hand":
                distance = self.left_hand_distances[-1]
            elif self.tracked_body_part == "right hand":
                distance = self.right_hand_distances[-1]

            # Smooth transitions with a moving average
            #if not hasattr(self, 'smoothed_distance'):
            #    self.smoothed_distance = distance
            #    print(f"Smoothed distance initialized to {self.smoothed_distance}")
            #else:
            #    smoothing_factor = 0.2  # Adjust for smoother transitions (lower = smoother)
            #    self.smoothed_distance = self.smoothed_distance * (1 - smoothing_factor) + self.distance * smoothing_factor
            #    print(f"Smoothed distance updated to {self.smoothed_distance}")
            #self.distance = self.update_distance(self.smoothed_distance)

            self.distance = self.update_distance(distance)

            # Log the tracked information
            # print(f"Tracked: {self.tracked_body_part}, Position: {self.sex_position}, Distance: {int(self.smoothed_distance)}")

    def handle_Footjob_position(self, reason, current_frame_id):
        distance = 100

        if self.tracking_states['left foot'] and self.tracked_objects['left foot']['touching']:
            if len(self.left_foot_distances) > 3:
                self.left_foot_distances.pop(0)
            self.left_foot_distances.append(
                self.calculate_distance(self.locked_penis_box, self.boxes['left foot']))
            self.foot_movements['left foot'] = abs(max(self.left_foot_distances) - min(self.left_foot_distances))

        if self.tracking_states['right foot'] and self.tracked_objects['right foot']['touching']:
            if len(self.right_foot_distances) > 3:
                self.right_foot_distances.pop(0)
            self.right_foot_distances.append(
                self.calculate_distance(self.locked_penis_box, self.boxes['right foot']))
            self.foot_movements['right foot'] = abs(max(self.right_foot_distances) - min(self.right_foot_distances))

        # Determine the most moving foot
        most_moving_foot = max(self.foot_movements, key=self.foot_movements.get, default=None)
        max_foot_movement = self.foot_movements.get(most_moving_foot, 0)
        current_tracked_foot_movement = self.foot_movements.get(self.tracked_body_part, 0)

        if most_moving_foot and most_moving_foot != self.tracked_body_part:
            # Check if movement is significantly higher than the currently tracked foot
            if (
                    max_foot_movement > current_tracked_foot_movement * self.foot_switch_threshold_multiplier and
                    self.foot_switch_cooldown == 0
            ):
                # Increment confidence counter
                if most_moving_foot == self.last_tracked_foot:
                    self.consecutive_foot_switch_frames += 1
                else:
                    self.consecutive_foot_switch_frames = 1  # Reset counter if different foot

                self.last_tracked_foot = most_moving_foot

                # Switch only if sustained for required frames
                if self.consecutive_foot_switch_frames >= self.required_foot_frames_to_switch:
                    self.tracked_body_part = most_moving_foot
                    self.consecutive_foot_switch_frames = 0
                    self.foot_switch_cooldown = self.foot_cooldown_frames  # Set cooldown
                    # print(f"Switching to: {self.tracked_body_part} due to sustained foot movement")
            else:
                self.consecutive_foot_switch_frames = 0  # Reset counter if condition not met
        else:
            self.consecutive_foot_switch_frames = 0  # Reset if no significant movement detected

        # Cooldown logic
        if self.foot_switch_cooldown > 0:
            self.foot_switch_cooldown -= 1

        # Update position and distance for the tracked foot
        if self.tracked_body_part in ['left foot', 'right foot']:
            self.detect_sex_position_change("Footjob", reason)
            if self.tracked_body_part == "left foot":
                distance = self.left_foot_distances[-1]
            elif self.tracked_body_part == "right foot":
                distance = self.right_foot_distances[-1]

        # Smooth transitions with a moving average
        foot_smoothing_factor = 0.2  # Adjust for smoother transitions (lower = smoother)
        if hasattr(self, 'smoothed_foot_distance'):
            self.smoothed_foot_distance = self.smoothed_foot_distance * (
                        1 - foot_smoothing_factor) + distance * foot_smoothing_factor
        else:
            self.smoothed_foot_distance = distance

        # Log the tracked foot information
        # print(f"Tracked: {self.tracked_body_part}, Position: {self.sex_position}, Distance: {self.smoothed_foot_distance}")
        self.distance = self.update_distance(self.smoothed_foot_distance)

    def handle_Boobjob_position(self, reason, current_frame_id):
        #distance = self.calculate_distance(self.locked_penis_box, self.boxes['breast'], 'breast')
        distance = self.normalized_distances['breast'][-1]
        self.distance = self.update_distance(distance)
        self.tracked_body_part = "breast"
        self.detect_sex_position_change("Boobjob", reason)

