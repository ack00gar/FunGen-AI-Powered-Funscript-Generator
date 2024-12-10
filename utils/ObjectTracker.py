from utils import KalmanFilter as KF
from collections import deque
import numpy as np
import cv2
from utils.config import class_names

class ObjectTracker:
    def __init__(self, state=None):
        self.class_names = class_names
        self.tracked_objects = {
            class_name: {'kf': KF.KalmanFilter(),
                         'position': None, 'detected': False, 'touching': False, 'prediction_count': 0}
            for class_name in class_names
        }
        self.detection_interval = 10
        self.frame_count = 0
        self.trackers = {}

        self.distance_kf = KF.KalmanFilter()

        self.frame = None
        self.image_y_size = 0

        self.occlusion = False

        self.minimum_penis_consecutive_detections = 2

        self.penis_box, self.locked_penis_box = None, None
        self.glans_detected = False
        self.locked_penis_height = 0
        self.breast_tracking = False
        self.distance = 100
        self.raw_distance = 100
        self.previous_distances = [100, 100, 100]
        self.tracked_body_part = "Nothing"

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

        #self.positions = {class_name: [] for class_name in class_names}
        self.positions = {class_name: deque(maxlen=60) for class_name in class_names}

        # Initialize normalized_positions as a dictionary of dictionaries
        self.normalized_positions = {class_name: deque(maxlen=10) for class_name in class_names}
        self.normalized_positions['navel'].append(100)
        self.normalized_positions['breast'].append(100)
        #    'breast': deque(maxlen=10),
        #    'navel': deque(maxlen=10),
            # add other classes as needed
        #}

        #self.normalized_positions = []

        self.moving_average_window = 5  # Number of frames to consider for moving average

        # Counters for consecutive detections and non-detections
        self.consecutive_penis_detections = 0
        self.consecutive_no_penis_detections = 0

        self.penetration = False
        self.penetration_mode = 'No penetration'
        self.tracked_body_part = None

        self.max_predictions = 600

        if state:
            tracked_objects_state = state.get('tracked_objects_state', {})
            for class_name, obj_state in tracked_objects_state.items():
                if class_name in self.tracked_objects:
                    self.tracked_objects[class_name]['kf'].mean = obj_state['mean']
                    self.tracked_objects[class_name]['kf'].covariance = obj_state['covariance']
                    self.tracked_objects[class_name]['position'] = obj_state['position']
                    self.tracked_objects[class_name]['detected'] = obj_state['detected']
                    self.tracked_objects[class_name]['touching'] = obj_state['touching']
                    self.tracked_objects[class_name]['prediction_count'] = obj_state['prediction_count']

            distance_kf_state = state.get('distance_kf_state', {})
            if distance_kf_state:
                self.distance_kf.mean = distance_kf_state['mean']
                self.distance_kf.covariance = distance_kf_state['covariance']

            self.tracked_body_part = state.get('tracked_body_part', "Nothing")
            self.sex_position = state.get('sex_position', "Not relevant")
            self.sex_position_reason = state.get('sex_position_reason', "")

            positions = state.get('positions', {})
            for class_name, pos_list in positions.items():
                if class_name in self.positions:
                    self.positions[class_name].extend(pos_list)

            normalized_positions = state.get('normalized_positions', {})
            for class_name, norm_pos_list in normalized_positions.items():
                if class_name in self.normalized_positions:
                    self.normalized_positions[class_name].extend(norm_pos_list)

            self.face_hands_movements = state.get('face_hands_movements', {})
            self.foot_movements = state.get('foot_movements', {})
            self.right_hand_distances = state.get('right_hand_distances', [])
            self.left_hand_distances = state.get('left_hand_distances', [])
            self.right_foot_distances = state.get('right_foot_distances', [])
            self.left_foot_distances = state.get('left_foot_distances', [])
            self.consecutive_penis_detections = state.get('consecutive_penis_detections', 0)
            self.consecutive_no_penis_detections = state.get('consecutive_no_penis_detections', 0)
            self.penetration = state.get('penetration', False)
            self.penetration_mode = state.get('penetration_mode', 'No penetration')
            self.switch_cooldown = state.get('switch_cooldown', 0)
            self.foot_switch_cooldown = state.get('foot_switch_cooldown', 0)

    def get_state(self):
        state = {
            'tracked_objects_state': {
                class_name: {
                    'mean': obj['kf'].mean,
                    'covariance': obj['kf'].covariance,
                    'position': obj['position'],
                    'detected': obj['detected'],
                    'touching': obj['touching'],
                    'prediction_count': obj['prediction_count']
                }
                for class_name, obj in self.tracked_objects.items()
            },
            'distance_kf_state': {
                'mean': self.distance_kf.mean,
                'covariance': self.distance_kf.covariance
            },
            'tracked_body_part': self.tracked_body_part,
            'sex_position': self.sex_position,
            'sex_position_reason': self.sex_position_reason,
            'positions': {
                class_name: list(pos)
                for class_name, pos in self.positions.items()
            },
            'normalized_positions': {
                class_name: list(norm_pos)
                for class_name, norm_pos in self.normalized_positions.items()
            },
            'face_hands_movements': self.face_hands_movements,
            'foot_movements': self.foot_movements,
            'right_hand_distances': self.right_hand_distances,
            'left_hand_distances': self.left_hand_distances,
            'right_foot_distances': self.right_foot_distances,
            'left_foot_distances': self.left_foot_distances,
            'consecutive_penis_detections': self.consecutive_penis_detections,
            'consecutive_no_penis_detections': self.consecutive_no_penis_detections,
            'penetration': self.penetration,
            'penetration_mode': self.penetration_mode,
            'switch_cooldown': self.switch_cooldown,
            'foot_switch_cooldown': self.foot_switch_cooldown
        }
        return state

    def reinit_with_distance(self, distance):
        self.__init__()
        self.distance = distance

    def initialize_trackers(self, sorted_boxes):
        params = cv2.TrackerVit_Params()
        params.net = "models/vit/object_tracking_vittrack_2023sep.onnx"

        for box, conf, cls, class_name in sorted_boxes:
            if class_name not in self.trackers:
                tracker = cv2.TrackerVit_create(params)
                tracker.init(self.frame, tuple(box))
                self.trackers[class_name] = tracker

    def update_trackers(self):
        for class_name, tracker in self.trackers.items():
            success, box = tracker.update(self.frame)
            if success:
                self.tracked_objects[class_name]['position'] = box
                self.tracked_objects[class_name]['detected'] = True
            else:
                self.tracked_objects[class_name]['detected'] = False

    def update_distance(self, raw_distance):
        rounded_distance = round(raw_distance/5)*5
        #self.previous_distances.pop(0)
        #self.previous_distances.append(raw_distance)
        ema_distance = int(0.7 * rounded_distance + 0.1 * self.previous_distances[1] + 0.2 * self.previous_distances[2])
        # cap the change in distance to a maximum of +5 or -5 vs previous distance
        if abs(ema_distance - self.previous_distances[-1]) > 15:
            ema_distance = self.previous_distances[-1] + np.sign(ema_distance - self.previous_distances[-1]) * 15
            #print(f"Capping distance change to {ema_distance} vs {raw_distance}")
        elif abs(ema_distance - self.previous_distances[-1]) < 2:
            # if the change is less than 2, use the previous distance to eliminate jitter
            #print(f"Skipping distance change to {ema_distance} vs {self.previous_distances[-1]} because change is less than 4")
            ema_distance = self.previous_distances[-1]
        self.previous_distances.pop(0)
        self.previous_distances.append(ema_distance)
        #self.raw_distance = raw_distance
        if raw_distance is not None:
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
        # Only perform prediction for the tracked item
        elif class_name == tracked_item:
            # Object not detected: Predict position based on the Kalman filter
            # print(f"Kalman prediction for {class_name}: {tracked['prediction_count'] + 1} / {int(self.max_predictions)}")
            tracked['kf'].predict()
            # print(f"Kalman prediction for {class_name}: {tracked['prediction_count'] + 1} / {int(self.max_predictions)}")
            tracked['detected'] = False
            tracked['position'] = tracked['kf'].position
            tracked['touching'] = self.boxes_overlap(self.locked_penis_box, tracked['position'])

            # Use fallback if primary tracked object is lost and prediction count exceeds 60
            if tracked['prediction_count'] >= self.max_predictions:
                if fallback_class and self.tracked_objects[fallback_class]['detected']:
                    print(f"Handling occlusion of {class_name}, fallback to {fallback_class}")
                    tracked['position'] = self.tracked_objects[fallback_class]['position']
                else:
                    print(f"No fallback available for {class_name}, deactivating tracking")
                    tracked['position'] = None
                    tracked['detected'] = False
                    tracked['touching'] = False
                    tracked['prediction_count'] = 0
                    self.tracked_body_part = None

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

    def calculate_distance(self, penis_box, other_box, class_name):
        """Calculate distance between penis box and other body part box."""
        px1, py1, px2, py2 = penis_box
        ox1, oy1, ox2, oy2 = other_box

        min_reevaluation = py1
        max_reevaluation = py2

        # Retrieve normalized distance for class_name
        normalized_distance = self.normalized_positions[class_name][-1]

        if class_name in ['left foot', 'right foot', 'breast', 'left hand', 'right hand']:
            # Calculate distance from middle of hand/foot box to bottom of penis box
            pos = (oy1 + oy2) // 2
            min_reevaluation = py1 + ((oy2 - oy1) * 0.2)
            max_reevaluation = py2 - ((oy2 - oy1) / 3)

        else:
            # Calculate distance from bottom of other box to bottom of penis box
            pos = oy2
            if class_name == 'butt':
                min_reevaluation = py1 + ((oy2 - oy1) * 0.2)
            elif class_name == 'face':
                min_reevaluation = py1 + ((py2 - py1) * 0.1)
                max_reevaluation = py1 + ((py2 - py1) * 0.6)
            elif class_name == 'pussy':
                max_reevaluation = py2 - ((oy2 - oy1) // 2)

        secured_distance = max(0, max_reevaluation - pos)
        # Normalize distance to 0-100 scale
        max_distance = max_reevaluation - min_reevaluation

        if normalized_distance:
            if max_distance == 0:
                return normalized_distance
            else:
                normalized_distance = (int(max(0, min(100, (secured_distance / max_distance) * 100))) + normalized_distance) //2
        else:
            if max_distance == 0:
                return 100
            else:
                normalized_distance = int(max(0, min(100, (secured_distance / max_distance) * 100)))
        # print(f"{class_name} distance: secured : {secured_distance}, max distance: {max_distance},  normalized: {normalized_distance}")
        return normalized_distance

    def detect_sex_position_change(self, sex_position, reason):
        if sex_position != self.sex_position and sex_position == self.prev_sex_position:
            print(f"Sex position changed from {self.sex_position} to {sex_position} given {reason}")
            self.sex_position = sex_position
            self.sex_position_reason = reason
        else:
            self.prev_sex_position = sex_position

    def tracking_logic(self, sorted_boxes, current_frame_id, image_y_size):
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
            if class_name == "glans" and conf > 0.5:
                self.glans_detected = True
                if conf > 0.8 and self.penetration:
                    self.penetration = False
                    # print(f"Penetration ended at frame {current_frame_id}")

            elif class_name == "penis" and conf > 0.5:
                self.boxes['penis'] = box
                self.penis_handling(box, conf, current_frame_id)

            elif class_name == "navel" and conf > 0.5:
                self.navel_handling(box)

            if class_name in ["breast", "navel"] and conf > 0.5:
                self.log_and_normalize_pos(box, class_name)

            elif self.locked_penis_box and class_name not in ["penis", "glans"]:
                if conf > 0.4:
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

    def penis_handling(self, box, conf, current_frame_id):
        if self.penis_box is None:
            print(f"Penis detected at frame {current_frame_id} with confidence {conf}")

        self.penis_box = box

        px1, py1, px2, py2 = self.penis_box
        current_height = py2 - py1

        if self.glans_detected:
            self.locked_penis_height = current_height
        elif current_height > self.locked_penis_height:
            self.locked_penis_height = current_height

        if self.locked_penis_box:
            self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)

        self.consecutive_penis_detections += 1
        self.consecutive_no_penis_detections = 0

        if self.consecutive_penis_detections >= self.minimum_penis_consecutive_detections and self.locked_penis_box is None:
            self.locked_penis_box = (px1, py2 - self.locked_penis_height, px2, py2)
            # print(f"Locked penis box activated at frame {current_frame_id}")

        if (self.image_y_size - py1) / self.image_y_size < 0.1 and not self.breast_tracking and self.tracked_objects['breast']['detected']:
            # print("Breast tracking mode activated given penis position in lower 10th of frame")
            self.breast_tracking = True

        return self.penis_box, self.locked_penis_box, self.breast_tracking

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
        #self.normalized_positions[class_name][current_frame_id] = normalized_y
        self.normalized_positions[class_name].append(normalized_y)

    def handle_closeup_position(self, reason):
        self.distance = 100
        self.update_distance(self.distance)
        self.tracked_body_part = "Nothing"
        self.detect_sex_position_change("Closeup", reason)

    def handle_CowGirl_Missionary_position(self, reason, current_frame_id, body_part):
        if body_part == "pussy":
            distance = self.calculate_distance(self.locked_penis_box, self.boxes[body_part], body_part)
            # trying to add up the navel distance and average
            #if self.normalized_positions['navel'].get(current_frame_id, 100) is not None:
            #    distance = 0.3 * distance + 0.7 * self.normalized_positions['navel'].get(current_frame_id, 100)
            if self.normalized_positions['navel'][-1] is not None:
                distance = 0.3 * distance + 0.7 * self.normalized_positions['navel'][-1]
        elif body_part == "navel":
            #distance = self.normalized_positions['navel'].get(current_frame_id, 100)
            distance = self.normalized_positions['navel'][-1]
        self.distance = self.update_distance(distance)
        self.tracked_body_part = body_part
        self.detect_sex_position_change("Cowgirl / Missionary", reason)

    def handle_RevCowGirl_Doggy_Pronebone_position(self, position, reason, current_frame_id):
        if self.boxes['butt'][3] > self.locked_penis_box[3]:
            # print(f"Actuating locked penis box to fit butt lower position at frame {current_frame_id}")
            self.locked_penis_box = (
                self.locked_penis_box[0], self.locked_penis_box[1],
                self.locked_penis_box[2], self.boxes['butt'][3])
        distance = self.calculate_distance(self.locked_penis_box, self.boxes['butt'], 'butt')
        self.distance = self.update_distance(distance)
        self.tracked_body_part = "butt"
        self.detect_sex_position_change(position, reason)

    def handle_Blowjob_Handjob_position(self, reason, current_frame_id):
        distance = None

        if self.tracking_states['face'] and self.tracked_objects['face']['touching']:
            if len(self.face_distances) > 3:
                self.face_distances.pop(0)
            distance = self.calculate_distance(self.locked_penis_box, self.boxes['face'], 'face')
            #print(f"Face distance: {distance}")
            self.face_distances.append(distance)
            self.face_hands_movements['face'] = abs(max(self.face_distances) - min(self.face_distances))
            # if no left hand or no right hand, return
            if not self.tracked_objects['left hand']['touching'] and not self.tracked_objects['right hand']['touching']:
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "face"
                self.detect_sex_position_change("Blowjob / Handjob", "presence of face only")
                return
        else:
            self.face_hands_movements['face'] = 0

        if self.tracking_states['left hand'] and self.tracked_objects['left hand']['touching']:
            if len(self.left_hand_distances) > 3:
                self.left_hand_distances.pop(0)
            distance = self.calculate_distance(self.locked_penis_box, self.boxes['left hand'], 'left hand')
            #print(f"Left hand distance: {distance}")
            self.left_hand_distances.append(distance)
            self.face_hands_movements['left hand'] = abs(max(self.left_hand_distances) - min(self.left_hand_distances))
            if not self.tracked_objects['right hand']['touching'] and not self.tracked_objects['face']['touching']:
                self.distance = self.update_distance(distance)
                self.tracked_body_part = "left hand"
                self.detect_sex_position_change("Blowjob / Handjob", "presence of left hand only")
                return
        else:
            self.face_hands_movements['left hand'] = 0

        if self.tracking_states['right hand'] and self.tracked_objects['right hand']['touching']:
            if len(self.right_hand_distances) > 3:
                self.right_hand_distances.pop(0)
            distance = self.calculate_distance(self.locked_penis_box, self.boxes['right hand'], 'right hand')
            #print(f"Right hand distance: {distance}")
            self.right_hand_distances.append(distance)
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
                self.calculate_distance(self.locked_penis_box, self.boxes['left foot'], 'left foot'))
            self.foot_movements['left foot'] = abs(max(self.left_foot_distances) - min(self.left_foot_distances))

        if self.tracking_states['right foot'] and self.tracked_objects['right foot']['touching']:
            if len(self.right_foot_distances) > 3:
                self.right_foot_distances.pop(0)
            self.right_foot_distances.append(
                self.calculate_distance(self.locked_penis_box, self.boxes['right foot'], 'right foot'))
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
        distance = self.calculate_distance(self.locked_penis_box, self.boxes['breast'], 'breast')
        self.distance = self.update_distance(distance)
        self.tracked_body_part = "breast"
        self.detect_sex_position_change("Boobjob", reason)

