import time
import os
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

from utils.config import class_names, class_priority_order, class_reverse_match, class_colors
from utils.lib_ObjectTracker_v2 import ObjectTracker
from utils.lib_FunscriptHandler import FunscriptGenerator
from utils.lib_Visualizer import Visualizer
from utils.lib_Debugger import Debugger
from utils.lib_SceneCutsDetect import detect_scene_changes
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg

# Define the BoxRecord class
class BoxRecord:
    def __init__(self, box, conf, cls, class_name, track_id):
        self.box = box
        self.conf = conf
        self.cls = cls
        self.class_name = class_name
        self.track_id = int(track_id)

    def __iter__(self):
        return iter((self.box, self.conf, self.cls, self.class_name, self.track_id))


# Define the Result class
class Result:
    def __init__(self, image_width):
        self.frame_data = {}
        self.image_width = image_width

    def add_record(self, frame_id, box_record):
        if frame_id in self.frame_data:
            self.frame_data[frame_id].append(box_record)
        else:
            self.frame_data[frame_id] = [box_record]

    def map_class_type_to_name(self, class_type, x1, x2, image_width):
        #if class_type in ['foot', 'hand']:
        #    # Call it left if it is mainly on the left of the frame, right otherwise
        #    if (x1 + x2) / 2 < image_width / 2:
        #        class_name = 'right ' + class_type
        #    else:
        #        class_name = 'left ' + class_type
        #else:
        class_name = class_type
        return class_name

    def get_boxes(self, frame_id):
        itemized_boxes = []
        if frame_id not in self.frame_data:
            return itemized_boxes
        boxes = self.frame_data[frame_id]
        for box, conf, cls, class_name, track_id in boxes:
            itemized_boxes.append((box, conf, cls, class_name, track_id))
        sorted_boxes = sorted(
            itemized_boxes,
            key=lambda x: class_priority_order.get(x[3], 7)  # Sort by class name priority
        )
        return sorted_boxes

    def get_all_frame_ids(self):
        return list(self.frame_data.keys())


def write_dataset(file_path, data):
    print(f"Exporting data...")
    export_start = time.time()
    # if file exists, rename it
    if os.path.exists(file_path):
        os.rename(file_path, file_path + ".bak")
    with open(file_path, 'w') as f:
        json.dump(data, f)
    export_end = time.time()
    print(f"Done in {export_end - export_start}.")

def extract_yolo_data(model_file, video_path, frame_start, frame_end=None, TestMode = False, isVR = False):
    # if file already exists, ask if overwrite
    if os.path.exists(video_path[:-4] + f"_rawyolo.json"):
        return
        #print(f"File {video_path[:-4] + f'_rawyolo.json'} already exists. Overwrite? (y/n)")
        #if input() != 'y':
        #    return
        #else:
        #    # make a backup before continuing
        #    os.rename(video_path[:-4] + f"_rawyolo.json", video_path[:-4] + f"_rawyolo.json.bak")

    records = []
    test_result = Result(320)

    #cap = cv2.VideoCapture(video_path)
    cap = VideoReaderFFmpeg(video_path, is_VR=isVR)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    if frame_end:
        last_frame = frame_end
    else:
        last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load the YOLO11 model
    model = YOLO(model_file, task="detect")

    # Loop through the video frames
    for frame_pos in tqdm(range(frame_start, last_frame), ncols=None, desc="Performing YOLO detection on frames"):
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            if isVR:
                frame = frame[:, frame.shape[1] // 3:2*frame.shape[1] // 3, :]

            # Run YOLO11 tracking on the frame, persisting tracks between frames
            yolo_results = model.track(frame, persist=True, conf=0.3, verbose=False)

            if yolo_results[0].boxes.id is None:  # in case of tracking
                continue

            if len(yolo_results[0].boxes) == 0 and not TestMode:
                continue

            # Get the boxes and track IDs
            track_ids = yolo_results[0].boxes.id.cpu().tolist()
            boxes = yolo_results[0].boxes.xywh.cpu()
            classes = yolo_results[0].boxes.cls.cpu().tolist()
            confs = yolo_results[0].boxes.conf.cpu().tolist()

            for track_id, cls, conf, box in zip(track_ids, classes, confs, boxes):
                track_id = int(track_id)
                x, y, w, h = box.int().tolist()
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                record = [frame_pos, int(cls), round(conf, 1), x1, y1, x2, y2, track_id]
                records.append(record)
                if TestMode:
                    print(f"Record : {record}")
                    print(f"For class id: {int(cls)}, getting: {class_reverse_match.get(int(cls), 'unknown')}")
                    test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls), class_reverse_match.get(int(cls), 'unknown'), track_id]
                    print(f"Test box: {test_box}")
                    test_result.add_record(frame_pos, test_box)

            if TestMode:
                # display the results from YOLO
                yolo_results[0].plot()
                cv2.imshow("YOLO11", yolo_results[0].plot())
                cv2.waitKey(1)
                # Making sure we did not mess the classes
                sorted_boxes = test_result.get_boxes(frame_pos)
                print(f"Sorted boxes : {sorted_boxes}")

                frame_display = frame.copy()

                for box in sorted_boxes:
                    color = class_colors.get(box[3])
                    cv2.rectangle(frame_display, (box[0][0], box[0][1]), (box[0][2], box[0][3]), color, 2)
                    cv2.putText(frame_display, f"{box[4]}: {box[3]}", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("YOLO11 test boxes Tracking", frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    write_dataset(video_path[:-4] + f"_rawyolo.json", records)
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def load_yolo_data_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"Loaded data from {file_path}, length: {len(data)}")
    return data

def make_data_boxes(records, image_x_size):
    # Create Result instance
    result = Result(image_x_size)

    # Populate the Result instance with BoxRecord instances
    for record in records:
        frame_idx, cls, conf, x1, y1, x2, y2, track_id = record
        box = [x1, y1, x2, y2]
        class_name = class_reverse_match.get(cls, 'unknown')
        box_record = BoxRecord(box, conf, cls, class_name, track_id)
        result.add_record(frame_idx, box_record)
    return result

def analyze_tracking_results(results, image_y_size, video_path, frame_start = None, frame_end = None, TestMode = False):
    list_of_frames = results.get_all_frame_ids()
    visualizer = Visualizer()

    #cap = cv2.VideoCapture(video_path)
    cap = VideoReaderFFmpeg(video_path, is_VR=isVR)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    if not frame_start:
        frame_start = 0

    if not frame_end:
        frame_end = nb_frames

    if TestMode:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    else:
        cap.release()

    # if a file already exists, directly load the cuts
    if os.path.exists(video_path[:-4] + f"_cuts.json"):
        print(f"Loading cuts from {video_path[:-4] + f'_cuts.json'}")
        with open(video_path[:-4] + f"_cuts.json", 'r') as f:
            cuts = json.load(f)
        print(f"Loaded {len(cuts)} cuts : {cuts}")

    else:
        scene_list = detect_scene_changes(video_path, "Left", 0.9, frame_start, frame_end)
        print(f"Analyzing frames {frame_start} to {frame_end}")
        cuts = [scene[1] for scene in scene_list]
        # remove last entry in cuts
        cuts = cuts[:-1]
        # save to a file
        with open(video_path[:-4] + f"_cuts.json", 'w') as f:
            json.dump(cuts, f)

    funscript_frames = []
    tracker = ObjectTracker(fps, frame_start)

    for frame_pos in tqdm(range(frame_start, frame_end), ncols=80, desc="Analyzing tracking results"):

        if frame_pos in cuts:
            print(f"Reaching cut at frame {frame_pos}")
            previous_distances = tracker.previous_distances
            # tqdm.write
            print(f"Reinitializing tracker with previous distances: {previous_distances}")
            tracker = ObjectTracker(fps, frame_pos)
            tracker.previous_distances = previous_distances

        if frame_pos in list_of_frames:
            sorted_boxes = results.get_boxes(frame_pos)

            tracker.tracking_logic(sorted_boxes, frame_pos, image_y_size)

            if tracker.distance:
                funscript_frames.append(frame_pos)
                funscript_distances.append(int(tracker.distance))

            if DebugMode:
                bounding_boxes = []  # Initialize an empty list for bounding boxes
                for box in sorted_boxes:
                    if box[4] in tracker.tracked_positions:
                        position = tracker.tracked_positions[box[4]][-1]
                    else:
                        position = None
                    bounding_boxes.append({
                        'box': box[0],
                        'conf': box[1],
                        'class_name': box[3],
                        'track_id': box[4],
                        'position': position,
                    })
                debugger.log_frame(frame_pos,
                                   bounding_boxes=bounding_boxes,  # Pass the list of bounding boxes
                                   variables={
                                       'distance': tracker.distance,
                                       'Penetration': tracker.penetration,
                                       #'Grinding': tracker.grinding,
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
                                   })

        if TestMode:
            ret, frame = cap.read()
            frame_display = frame.copy()

            for box in tracker.tracked_boxes:
                frame_display = visualizer.draw_bounding_box(frame_display,
                                                             box[0],
                                                             str(box[2]) + ": " + box[1],
                                                             class_colors[str(box[1])],
                                                             offset_x)
            if tracker.locked_penis_box is not None:
                frame_display = visualizer.draw_bounding_box(frame_display, tracker.locked_penis_box,
                                                              "Locked_Penis",
                                                              class_colors['penis'],
                                                              offset_x)
            if tracker.glans_detected:
                frame_display = visualizer.draw_bounding_box(frame_display, tracker.boxes['glans'],
                                                              "Glans",
                                                              class_colors['glans'],
                                                              offset_x)
            if funscript_distances:
                frame_display = visualizer.draw_gauge(frame_display, funscript_distances[-1])
                #frame_display = visualizer.draw_limited_graph(frame_display, funscript_distances,
                #                                               funscript_frames, 200)

            cv2.imshow("Combined Results", frame_display)
            cv2.waitKey(1)

    funscript_data = list(zip(funscript_frames, funscript_distances))

    points = "["
    for i in range(len(funscript_frames)):
        if i != 0:
            points += ","
        points += f"[{funscript_frames[i]}, {funscript_distances[i]}]"
    points += "]"
    # write the raw data to a json file
    with open(video_path[:-4] + f"_rawfunscript.json", 'w') as f:
        json.dump(funscript_data, f)
    return funscript_data


def parse_yolo_data_looking_for_penis(data, start_frame):
    consecutive_frames = 0
    frame_detected = 0
    penis_frame = 0
    for line in data:
        if line[0] >= start_frame and line[1] == 0 and line[2] >= 0.5:
            penis_frame = line[0]
        if line[0] == penis_frame and line[1] == 1 and line[2] >= 0.5:
            if frame_detected == 0:
                frame_detected = line[0]
                consecutive_frames += 1
            elif line[0] == frame_detected + 1:
                consecutive_frames += 1
                frame_detected = line[0]
            else:
                consecutive_frames = 0
                frame_detected = 0

            if consecutive_frames >= 2:
                print(f"First instance of Glans/Penis found in frame {line[0] - 4}")
                return line[0] - 4

# MAIN logic

if __name__ == '__main__':

    # YOLO model file
    yolo_model = "models/k00gar-11n-200ep-best.mlpackage"

    funscript_data = []

    video_list = []

    video_list.append("/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4")


    TestMode = False
    DebugMode = True
    isVR = True

    for video_path in video_list:
        print(f"Processing video: {video_path}")
        #funscript_data = []

        debugger = Debugger(video_path, output_dir=video_path[:-4])

        #cap = cv2.VideoCapture(video_path)
        cap = VideoReaderFFmpeg(video_path, is_VR=isVR)
        fps = cap.get(cv2.CAP_PROP_FPS)
        image_y_size = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image_x_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        if isVR:
            offset_x = image_x_size = image_x_size // 3

        print(f"Image size: {image_x_size}x{image_y_size}")
        cap.release()

        frame_start = 0   # 61000  # 0
        frame_end = None  #64000  # None

        position_changes = []
        sex_position = ""
        funscript_frames = []
        funscript_distances = []

        # Process the video

        # Run the YOLO detection and saves result to _rawyolo.json file
        #extract_yolo_data(yolo_model, video_path, frame_start, frame_end, TestMode, isVR)

        # Load YOLO detection results from file
        yolo_data = load_yolo_data_from_file(video_path[:-4] + f"_rawyolo.json")

        results = make_data_boxes(yolo_data, image_x_size)

        # Looking for the first instance of penis within the YOLO results
        first_penis_frame = parse_yolo_data_looking_for_penis(yolo_data, 0)

        if first_penis_frame is None:
            print(f"No penis found in video: {video_path}")
            first_penis_frame = 0

        # Deciding whether we start from there or from a user specified later frame
        frame_start = max(max(first_penis_frame - int(fps), frame_start - int(fps)), 0)

        # Performing the tracking part and generation of the raw funscript data
        funscript_data = analyze_tracking_results(results, image_y_size, video_path, frame_start, frame_end, TestMode)

        debugger.save_logs()

        funscript_handler = FunscriptGenerator()

        # Simplifying the funscript data and generating the file
        funscript_handler.generate(video_path[:-4] + f"_rawfunscript.json", funscript_data, fps, TestMode)

        print(f"Finished processing video: {video_path}")
