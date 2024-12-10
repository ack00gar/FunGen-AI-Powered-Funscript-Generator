import time
import os
import cv2
import json
from utils.config import class_names, class_priority_order, class_reverse_match, class_colors
from ultralytics import YOLO
from utils import ObjectTracker as OT
from utils import FunscriptHandler as FH
from tqdm import tqdm
from utils import Visualizer as VS
from utils.SceneCutsDetect import detect_scene_changes


# Define the BoxRecord class
class BoxRecord:
    def __init__(self, box, conf, cls, class_name):
        self.box = box
        self.conf = conf
        self.cls = cls
        self.class_name = class_name

    def __iter__(self):
        return iter((self.box, self.conf, self.cls, self.class_name))


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
        if class_type in ['foot', 'hand']:
            # Call it left if it is mainly on the left of the frame, right otherwise
            if (x1 + x2) / 2 < image_width / 2:
                class_name = 'right ' + class_type
            else:
                class_name = 'left ' + class_type
        else:
            class_name = class_type
        return class_name

    def get_boxes(self, frame_id):
        itemized_boxes = []
        if frame_id not in self.frame_data:
            raise KeyError(f"No records found for frame ID {frame_id}")
        boxes = self.frame_data[frame_id]
        for box, conf, cls, class_name in boxes:
            target_class_name = self.map_class_type_to_name(class_name, box[0], box[2], self.image_width)
            itemized_boxes.append((box, conf, cls, target_class_name))
        sorted_boxes = sorted(
            itemized_boxes,
            key=lambda x: class_priority_order.get(x[3], 7)  # Sort by class name priority
        )
        return sorted_boxes  # sorted_boxes

    def get_all_frame_ids(self):
        return list(self.frame_data.keys())

def update_sex_position(position, new_position, frame_id, reason=None):
    if position is not None and new_position is not None and position != new_position:
        due_to = " given " + reason if reason else ""
        print(f"Sex position changed from {position} to {new_position}{due_to}")
        position_changes.append((frame_id, position, new_position, reason))
        return new_position
    return position

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
    records = []
    test_result = Result(320)

    cap = cv2.VideoCapture(video_path)
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
                # only keep the center third of the left half of the frame
                frame = frame[:, frame.shape[1] // 6:frame.shape[1] // 3, :]

            # Run YOLO11 tracking on the frame, persisting tracks between frames
            # yolo_results = model.predict(frame, verbose=False, conf=0.5)  #track(frame, persist=True, verbose=False)
            yolo_results = model.track(frame, persist=True, conf=0.5, verbose=False)

            #if results[0].boxes.id is None:
            if yolo_results[0].boxes.cls is None:
                continue

            # Get the boxes and track IDs
            #track_ids = results[0].boxes.id.cpu().tolist()
            boxes = yolo_results[0].boxes.xywh.cpu()
            classes = yolo_results[0].boxes.cls.cpu().tolist()
            confs = yolo_results[0].boxes.conf.cpu().tolist()

            # for track_id, cls, conf, box in zip(track_ids, classes, confs, boxes):
            for cls, conf, box in zip(classes, confs, boxes):
                x, y, w, h = box.int().tolist()
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                #record = [frame_pos, int(track_id), int(cls), round(conf, 1), x1, y1, x2, y2]
                record = [frame_pos, int(cls), round(conf, 1), x1, y1, x2, y2]
                records.append(record)
                if TestMode:
                    print(f"Record : {record}")
                    print(f"For class id: {int(cls)}, getting: {class_reverse_match.get(int(cls), 'unknown')}")
                    test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls), class_reverse_match.get(int(cls), 'unknown')]
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

                for box in sorted_boxes:
                    cv2.rectangle(frame, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{box[3]}", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv2.putText(frame, f"{int(cls)}: {int(conf*100)}%", (record[4], record[5] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("YOLO11 test boxes Tracking", frame)
                cv2.waitKey(1)
            if frame_pos == last_frame-1:
                print("Triggered here")
                write_dataset(video_path[:-4] + f"_rawyolo.json", records)
                break
        else:
            # Break the loop if the end of the video is reached
            print("Triggered there")
            write_dataset(video_path[:-4] + f"_rawyolo.json", records)
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def load_yolo_data_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def make_data_boxes(records, image_x_size):
    # Create Result instance
    result = Result(image_x_size)

    # Populate the Result instance with BoxRecord instances
    for record in records:
        frame_idx, cls, conf, x1, y1, x2, y2 = record
        box = [x1, y1, x2, y2]
        class_name = class_reverse_match.get(cls, 'unknown')
        box_record = BoxRecord(box, conf, cls, class_name)
        result.add_record(frame_idx, box_record)
    return result

def analyze_tracking_results(results, image_y_size, video_path, frame_start = None, frame_end = None, TestMode = False):
    list_of_frames = results.get_all_frame_ids()
    visualizer = VS.Visualizer()

    cap = cv2.VideoCapture(video_path)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not frame_start:
        frame_start = 0

    if not frame_end:
        frame_end = nb_frames

    if TestMode:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    # if a file already exists, directly load the cuts
    if os.path.exists(video_path[:-4] + f"_cuts.json"):
        print(f"Loading cuts from {video_path[:-4] + f'_cuts.json'}")
        with open(video_path[:-4] + f"_cuts.json", 'r') as f:
            cuts = json.load(f)
    else:
        scene_list = detect_scene_changes(video_path, "Left", 0.9, frame_start, frame_end)
        print(f"Analyzing frames {frame_start} to {frame_end}")
        cuts = [scene[1] for scene in scene_list]
        # remove last entry in cuts
        cuts = cuts[:-1]
        # save to a file
        with open(video_path[:-4] + f"_cuts.json", 'w') as f:
            json.dump(cuts, f)

    tracker = OT.ObjectTracker()

    for frame_pos in tqdm(range(frame_start, frame_end), ncols=None, desc="Analyzing tracking results"):

        if frame_pos in cuts:
            print(f"Reaching cut at frame {frame_pos}")
            previous_distances = tracker.previous_distances
            tqdm.write(f"Reinitializing tracker with previous distances: {previous_distances}")
            tracker = OT.ObjectTracker()
            tracker.previous_distances = previous_distances

        if frame_pos in list_of_frames:
            sorted_boxes = results.get_boxes(frame_pos)

            tracker.tracking_logic(sorted_boxes, frame_pos, image_y_size)

            if tracker.distance:
                funscript_frames.append(frame_pos)
                funscript_distances.append(tracker.distance)

        if TestMode:
            ret, frame = cap.read()
            frame = frame[:, :frame.shape[1] // 2, :]
            if tracker.tracked_body_part in class_names and tracker.boxes[tracker.tracked_body_part] is not None:
                frame = visualizer.draw_bounding_box(frame,
                                                          tracker.boxes[tracker.tracked_body_part],
                                                          tracker.tracked_body_part,
                                                          class_colors[tracker.tracked_body_part],
                                                          int(image_x_size / 2))
            if tracker.locked_penis_box is not None:
                frame = visualizer.draw_bounding_box(frame, tracker.locked_penis_box,
                                                              "Locked_Penis",
                                                              class_colors['penis'],
                                                              int(image_x_size / 2))
            if funscript_distances:
                frame = visualizer.draw_gauge(frame, funscript_distances[-1])
                #frame = visualizer.draw_limited_graph(frame, funscript_distances,
                #                                               funscript_frames, 200)

            cv2.imshow("Combined Results", frame)
            cv2.waitKey(1)
    return funscript_frames, funscript_distances


def parse_yolo_data_looking_for_penis(data, start_frame):
    for line in data:
        if line[0] >= start_frame and line[1] == 0:  # class_types.get("penis"):
            print(f"First instance of Penis found in frame {line[0]}")
            return line[0]




# MAIN logic

# YOLO model file
yolo_model = "models/k00gar-11n-200ep-best.mlpackage"

video_list = []
# video_list.append("/Users/k00gar/Downloads/wankzvr-shocum-180_180x180_3dh_LR.mp4")
# video_list.append("/Users/k00gar/Downloads/ARPorn_Angel Youngs_Angel Of Nurse_4000p_2K_original_FISHEYE190_alpha.mp4")
# video_list.append("/Users/k00gar/Downloads/Milfvr - Honeymoon In Vega - Vanessa Vega_1080p.mp4")
# video_list.append("/Users/k00gar/Downloads/SLR_SLR Originals_Sugar-baby_ Drooling Kitty_1920p_48559_FISHEYE190.mp4")
# video_list.append("/Volumes/Crucial/pn/EBVR-018-B_AI_upscaled.mp4")
# video_list.append("/Volumes/Crucial/pn/NaugthyAmericaVR.My.Wifes.Hot.Friend.23.09.15.Blake.Blossom.Oculus.8k.4096.mp4")
# video_list.append("/Volumes/WD Elements/VRPn/CzechVR.723.I.m.Here.to.Serve.Agatha.Vega.Oculus.8K..mp4")
# video_list.append("/Volumes/WD Elements/VRPn/721.czechvr.3d.7680x3840.60fps.oculusrift_uhq_h265.mp4")
# video_list.append("/Users/k00gar/Downloads/JVR_100199.mp4")
# video_list.append("/Users/k00gar/Downloads/JVR_100200.mp4")
# video_list.append("/Users/k00gar/Downloads/JVR_100201.mp4")
video_list.append("/Users/k00gar/Downloads/SLR_SLR Originals_Poolside Romance_1920p_39107_FISHEYE190.mp4")

TestMode = False

for video_path in video_list:
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    image_y_size = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image_x_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 3
    print(f"Image size: {image_x_size}x{image_y_size}")
    cap.release()

    frame_start = 0   # 61000  # 0
    frame_end = None  #64000  # None

    position_changes = []
    sex_position = ""
    funscript_frames = []
    funscript_distances = []

    # Process the video
    extract_yolo_data(yolo_model, video_path, frame_start, frame_end, TestMode)

    yolo_data = load_yolo_data_from_file(video_path[:-4] + f"_rawyolo.json")

    results = make_data_boxes(yolo_data, image_x_size)

    first_penis_frame = parse_yolo_data_looking_for_penis(yolo_data, 0)

    frame_start = max(max(first_penis_frame - int(fps), frame_start - int(fps)), 0)

    analyze_tracking_results(results, image_y_size, video_path, frame_start, frame_end, TestMode)

    funscript_handler = FH.FunscriptGenerator()
    funscript_handler.generate(funscript_frames, funscript_distances, video_path[:-4] + "_kAI.funscript", fps, TestMode)

    print(f"Finished processing video: {video_path}")

