# Import necessary libraries
import time  # For measuring execution time
import os  # For file and directory operations
import cv2  # OpenCV for image and video processing
import json  # For handling JSON data
from tqdm import tqdm  # For progress bars
from ultralytics import YOLO  # YOLO model for object detection
import argparse  # For command-line argument parsing
import sys  # For accessing command-line arguments

# Import custom modules and configurations
from params.config import class_priority_order, class_reverse_match, class_colors  # Configuration for class priorities, reverse matching, and colors
from utils.lib_ObjectTracker import ObjectTracker  # Custom object tracking logic
from utils.lib_FunscriptHandler import FunscriptGenerator  # For generating Funscript files
from utils.lib_Visualizer import Visualizer  # For visualizing results
from utils.lib_Debugger import Debugger  # For debugging and logging
from utils.lib_SceneCutsDetect import detect_scene_changes  # For detecting scene changes in videos
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg  # Custom video reader using FFmpeg

# Define the BoxRecord class to store bounding box information
class BoxRecord:
    def __init__(self, box, conf, cls, class_name, track_id):
        """
        Initialize a BoxRecord object.
        :param box: Bounding box coordinates [x1, y1, x2, y2].
        :param conf: Confidence score of the detection.
        :param cls: Class ID of the detected object.
        :param class_name: Class name of the detected object.
        :param track_id: Track ID for object tracking.
        """
        self.box = box
        self.conf = conf
        self.cls = cls
        self.class_name = class_name
        self.track_id = int(track_id)

    def __iter__(self):
        """
        Make the BoxRecord object iterable.
        :return: An iterator over the box, confidence, class, class name, and track ID.
        """
        return iter((self.box, self.conf, self.cls, self.class_name, self.track_id))

# Define the Result class to store and manage detection results
class Result:
    def __init__(self, image_width):
        """
        Initialize a Result object.
        :param image_width: Width of the image/frame.
        """
        self.frame_data = {}  # Dictionary to store data for each frame
        self.image_width = image_width

    def add_record(self, frame_id, box_record):
        """
        Add a BoxRecord to the frame_data dictionary.
        :param frame_id: The frame ID to which the record belongs.
        :param box_record: The BoxRecord object to add.
        """
        if frame_id in self.frame_data:
            self.frame_data[frame_id].append(box_record)
        else:
            self.frame_data[frame_id] = [box_record]

    def map_class_type_to_name(self, class_type, x1, x2, image_width):
        """
        Map class type to a class name (currently unused).
        :param class_type: The class type to map.
        :param x1: The x-coordinate of the bounding box's top-left corner.
        :param x2: The x-coordinate of the bounding box's bottom-right corner.
        :param image_width: The width of the image/frame.
        :return: The mapped class name.
        """
        class_name = class_type
        return class_name

    def get_boxes(self, frame_id):
        """
        Retrieve and sort bounding boxes for a specific frame.
        :param frame_id: The frame ID to retrieve boxes for.
        :return: A list of sorted bounding boxes.
        """
        itemized_boxes = []
        if frame_id not in self.frame_data:
            return itemized_boxes
        boxes = self.frame_data[frame_id]
        for box, conf, cls, class_name, track_id in boxes:
            itemized_boxes.append((box, conf, cls, class_name, track_id))
        # Sort boxes based on class priority order
        sorted_boxes = sorted(
            itemized_boxes,
            key=lambda x: class_priority_order.get(x[3], 7)  # Default priority is 7 if class not found
        )
        return sorted_boxes

    def get_all_frame_ids(self):
        """
        Get a list of all frame IDs in the frame_data dictionary.
        :return: A list of frame IDs.
        """
        return list(self.frame_data.keys())

def write_dataset(file_path, data):
    """
    Write data to a JSON file.
    :param file_path: The path to the output file.
    :param data: The data to write.
    """
    print(f"Exporting data...")
    export_start = time.time()
    # If the file already exists, rename it as a backup
    if os.path.exists(file_path):
        os.rename(file_path, file_path + ".bak")
    # Write the data to the file
    with open(file_path, 'w') as f:
        json.dump(data, f)
    export_end = time.time()
    print(f"Done in {export_end - export_start}.")

def extract_yolo_data(model_file, video_path, frame_start, frame_end=None, TestMode=False, isVR=False):
    """
    Extract YOLO detection data from a video.
    :param model_file: Path to the YOLO model file.
    :param video_path: Path to the input video file.
    :param frame_start: The starting frame for processing.
    :param frame_end: The ending frame for processing (optional).
    :param TestMode: Enable/disable test mode for debugging.
    :param isVR: Enable/disable VR mode for video processing.
    """
    # Check if the output file already exists
    if os.path.exists(video_path[:-4] + f"_rawyolo.json"):
        return

    records = []  # List to store detection records
    test_result = Result(320)  # Test result object for debugging

    # Initialize the video reader
    cap = VideoReaderFFmpeg(video_path, is_VR=isVR)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    # Determine the last frame to process
    if frame_end:
        last_frame = frame_end
    else:
        last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load the YOLO model
    model = YOLO(model_file, task="detect")

    # Loop through the video frames
    for frame_pos in tqdm(range(frame_start, last_frame), ncols=None, desc="Performing YOLO detection on frames"):
        success, frame = cap.read()  # Read a frame from the video

        if success:
            if isVR:
                # For VR videos, crop the frame to the middle third
                frame = frame[:, frame.shape[1] // 3:2 * frame.shape[1] // 3, :]

            # Run YOLO tracking on the frame
            yolo_results = model.track(frame, persist=True, conf=0.3, verbose=False)

            if yolo_results[0].boxes.id is None:  # Skip if no tracks are found
                continue

            if len(yolo_results[0].boxes) == 0 and not TestMode:  # Skip if no boxes are detected
                continue

            # Extract track IDs, boxes, classes, and confidence scores
            track_ids = yolo_results[0].boxes.id.cpu().tolist()
            boxes = yolo_results[0].boxes.xywh.cpu()
            classes = yolo_results[0].boxes.cls.cpu().tolist()
            confs = yolo_results[0].boxes.conf.cpu().tolist()

            # Process each detection
            for track_id, cls, conf, box in zip(track_ids, classes, confs, boxes):
                track_id = int(track_id)
                x, y, w, h = box.int().tolist()
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                # Create a detection record
                record = [frame_pos, int(cls), round(conf, 1), x1, y1, x2, y2, track_id]
                records.append(record)
                if TestMode:
                    # Print and test the record
                    print(f"Record : {record}")
                    print(f"For class id: {int(cls)}, getting: {class_reverse_match.get(int(cls), 'unknown')}")
                    test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls), class_reverse_match.get(int(cls), 'unknown'), track_id]
                    print(f"Test box: {test_box}")
                    test_result.add_record(frame_pos, test_box)

            if TestMode:
                # Display the YOLO results for testing
                yolo_results[0].plot()
                cv2.imshow("YOLO11", yolo_results[0].plot())
                cv2.waitKey(1)
                # Verify the sorted boxes
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

    # Write the detection records to a JSON file
    write_dataset(video_path[:-4] + f"_rawyolo.json", records)
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def load_yolo_data_from_file(file_path):
    """
    Load YOLO data from a JSON file.
    :param file_path: Path to the JSON file.
    :return: The loaded data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"Loaded data from {file_path}, length: {len(data)}")
    return data

def make_data_boxes(records, image_x_size):
    """
    Convert YOLO records into BoxRecord objects.
    :param records: List of YOLO detection records.
    :param image_x_size: Width of the image/frame.
    :return: A Result object containing BoxRecord instances.
    """
    result = Result(image_x_size)  # Create a Result instance
    for record in records:
        frame_idx, cls, conf, x1, y1, x2, y2, track_id = record
        box = [x1, y1, x2, y2]
        class_name = class_reverse_match.get(cls, 'unknown')
        box_record = BoxRecord(box, conf, cls, class_name, track_id)
        result.add_record(frame_idx, box_record)
    return result

def analyze_tracking_results(results, image_y_size, video_path, frame_start=None, frame_end=None, TestMode=False):
    """
    Analyze tracking results and generate Funscript data.
    :param results: The Result object containing detection data.
    :param image_y_size: Height of the image/frame.
    :param video_path: Path to the input video file.
    :param frame_start: The starting frame for processing (optional).
    :param frame_end: The ending frame for processing (optional).
    :param TestMode: Enable/disable test mode for debugging.
    :return: A list of Funscript data.
    """
    list_of_frames = results.get_all_frame_ids()  # Get all frame IDs with detections
    visualizer = Visualizer()  # Initialize the visualizer

    cap = VideoReaderFFmpeg(video_path, is_VR=isVR)  # Initialize the video reader
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's FPS
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames

    if not frame_start:
        frame_start = 0

    if not frame_end:
        frame_end = nb_frames

    if TestMode:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    else:
        cap.release()

    # Load scene cuts if the file exists
    if os.path.exists(video_path[:-4] + f"_cuts.json"):
        print(f"Loading cuts from {video_path[:-4] + f'_cuts.json'}")
        with open(video_path[:-4] + f"_cuts.json", 'r') as f:
            cuts = json.load(f)
        print(f"Loaded {len(cuts)} cuts : {cuts}")
    else:
        # Detect scene changes if the cuts file does not exist
        scene_list = detect_scene_changes(video_path, "Left", 0.9, frame_start, frame_end)
        print(f"Analyzing frames {frame_start} to {frame_end}")
        cuts = [scene[1] for scene in scene_list]
        cuts = cuts[:-1]  # Remove the last entry
        # Save the cuts to a file
        with open(video_path[:-4] + f"_cuts.json", 'w') as f:
            json.dump(cuts, f)

    funscript_frames = []  # List to store Funscript frames
    tracker = ObjectTracker(fps, frame_start)  # Initialize the object tracker

    for frame_pos in tqdm(range(frame_start, frame_end), ncols=80, desc="Analyzing tracking results"):
        if frame_pos in cuts:
            # Reinitialize the tracker at scene cuts
            print(f"Reaching cut at frame {frame_pos}")
            previous_distances = tracker.previous_distances
            print(f"Reinitializing tracker with previous distances: {previous_distances}")
            tracker = ObjectTracker(fps, frame_pos)
            tracker.previous_distances = previous_distances

        if frame_pos in list_of_frames:
            # Get sorted boxes for the current frame
            sorted_boxes = results.get_boxes(frame_pos)
            tracker.tracking_logic(sorted_boxes, frame_pos, image_y_size)  # Apply tracking logic

            if tracker.distance:
                # Append Funscript data if distance is available
                funscript_frames.append(frame_pos)
                funscript_distances.append(int(tracker.distance))

            if DebugMode:
                # Log debugging information
                bounding_boxes = []
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
                                   bounding_boxes=bounding_boxes,
                                   variables={
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
                                   })

        if TestMode:
            # Display the tracking results for testing
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

            cv2.imshow("Combined Results", frame_display)
            cv2.waitKey(1)

    # Prepare Funscript data
    funscript_data = list(zip(funscript_frames, funscript_distances))

    points = "["
    for i in range(len(funscript_frames)):
        if i != 0:
            points += ","
        points += f"[{funscript_frames[i]}, {funscript_distances[i]}]"
    points += "]"
    # Write the raw Funscript data to a JSON file
    with open(video_path[:-4] + f"_rawfunscript.json", 'w') as f:
        json.dump(funscript_data, f)
    return funscript_data

def parse_yolo_data_looking_for_penis(data, start_frame):
    """
    Parse YOLO data to find the first instance of a penis.
    :param data: The YOLO detection data.
    :param start_frame: The starting frame for the search.
    :return: The frame ID where the penis is first detected.
    """
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
    video_list = []  # List of videos to process

    # Default values for IDE usage
    video_list.append("/Users/k00gar/Downloads/730-czechvr-3d-7680x3840-60fps-oculusrift_uhq_h265.mp4")
    yolo_model = "models/k00gar-11n-200ep-best.mlpackage"
    DebugMode = True
    TestMode = False
    isVR = True

    # Check if the script is run from the command line
    if len(sys.argv) > 1:  # Command-line arguments are present
        # Set up argument parsing for command-line usage
        parser = argparse.ArgumentParser(description="Generate Funscript data from a video using YOLO object detection.")
        parser.add_argument("video_path", type=str, help="Path to the input video file.")
        parser.add_argument("--yolo_model", type=str, default=yolo_model, help="Path to the YOLO model file.")
        parser.add_argument("--test_mode", action="store_true", help="Enable test mode for displaying.")
        parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode for logging.")
        parser.add_argument("--is_vr", action="store_true", help="Enable VR mode for video processing.")
        args = parser.parse_args()

        # Override default values with command-line arguments
        video_list.append(args.video_path)
        yolo_model = args.yolo_model
        TestMode = args.test_mode
        DebugMode = args.debug_mode
        isVR = args.is_vr

    funscript_data = []  # List to store Funscript data

    for video_path in video_list:
        print(f"Processing video: {video_path}")

        debugger = Debugger(video_path, output_dir=video_path[:-4])  # Initialize the debugger

        cap = VideoReaderFFmpeg(video_path, is_VR=isVR)  # Initialize the video reader
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's FPS
        image_y_size = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Get the video's height
        image_x_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Get the video's width

        if isVR:
            offset_x = image_x_size = image_x_size // 3  # Adjust for VR videos

        print(f"Image size: {image_x_size}x{image_y_size}")
        cap.release()

        frame_start = 0   # 0 for analysis from the beginning
        frame_end = None  # None for analysis until the end

        funscript_frames = []
        funscript_distances = []

        # Process the video

        # Run the YOLO detection and saves result to _rawyolo.json file
        extract_yolo_data(yolo_model, video_path, frame_start, frame_end, TestMode, isVR)

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
