# Import necessary libraries
import time  # For measuring execution time
import os  # For file and directory operations
import cv2  # OpenCV for image and video processing
import json  # For handling JSON data
from tqdm import tqdm  # For progress bars
from ultralytics import YOLO  # YOLO model for object detection
import platform  # For identifying the operating system
import torch  # PyTorch for use of .pt model if not on Apple device
import tkinter as tk  # GUI library for macOS for basic use in our case
from tkinter import filedialog, messagebox  # here, what was I saying...
import subprocess  # For running shell commands

# Import custom modules and configurations
from params.config import class_priority_order, class_reverse_match, class_colors, yolo_models  # Configuration for class priorities, reverse matching, and colors
from utils.lib_ObjectTracker import ObjectTracker  # Custom object tracking logic
from utils.lib_FunscriptHandler import FunscriptGenerator  # For generating Funscript files
from utils.lib_Visualizer import Visualizer  # For visualizing results
from utils.lib_Debugger import Debugger  # For debugging and logging
from utils.lib_SceneCutsDetect import detect_scene_changes  # For detecting scene changes in videos
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg  # Custom video reader using FFmpeg

# Define a GlobalState class to manage global variables
class GlobalState:
    def __init__(self):
        self.yolo_det_model = ""
        self.yolo_pose_model = ""
        self.video_file = ""
        self.DebugMode = False
        self.LiveDisplayMode = False
        self.isVR = True
        self.frame_start = 0
        self.frame_end = None
        self.reference_script = ""
        self.offset_x = 0
        self.funscript_data = []  # List to store Funscript data
        self.funscript_frames = []
        self.funscript_distances = []
        self.debugger = None

# Initialize global state
global_state = GlobalState()

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

def get_yolo_model_path():
    # Check if the device is an Apple device
    if platform.system() == 'Darwin':
        print(f"Apple device detected, loading {yolo_models[0]} for MPS inference.")
        return yolo_models[0]

    # Check if CUDA is available (for GPU support)
    elif torch.cuda.is_available():
        print(f"CUDA is available, loading {yolo_models[1]} for GPU inference.")
        return yolo_models[1]

    # Fallback to ONNX model for other platforms without CUDA
    else:
        print(f"Falling back to CPU inference, loading {yolo_models[2]}.")
        return yolo_models[2]

def extract_yolo_data():
    """
    Extract YOLO detection data from a video.
    """
    # Check if the output file already exists
    if os.path.exists(global_state.video_file[:-4] + f"_rawyolo.json"):
        # messagebox to ask if user wants to overwrite or reuse
        # file name without path
        file_name = os.path.basename(global_state.video_file[:-4] + f"_rawyolo.json")
        skip_detection = messagebox.askyesno("Detection file already exists",
                f"File {file_name} already exists.\n\nClick Yes to reuse the existing detections file.\nClick No to perform detections again.")
        if skip_detection:
            print(
                f"File {global_state.video_file[:-4] + f'_rawyolo.json'} already exists. Skipping detections and loading file content...")
            return
        else:
            os.remove(global_state.video_file[:-4] + f"_rawyolo.json")

    records = []  # List to store detection records
    test_result = Result(320)  # Test result object for debugging

    # Initialize the video reader
    cap = VideoReaderFFmpeg(global_state.video_file, is_VR=global_state.isVR)
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_state.frame_start)

    # Determine the last frame to process
    if global_state.frame_end:
        last_frame = global_state.frame_end
    else:
        last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load the YOLO model
    det_model = YOLO(global_state.yolo_det_model, task="detect")

    # make the pose model optional
    if len(global_state.yolo_pose_model) > 0:
        run_pose_model = True
        print("Activating pose model")
    else:
        run_pose_model = False
        print("Discarding pose model part of the code")
    if run_pose_model:
        pose_model = YOLO(global_state.yolo_pose_model, task="pose")

    # Loop through the video frames
    for frame_pos in tqdm(range(global_state.frame_start, last_frame), ncols=None, desc="Performing YOLO detection on frames"):
        success, frame = cap.read()  # Read a frame from the video

        if success:

            # Run YOLO tracking on the frame
            yolo_det_results = det_model.track(frame, persist=True, conf=0.3, verbose=False)
            if run_pose_model:
                yolo_pose_results = pose_model.track(frame, persist=True, conf=0.3, verbose=False)

            if yolo_det_results[0].boxes.id is None:  # Skip if no tracks are found
                continue

            if len(yolo_det_results[0].boxes) == 0 and not global_state.LiveDisplayMode:  # Skip if no boxes are detected
                continue

            ### DETECTION of BODY PARTS
            # Extract track IDs, boxes, classes, and confidence scores
            track_ids = yolo_det_results[0].boxes.id.cpu().tolist()
            boxes = yolo_det_results[0].boxes.xywh.cpu()
            classes = yolo_det_results[0].boxes.cls.cpu().tolist()
            confs = yolo_det_results[0].boxes.conf.cpu().tolist()

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
                if global_state.LiveDisplayMode:
                    # Print and test the record
                    print(f"Record : {record}")
                    print(f"For class id: {int(cls)}, getting: {class_reverse_match.get(int(cls), 'unknown')}")
                    test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls), class_reverse_match.get(int(cls), 'unknown'), track_id]
                    print(f"Test box: {test_box}")
                    test_result.add_record(frame_pos, test_box)

            if run_pose_model:
                ### POSE DETECTION - Hips and wrists
                # Extract track IDs, boxes, classes, and confidence scores
                if len(yolo_pose_results[0].boxes) > 0 and yolo_pose_results[0].boxes.id is not None:
                    pose_track_ids = yolo_pose_results[0].boxes.id.cpu().tolist()

                    # Check if keypoints are detected
                    if yolo_pose_results[0].keypoints is not None:
                        # print("We have keypoints")
                        # pose_keypoints = yolo_pose_results[0].keypoints.cpu()
                        # pose_track_ids = yolo_pose_results[0].boxes.id.cpu().tolist()
                        # pose_boxes = yolo_pose_results[0].boxes.xywh.cpu()
                        # pose_classes = yolo_pose_results[0].boxes.cls.cpu().tolist()
                        pose_confs = yolo_pose_results[0].boxes.conf.cpu().tolist()

                        pose_keypoints = yolo_pose_results[0].keypoints.cpu()
                        pose_keypoints_list = pose_keypoints.xy.cpu().tolist()
                        left_hip = pose_keypoints_list[0][11]
                        right_hip = pose_keypoints_list[0][12]

                        middle_x_frame = frame.shape[1] // 2
                        mid_hips = [middle_x_frame, (int(left_hip[1])+ int(right_hip[1]))//2]
                        x1 = mid_hips[0]-5
                        y1 = mid_hips[1]-5
                        x2 = mid_hips[0]+5
                        y2 = mid_hips[1]+5
                        cls = 10  # hips center
                        # print(f"pose_confs: {pose_confs}")
                        conf = pose_confs[0]

                        record = [frame_pos, 10, round(conf, 1), x1, y1, x2, y2, 0]
                        records.append(record)
                        if global_state.LiveDisplayMode:
                            # Print and test the record
                            print(f"Record : {record}")
                            print(f"For class id: {int(cls)}, getting: {class_reverse_match.get(int(cls), 'unknown')}")
                            test_box = [[x1, y1, x2, y2], round(conf, 1), int(cls),
                                        class_reverse_match.get(int(cls), 'unknown'), 0]
                            print(f"Test box: {test_box}")
                            test_result.add_record(frame_pos, test_box)

            if global_state.LiveDisplayMode:
                # Display the YOLO results for testing
                yolo_det_results[0].plot()
                cv2.imshow("YOLO11", yolo_det_results[0].plot())
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
    write_dataset(global_state.video_file[:-4] + f"_rawyolo.json", records)
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

def analyze_tracking_results(results, image_y_size):
    """
    Analyze tracking results and generate Funscript data.
    :param results: The Result object containing detection data.
    :param image_y_size: Height of the image/frame.
    :return: A list of Funscript data.
    """
    list_of_frames = results.get_all_frame_ids()  # Get all frame IDs with detections
    visualizer = Visualizer()  # Initialize the visualizer

    cap = VideoReaderFFmpeg(global_state.video_file, is_VR=global_state.isVR)  # Initialize the video reader
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's FPS
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames

    cuts = []

    image_area = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not global_state.frame_start:
        global_state.frame_start = 0

    if not global_state.frame_end:
        global_state.frame_end = nb_frames

    if global_state.LiveDisplayMode:
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_state.frame_start)
    else:
        cap.release()

    # Load scene cuts if the file exists
    if os.path.exists(global_state.video_file[:-4] + f"_cuts.json"):
        print(f"Loading cuts from {global_state.video_file[:-4] + f'_cuts.json'}")
        with open(global_state.video_file[:-4] + f"_cuts.json", 'r') as f:
            cuts = json.load(f)
        print(f"Loaded {len(cuts)} cuts : {cuts}")
    else:
        # Detect scene changes if the cuts file does not exist
        scene_list = detect_scene_changes(global_state.video_file, global_state.isVR, 0.9, global_state.frame_start, global_state.frame_end)
        print(f"Analyzing frames {global_state.frame_start} to {global_state.frame_end}")
        cuts = [scene[1] for scene in scene_list]
        cuts = cuts[:-1]  # Remove the last entry
        # Save the cuts to a file
        with open(global_state.video_file[:-4] + f"_cuts.json", 'w') as f:
            json.dump(cuts, f)

    global_state.funscript_frames = []  # List to store Funscript frames
    tracker = ObjectTracker(fps, global_state.frame_start, image_area)  # Initialize the object tracker

    for frame_pos in tqdm(range(global_state.frame_start, global_state.frame_end), unit="f"):
        if frame_pos in cuts:
            # Reinitialize the tracker at scene cuts
            print(f"Reaching cut at frame {frame_pos}")
            previous_distances = tracker.previous_distances
            print(f"Reinitializing tracker with previous distances: {previous_distances}")
            tracker = ObjectTracker(fps, frame_pos, image_area)
            tracker.previous_distances = previous_distances

        if frame_pos in list_of_frames:
            # Get sorted boxes for the current frame
            sorted_boxes = results.get_boxes(frame_pos)
            tracker.tracking_logic(sorted_boxes, frame_pos, image_y_size)  # Apply tracking logic

            if tracker.distance:
                # Append Funscript data if distance is available
                global_state.funscript_frames.append(frame_pos)
                global_state.funscript_distances.append(int(tracker.distance))

            if global_state.DebugMode:
                # Log debugging information
                bounding_boxes = []
                for box in sorted_boxes:
                    #if box[4] in tracker.tracked_positions:
                    if box[4] in tracker.normalized_absolute_tracked_positions:
                        #position = tracker.tracked_positions[box[4]][-1]
                        if box[4] == 0:  # generic track_id for 'hips center'
                            str_dist_penis = 'None'
                        else:
                            if box[4] in tracker.normalized_distance_to_penis:
                                # print(f"box[4]: {box[4]}")
                                str_dist_penis = str(int(tracker.normalized_distance_to_penis[box[4]][-1]))
                            else:
                                str_dist_penis = 'None'
                        str_abs_pos = str(int(tracker.normalized_absolute_tracked_positions[box[4]][-1]))
                        position = 'p: ' + str_dist_penis + ' | ' + 'a: ' + str_abs_pos
                    else:
                        position = None
                    bounding_boxes.append({
                        'box': box[0],
                        'conf': box[1],
                        'class_name': box[3],
                        'track_id': box[4],
                        'position': position,
                    })
                global_state.debugger.log_frame(frame_pos,
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

        if global_state.LiveDisplayMode:
            # Display the tracking results for testing
            ret, frame = cap.read()
            frame_display = frame.copy()

            for box in tracker.tracked_boxes:
                frame_display = visualizer.draw_bounding_box(frame_display,
                                                             box[0],
                                                             str(box[2]) + ": " + box[1],
                                                             class_colors[str(box[1])],
                                                             global_state.offset_x)
            if tracker.locked_penis_box is not None and tracker.locked_penis_box.is_active():
                frame_display = visualizer.draw_bounding_box(frame_display, tracker.locked_penis_box.box,
                                                             "Locked_Penis",
                                                             class_colors['penis'],
                                                             global_state.offset_x)
            else:
                print("No active locked penis box to draw.")

            if tracker.glans_detected:
                frame_display = visualizer.draw_bounding_box(frame_display, tracker.boxes['glans'],
                                                              "Glans",
                                                              class_colors['glans'],
                                                              global_state.offset_x)
            if global_state.funscript_distances:
                frame_display = visualizer.draw_gauge(frame_display, global_state.funscript_distances[-1])

            cv2.imshow("Combined Results", frame_display)
            cv2.waitKey(1)

    # Prepare Funscript data
    global_state.funscript_data = list(zip(global_state.funscript_frames, global_state.funscript_distances))

    points = "["
    for i in range(len(global_state.funscript_frames)):
        if i != 0:
            points += ","
        points += f"[{global_state.funscript_frames[i]}, {global_state.funscript_distances[i]}]"
    points += "]"
    # Write the raw Funscript data to a JSON file
    with open(global_state.video_file[:-4] + f"_rawfunscript.json", 'w') as f:
        json.dump(global_state.funscript_data, f)
    return global_state.funscript_data

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

def select_video_file():
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if file_path:
        video_path.set(file_path)
        check_video_resolution(file_path)

def select_reference_script():
    file_path = filedialog.askopenfilename(
        title="Select a reference funscript file",
        filetypes=[("Funscript Files", "*.funscript")]
    )
    if file_path:
        reference_script_path.set(file_path)

def check_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the video file.")
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if height > 1920:
        resize_options = ["1920", "1440", "1080"]
        choice = messagebox.askquestion(
            "Resize Video",
            f"The video height is {height}p. Do you want to resize it to a lower resolution?",
            detail="Options: " + ", ".join(resize_options)
        )
        if choice == "yes":
            resize_video(video_path, resize_options)

def resize_video(video_path, options):

    def on_resize_select(option):
        output_path = video_path.replace(".mp4", f"_{option}p.mp4")
        command = [
            "ffmpeg", "-i", video_path,
            "-vf", f"scale=-1:{option}",
            # "-c:a", "copy",  # if you need the audio, uncomment this line and comment the next one
            "-an",  # skipping the audio
            output_path
        ]
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", f"Video resized to {option}p and saved as {output_path}")
        resize_window.destroy()

    resize_window = tk.Toplevel()
    resize_window.title("Select Resize Option")
    for option in options:
        btn = tk.Button(resize_window, text=f"{option}p", command=lambda o=option: on_resize_select(o))
        btn.pack(pady=5)

def start_processing():
    global_state.video_file = video_path.get()
    if not global_state.video_file:
        messagebox.showerror("Error", "Please select a video file.")
        return

    global_state.yolo_det_model = get_yolo_model_path()
    global_state.yolo_pose_model = ""  # "models/yolo11n-pose.mlpackage"
    global_state.DebugMode = debug_mode_var.get()
    global_state.LiveDisplayMode = live_display_mode_var.get()
    global_state.isVR = is_vr_var.get()
    global_state.frame_start = 0 if frame_start_entry.get() == "" else int(frame_start_entry.get())
    global_state.frame_end = None if frame_end_entry.get() == "" else int(frame_end_entry.get())
    global_state.reference_script = reference_script_path.get()

    print(f"Processing video: {global_state.video_file}")
    print(f"YOLO Detection Model: {global_state.yolo_det_model}")
    print(f"YOLO Pose Model: {global_state.yolo_pose_model}")
    print(f"Debug Mode: {global_state.DebugMode}")
    print(f"Live Display Mode: {global_state.LiveDisplayMode}")
    print(f"VR Mode: {global_state.isVR}")
    print(f"Frame Start: {global_state.frame_start}")
    print(f"Frame End: {global_state.frame_end}")
    print(f"Reference Script: {global_state.reference_script}")

    # Processing logic

    global_state.debugger = Debugger(global_state.video_file, output_dir=global_state.video_file[:-4])  # Initialize the debugger

    cap = VideoReaderFFmpeg(global_state.video_file, is_VR=global_state.isVR)  # Initialize the video reader
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's FPS
    image_y_size = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Get the video's height
    image_x_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Get the video's width

    print(f"Image size: {image_x_size}x{image_y_size}")
    cap.release()

    # Process the video

    # Run the YOLO detection and saves result to _rawyolo.json file
    extract_yolo_data()
    #
    # Load YOLO detection results from file
    yolo_data = load_yolo_data_from_file(global_state.video_file[:-4] + f"_rawyolo.json")

    results = make_data_boxes(yolo_data, image_x_size)

    # Looking for the first instance of penis within the YOLO results
    first_penis_frame = parse_yolo_data_looking_for_penis(yolo_data, 0)

    if first_penis_frame is None:
        print(f"No penis found in video: {global_state.video_file}")
        first_penis_frame = 0

    # Deciding whether we start from there or from a user specified later frame
    global_state.frame_start = max(max(first_penis_frame - int(fps), global_state.frame_start - int(fps)), 0)

    print(f"Frame Start adjusted to: {global_state.frame_start}")

    # Performing the tracking part and generation of the raw funscript data
    global_state.funscript_data = analyze_tracking_results(results, image_y_size)

    global_state.debugger.save_logs()

    funscript_handler = FunscriptGenerator()

    # Simplifying the funscript data and generating the file
    funscript_handler.generate(global_state.video_file[:-4] + f"_rawfunscript.json", global_state.funscript_data, fps, global_state.LiveDisplayMode)

    # Optional, compare generated funscript with reference funscript if specified, or a simple generic report
    funscript_handler.compare_funscripts(global_state.reference_script, global_state.video_file[:-3] + "funscript", global_state.video_file, global_state.isVR,
                                             global_state.video_file[:-4] + "_comparefunscripts.png",
                                             global_state.video_file[:-4] + "_adjusted.funscript", )

    print(f"Finished processing video: {global_state.video_file}")

def debug_function():
    """
    Debugging function to perform specific debugging tasks.
    """
    global_state.video_file = video_path.get()
    if not global_state.video_file:
        messagebox.showerror("Error", "Please select a video file.")
        return

    global_state.yolo_det_model = get_yolo_model_path()
    global_state.yolo_pose_model = ""  # "models/yolo11n-pose.mlpackage"
    global_state.DebugMode = debug_mode_var.get()
    global_state.LiveDisplayMode = live_display_mode_var.get()
    global_state.isVR = is_vr_var.get()
    global_state.frame_start = 0 if frame_start_entry.get() == "" else int(frame_start_entry.get())
    global_state.frame_end = None if frame_end_entry.get() == "" else int(frame_end_entry.get())
    global_state.reference_script = reference_script_path.get()

    print(f"Processing video: {global_state.video_file}")
    print(f"YOLO Detection Model: {global_state.yolo_det_model}")
    print(f"YOLO Pose Model: {global_state.yolo_pose_model}")
    print(f"Debug Mode: {global_state.DebugMode}")
    print(f"Live Display Mode: {global_state.LiveDisplayMode}")
    print(f"VR Mode: {global_state.isVR}")
    print(f"Frame Start: {global_state.frame_start}")
    print(f"Frame End: {global_state.frame_end}")
    print(f"Reference Script: {global_state.reference_script}")

    # Processing logic

    global_state.debugger = Debugger(global_state.video_file, output_dir=global_state.video_file[:-4])  # Initialize the debugger

    # if the debug_logs.json file exists, load it
    if os.path.exists(global_state.video_file[:-4] + f"_debug_logs.json"):
        global_state.debugger.load_logs()
        global_state.debugger.play_video(global_state.frame_start)
    else:
        messagebox.showinfo("Info", f"Debug logs file not found: {global_state.video_file[:-4] + f'_debug_logs.json'}")

def quit_application():
    """
    Quit the application.
    """
    print("Quitting the application...")
    root.quit()  # Close the Tkinter main loop
    root.destroy()  # Destroy the root window

# GUI Setup
root = tk.Tk()
root.title("VR funscript generation helper")

# Variables
video_path = tk.StringVar()
reference_script_path = tk.StringVar()
debug_mode_var = tk.BooleanVar()
live_display_mode_var = tk.BooleanVar()
is_vr_var = tk.BooleanVar()

# Video File Selection
tk.Label(root, text="Video File:").grid(row=0, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=video_path, width=50).grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=select_video_file).grid(row=0, column=2, padx=5, pady=5)

# Reference Script Selection
tk.Label(root, text="Reference Script:").grid(row=1, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=reference_script_path, width=50).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=select_reference_script).grid(row=1, column=2, padx=5, pady=5)

# Debug Mode
tk.Checkbutton(root, text="Debug Mode (save logs)", variable=debug_mode_var).grid(row=2, column=0, padx=5, pady=5)

# Live Display Mode
tk.Checkbutton(root, text="Live Display Mode (plays video during detection, costs time)", variable=live_display_mode_var).grid(row=2, column=1, padx=5, pady=5)

# VR Mode, activated by default
is_vr_var.set(True)
tk.Checkbutton(root, text="VR Mode", variable=is_vr_var).grid(row=2, column=2, padx=5, pady=5)

# Frame Start
tk.Label(root, text="Frame Start (or blank):").grid(row=3, column=0, padx=5, pady=5)
frame_start_entry = tk.Entry(root, width=10)
frame_start_entry.grid(row=3, column=1, padx=5, pady=5)

# Frame End
tk.Label(root, text="Frame End (or blank):").grid(row=4, column=0, padx=5, pady=5)
frame_end_entry = tk.Entry(root, width=10)
frame_end_entry.grid(row=4, column=1, padx=5, pady=5)

# Start Button
tk.Button(root, text="Start Processing", command=start_processing).grid(row=5, column=1, padx=5, pady=20)

# Debug Button
tk.Button(root, text="Debug (q to quit)", command=debug_function).grid(row=5, column=2, padx=5, pady=20)

# Quit Button
tk.Button(root, text="Quit", command=quit_application).grid(row=5, column=0, padx=5, pady=20)

tk.Label(root, text="Not for commercial use. Individual and personal use only.\nk00gar © 2024 - https://github.com/ack00gar", font=("Arial", 10, "italic")).grid(row=6, column=0, columnspan=3, padx=5, pady=5)

root.mainloop()