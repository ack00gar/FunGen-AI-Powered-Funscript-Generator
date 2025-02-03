from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict
from script_generator.constants import CLASS_REVERSE_MATCH
from script_generator.debug.logger import log

#video_path = "/Users/k00gar/Downloads/Katrina Jade [ADD] Oct 16, 2024R_6kvr265_reenc.mp4"
#video_path = "/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4"
video_path = "/Users/k00gar/Downloads/730-czechvr-3d-7680x3840-60fps-oculusrift_uhq_h265.mp4"

# Load a model
#pose_model = YOLO("../models/yolo11x-pose.mlpackage")
pose_model = YOLO("../models/yolo11n-pose.mlpackage")
detect_model = YOLO("../models/k00gar-11n-200ep-best.mlpackage")

timestamp_min = 25 #15
timestamp_sec = 0

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)


frame_pos = timestamp_min * 60 * fps + timestamp_sec * fps
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

# Store the track history
track_history = defaultdict(lambda: [])

success, frame = cap.read()

output_path = video_path.replace(".mp4", "_posedemo.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # only half left of the frame
        frame = frame[:, :frame.shape[1] // 2, :]

        middle_x_frame = frame.shape[1] // 2

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        pose_results = pose_model.track(frame, persist=True, verbose=False)
        detect_results = detect_model.track(frame, persist=True, verbose=False)

        frame_1 = pose_results[0].plot()

        # display the results of detect model on the frame also
        frame_2 = detect_results[0].plot("hand")

        if pose_results[0].boxes.id is not None:

            keypoints = pose_results[0].keypoints.cpu()
            keypoints_list = keypoints.xy.cpu().tolist()
            left_hip = keypoints_list[0][11]
            right_hip = keypoints_list[0][12]
            left_wrist = keypoints_list[0][9]
            right_wrist = keypoints_list[0][10]

            mid_point = [middle_x_frame, (left_hip[1]+ right_hip[1])/2]

            track = track_history[1]

            track.append((int(middle_x_frame), int(mid_point[1])))  # x, y center point
            if len(track) > 5:  # retain 90 tracks for 90 frames
                track.pop(0)

            cv2.circle(frame, (int(middle_x_frame), int(mid_point[1])), 2, (0, 0, 0), -1)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_1, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            track = track_history[2]

            track.append((int(left_wrist[0]), int(left_wrist[1])))  # x, y center point
            if len(track) > 5:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_2 - 1, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            track = track_history[3]

            track.append((int(right_wrist[0]), int(right_wrist[1])))  # x, y center point
            if len(track) > 5:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_2 - 1, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        hand_counter = 0

        boxes = detect_results[0].boxes.xywh.cpu()
        classes = detect_results[0].boxes.cls.cpu().tolist()
        confs = detect_results[0].boxes.conf.cpu().tolist()

        # if less than 2 boxes of class "hand" in detect_results, print a message
        for cls, conf, box in zip(classes, confs, boxes):
            if CLASS_REVERSE_MATCH.get(int(cls), 'unknown') == "hand":
                hand_counter += 1

        if hand_counter < 2:
            log.warning(f"WARNING: {hand_counter} hand detected")

        frame = np.hstack((frame_1, frame_2))

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)

        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()