from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict
from utils.config import class_names, class_priority_order, class_reverse_match, class_colors

#video_path = "/Users/k00gar/Downloads/Katrina Jade [ADD] Oct 16, 2024R_6kvr265_reenc.mp4"
video_path = "/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4"

# Load a model
pose_model = YOLO("yolo11x-pose.pt")
detect_model = YOLO("../models/k00gar-11n-200ep-best.mlpackage")

timestamp_min = 15
timestamp_sec = 0

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)


frame_pos = timestamp_min * 60 * fps + timestamp_sec * fps
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
"""
ret, frame = cap.read()


results = model(frame)
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
"""

# Store the track history
track_history = defaultdict(lambda: [])

success, frame = cap.read()

output_path = video_path.replace(".mp4", "_posedemo.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
#out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1] // 2, frame.shape[0]))
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



        # print(pose_results)
        keypoints = pose_results[0].keypoints.cpu()
        #print(f"keypoints: {keypoints}")
        # print("next : ")
        keypoints_list = keypoints.xy.cpu().tolist()
        #print(f"keypoints list: {keypoints_list}")
        left_hip = keypoints_list[0][11]
        right_hip = keypoints_list[0][12]
        left_wrist = keypoints_list[0][9]
        right_wrist = keypoints_list[0][10]

        mid_point = [middle_x_frame, (left_hip[1]+ right_hip[1])/2]

        track = track_history[1]

        track.append((int(middle_x_frame), int(mid_point[1])))  # x, y center point
        if len(track) > 5:  # retain 90 tracks for 90 frames
            track.pop(0)

        #cv2.circle(frame, (int(left_hip[0]), int(left_hip[1])), 5, (0, 255, 0), -1)
        #cv2.circle(frame, (int(right_hip[0]), int(right_hip[1])), 5, (0, 0, 255), -1)
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
            if class_reverse_match.get(int(cls), 'unknown') == "hand":
                hand_counter += 1

        if hand_counter < 2:
            print(f"WARNING: {hand_counter} hand detected")


        frame = np.hstack((frame_1, frame_2))

        """
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        keypoints = results[0].keypoints.cpu()

        # Get the track history for each keypoint 11 and 12
        track_history_11 = track_history[11]
        track_history_12 = track_history[12]

        # Visualize the results on the frame, plotting only keypoints 11 and 12
        annotated_frame = results[0].plot(keypoints=[11, 12])
        #annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        """
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