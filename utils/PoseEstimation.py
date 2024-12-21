from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict

video_path = "/Users/k00gar/Downloads/Katrina Jade [ADD] Oct 16, 2024R_6kvr265_reenc.mp4"

# Load a model
model = YOLO("yolo11l-pose.pt")

timestamp_min = 35
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
out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1] // 2, frame.shape[0]))


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # only half left of the frame
        frame = frame[:, :frame.shape[1] // 2, :]

        middle_x_frame = frame.shape[1] // 2

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)


        # frame = results[0].plot()

        print(results)
        keypoints = results[0].keypoints.cpu()
        print(f"keypoints: {keypoints}")
        print("next : ")
        keypoints_list = keypoints.xy.cpu().tolist()
        print(f"keypoints list: {keypoints_list}")
        left_hip = keypoints_list[0][11]
        right_hip = keypoints_list[0][12]

        mid_point = [middle_x_frame, (left_hip[1]+ right_hip[1])/2]

        cv2.circle(frame, (int(left_hip[0]), int(left_hip[1])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_hip[0]), int(right_hip[1])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(middle_x_frame), int(mid_point[1])), 2, (0, 0, 0), -1)

        track = track_history[1]

        track.append((int(middle_x_frame), int(mid_point[1])))  # x, y center point
        if len(track) > 5:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

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