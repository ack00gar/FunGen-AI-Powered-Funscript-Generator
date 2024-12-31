from ultralytics import YOLO
import cv2
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg
import numpy as np

# Paths to the model and input image
model_file = "../models/k00gar-11n-200ep-best.mlpackage"

#video_path = video = "/Users/k00gar/Downloads/EBVR-093-C.mp4"
video_path = video = "/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4"
#video_path = video = "/Users/k00gar/Downloads/730-czechvr-3d-7680x3840-60fps-oculusrift_uhq_h265.mp4"

frame_id = 79080  # 21*60*60

output_path_1 = video_path[:-4] + "_" + str(frame_id) + "_ffmpeg_unwarped.jpg"  # Save location for the annotated frame
output_path_2 = video_path[:-4] + "_" + str(frame_id) + "_original.jpg"
output_path_3 = video_path[:-4] + "_" + str(frame_id) + "_results.jpg"

cap1 = VideoReaderFFmpeg(video_path, is_VR=True)
cap2 = VideoReaderFFmpeg(video_path, is_VR=True, unwarp=False)

cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

# we will only keep left half of frame2
# frame2 = frame2[:, :frame2.shape[1] // 2, :]

cap1.release()
cap2.release()

# Load the YOLO model
model = YOLO(model_file, task="detect")

# Perform inference
results = model.predict(frame1)

# Annotate the frame
annotated_frame_1 = results[0].plot()
cv2.imwrite(output_path_1, annotated_frame_1)

# Perform inference
results = model.predict(frame2)

# Annotate the frame
annotated_frame_2 = results[0].plot()

# paste both image side by side
annotated_frame_3 = np.concatenate((annotated_frame_2, annotated_frame_1), axis=1)

# Display the annotated frame
cv2.imshow("Concatenated result", annotated_frame_3)
cv2.imwrite(output_path_3, annotated_frame_3)

# Wait for a key press to close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()