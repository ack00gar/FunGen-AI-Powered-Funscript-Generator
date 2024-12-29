import cv2
from ultralytics import YOLO

# Example usage
if __name__ == "__main__":

    frame_id=32900
    # Replace with your video paths and YOLO models
    video = []
    video.append("/Users/k00gar/Downloads/ARPorn_Sasha Tatcha_Fit and Fired Up_4000p_8K_original_FISHEYE190_alpha_orig.mp4")

    models = []
    models.append("models/k00gar-11n-200ep-best.mlpackage")
    models.append("models/k00gar-11s-198ep-best.mlpackage")
    models.append("models/k00gar-11m-134ep-best.mlpackage")

    for model in models:
        for video_path in video:
            cap = cv2.VideoCapture(video_path)
            # position at frame 32000
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            frame = frame[:, frame.shape[1] // 6:frame.shape[1] // 3, :]
            # Load the YOLO model
            yolo_instance = YOLO(model)
            results = yolo_instance(frame)
            for result in results:
                print(f"Results for model {model} and video {video_path}")
                result.show()
