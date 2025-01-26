import cv2
from PIL import Image

from script_generator.debug.logger import logger
from utils.lib_VideoReaderFFmpeg import VideoReaderFFmpeg
from ultralytics import YOLO

def save_frames_as_gif(frames, output_path, duration=100, loop=0):
    #frames = [Image.fromarray(frame) for frame in frames]
    #frames[0].
    #for i in range(len(frames)):
    frames[0].save(output_path,
                  save_all=True,
                  append_images=frames[1:],
                  duration=duration,
                  loop=loop,
                  optimize=True)

    logger.info(f"GIF saved to {output_path}")


# Example usage
if __name__ == "__main__":

    frame_id= 170000 #28380 #28321
    duration = 5
    # Replace with your video paths and YOLO models
    video = []
    video.append("/Users/k00gar/Downloads/OrgyVR -  Linda Lan, Lulu Chu, Ember Snow, Nicole Doshi - The Pussycat Girls.mp4")

    models = []
    models.append("../models/k00gar-11n-200ep-best.mlpackage")
    models.append("../models/k00gar-obb-11n-92ep.mlpackage")
    #models.append("models/k00gar-11s-198ep-best.mlpackage")
    #models.append("models/k00gar-11m-134ep-best.mlpackage")

    """
    yolo_instance = YOLO(models[1])

    for video_path in video:
        # read video until model finds an item of class == 'penis'
        cap = VideoReaderFFmpeg(video_path, is_vr=True)
        # position at frame 32000
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        while True:
            ret, frame = cap.read()
            results = yolo_instance(frame, conf=0.1)
            for result in results:
                print(f"@{frame_id} - Results for model {models[1]} and video {video_path}")
                #print(f"@{frame_id} Results: {result.obb}")
                obb = result.obb
                class_values = obb.cls
                print(f"Classes: {class_values}")
                if 0 in class_values:
                    print(f"Found penis at frame {frame_id}")
                    cap.release()
                    break
            frame_id += 1

    """
    cap = VideoReaderFFmpeg(video[0], is_vr=True)
    # position at frame 32000
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    fps = cap.get(cv2.CAP_PROP_FPS)
    yolo_instance = YOLO(models[1])
    frames = []
    i = 0

    while i < duration * int(fps):
        ret, frame = cap.read()
        results = yolo_instance(frame, conf=0.4, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        if i % 3 == 0:
            # divide size by 2
            annotated_frame = cv2.resize(annotated_frame, (annotated_frame.shape[1] // 3, annotated_frame.shape[0] // 3))
            img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
        i += 1
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    save_frames_as_gif(frames, video[0].replace(".mp4", ".gif"), duration=duration, loop=0)


    """
    i = 0
    for model in models:
        for video_path in video:
            #cap = cv2.VideoCapture(video_path)
            cap = VideoReaderFFmpeg(video_path, is_vr=True)
            # position at frame 32000
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            #frame = frame[:, frame.shape[1] // 6:frame.shape[1] // 3, :]
            # Load the YOLO model
            yolo_instance = YOLO(model)
            results = yolo_instance(frame, conf=0.1, save=True)
            # save the result as an image

            for result in results:
                print(f"Results for model {model} and video {video_path}")
                result.show()
                result.save(f"/Users/k00gar/Downloads/test{i}.png")
            i+=1
                
    #"""
