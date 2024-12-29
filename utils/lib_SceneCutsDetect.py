import cv2
import numpy as np
import tqdm

def compute_histogram(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Compute the histogram for the H channel
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    return hist

def compare_histograms(hist1, hist2):
    # Use the correlation method to compare histograms
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def detect_scene_changes(video_path, crop = None, threshold=0.97, frame_start = 0, frame_end = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    prev_cut = 0
    fps_step = 2 * int(cap.get(cv2.CAP_PROP_FPS))
    total_frames_base = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_start != 0:
        if frame_end is None:
            total_frames = total_frames_base - frame_start
        else:
            total_frames = frame_end - frame_start
    else:
        total_frames = total_frames_base

    total_frames_to_parse = int(total_frames / fps_step)
    scene_changes = []
    prev_hist = None

    for frame_pos in tqdm.tqdm(range(total_frames_to_parse)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos * fps_step)
        ret, frame = cap.read()
        if not ret:
            break

        if crop == "Left":
            # only left side of the frame
            frame = frame[:, :frame.shape[1] // 2]
        elif crop == "Right":
            # only right side of the frame
            frame = frame[:, frame.shape[1] // 2:]

        current_hist = compute_histogram(frame)

        if prev_hist is not None:
            similarity = compare_histograms(prev_hist, current_hist)
            if similarity < threshold:
                # we have a scene cut in between (frame_pos - 1)*fps and frame_pos * fps
                tqdm.tqdm.write(
                    f"Raw Scene change detected at frame {frame_pos * fps_step}, time: {int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 // 60)} min {int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 % 60)} sec")
                scene_changes.append([prev_cut, frame_pos * fps_step])
                prev_cut = frame_pos * fps_step

        prev_hist = current_hist

    scenes = []
    for scene in scene_changes:
        if len(scenes) == 0:
            scenes.append(scene)
        else:
            if scene[1] - scenes[-1][1] < 5000:
                scenes[-1][1] = scene[1]
            else:
                scenes.append(scene)

    # add the last scene
    if len(scenes) == 0:
        scenes.append([0, total_frames_base])
    else:
        if scenes[-1][1] != total_frames_base:
            scenes.append([scenes[-1][1], total_frames_base])

    print(f"Found {len(scenes)} relevant scenes: {scenes}.")

    cap.release()
    return scenes


